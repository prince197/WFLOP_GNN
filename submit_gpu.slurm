#!/bin/bash
#SBATCH --job-name=WFLOP_GPU
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gpu_output_%j.out
#SBATCH --error=logs/gpu_error_%j.err

# ---------------------------------------------------------------------------
# NOTES
#
# --partition and --gres are cluster-specific. On CESGA/FinisTerrae III the
# GPU partition and A100 request are typically:
#     #SBATCH --partition=gpu
#     #SBATCH --gres=gpu:a100:1
# Check with `sinfo -o "%P %G"` before submitting.
#
# Only ONE GPU is requested. The campaign is a sequence of dependent
# iterations; a second GPU would sit idle unless you split the case grid
# across two jobs (see the bottom of this file).
#
# --cpus-per-task matters even on a GPU job: the runner distributes the
# (radius, turbines, algorithm) groups over worker processes, and on the CPU
# backend it uses every core the job is given. Ask for as many as the queue
# will grant; the code adapts to whatever it actually receives.
#
# This job runs BOTH wind data sets, one after the other, so the wall clock
# covers two full campaigns. 24 hours is generous - the batched code needs
# ~2.3 hours per data set on a single CPU core, so a GPU should finish far
# inside it. If the partition rejects 24 hours, lower it: sbatch says so at
# submission, and resume makes a wall-clock timeout harmless anyway - the
# checkpoint is per data set, so resubmitting continues where it stopped.
#
# IMPORTANT: SLURM does NOT create the directory in --output. Run
#     mkdir -p logs results curves
# once, in this directory, before the first sbatch.
# ---------------------------------------------------------------------------

echo "==========================================="
echo "WFLOP GPU JOB"
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "Date   : $(date)"
echo "==========================================="

module purge
module load python
# module load cuda cudnn          # uncomment / adjust to your site's modules

nvidia-smi || echo "WARNING: nvidia-smi not found - is this really a GPU node?"
python --version

# Thread counts are deliberately NOT set here. run_experiments_gpu.py reads the
# job's CPU affinity mask, decides how many worker processes to start, and then
# sets OMP/MKL/OpenBLAS threads to cores // workers so the cores are used once
# rather than oversubscribed. Setting them here would override that and pin the
# job to a single thread per process.
#
# To override the worker count (e.g. if a GNN worker runs out of memory):
# export WFLOP_WORKERS=8

# Fail loudly rather than silently falling back to the CPU on a GPU job.
export WFLOP_BACKEND=gpu
export WFLOP_DTYPE=float64
export WFLOP_MEM_BUDGET_MB=2048

# Wind data sets to run, in order. Both are done in ONE job: data set 1 runs to
# completion and its results are written and sealed before data set 2 starts.
# Each data set keeps its own results file, its own checkpoint and its own curve
# files, so they cannot contaminate each other and a resume picks up wherever
# the job stopped. Set to "1" or "2" alone to run only one.
WFLOP_DATASETS="${WFLOP_DATASETS:-1 2}"

# The GNN family costs orders of magnitude more than the rest. Consider
# splitting the campaign into two jobs, e.g.
#   export WFLOP_ALGOS=GA,PSO,DE,GWO,BBO,SSA,LXSSA,QASSA,ACO,PF
#   export WFLOP_ALGOS=GNNLXSSA,GNNQASSA,GNNLXSSA_UQ,GNNQASSA_UQ
# and give the second one a longer wall-clock limit.
# export WFLOP_ALGOS=...

mkdir -p logs results curves layouts

echo "--- validation -------------------------------------------"
python validate_gpu.py || { echo "VALIDATION FAILED - stopping"; exit 1; }

# ---------------------------------------------------------------------------
# ONE DATA SET AT A TIME, IN ORDER
#
# Campaign then statistics, data set by data set, rather than both campaigns
# followed by both analyses. The results of data set 1 are therefore complete,
# sealed and analysed before data set 2 starts - so if the job hits the wall
# clock half way through, what finished is finished, and resubmitting resumes
# from the checkpoint instead of starting over.
# ---------------------------------------------------------------------------
for DS in $WFLOP_DATASETS; do
    echo "==========================================================="
    echo "WIND DATA SET $DS"
    echo "==========================================================="

    echo "--- campaign, wind data set $DS --------------------------"
    WFLOP_DATASET=$DS python run_experiments_gpu.py

    echo "--- statistics, wind data set $DS ------------------------"
    # One campaign feeds all four papers; only the ANALYSIS is per paper, so
    # the same raw results file is analysed four times with a different
    # comparison set and a different pre-specified reference method each time:
    #
    #     --paper 1   GNN-LX-SSA      vs GA PSO DE ACO PF BBO GWO SSA LX-SSA
    #     --paper 2   GNN-QA-SSA      vs GA PSO DE ACO PF BBO GWO SSA QA-SSA
    #     --paper 3   GNN-LX-SSA-UQ   vs the paper-1 set + GNN-LX-SSA
    #     --paper 4   GNN-QA-SSA-UQ   vs the paper-2 set + GNN-QA-SSA
    #
    # report.py refuses to run without --paper or --reference: falling back to
    # the best-ranked algorithm would be selection on the outcome. Each call
    # writes its own WFLOP_Report_ds<D>_paper<N>.xlsx, so they do not overwrite
    # each other. The statistics are computed WITHIN a data set and are never
    # pooled across the two: they are different wind regimes, not replications.
    for P in 1 2 3 4; do
        python report.py --dataset $DS --paper $P
    done
done

echo "--- combined raw results ---------------------------------"
# Both data sets side by side in one workbook, one sheet each
# (RawResults_DS1, RawResults_DS2). Statistics stay in the per-paper
# workbooks; this is a transport step, written after the seals are verified.
python combine_results.py --datasets $WFLOP_DATASETS

# No figure step: turbine coordinates live in the Coordinates column of the
# results CSV and the convergence curves in curves/Conv_ds*_R*_T*.npz, so the
# plots are yours to draw. report.py writes the numbers behind them.

echo "Finished."
date

# ---------------------------------------------------------------------------
# SPLITTING ACROSS SEVERAL GPUs
#
# The checkpoint makes this safe as long as each job owns a disjoint slice of
# the grid. Add a filter on the radius to run_experiments_gpu.py, e.g.
#
#     FARM_CASES = {int(os.environ["WFLOP_RADIUS"]): FARM_CASES_ALL[...]}
#
# then submit three jobs with WFLOP_RADIUS=500 / 750 / 1000 and separate
# checkpoint paths. Do not point two concurrent jobs at the same checkpoint.
# ---------------------------------------------------------------------------
