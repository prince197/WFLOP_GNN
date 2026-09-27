#!/bin/bash
#SBATCH --job-name=WFLOP_CPU
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/cpu_output_%j.out
#SBATCH --error=logs/cpu_error_%j.err

# ---------------------------------------------------------------------------
# CPU campaign - no GPU required.
#
# --ntasks=1 with --cpus-per-task=64 is the right shape: ONE python process
# starts the worker pool itself and spreads the (radius, turbines, algorithm)
# groups over all 64 cores. Asking for --ntasks=64 would request 64 separate
# tasks that this script never launches, leaving 63 idle.
#
# Raise --cpus-per-task to a whole node if the queue allows; the runner reads
# the CPU affinity mask, so it uses exactly what SLURM grants, no more and no
# less. Nothing is pinned to a particular core.
#
# Thread counts are deliberately NOT exported here - the runner sets
# OMP/MKL/OpenBLAS threads to cores // workers so the cores are used once
# rather than oversubscribed.
#
# If a GNN worker runs out of memory, cap the pool:
#     export WFLOP_WORKERS=16
# ---------------------------------------------------------------------------

echo "==========================================="
echo "WFLOP CPU JOB"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "Cores     : $SLURM_CPUS_PER_TASK"
echo "Date      : $(date)"
echo "==========================================="

module purge
module load python
python --version

export WFLOP_BACKEND=cpu
export WFLOP_DTYPE=float64
export WFLOP_MEM_BUDGET_MB=1024

mkdir -p logs results curves

echo "--- validation -------------------------------------------"
python validate_gpu.py || { echo "VALIDATION FAILED - stopping"; exit 1; }

# Both wind data sets, one after the other. Each writes its own results file
# and its own checkpoint, so they cannot contaminate each other.
for DS in 1 2; do
    echo "--- campaign, wind data set $DS --------------------------"
    WFLOP_DATASET=$DS python run_experiments_gpu.py
    echo "--- statistics, wind data set $DS ------------------------"
    # Same four-paper reporting as submit_gpu.slurm: report.py requires a
    # pre-specified reference (--paper or --reference) and would exit 1 without
    # one. Each call writes WFLOP_Report_ds<D>_paper<N>.xlsx.
    python report.py --dataset $DS --paper 1
    python report.py --dataset $DS --paper 2
    python report.py --dataset $DS --paper 3
    python report.py --dataset $DS --paper 4
done

echo "--- combined raw results ---------------------------------"
# Both data sets side by side in one workbook, one sheet each
# (RawResults_DS1, RawResults_DS2). Statistics stay in the per-paper
# workbooks; this is a transport step, written after the seals are verified.
python combine_results.py --datasets 1 2

echo "Finished."
date

# ---------------------------------------------------------------------------
# SPLITTING THE GNN FAMILY INTO ITS OWN JOB
#
# The four GNN optimizers cost far more than the other ten and hold much larger
# tensors. A common split:
#
#   export WFLOP_ALGOS=GA,PSO,DE,GWO,BBO,SSA,LXSSA,QASSA,ACO,PF   # short job
#   export WFLOP_ALGOS=GNNLXSSA,GNNQASSA,GNNLXSSA_UQ,GNNQASSA_UQ  # long job
#
# The checkpoint makes this safe: the second job skips what the first finished
# and adds its algorithms to the same results file and the same curve files.
# ---------------------------------------------------------------------------
