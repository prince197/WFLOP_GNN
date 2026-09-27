"""
============================================================
WFLOP EXPERIMENT RUNNER - GPU / batched
============================================================

Same campaign as run_experiments.py:

    500 m : T2-T10     750 m : T2-T14      1000 m : T2-T18
    39 cases x 30 runs x 14 algorithms  =  16,380 runs

but organised for a GPU. The unit of work is no longer one
(radius, turbines, seed, algorithm) task on one CPU core; it is
one (radius, turbines, algorithm) task with all 30 seeds advanced
in lockstep, so each objective call evaluates 30 x 30 = 900
layouts at once.

    python run_experiments_gpu.py                # auto-detect GPU
    WFLOP_BACKEND=cpu python run_experiments_gpu.py   # force CPU

PARALLELISM - TWO LEVELS, BOTH ON BY DEFAULT
--------------------------------------------
1. INSIDE a group: 30 seeds x 30 individuals are advanced in
   lockstep, so one objective call evaluates 900 layouts. This is
   the level that suits a GPU.
2. ACROSS groups: the (radius, turbines, algorithm) groups are
   independent, so they are distributed over worker processes.

By default the runner uses EVERY core the job actually has - taken
from the CPU affinity mask, so it honours whatever SLURM granted,
without pinning anything. Set WFLOP_WORKERS to override. BLAS
threads per worker are set automatically to cores // workers, so
the cores are used once, not oversubscribed.

On a GPU the default is one worker per visible GPU; the batching
inside a group already fills the device, and extra processes on one
GPU only contend for it.

Resume works at (radius, turbines, algorithm) granularity: the
checkpoint is read on startup and completed groups are skipped.

Grid: 500 m carries 2-10 turbines, 750 m carries 2-14 and 1000 m
carries 2-18, i.e. 39 cases per wind data set.

Output schema:

    Radius, Turbines, Seed, Algorithm, WakeLoss, EnergyProduction,
    Runtime, Evaluations, SurrogateEvaluations, Dataset, Coordinates

TURBINE COORDINATES
-------------------
The best layout of every run is stored IN the results file, in the
`Coordinates` column, as "x1 y1;x2 y2;...;xN yN" (metres, at full
round-trip precision, no commas so the CSV needs no quoting). No
layouts/*.npy files are
written any more - one CSV holds everything, and you can plot from it
yourself. `parse_coordinates()` below turns a cell back into an (N,2)
array. Convergence curves go to one file per case (see below).

RUNTIME COLUMNS
---------------
`Runtime` is the batched group time divided by NUM_RUNS: 30 seeds
share every kernel launch, so per-seed time is not observable. It is
fair for comparing algorithms on this hardware, but it is NOT
comparable to a sequential implementation.

`RuntimeSingleRun` fixes that. With WFLOP_TIMING=1 the runner also
executes ONE run on its own per (case, algorithm) and times it, which
IS a like-for-like wall time. It costs about a thirtieth of the
campaign. Without the flag the column is NaN.

`Evaluations` is exact either way, and is the budget measure that
belongs in a paper.
============================================================
"""

import hashlib
import json
import os
import csv
import time
import zlib
import importlib.util

# ---------------------------------------------------------------------------
# CORES, WORKERS AND THREADS  -  configured BEFORE numpy is imported, because
# the BLAS thread pools read these variables once, at import time.
# ---------------------------------------------------------------------------


def _available_cores():
    """Cores this process may actually use.

    sched_getaffinity reflects cgroup / taskset / SLURM binding, so it is the
    honest number on a shared HPC node; SLURM_CPUS_PER_TASK is used when the
    affinity mask is unavailable (e.g. macOS).
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        pass
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        v = os.environ.get(var)
        if v and v.isdigit() and int(v) > 0:
            return int(v)
    return max(1, os.cpu_count() or 1)


def _visible_gpus():
    """How many GPUs this job may use. Probes in order of reliability and
    never creates a CUDA context, so it is safe before fork."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis is not None:
        return len([x for x in vis.split(",") if x.strip() != ""])
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, timeout=10)
        n = len([l for l in out.stdout.splitlines() if l.strip()])
        if n:
            return n
    except Exception:                        # noqa: BLE001
        pass
    try:                                     # driver query only, no context
        import cupy
        return max(1, int(cupy.cuda.runtime.getDeviceCount()))
    except Exception:                        # noqa: BLE001
        return 0


CORES = _available_cores()

_MODE = os.environ.get("WFLOP_BACKEND", "auto").lower()
_CUPY_PRESENT = importlib.util.find_spec("cupy") is not None
_N_GPUS = _visible_gpus() if _MODE in ("auto", "gpu") else 0
# The runner must agree with backend.py about which device will be used, or the
# worker count is wrong: "auto" on a GPU node would otherwise start one process
# per CPU core, all contending for the same card.
_GPU_MODE = (_MODE == "gpu") or (_MODE == "auto" and _CUPY_PRESENT and _N_GPUS > 0)

if os.environ.get("WFLOP_WORKERS"):
    WORKERS = max(1, int(os.environ["WFLOP_WORKERS"]))
elif _GPU_MODE:
    WORKERS = max(1, _N_GPUS)                # one process per GPU
else:
    WORKERS = CORES                          # CPU: use everything we are given

# Give each worker its fair share of the cores; with one worker that is all of
# them, so a single-process run is still fully threaded. Never pin to core 0.
THREADS_PER_WORKER = max(1, CORES // max(1, WORKERS))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, str(THREADS_PER_WORKER))

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from backend import xp, asnumpy, device_info, USING_GPU
from objective_gpu import make_objective, energy_production_batch
from algorithms_gpu import build
import gnn_algorithms_gpu  # noqa: F401  (registers the four GNN optimizers)

# ------------------------------------------------------------------
FARM_CASES = {500: range(2, 11), 750: range(2, 15), 1000: range(2, 19)}
NUM_RUNS = 30
POP = 30
ITER = 100
ALGORITHM_NAMES = ["GA", "PSO", "DE", "GWO", "BBO", "SSA", "LXSSA", "QASSA",
                   "ACO", "PF",
                   "GNNLXSSA", "GNNQASSA", "GNNLXSSA_UQ", "GNNQASSA_UQ"]

# WFLOP_ALGOS="GA,SSA,GNNLXSSA" runs a subset (the GNN family is far more
# expensive than the rest, so you will often want to split the campaign).
_sel = os.environ.get("WFLOP_ALGOS")
if _sel:
    want = [a.strip() for a in _sel.split(",") if a.strip()]
    unknown = [a for a in want if a not in ALGORITHM_NAMES]
    if unknown:
        raise SystemExit(f"unknown algorithm(s) in WFLOP_ALGOS: {unknown}")
    ALGORITHM_NAMES = want

# Wind data set: 1 = Data Set I, 2 = Data Set II
DATASET = int(os.environ.get("WFLOP_DATASET", 1))
if DATASET not in (1, 2):
    raise SystemExit(f"WFLOP_DATASET must be 1 or 2 (got {DATASET})")

# WFLOP_SMOKE=1 runs a two-minute version of the campaign so you can prove the
# whole pipeline works on the GPU node before queuing the real job.
if os.environ.get("WFLOP_SMOKE") == "1":
    FARM_CASES = {500: range(4, 6)}
    NUM_RUNS, POP, ITER = 4, 8, 10

# WFLOP_CASES restricts the grid to named cases, e.g. "500:5,500:9,1000:18".
# WFLOP_RUNS overrides the seed count. Both exist so an ablation or a pilot can
# be scoped down without editing the file the campaign is frozen from.
if os.environ.get("WFLOP_CASES"):
    _sel = {}
    for _c in os.environ["WFLOP_CASES"].split(","):
        _r, _n = (int(v) for v in _c.strip().split(":"))
        _sel.setdefault(_r, []).append(_n)
    FARM_CASES = {r: sorted(v) for r, v in _sel.items()}
if os.environ.get("WFLOP_RUNS"):
    NUM_RUNS = int(os.environ["WFLOP_RUNS"])

# ---------------------------------------------------------------------------
# TWO EXPERIMENT REGIMES
# ---------------------------------------------------------------------------
# Regime A (default): fixed ITERATIONS. Every algorithm gets 100 iterations.
#   Answers "which algorithm is better per iteration", and is the right setting
#   for convergence behaviour - but the algorithms do NOT spend the same number
#   of exact objective evaluations, so it cannot support an efficiency claim.
#
# Regime B: WFLOP_BUDGET=<n> is a HARD CAP on exact objective evaluations per
#   run. Answers "which algorithm is better per evaluation", which is what an
#   efficiency claim needs.
#
#   The cap is never exceeded. Iteration counts are derived with a FLOOR, not
#   by rounding: rounding to the nearest iteration overshoots by up to half an
#   iteration's cost, which for population 30 is up to 15 evaluations - so a
#   "500-evaluation" comparison silently ran some algorithms at 510. Every
#   algorithm now gets the largest whole number of iterations whose total cost
#   is <= the budget, and spends at most the budget.
#
#   Cost model, in exact objective evaluations per run:
#
#     algorithm             mandatory (before iterating)   per iteration
#     GA PSO DE GWO BBO SSA PF   pop                       pop
#     LX-SSA, QA-SSA             pop                       2*pop
#     ACO                        1                         N + pop
#     GNN-LX/QA-SSA              pop + n_pretrain          n_exact + 2
#                                + 4*N*n_fd  (= 70 + 64N)
#     GNN-*-UQ                   same as above             data-dependent
#
#   The GNN mandatory cost GROWS WITH N - 198 at N=2, 1,222 at N=18 - because
#   the finite-difference direction labels cost 4N evaluations per sampled
#   layout. A budget smaller than the mandatory cost plus one iteration cannot
#   be honoured at all. Such a (case, algorithm) pair is marked BUDGET-
#   INFEASIBLE and skipped, and is listed in results/budget_infeasible_ds<D>
#   .csv - it is never run over budget, and never silently dropped either.
#
#   Because of that floor, a common budget for the whole N = 2..18 campaign
#   must clear 1,222 + one iteration for the GNN family. Around 3,000 is a
#   sound choice: it is feasible for every algorithm at every N, and leaves the
#   GNN family 127 iterations at N=18. Confirm with:
#
#       python test_budget.py --budget 3000
BUDGET = int(os.environ.get("WFLOP_BUDGET", 0)) or None

# Optimizer settings that enter the cost model. Kept here as named constants
# rather than repeated as literals, so the cost model and the algorithms
# cannot drift apart.
N_PRETRAIN = 40          # GNN pre-training layouts
FD_FRACTION = 0.25       # fraction of the buffer given finite-difference labels
FD_MAX = 16              # cap on those samples
MU = 0.2                 # fraction of candidates exactly evaluated per iteration


# ---------------------------------------------------------------------------
# ABLATIONS
# ---------------------------------------------------------------------------
# WFLOP_ALGO_KWARGS is a JSON object of per-algorithm keyword overrides, so an
# ablation needs no code edit. The four papers each propose a method built from
# several components, and a reviewer will ask which component earns the result.
#
#   guidance off (does the direction head matter?)
#     WFLOP_ALGO_KWARGS='{"GNNQASSA": {"lambda_g": 0.0}}'
#   single model instead of the 5-member ensemble (does the ensemble matter?)
#     WFLOP_ALGO_KWARGS='{"GNNQASSA_UQ": {"n_models": 1}}'
#   gate threshold sweep (how sensitive is the saving to sigma?)
#     WFLOP_ALGO_KWARGS='{"GNNQASSA_UQ": {"sigma_threshold": 0.02}}'
#
# run_ablations.py drives the full set required for Papers 2 and 4.
ALGO_KWARGS = json.loads(os.environ.get("WFLOP_ALGO_KWARGS", "{}"))

# A label for the ablation, so its outputs do not overwrite the main campaign's.
RUN_TAG = os.environ.get("WFLOP_TAG", "")


def mandatory_cost(alg_name, n_turb):
    """Exact evaluations spent BEFORE the iterative search begins."""
    if alg_name == "ACO":
        return 1                                  # single pheromone baseline
    if alg_name.startswith("GNN"):
        n_fd = min(FD_MAX, max(1, int(FD_FRACTION * (N_PRETRAIN + POP))))
        return POP + N_PRETRAIN + 4 * n_turb * n_fd
    return POP                                    # initial population


def per_iteration_cost(alg_name, n_turb):
    """Exact evaluations per iteration; None when data-dependent (UQ)."""
    if alg_name in ("LXSSA", "QASSA"):
        return 2 * POP
    if alg_name == "ACO":
        return n_turb + POP
    if alg_name.startswith("GNN"):
        if alg_name.endswith("_UQ"):
            return None                           # gate decides, run to run
        return int(np.ceil(MU * 2 * POP)) + 2
    return POP


def budget_cost(alg_name, n_turb, iters):
    """Total exact evaluations for `iters` iterations. UQ: minimum possible."""
    per = per_iteration_cost(alg_name, n_turb)
    return mandatory_cost(alg_name, n_turb) + iters * (1 if per is None else per)


def iterations_for(alg_name, n_turb, budget):
    """(iterations, hard_cap, status) honouring `budget` as a strict maximum.

    status is "ok", or "infeasible" when the budget cannot fund the mandatory
    start-up cost plus a single iteration. An infeasible pair is skipped and
    recorded, never run over budget.
    """
    if budget is None:
        return ITER, None, "ok"

    mand = mandatory_cost(alg_name, n_turb)
    per = per_iteration_cost(alg_name, n_turb)
    floor_per = 1 if per is None else per         # UQ spends >=1 per iteration
    if budget < mand + floor_per:
        return 0, None, "infeasible"

    if per is None:
        # Data-dependent spend. The optimizer is given the cap itself and
        # truncates its gate admissions to whatever budget is left, so it stops
        # at the cap exactly rather than after overshooting it.
        #
        # It also needs an iteration ceiling, and that ceiling must not be an
        # arbitrary number: an under-generous one would stop the gated method
        # before it had spent its budget, making it look artificially cheap
        # while denying it the search the others got. The principled value is
        # the point past which the budget cannot stretch even if the gate
        # admitted the bare minimum of one candidate per iteration - so the
        # budget, not the ceiling, is always what binds.
        ceiling = int(os.environ.get("WFLOP_UQ_MAX_ITERS", 0)) or (budget - mand)
        return max(1, ceiling), budget, "ok"

    return (budget - mand) // per, None, "ok"     # floor: never overshoot


RESULT_COLUMNS = ["Radius", "Turbines", "Seed", "Algorithm",
                  "WakeLoss", "EnergyProduction", "Runtime", "RuntimeSingleRun",
                  "Evaluations", "ObjectiveCalls", "SurrogateInferences",
                  "SurrogateTrainPasses", "Iterations", "GateAdmissionRate",
                  "Dataset", "Coordinates"]

# COUNTING COLUMNS - three different quantities, deliberately separated:
#
#   Evaluations         exact objective values the optimizer ADMITTED into its
#                       decisions. This is the algorithmic budget and the
#                       number that belongs in a paper.
#   ObjectiveCalls      exact objective computations actually PERFORMED. Equal
#                       to Evaluations for every optimizer except the two UQ
#                       variants, where the gated batch is padded to a
#                       rectangle and the padding costs a few extra columns.
#   SurrogateInferences surrogate FORWARD passes used for screening and
#                       guidance (candidates x ensemble members).
#   SurrogateTrainPasses surrogate forward passes spent TRAINING the model.
#                       Counted apart from inference because they are the cost
#                       of building the surrogate, not a substitute for an
#                       exact evaluation. Never add either surrogate count to
#                       the evaluation budget.
#   GateAdmissionRate   fraction of candidates the uncertainty gate sent for
#                       exact evaluation, averaged over iterations, per run.
#                       0 for every non-gated algorithm. This is the number
#                       behind the efficiency claim in Papers 3 and 4: it says
#                       how selective the gate actually was.

# WFLOP_TIMING=1 additionally times ONE run on its own per (case, algorithm)
# and records it as RuntimeSingleRun. `Runtime` is the batched group time
# divided by NUM_RUNS - fair between algorithms here, but not comparable to a
# sequential implementation. RuntimeSingleRun is a genuine single-run wall time
# and IS comparable. It costs about one thirtieth of the campaign.
TIME_SINGLE_RUN = os.environ.get("WFLOP_TIMING", "0") == "1"

# Turbine coordinates are written straight into the results file instead of
# one .npy per run. Format: "x1 y1;x2 y2;...;xN yN", metres.
# Read them back with:
#
#     import numpy as np, pandas as pd
#     df = pd.read_csv("results/RawResults_ds1.csv")
#     xy = parse_coordinates(df.loc[0, "Coordinates"])      # (N, 2) array
#
# PRECISION: values are written with repr(), i.e. the shortest string that
# converts back to exactly the same float. Rounding would be dangerous here:
# the constraint penalty has a coefficient of 1e10, and the GNN repair step
# leaves turbines sitting about 1e-6 m outside the spacing limit, so a layout
# rounded to a few decimals could re-evaluate with a huge spurious penalty.
# Full precision costs a few extra characters and removes that risk entirely.


def format_coordinates(xy):
    """xy : (N,2) array -> 'x1 y1;x2 y2;...' (no commas, so no CSV quoting)."""
    return ";".join(f"{float(x)!r} {float(y)!r}" for x, y in xy)


def parse_coordinates(text):
    """Inverse of format_coordinates: 'x1 y1;x2 y2;...' -> (N,2) array."""
    return np.array([[float(v) for v in pair.split()]
                     for pair in str(text).split(";")], dtype=float)

# WFLOP_TAG keeps an ablation's outputs beside the main campaign's instead of
# on top of them: every path below picks up the tag, so an ablation can never
# overwrite the frozen results the papers are built from.
_TAG = f"_{RUN_TAG}" if RUN_TAG else ""
CHECKPOINT_PATH = f"results/RawResults_checkpoint_ds{DATASET}{_TAG}.csv"
FINAL_PATH = f"results/RawResults_ds{DATASET}{_TAG}.csv"

# ---------------------------------------------------------------------------
# CONVERGENCE CURVES
# ---------------------------------------------------------------------------
# One file per CASE, not per run: curves/Conv_ds<D>_R<radius>_T<n>.npz
# 39 cases per data set, so 39 files per campaign and 78 for both - instead of
# tens of thousands of per-run .npy files.
#
# Each file stores, for every algorithm, the convergence curve of that
# algorithm's BEST of the 30 runs (lowest final wake loss), which is exactly
# what a "Convergence (R, T)" figure plots:
#
#     curves      (n_algorithms, iters+1)  best-so-far exact wake loss
#     algorithms  (n_algorithms,)          algorithm names, same order
#     best_seed   (n_algorithms,)          which seed produced each curve
#     best_wake   (n_algorithms,)          its final wake loss
#     radius, turbines, dataset            scalars
#
# The file is updated in place after each algorithm group finishes, so it works
# with resume and with WFLOP_ALGOS subsets. Read one with:
#
#     d = np.load("curves/Conv_ds1_R500_T10.npz", allow_pickle=False)
#     for name, curve in zip(d["algorithms"], d["curves"]):
#         plt.plot(curve, label=str(name))
#
SAVE_CURVES = os.environ.get("WFLOP_SAVE_ARTIFACTS", "1") != "0"

for d in ("results", "curves", "logs"):
    os.makedirs(d, exist_ok=True)


def curve_path(radius, n_turb):
    return f"curves/Conv_ds{DATASET}{_TAG}_R{radius}_T{n_turb}.npz"


def _pad(rows):
    """Stack curves of unequal length by holding each one's final value.

    Curves have equal length within a regime of fixed iterations, but under
    WFLOP_BUDGET the iteration count is derived per algorithm, so a case's
    curves can differ in length. Holding the final value keeps the array
    rectangular without inventing improvement.
    """
    w = max(len(r) for r in rows)
    return np.array([np.concatenate([r, np.full(w - len(r), r[-1])])
                     if len(r) < w else r for r in rows], dtype=float)


def store_best_curve(radius, n_turb, alg_name, curves, wake, axis=None):
    """One file per case. Per algorithm it stores:

        curves     the BEST run's curve (the run with the lowest wake loss)
        median     the across-run median curve,
        q25, q75   the inter-quartile band, and
        evals      cumulative exact evaluations at each point of the best
                   run's curve,

    all over the same 30 runs. `evals` is what makes an efficiency figure
    possible: plot curves against it instead of against the iteration index
    and you get best-so-far wake loss versus exact objective evaluations,
    which is the comparison a fixed-iteration campaign cannot show. The best-of-30 curve is what the user asked
    for; the median and band are stored beside it because a best-of-30 curve
    alone is a favourable order statistic and a reviewer will ask what the
    typical run did. Storing them costs three extra rows per algorithm.
    """
    C = np.asarray(curves, dtype=float)          # (runs, iters+1)
    k = int(np.argmin(wake))
    A = (np.asarray(axis, dtype=float)[:, k] if axis is not None
         else np.arange(C.shape[1], dtype=float))

    path = curve_path(radius, n_turb)
    names, rows, med, lo, hi, ax, seeds, bests = [], [], [], [], [], [], [], []
    if os.path.exists(path):
        old = np.load(path, allow_pickle=False)
        names = [str(x) for x in old["algorithms"]]
        rows = [np.asarray(r, dtype=float) for r in old["curves"]]
        med = [np.asarray(r, dtype=float) for r in old["median"]]
        lo = [np.asarray(r, dtype=float) for r in old["q25"]]
        hi = [np.asarray(r, dtype=float) for r in old["q75"]]
        ax = [np.asarray(r, dtype=float) for r in old["evals"]]
        seeds = [int(x) for x in old["best_seed"]]
        bests = [float(x) for x in old["best_wake"]]

    entry = (alg_name, C[k], np.median(C, axis=0),
             np.percentile(C, 25, axis=0), np.percentile(C, 75, axis=0),
             A, k + 1, float(wake[k]))
    if alg_name in names:                       # re-run: replace in place
        i = names.index(alg_name)
        (names[i], rows[i], med[i], lo[i], hi[i], ax[i],
         seeds[i], bests[i]) = entry
    else:
        for lst, val in zip((names, rows, med, lo, hi, ax, seeds, bests), entry):
            lst.append(val)

    np.savez_compressed(path,
                        curves=_pad(rows), median=_pad(med),
                        q25=_pad(lo), q75=_pad(hi), evals=_pad(ax),
                        algorithms=np.array(names),
                        best_seed=np.array(seeds, dtype=int),
                        best_wake=np.array(bests, dtype=float),
                        n_runs=np.array(C.shape[0]),
                        radius=np.array(radius),
                        turbines=np.array(n_turb),
                        dataset=np.array(DATASET))


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        return [], set()
    df = pd.read_csv(CHECKPOINT_PATH)
    for col in ("Radius", "Turbines", "Seed"):
        df[col] = df[col].astype(int)
    df["Algorithm"] = df["Algorithm"].astype(str)
    rows = df[RESULT_COLUMNS].values.tolist()
    done = set(zip(df["Radius"].tolist(), df["Turbines"].tolist(),
                   df["Algorithm"].tolist()))
    return rows, done


def append_rows(rows):
    exists = os.path.exists(CHECKPOINT_PATH)
    with open(CHECKPOINT_PATH, "a", newline="") as fh:
        w = csv.writer(fh)
        if not exists:
            w.writerow(RESULT_COLUMNS)
        w.writerows(rows)


def group_seed_of(radius, n_turb, alg_name):
    """Deterministic per-group seed.

    Python's hash() is salted per process, so it must NOT be used here: with
    several worker processes it would not even be consistent within one job,
    let alone across resumes.
    """
    return zlib.crc32(f"{radius}|{n_turb}|{alg_name}".encode()) % (2 ** 31)


def _init_worker():
    """Bind each worker to its own GPU when several are visible."""
    if not USING_GPU:
        return
    try:
        import multiprocessing as mp
        ident = mp.current_process()._identity
        idx = (ident[0] - 1) if ident else 0
        xp.cuda.Device(idx % max(1, _visible_gpus())).use()
    except Exception:                        # noqa: BLE001
        pass                                 # single GPU: nothing to bind


def run_group(group):
    """One (radius, turbines, algorithm) group: 30 seeds in lockstep.

    Runs in a worker process. Returns everything the parent needs, so that all
    file writing stays in the parent and no two processes touch a file.
    """
    radius, n_turb, alg_name = group
    dim = 2 * n_turb

    iters, cap, status = iterations_for(alg_name, n_turb, BUDGET)
    if status == "infeasible":
        # Budget smaller than this algorithm's mandatory start-up cost plus one
        # iteration. Reported, not run: silently exceeding the cap would
        # corrupt the very comparison the fixed-budget regime exists to make.
        return group, None, None, None, 0.0, None, None, None

    f = make_objective(radius, dataset=DATASET)
    kw = dict(dataset=DATASET) if alg_name.startswith("GNN") else {}
    kw.update(ALGO_KWARGS.get(alg_name, {}))
    if cap is not None:
        kw["max_evals"] = cap
    algo = build(alg_name, **kw)

    t0 = time.perf_counter()
    best_x, best_f, curves, n_evals = algo.optimize(
        f, dim, -radius, radius, NUM_RUNS, POP, iters,
        seed=group_seed_of(radius, n_turb, alg_name))
    if USING_GPU:
        xp.cuda.runtime.deviceSynchronize()
    dt = time.perf_counter() - t0

    single_dt = float("nan")
    if TIME_SINGLE_RUN:
        solo = build(alg_name, **kw)
        t1 = time.perf_counter()
        solo.optimize(f, dim, -radius, radius, 1, POP, iters,
                      seed=group_seed_of(radius, n_turb, alg_name))
        if USING_GPU:
            xp.cuda.runtime.deviceSynchronize()
        single_dt = time.perf_counter() - t1

    ep = asnumpy(energy_production_batch(best_x, dataset=DATASET))
    bx = asnumpy(best_x).reshape(NUM_RUNS, n_turb, 2)
    bf = asnumpy(best_f)
    cv = asnumpy(curves)

    # The uncertainty-gated optimizers spend a different number of exact
    # evaluations in each run, so n_evals may be a per-run vector.
    ev = np.broadcast_to(np.atleast_1d(asnumpy(n_evals)), (NUM_RUNS,))
    oc = np.broadcast_to(
        np.atleast_1d(asnumpy(getattr(algo, "n_objective_calls", n_evals))),
        (NUM_RUNS,))
    surro = int(getattr(algo, "n_surrogate_evals", 0)) // max(1, NUM_RUNS)
    strain = int(getattr(algo, "n_surrogate_train", 0)) // max(1, NUM_RUNS)
    grate = np.broadcast_to(
        np.atleast_1d(np.asarray(getattr(algo, "gate_rate", 0.0), dtype=float)),
        (NUM_RUNS,))

    rows = [[radius, n_turb, k + 1, alg_name,
             float(bf[k]), float(ep[k]),
             dt / NUM_RUNS, single_dt, int(ev[k]), int(oc[k]), surro,
             strain, int(cv.shape[1] - 1), float(grate[k]),
             DATASET, format_coordinates(bx[k])]
            for k in range(NUM_RUNS)]

    axis = np.asarray(algo.eval_axis, dtype=float)          # (iters+1, runs)
    return group, rows, cv, bf, dt, ev, axis, calibration_bins(group, algo)


def calibration_bins(group, algo, n_bins=10):
    """Bin the gated candidates of one group by predicted ensemble spread.

    For each gated candidate the UQ optimizers keep a matched pair: the
    ensemble's predicted mean and spread, and the exact value that came back.
    Binning by spread and reporting the empirical error in each bin is what
    shows whether the uncertainty is INFORMATIVE - a gate that is cheap but
    whose sigma does not track the real error saves evaluations by luck.

    Infeasible layouts are excluded from the statistics. They carry the 1e10
    constraint penalty, so their objective is ~1e23 and no surrogate trained on
    feasible layouts can track them; leaving them in would swamp every number
    here. The count is reported instead, so the exclusion is visible.
    """
    cal = np.asarray(getattr(algo, "uq_calibration", np.zeros((0, 5))))
    if cal.size == 0:
        return None
    radius, n_turb, alg_name = group
    feas = cal[:, 4] > 0.5
    sig, err = cal[feas, 0], cal[feas, 1]
    if sig.size < n_bins * 2:
        return None

    rho = float(pd.Series(sig).corr(pd.Series(err), method="spearman"))
    edges = np.quantile(sig, np.linspace(0, 1, n_bins + 1))
    rows = []
    for i in range(n_bins):
        m = ((sig >= edges[i]) & (sig <= edges[i + 1])) if i == n_bins - 1 \
            else ((sig >= edges[i]) & (sig < edges[i + 1]))
        if not m.any():
            continue
        rows.append([radius, n_turb, alg_name, DATASET, i + 1,
                     float(edges[i]), float(edges[i + 1]), int(m.sum()),
                     float(sig[m].mean()), float(err[m].mean()),
                     float(np.median(err[m])), rho,
                     int(cal.shape[0]), int((~feas).sum())])
    return rows


CALIBRATION_COLUMNS = ["Radius", "Turbines", "Algorithm", "Dataset",
                       "SigmaBin", "SigmaLow", "SigmaHigh", "N",
                       "MeanSigma", "MeanAbsError", "MedianAbsError",
                       "SpearmanSigmaError", "GatedPairs", "InfeasibleExcluded"]


def write_manifest():
    """Provenance record for the campaign, written beside the results.

    A results CSV on its own is not reproducible: it does not say which code
    produced it, with which settings, on which backend, or at what numerical
    precision. This file closes that gap - it hashes every source file, so a
    later reader can prove the results and the code match.
    """
    import json, platform
    src = {}
    here = os.path.dirname(os.path.abspath(__file__))
    for fn in sorted(f for f in os.listdir(here) if f.endswith(".py")):
        with open(os.path.join(here, fn), "rb") as fh:
            src[fn] = hashlib.sha256(fh.read()).hexdigest()[:16]

    try:
        import cupy
        cupy_ver = cupy.__version__
    except Exception:
        cupy_ver = None

    # Everything the checklist requires frozen before the final campaign, in
    # one machine-readable place: algorithm lists, population, iteration and
    # budget rules, optimizer parameters, GNN architecture, UQ ensemble and
    # threshold, the seed list, the farm cases, the wind arrays, the objective
    # constants, the statistical protocol and the source hashes.
    import algorithms_gpu as _A
    import objective_gpu as _O
    try:
        import gnn_algorithms_gpu as _G                       # noqa: F401
    except Exception:
        pass

    def _arr_hash(a):
        return hashlib.sha256(np.ascontiguousarray(
            asnumpy(a), dtype=np.float64).tobytes()).hexdigest()[:16]

    params = {k: dict(v) for k, v in sorted(_A.DEFAULT_KWARGS.items())
              if k in ALGORITHM_NAMES}
    cases = [(r, n) for r, rng_ in FARM_CASES.items() for n in rng_]

    manifest = {
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": DATASET,
        "regime": ("fixed_budget" if BUDGET else "fixed_iterations"),
        "budget": BUDGET,
        "budget_rule": ("hard cap; iterations = floor((budget - mandatory) / "
                        "per_iteration); a pair whose mandatory cost plus one "
                        "iteration exceeds the budget is marked infeasible and "
                        "skipped, never run over budget"),
        "settings": {"runs": NUM_RUNS, "pop": POP, "iters": ITER,
                     "cases": {str(k): [min(v), max(v)]
                               for k, v in FARM_CASES.items()},
                     "n_cases": len(cases),
                     "algorithms": ALGORITHM_NAMES},
        "optimizer_parameters": params,
        "algo_kwarg_overrides": ALGO_KWARGS,
        "cost_model": {
            "n_pretrain": N_PRETRAIN, "fd_fraction": FD_FRACTION,
            "fd_max": FD_MAX, "mu": MU,
            "mandatory": {a: {str(n): mandatory_cost(a, n)
                              for n in (2, 9, 18)} for a in ALGORITHM_NAMES},
            "per_iteration": {a: {str(n): per_iteration_cost(a, n)
                                  for n in (2, 9, 18)}
                              for a in ALGORITHM_NAMES}},
        "surrogate": {
            "hidden": 64, "message_passing_layers": 3,
            "dtype": "float32",
            "node_features": 7, "edge_features": 5,
            "uq_ensemble_members": 5,
            "uq_sigma_threshold": _A.DEFAULT_KWARGS.get(
                "GNNLXSSA_UQ", {}).get("sigma_threshold"),
            "note": "values as declared in DEFAULT_KWARGS; overrides above "
                    "take precedence"},
        "seeding": {"scheme": "crc32(radius|turbines|algorithm) -> group seed",
                    "runs_per_group": NUM_RUNS,
                    "seed_list": list(range(1, NUM_RUNS + 1)),
                    "group_seeds": {f"{r}_{n}_{a}": group_seed_of(r, n, a)
                                    for (r, n) in cases[:3]
                                    for a in ALGORITHM_NAMES[:3]},
                    "note": "run k of a group is row k of the batch; the same "
                            "group seed reproduces all runs exactly. A sample "
                            "of group seeds is listed so a reader can verify "
                            "the derivation without running anything."},
        "objective": {
            "rotor_radius": _O.R, "wake_decay": _O.K, "thrust_coefficient":
                _O.CT, "min_spacing": _O.MIN_SPACING, "penalty": _O.PENALTY,
            "ideal_power_ds1": _O.IDEAL_POWER_SCEN1,
            "ideal_power_ds2": _O.IDEAL_POWER_SCEN2,
            "wind_arrays_sha256": {
                "omega_ds1": _arr_hash(_O.OMEGA_1),
                "psi_ds1": _arr_hash(_O.PSI_1),
                "omega_ds2": _arr_hash(_O.OMEGA_2),
                "psi_ds2": _arr_hash(_O.PSI_2)},
            "omega_sums": {"ds1": float(asnumpy(_O.OMEGA_1).sum()),
                           "ds2": float(asnumpy(_O.OMEGA_2).sum())},
            "note": "the Data Set II direction weights sum to 0.9999 by "
                    "design - the published table, used without "
                    "renormalisation"},
        "statistical_protocol": {
            "primary_endpoint": "WakeLoss",
            "secondary": "EnergyProduction",
            "runtime": "descriptive only, never significance-tested",
            "omnibus": "Friedman per case",
            "pairwise": "Wilcoxon signed-rank vs a PRESPECIFIED reference, "
                        "gated on Friedman",
            "multiplicity": "Holm within each declared family",
            "effect_size": "rank-biserial with average ranks for ties",
            "alpha": 0.05},
        "backend": {"using_gpu": bool(USING_GPU),
                    "dtype": os.environ.get("WFLOP_DTYPE", "float64"),
                    "device": device_info(), "workers": WORKERS,
                    "threads_per_worker": THREADS_PER_WORKER},
        "versions": {"python": platform.python_version(),
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "cupy": cupy_ver, "platform": platform.platform()},
        "source_sha256": src,
    }
    if RUN_TAG:
        manifest["tag"] = RUN_TAG
    os.makedirs("results", exist_ok=True)
    path = f"results/manifest_ds{DATASET}{_TAG}.json"
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return path


def main():
    print("=" * 74)
    print("WFLOP GPU CAMPAIGN")
    print(device_info())
    print(f"batch per objective call : {NUM_RUNS} seeds x {POP} individuals "
          f"= {NUM_RUNS * POP} layouts")
    print(f"cores available          : {CORES}")
    print(f"worker processes         : {WORKERS} "
          f"({THREADS_PER_WORKER} BLAS thread(s) each)")
    print(f"wind data set            : {DATASET}")
    print(f"algorithms               : {', '.join(ALGORITHM_NAMES)}")
    print(f"regime                   : "
          + (f"fixed budget, {BUDGET} exact evaluations per run"
             if BUDGET else f"fixed iterations ({ITER})"))
    print(f"manifest                 : {write_manifest()}")
    if WORKERS > 1 and any(a.startswith("GNN") for a in ALGORITHM_NAMES):
        print("note: the GNN optimizers hold large surrogate tensors; if a "
              "worker runs out of\n      memory, lower WFLOP_WORKERS.")
    print("=" * 74)

    groups = [(r, n, a) for r, rng_ in FARM_CASES.items()
              for n in rng_ for a in ALGORITHM_NAMES]

    results, done = load_checkpoint()
    pending = [g for g in groups if g not in done]
    infeasible = []
    calibration = []

    print(f"groups total {len(groups)} | done {len(done)} | pending {len(pending)}")
    print("=" * 74)

    t_all = time.perf_counter()
    gi = 0

    def absorb(payload):
        """Record one finished group. Parent-only, so writes never race."""
        nonlocal gi
        group, rows, cv, bf, dt, n_evals, axis, calib = payload
        radius, n_turb, alg_name = group
        gi += 1

        if calib:
            calibration.extend(calib)

        if rows is None:                      # budget-infeasible: record, skip
            mand = mandatory_cost(alg_name, n_turb)
            per = per_iteration_cost(alg_name, n_turb)
            infeasible.append([radius, n_turb, alg_name, BUDGET, mand,
                               "n/a" if per is None else per,
                               mand + (1 if per is None else per)])
            print(f"[{gi}/{len(pending)}] R{radius} T{n_turb:2d} {alg_name:11s} "
                  f"BUDGET-INFEASIBLE: needs at least "
                  f"{mand + (1 if per is None else per)} evaluations "
                  f"(mandatory {mand}), budget is {BUDGET}. Skipped.",
                  flush=True)
            return

        if SAVE_CURVES:
            store_best_curve(radius, n_turb, alg_name, cv, bf, axis)
        results.extend(rows)
        append_rows(rows)

        elapsed = time.perf_counter() - t_all
        eta = elapsed / (gi / len(pending)) - elapsed
        print(f"[{gi}/{len(pending)}] R{radius} T{n_turb:2d} {alg_name:11s} "
              f"best={bf.min():12.4g}  group={dt:6.2f}s  "
              f"evals/run={int(np.mean(n_evals))}  elapsed={elapsed/60:5.1f}min  "
              f"ETA={eta/60:5.1f}min", flush=True)

    if pending:
        # Longest jobs first: the big turbine counts dominate the wall clock,
        # so starting them early keeps every worker busy to the end.
        order = sorted(pending, key=lambda g: (-g[1], g[0], g[2]))

        if WORKERS == 1:
            for g in order:
                absorb(run_group(g))
        else:
            import multiprocessing as mp
            ctx = mp.get_context("spawn" if USING_GPU else "fork")
            with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx,
                                     initializer=_init_worker) as pool:
                for payload in pool.map(run_group, order, chunksize=1):
                    absorb(payload)

    pd.DataFrame(results, columns=RESULT_COLUMNS).to_csv(FINAL_PATH, index=False)

    # Seal the raw results. The four paper analyses are derived FROM this file
    # and must never edit it; the seal makes an accidental edit detectable
    # rather than merely forbidden. report.py verifies it and refuses to run
    # silently on modified raw data.
    with open(FINAL_PATH, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    seal = f"results/RawResults_ds{DATASET}{_TAG}.sha256"
    with open(seal, "w") as fh:
        fh.write(f"{digest}  {os.path.basename(FINAL_PATH)}\n")
    print(f"raw results sealed: {seal}")

    if calibration:
        path = f"results/uq_calibration_ds{DATASET}{_TAG}.csv"
        pd.DataFrame(calibration, columns=CALIBRATION_COLUMNS).to_csv(
            path, index=False)
        print(f"UQ calibration written to {path} "
              f"({len(calibration)} bin rows)")

    if infeasible:
        path = f"results/budget_infeasible_ds{DATASET}{_TAG}.csv"
        pd.DataFrame(infeasible, columns=[
            "Radius", "Turbines", "Algorithm", "Budget", "MandatoryCost",
            "PerIterationCost", "MinimumFeasibleBudget"]).to_csv(path,
                                                                index=False)
        need = max(r[6] for r in infeasible)
        print("=" * 74)
        print(f"WARNING: {len(infeasible)} (case, algorithm) pairs could not be "
              f"run within the budget of {BUDGET}.")
        print(f"         They are listed in {path} and are ABSENT from the "
              f"results file.")
        print(f"         The smallest budget that covers every pair here is "
              f"{need}.")

    print("=" * 74)
    print(f"{FINAL_PATH} saved ({len(results)} rows) in "
          f"{(time.perf_counter() - t_all)/60:.1f} min")
    print("=" * 74)


if __name__ == "__main__":
    main()
