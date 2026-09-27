# WFLOP on GPU — batched port of `WFLOP_HPC_FIXED`

**Fourteen optimizers:** GA, PSO, DE, GWO, BBO, SSA, LX-SSA, QA-SSA,
ACO, PF, GNN-LX-SSA, GNN-QA-SSA, GNN-LX-SSA-UQ, GNN-QA-SSA-UQ.
**Two wind data sets:** I and II.

## 1. How it is parallelised

Two levels, both on by default:

| Level | What runs together | Size at campaign settings |
|---|---|---|
| Inside a group | 30 seeds x 30 individuals, advanced in lockstep | **900 layouts per objective call** |
| Across groups | independent (radius, turbines, algorithm) groups spread over worker processes | **one per core** |

**The runner uses every core the job actually has.** It reads the CPU affinity
mask (`sched_getaffinity`), which reflects whatever SLURM, cgroups or `taskset`
granted, so nothing is pinned and nothing is left idle. `WFLOP_WORKERS`
overrides the count. BLAS threads per worker are then set automatically to
`cores // workers`, so the cores are used once rather than oversubscribed — with
one worker that is all threads, with 64 workers it is one each.

Do **not** export `OMP_NUM_THREADS` yourself in the job script; that would
override this and pin every worker to a single thread.

On a GPU the default is one worker per visible GPU: the batching inside a group
already fills the device, and extra processes on one GPU only contend for it.
With several GPUs each worker binds to its own.

Parallelism does not change results. Every group is seeded deterministically
from its own identity, so a 1-worker run and a 64-worker run produce
**identical** output — verified here by diffing the results file and every
curve file between a serial and a 4-worker run.

## 2. What was actually done, and why

The original code is CPU-shaped in one specific way: it calls the objective
**one layout at a time**. For `N = 18` that call touches a 24 × 18 × 18 wake
tensor — about 7,800 numbers. Handing 7,800 numbers to an A100 is slower than
doing them on a CPU, because the kernel launch costs more than the arithmetic.

So this port does **not** translate the code call-by-call. It restructures the
work so that one call evaluates many layouts:

| Level | What is batched | Batch size at campaign settings |
|---|---|---|
| Population | all individuals of one run | 30 |
| Runs | all 30 seeds of a case, advanced in lockstep | ×30 |
| **Total** | **layouts per objective call** | **900** |

900 layouts at `N = 18` is a 7-million-element tensor — a real GPU workload.
The same restructuring also makes the CPU dramatically faster, which matters:
**even with `WFLOP_BACKEND=cpu` the full campaign drops from ~330 core-hours
to ~2.3 hours single-core** — measured, not estimated. If your cluster queue
for GPUs is long, the CPU path alone may already solve your problem.

## 3. Files

| File | Purpose |
|---|---|
| `backend.py` | Selects CuPy (GPU) or NumPy (CPU). Everything else imports `xp` from here. |
| `objective_gpu.py` | Batched objective, wind data sets I and II. |
| `surrogate_gpu.py` | Batched dense GNWM surrogate — forward, manual backprop, Adam. |
| `algorithms_gpu.py` | GA, PSO, DE, GWO, BBO, SSA, LX-SSA, QA-SSA, ACO, PF. |
| `gnn_algorithms_gpu.py` | The four GNN-guided salp optimizers. |
| `run_experiments_gpu.py` | Campaign driver. Same output schema as the CPU version. |
| `validate_gpu.py` | **Run this first.** Equivalence, sanity and throughput checks. |
| `run_all.py` | The same pipeline **without SLURM**, detached into the background so you get your shell back: `python run_all.py` starts it, `status` shows a progress bar per data set, `log` follows it, `stop` halts it (resume is safe). Survives logout. |
| `submit_gpu.slurm` | SLURM job for one GPU. |
| `submit_cpu.slurm` | SLURM job for a CPU node, using every core it is granted. |
| `report.py` | Statistics and Excel export: descriptive stats, Friedman, Holm-corrected Wilcoxon with effect sizes, ranks, evaluation budgets, best layouts, convergence summary. Has a `--paper` mode for the four planned papers. Requires a pre-specified reference algorithm; runtime is descriptive only, never tested. |
| `combine_results.py` | Both wind data sets in one workbook, one sheet each (`RawResults_DS1`, `RawResults_DS2`). A transport step, not an analysis: it verifies each seal, then copies the rows. |
| `test_budget.py` | Regression test for the fixed-budget regime: 8,190 analytic (algorithm, N, budget) probes plus real runs, asserting the cap is never exceeded. Also reports the smallest feasible common budget. |
| `run_ablations.py` | The five ablations Papers 2 and 4 need, as one command, with a summary table. |
| `preflight.py` | Executes the pre-results checklist — 26 checks — and reports what is still outstanding. |
| `tests/fake_cupy/cupy.py` | A strict CuPy stand-in that lets the GPU code path run on a machine with no GPU. Test utility only. |
| `objective.py`, `algorithms.py` | The CPU originals, bundled **only** so `validate_gpu.py` can compare against them. Nothing imports them at run time. `objective.py` differs from your v34 copy in one respect: the Horns Rev import and the dataset-3 branch are removed, so `dataset=3` raises. |

Turbine coordinates now live **inside the results CSV**, not in per-run
`.npy` files. The CSV keeps the original column order with six appended:
`Evaluations`, `ObjectiveCalls`, `SurrogateInferences`,
`SurrogateTrainPasses`, `Iterations`, `Dataset` and `Coordinates`. The `layouts/` directory is no longer created.

The three counting columns are three different quantities and are kept apart
deliberately:

| Column | Meaning | Use it for |
|---|---|---|
| `Evaluations` | Exact objective values the optimizer **admitted into its decisions**. | The algorithmic budget. This is the number that belongs in a paper. |
| `ObjectiveCalls` | Exact objective evaluations actually **computed**. Equal to `Evaluations` for every algorithm except the two UQ variants, whose gated batch is padded to a rectangle so 30 seeds can be evaluated in one call. | Compute cost of the batched implementation. |
| `SurrogateInferences` | Surrogate **forward passes** used for screening and guidance, per run. | Reporting surrogate load. Never add it to `Evaluations`: a surrogate call is orders of magnitude cheaper than an exact one. |
| `SurrogateTrainPasses` | Surrogate forward passes spent **training** the model, per run. | The cost of *building* the surrogate, kept apart from the cost of *using* it. Also never added to the budget. |
| `Iterations` | Iterations the algorithm actually ran. | Constant at 100 in the default regime; derived per algorithm under `WFLOP_BUDGET`. |
| `GateAdmissionRate` | Fraction of candidates the uncertainty gate sent for exact evaluation, averaged over iterations, per run. 0 for every non-gated algorithm. | The number behind the efficiency claim in Papers 3 and 4 — it says how selective the gate actually was. |

The `Coordinates` cell holds the best layout of that run as
`"x1 y1;x2 y2;...;xN yN"` in metres — no commas, so the CSV needs no quoting —
written at full round-trip precision. Read one back with:

```python
import pandas as pd
from run_experiments_gpu import parse_coordinates
df = pd.read_csv("results/RawResults_ds1.csv")
xy = parse_coordinates(df.loc[0, "Coordinates"])     # (N, 2) array in metres
```

Precision is deliberately not rounded: the penalty coefficient is 1e10 and the
GNN repair leaves turbines about 1e-6 m outside the spacing limit, so a layout
rounded to a few decimals could re-evaluate with a huge spurious penalty. As
written, recomputing the objective from a stored cell reproduces the stored
`WakeLoss` to floating-point precision. Expect roughly 13 MB per data set for
the full campaign — far less than one `.npy` per run.

**Convergence curves: one file per case, best run only.** Instead of one curve
per run, each case gets a single `curves/Conv_ds<D>_R<radius>_T<n>.npz`
containing, for every algorithm, the curve of that algorithm's best of the 30
runs. That is 39 files per wind data set — 78 in total — and it is exactly what
a "Convergence (R, T)" figure needs:

```python
import numpy as np, matplotlib.pyplot as plt
d = np.load("curves/Conv_ds1_R500_T10.npz")
for name, curve in zip(d["algorithms"], d["curves"]):
    plt.plot(curve, label=str(name))
```

Each file also carries `median`, `q25` and `q75` — the across-run median curve
and inter-quartile band over the same 30 runs — an `evals` axis giving the
cumulative exact evaluations at each point of the best run's curve, and
`best_seed` and `best_wake` per algorithm, so you can trace any curve back to
its row in the CSV.

The `evals` axis is what makes an efficiency figure possible. Plot a curve
against it instead of against the iteration index and you get **best-so-far
wake loss versus exact objective evaluations** — the comparison a
fixed-iteration campaign cannot show, because the algorithms spend very
different amounts per iteration. `report.py` also tabulates it (the
`Efficiency` sheet) at 25/50/75/100 % of each algorithm's budget.

The best-of-30 curve is what you asked for and is the right figure for "how
good can this method get". It is, however, a favourable order statistic: it
selects the luckiest of 30 runs for each algorithm, and a reviewer who notices
that will ask what a typical run did. The median curve with an IQR band answers
that from the same data at no extra cost, so both are stored. The
recommendation is a median-with-band figure in the paper and the best-of-30
curve alongside it or in supplementary material.

```python
d = np.load("curves/Conv_ds1_R500_T10.npz")
i = list(map(str, d["algorithms"])).index("GNNLXSSA")
plt.plot(d["median"][i], label="GNN-LX-SSA (median of 30)")
plt.fill_between(range(d["median"].shape[1]), d["q25"][i], d["q75"][i], alpha=.2)
```
 Files are updated in place as
algorithms finish, so resume and `WFLOP_ALGOS` subsets work normally. The old reporting scripts still read it,
but note the file is now `results/RawResults_ds<N>.csv`. The v34 reporting
pipeline (`run.py`, `reporting.py`, `studies.py`, `consolidate.py`) was **not**
part of this port — only the algorithms and the objective were.

## 4. The six algorithms added from v34

| Added | Source | Notes |
|---|---|---|
| `ACO` | ported | Eroglu & Seckiner (2012) pheromone = per-turbine leave-one-out wake contribution; ants relocate turbines, greedy accept-if-improved. Batched across runs only — the greedy chain inside a run is sequential by construction. |
| `PF` | ported | Eroglu & Seckiner (2013) particle filter: predict → elite-quantile weights → systematic resampling. Fully batched, semantics unchanged. |
| `GNNLXSSA` | ported | Algorithm 2 of the GNN-LX-SSA paper: GNWM surrogate screening, direction-head guidance (Eq. 32), repair, replay buffer, periodic fine-tuning. |
| `GNNLXSSA_UQ` | ported | Deep-ensemble trust gate: predict with 5 surrogates, evaluate exactly when the ensemble spread exceeds `sigma_threshold`; generation best always verified. |
| `GNNQASSA` | **derived, new code** | Not present in your zip. Identical GNN machinery with QA-SSA's quadratic-interpolation vertex replacing the Laplace perturbation. |
| `GNNQASSA_UQ` | **derived, new code** | Same, with the ensemble trust gate. |

> The uploaded package's own registry ends at `GNNLXSSA_UQ` — there is no
> GNN-QA-SSA in it. The two `QASSA` GNN variants above are my construction
> by analogy; check they match your intent before publishing results from them.

**What the GNN port had to change.** The CPU surrogate builds a sparse edge
list per layout, and edge counts differ per layout, which cannot be batched.
The port uses a dense `(N, N)` graph with a boolean wake mask — a masked sum
over `j` is identical to a sum over the edge list — and gives every run its own
weights on a leading batch axis. Activations are recomputed in the backward
pass rather than stored, and the surrogate runs in **float32** (it predicts a
percentage of ideal power, never the 1e30 penalties; the exact objective stays
float64). The manual backprop is checked against numerical gradients by
`validate_gpu.py`, and the forward pass against the CPU class with identical
weights.

**The UQ gate is a true gate.** The ensemble predicts first, the gate mask is
built from the predictive spread, and the exact objective is called **only on
the gated candidates**. Because the number that passes the gate differs per
run and a batch must be rectangular, the gather is padded to the widest run
with a duplicate of that run's first gated candidate; the duplicate costs one
column and carries the same value, so writing it twice is harmless. **No
ungated candidate is ever evaluated exactly.**

That is why there are two counters. `Evaluations` is what the gate admitted —
the algorithmic budget, and the number for the paper. `ObjectiveCalls` is what
was actually computed, including the padding — the compute cost of running 30
seeds in lockstep. Measured at full campaign settings (30 individuals, 100
iterations, `n_pretrain = 40`, 4 seeds in the batch, data set I, R = 500 m):

| N | `Evaluations` (range over runs) | `ObjectiveCalls` | evaluate-all would cost | exact calls saved | gate admits |
|---|---|---|---|---|---|
| 2 | 298 – 423 | 426 | 6,198 | **93.1 %** | 2.4 % of candidates |
| 5 | 636 – 958 | 977 | 6,390 | **84.7 %** | 5.6 % |
| 9 | 1,627 – 2,850 | 3,755 | 6,646 | **43.5 %** | 26.3 % |

Two honest caveats on that table. The gate admits more as N grows — the
surrogate finds larger layouts harder, which is the gate working as intended,
not a defect. And the padding overhead grows with how much the runs in a batch
disagree: the width of the padded batch is the *maximum* gate count over the
runs, so with 30 seeds instead of 4 the `ObjectiveCalls` column will sit
higher relative to `Evaluations`. The budget the optimizer spends is unaffected
either way.

**Reported best is always exact.** For the GNN family the surrogate score is
used for screening and for the greedy salp selection, exactly as on the CPU,
but `best_f` and the convergence curve only ever take values from exact
evaluations. `validate_gpu.py` re-evaluates each returned layout and checks it
equals the reported best.

## 5. GPU readiness (A100)

The code is written for a single CuPy namespace, so nothing is NumPy-specific
at run time. Specifically, for an A100:

| Item | Status |
|---|---|
| API surface | 49 distinct `xp.*` calls, all present in CuPy. No NumPy call ever touches a device array — the only `np.*` uses are host scalars and constants. |
| Scalars | `math.pi` / `math.inf` / `math.nan` are used instead of `xp.pi` / `xp.inf` / `xp.nan`, so nothing depends on NumPy alias re-exports surviving in the CuPy namespace. |
| Indexing | Integer gathers and `take_along_axis` only; no multi-axis boolean indexing, which CuPy supports unevenly. |
| RNG | `random`, `integers`, `normal`, `uniform` — the common subset of `cupy.random.Generator`. Permutations are done as `argsort` of random keys, so nothing relies on `Generator.permutation`. |
| Precision | Objective in float64 (the 1e10 penalty needs the range); the A100 does FP64 at 9.7 TFLOP/s, so this costs little. Surrogate in float32. |
| Device syncs | The two `bool(xp.any(...))` early-exits inside the repair loops are **skipped on GPU** — they save work on a CPU but each one stalls the pipeline, and the repair loop runs N(N-1)/2 times per call. |
| Workers | Default is one process per visible GPU. Extra processes on one GPU only contend for it. With several GPUs each worker binds to its own device. |

`validate_gpu.py` starts with a **GPU readiness** block that prints the device
name, compute capability, memory, CuPy/CUDA versions, an estimate of the peak
surrogate and objective tensors at campaign settings, and runs an FP64 kernel
smoke test. Run it first on the node.

Rough peak memory at campaign settings (30 seeds, population 30, N = 18,
hidden 64, 3 layers): about 0.9 GiB for screening without an ensemble and
4.4 GiB with the five-model ensemble, plus a few hundred MB for the objective.
Comfortable on a 40 GB A100; if you raise `n_models`, `hidden` or `NUM_RUNS`,
re-check that block before queuing.

## 5b. Reporting for the four papers

One campaign feeds all four papers; only the analysis is per paper.

```bash
python report.py --dataset 1 --paper 1      # GNN-LX-SSA study
python report.py --dataset 2 --paper 4      # GNN-QA-SSA-UQ study, data set II
```

| `--paper` | Proposed method | Compared against |
|---|---|---|
| 1 | GNN-LX-SSA | GA, PSO, DE, ACO, PF, BBO, GWO, SSA, LX-SSA |
| 2 | GNN-QA-SSA | GA, PSO, DE, ACO, PF, BBO, GWO, SSA, QA-SSA |
| 3 | GNN-LX-SSA-UQ | the paper-1 set + GNN-LX-SSA |
| 4 | GNN-QA-SSA-UQ | the paper-2 set + GNN-QA-SSA |

Each writes `WFLOP_Report_ds<D>_paper<N>.xlsx` and prints a headline: the
proposed method's average-rank position, how many wake-loss comparisons survive
Holm correction, in how many of those it is the better method, and its exact
evaluation budget against the baselines'.

**Both data sets in one job, and one workbook holding both.** `submit_gpu.slurm`
now runs wind data set 1 to completion — campaign, seal, four paper reports —
and only then starts data set 2, so a wall-clock timeout never leaves a data set
half-analysed and a resubmit resumes from that data set's own checkpoint. Set
`WFLOP_DATASETS="1"` to run only one. After both finish, the job writes:

```bash
python combine_results.py            # -> WFLOP_RawResults_AllDatasets.xlsx
```

with exactly two sheets, `RawResults_DS1` and `RawResults_DS2`. It is a
transport step, not an analysis: the seals are verified first, then the rows are
copied unchanged. The statistics stay in the eight per-paper workbooks, because
the Friedman omnibus, the ranks and the Holm family are computed *within* a data
set — the two are different wind regimes, not replications of one another, and
pooling them would be wrong.

One caveat worth knowing: xlsx stores a float at 16 significant digits, one
short of the 17 a float64 needs, so numeric cells in that workbook are within
~1e-16 of the CSV rather than bit-identical. `Coordinates` is a string and is
copied verbatim, so re-evaluating a layout from the workbook reproduces its wake
loss exactly (verified: worst relative error 1.5e-16). For anything that must be
exact, read `results/RawResults_ds<D>.csv`.

**Why this is not a slice of the 14-algorithm analysis.** The Friedman omnibus,
the average ranks and the size of the Holm family all depend on how many
algorithms are in the comparison. `--paper` filters the runs first and then
recomputes everything, which is the statistically correct order. The runs
themselves are unaffected — they do not depend on which subset you later choose
to compare.

**A point to prepare for, and the switch that answers it.** In every paper the
proposed method uses a *different* number of exact objective evaluations from
the baselines it is compared with. The `EvaluationBudget` sheet quantifies
that, but a reviewer will ask whether the advantage survives an equal-
evaluation budget. It is now one environment variable:

```bash
WFLOP_BUDGET=3000 python run_experiments_gpu.py    # equal-evaluation regime
python test_budget.py --budget 3000                # prove it before you run it
```

**The budget is a hard cap, and the cap is never exceeded.** Iteration counts
are derived with a **floor**, not by rounding — rounding overshoots by up to
half an iteration's cost, which at population 30 is up to 15 evaluations, so a
"500-evaluation" comparison quietly ran some algorithms at 510. Each algorithm
now receives the largest whole number of iterations whose total cost fits
inside the budget.

The cost model, in exact objective evaluations per run:

| Algorithm | Mandatory, before iterating | Per iteration |
|---|---|---|
| GA, PSO, DE, GWO, BBO, SSA, PF | `pop` | `pop` |
| LX-SSA, QA-SSA | `pop` | `2·pop` |
| ACO | 1 | `N + pop` |
| GNN-LX/QA-SSA | `pop + n_pretrain + 4N·n_fd` = **70 + 64N** | `n_exact + 2` |
| GNN-*-UQ | same | data-dependent |

**The GNN mandatory cost grows with N** — 198 at N=2, **1,222 at N=18** —
because the finite-difference direction labels cost 4N evaluations per sampled
layout. A budget below that plus one iteration cannot be honoured at all. Such
a (case, algorithm) pair is marked **budget-infeasible**, skipped, and listed in
`results/budget_infeasible_ds<D>.csv`. It is never run over budget, and never
silently dropped either.

That sets a floor on any common budget for the whole N = 2..18 campaign:

| | N=2 | N=5 | N=9 | N=14 | N=18 |
|---|---|---|---|---|---|
| GA (minimum feasible) | 60 | 60 | 60 | 60 | 60 |
| LX-SSA | 90 | 90 | 90 | 90 | 90 |
| GNN-LX-SSA | 212 | 404 | 660 | 980 | **1,236** |

**The smallest budget that works everywhere is 1,236.** A practical common
budget is **3,000**: feasible for every algorithm at every N, and it leaves the
GNN family 127 iterations at N=18 and the classical algorithms 99 — close
enough to the fixed-iteration regime that the two campaigns are comparable.
Confirm before freezing with `python test_budget.py --budget 3000`, and state
the chosen budget and this rationale in the papers.

The two UQ variants cannot have an iteration count derived in advance — their
spend is data-dependent — so they get the budget as a hard cap and truncate
their gate admissions to whatever budget is left, spending it on the
best-predicted candidates. A run with nothing left admits nothing and cannot
improve further (the incumbent only ever takes exact values), so it holds while
the others finish. That is precisely what "best found within budget B" means.
Their iteration ceiling is derived too — `budget − mandatory`, the point past
which the budget cannot stretch even at one admission per iteration — so the
budget always binds, never an arbitrary iteration limit.

Run the campaign both ways and report both: fixed iterations answers "better
per iteration" (the convergence claim), fixed budget answers "better per
evaluation" (the efficiency claim). They are different claims and reviewers
ask for the second one.

**Reference algorithm must be pre-specified.** `report.py` refuses to run
without `--paper` or `--reference`. It used to fall back to the best-ranked
algorithm, which is selection on the outcome: choosing the reference after
seeing who won biases every p-value that follows.

**Ablations — one command.** `run_ablations.py` runs the five ablations Papers
2 and 4 need and tabulates them:

```bash
python run_ablations.py                       # all five, QA family
python run_ablations.py --family LX           # the equivalents for Papers 1, 3
python run_ablations.py --budget 3000         # at an equal budget (do this)
python run_ablations.py --cases 500:5 --runs 6 --only A,C   # quick pilot
```

| | Ablation | The question |
|---|---|---|
| A | SSA → QA-SSA | Does the quadratic operator help before any GNN is involved? |
| B | GNN-LX-SSA → GNN-QA-SSA | Does it still help once the surrogate screens? Identical GNN settings, so only the operator differs. |
| C | GNN-QA-SSA with / without guidance | Does the direction-head guidance earn its place, or is the power head doing the work? |
| D | GNN-QA-SSA → GNN-QA-SSA-UQ | What does the gate change, in quality *and* in evaluations spent? |
| E | Gate-threshold sweep | Is the saving robust, or does it hold only at one hand-picked σ? |

Each runs with `WFLOP_TAG` set, so its outputs land **beside** the main
campaign's rather than on top of them, and each gets its own manifest. Nothing
in an ablation can touch the frozen raw results.

Under the hood it is `WFLOP_ALGO_KWARGS`, a JSON object of per-algorithm
keyword overrides, which you can also use directly:

```bash
# does the direction-head guidance earn its place?
WFLOP_ALGO_KWARGS='{"GNNLXSSA": {"lambda_g": 0.0}}' python run_experiments_gpu.py

# does the 5-member ensemble earn its place?
WFLOP_ALGO_KWARGS='{"GNNLXSSA_UQ": {"n_models": 1}}' python run_experiments_gpu.py

# how sensitive is the saving to the gate threshold?
WFLOP_ALGO_KWARGS='{"GNNLXSSA_UQ": {"sigma_threshold": 0.02}}' python run_experiments_gpu.py
```

The component ladder — SSA → LX-SSA → GNN-LX-SSA → GNN-LX-SSA-UQ — needs no
overrides at all; it is an algorithm-set choice, and each step adds exactly one
component.

**Provenance, and the freeze.** Each campaign writes
`results/manifest_ds<D>.json` — the complete record of what was frozen:

- SHA-256 of every source file
- the regime, the budget and the budget rule
- algorithm list, population, iterations, farm cases, run count
- every optimizer's parameters, and any ablation overrides in force
- the cost model (mandatory and per-iteration costs at N = 2, 9, 18)
- surrogate architecture, ensemble size and gate threshold
- the seeding scheme, the seed list, and a sample of derived group seeds
- objective constants, both wind arrays' SHA-256, and their probability sums
- the statistical protocol — primary endpoint, omnibus, pairwise, multiplicity,
  effect size, alpha
- backend, dtype, worker layout, and library versions

**Raw results are sealed.** The campaign writes
`results/RawResults_ds<D>.sha256` beside the CSV, and `report.py` verifies it
before analysing anything — if the raw file has changed since the campaign
wrote it, the report stops rather than quietly analysing edited data. All four
paper analyses derive from that one sealed file; none of them edits it.

**Check readiness before you spend the queue time:**

```bash
python preflight.py --results     # 26 checks, one per pre-results requirement
```

It executes each item rather than asserting it, and reports the real-GPU
validation as OUTSTANDING rather than passed, since only the A100 can close it.

## 6. Running it

```bash
python validate_gpu.py                      # always do this first
WFLOP_SMOKE=1 python run_experiments_gpu.py # 2-minute end-to-end proof
python run_experiments_gpu.py               # the real campaign
```

**Or run the whole thing in the background, no SLURM.** `run_all.py` chains
validation, both campaigns, the eight paper reports and the combined workbook,
detached, so the shell comes straight back and the run survives logout:

```bash
python run_all.py                 # start; returns immediately
python run_all.py status          # progress bar per data set, workbooks so far
python run_all.py log             # follow the output (Ctrl-C stops watching only)
python run_all.py stop            # halt; restarting resumes from the checkpoint
```

It refuses to start a second run while one is alive, because two runs sharing a
checkpoint would corrupt it. Run it on a **compute node** (e.g. inside `salloc
--partition=gpu --gres=gpu:a100:1`), not on a login node — the full campaign is
hours of compute and most sites kill that on a login node. For a batch queue,
`submit_gpu.slurm` is still the right tool.

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `WFLOP_BACKEND` | `auto` | `gpu` forces CuPy and fails loudly if CUDA is missing; `cpu` forces NumPy. |
| `WFLOP_DTYPE` | `float64` | `float32` halves memory and is much faster on consumer cards — read the float32 warning before using it. |
| `WFLOP_MEM_BUDGET_MB` | `256` | Peak size of the speed-bin tensor; the code auto-chunks to stay under it. |
| `WFLOP_WORKERS` | all cores (CPU) / one per GPU | Worker processes. Lower it if a GNN worker runs out of memory. |
| `WFLOP_SMOKE` | unset | `1` runs a tiny grid for a pipeline check. |
| `WFLOP_TIMING` | `0` | `1` also times one run on its own per case and algorithm, giving a `RuntimeSingleRun` column that IS comparable to a sequential implementation. Costs about 1/30 of the campaign. |
| `WFLOP_SAVE_ARTIFACTS` | `1` | `0` skips writing the convergence-curve files. |
| `WFLOP_DATASET` | `1` | Wind data set: `1` or `2`. Results go to `results/RawResults_ds<N>.csv`. |
| `WFLOP_ALGOS` | all 14 | Comma-separated subset, e.g. `GA,SSA,GNNLXSSA`. The GNN family is far more expensive than the rest — splitting the campaign is usually the right move. |
| `WFLOP_SURROGATE_DTYPE` | `float32` | `float64` for the strict gradient check. |

Resume works exactly as before, at (radius, turbines, algorithm) granularity.

## 7. What is identical, and what changed

**Identical update rules** — GA, GWO, BBO, SSA, LX-SSA, QA-SSA, ACO, PF. The
Laplace operator, the quadratic-interpolation vertex, BBO's migration and
elitism, the SSA follower chain, ACO's leave-one-out pheromone and greedy ant
acceptance, and PF's elite-quantile weighting and systematic resampling all
reproduce the original logic.

**The objective is numerically equivalent**, for both wind data sets.
`validate_gpu.py` checks it element-wise against the original `objective.py`:
worst relative error **1.1 × 10⁻¹⁴**, which is floating-point summation order,
not physics.

**Wind Data Set III (Horns Rev 1) has been removed** at your request. The
batched Horns Rev module is gone, `dataset=3` raises a clear error in both the
GPU objective and the bundled CPU reference, and `WFLOP_DATASET` accepts only
1 or 2.

**Three deliberate changes** — each one is a decision you should be able to
defend in a paper, so they are listed explicitly:

1. **PSO and DE are now synchronous (generational).** The originals update
   `gbest` (PSO) and the population (DE) *during* the sweep, so individual *i*
   already sees the improvement made by individual *i−1*. That is inherently
   sequential and cannot be batched. The batched versions evaluate the whole
   population, then update — the textbook synchronous variant. This changes
   results slightly; it is a well-established variant, but it must be stated
   in the methods section.

2. **GWO reports best-so-far.** The original returned the best of the *final*
   population, which — with no elitism — can be worse than what it found. This
   was the correctness bug in the code review. Pass
   `build("GWO", legacy_reporting=True)` to reproduce the original behaviour.

3. **QA-SSA's degeneracy guard is now relative.** The original tested
   `|denominator| < 1e-12` in absolute terms, which is meaningless when fitness
   values are ~10³⁰ inside the penalty region. The guard now scales with the
   fitness magnitude.

4. **The uncertainty-gated variants report a per-run evaluation count.** How
   many candidates the gate sends for exact evaluation differs from run to run,
   so `Evaluations` is now recorded per run rather than collapsed to one number
   for the whole group. Every other algorithm spends the same budget in every
   run, and still reports a single value.

**Random numbers differ.** The batched code draws in a different order, and
CuPy's bit generator differs from NumPy's. A GPU run and a CPU run with the
same seed are **statistically equivalent, not bit-identical**. Compare
distributions over the 30 seeds — never single runs.

## 8. The `Runtime` column — read this before using it

The 30 seeds share every kernel launch, so per-seed wall time is not
observable. `Runtime` is the batch time divided by 30: fair for comparing
algorithms against each other on the same hardware, **not** comparable to the
CPU campaign's per-run timings.

`Runtime` is therefore reported **descriptively only** and is **never
significance-tested**. `report.py` excludes it from the Friedman/Wilcoxon
metrics on purpose: the 30 seeds of a group share one wall-clock measurement,
so writing `group_time / 30` onto all 30 rows creates 30 identical numbers, and
a paired test on those is pseudo-replication — n = 1 dressed up as n = 30. The
descriptive `Runtime` sheet in the Excel report carries the group time, the
per-run share and the optional single-run timing.

The port therefore adds an **`Evaluations`** column, which is exact. This also
fixes the fairness problem from the code review: LX-SSA and QA-SSA consume
**6,030** evaluations per run against **3,030** for the other six. That column
is the budget measure that belongs in the comparison, and it makes an
equal-budget re-run easy — set `ITER = 50` for the two variants and they match
the others exactly.

## 9. float32 — a warning

The penalty term `(1 + 1e10·g)²` reaches ~10³² for a violation of a few metres.
float32 does not overflow, but it carries ~7 significant digits: once a penalty
is added to a wake loss of order 10³, **the wake-loss information is destroyed
entirely** and the search cannot distinguish good infeasible layouts from bad
ones. On an A100 (9.7 TFLOP/s FP64) there is no reason to use float32. On a
consumer card with crippled FP64, lower `PENALTY` to ~10⁴ first and re-validate
that feasible solutions are still found — do not simply flip the dtype.

## 10. Expected performance, and what I could not test

**Measured here** (CPU, single core, batched code):

| N | per-layout, batch 900 | original per-layout | speed-up |
|---|---|---|---|
| 9 | 0.18 ms | 31.9 ms | ~180× |
| 18 | 0.63 ms | 112.5 ms | ~178× |

One full group (30 seeds × 30 individuals × 100 iterations, N = 9, GA) takes
**17.5 s** batched, against ~48 min for the same work in the original code on
the same machine.

**Not measured:** this environment has no GPU, so the CuPy path has never
executed a CUDA kernel on real hardware. It has, however, been executed
end-to-end through a strict CuPy stand-in (`tests/fake_cupy`) that rejects any
attribute the real library does not expose, so the GPU branch is known to use
only supported API. The NumPy path — the same source — is fully validated.

The GNN family **has** now been run at full campaign settings (30 individuals,
100 iterations, `n_pretrain = 40`, hidden 64, three message-passing layers) at
N = 2, 5 and 9, which is where the budget table in §6 comes from. Treat the GPU numbers as unverified
until you run `validate_gpu.py` on the node. What you should see there is
throughput climbing steeply with batch size and then flattening; if it flattens
early, raise the batch by running more seeds in lockstep.

**Where the GPU will help least:** LX-SSA, QA-SSA and ACO. Their inner loops
are sequential chains — each follower (or ant) depends on the previous one — so
those steps run at batch 30–60 instead of 900. Expect them to gain less than
the fully batched algorithms.

**Where the GPU should help most:** the GNN family. Its cost is dominated by
surrogate forward and backward passes on `(runs × candidates × N × N × 133)`
tensors — dense batched matmuls, exactly what a GPU is for. These are also the
algorithms that are painfully slow on a CPU: in this environment a single tiny
GNN-UQ group (4 runs, 8 individuals, 5 iterations, hidden 16) takes seconds,
and the full campaign settings (hidden 64, three layers, 5-member ensemble)
are impractical on one core. If you only port one thing to the GPU, port these.

**A knob that matters:** `n_models`, `hidden` and `mp_layers` drive surrogate
memory as `runs × models × candidates × N² × (2·hidden + 5)`. At the campaign
defaults with N = 18 and 30 seeds that is a few hundred MB in float32 — fine on
an A100, tight on a 12 GB card. Lower `NUM_RUNS` before lowering `hidden`.

## 11. Is a GPU even the right answer?

Honestly: possibly not. The objective is memory-bound elementwise arithmetic on
modest tensors, and the batched CPU version already brings the campaign to a
couple of hours on **one core** — minutes across the 64 cores you already
request. A GPU will beat that, but the decisive gain came from batching, not
from CUDA.

The strongest argument for the GPU path is what it makes affordable *next*:
equal-budget re-runs, sensitivity analysis over the penalty coefficient, larger
turbine counts, and the ≥30-run statistical certification protocol at several
budgets — all of which need the objective called tens of millions more times.
