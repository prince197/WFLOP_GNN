"""Regression test for the fixed-evaluation-budget regime.

WHY THIS FILE EXISTS

Under WFLOP_BUDGET the budget is a HARD CAP: no run may spend more exact
objective evaluations than the budget allows. That property is easy to break
by accident - an earlier version derived iteration counts by ROUNDING, which
overshot a 500-evaluation budget by 10 on every population-30 algorithm and by
736 for the GNN family at N=18, where the mandatory start-up cost alone exceeds
the budget. A comparison run at "500 evaluations" where some algorithms
actually spent 1,236 is not a fair comparison, and nothing in the output would
have revealed it.

So the cap is tested rather than asserted, in three layers:

  [1] Cost model, exhaustively. For all 14 algorithms at N = 2, 5, 9, 14, 18
      and budgets sampled just below, exactly at, and just above every
      iteration boundary, check that the derived iteration count both fits
      inside the budget AND is the largest one that does (no wasted budget,
      no overshoot).

  [2] Feasibility. Report the smallest budget that is feasible for every
      algorithm at every N, which is what a common campaign budget must clear.

  [3] Execution. Actually run every algorithm and assert that the measured
      Evaluations column never exceeds the budget - including the two UQ
      variants, whose spend is data-dependent and cannot be derived in
      advance.

Usage:
    python test_budget.py                  # layers 1 and 2, plus a quick [3]
    python test_budget.py --budget 3000    # feasibility of one candidate budget
    python test_budget.py --full           # [3] over more algorithms and N
"""

import argparse
import sys

import numpy as np

import run_experiments_gpu as R
from objective_gpu import make_objective
from algorithms_gpu import build

ALGOS = ["GA", "PSO", "DE", "GWO", "BBO", "SSA", "LXSSA", "QASSA", "ACO", "PF",
         "GNNLXSSA", "GNNQASSA", "GNNLXSSA_UQ", "GNNQASSA_UQ"]
NS = [2, 5, 9, 14, 18]

fail = []


def report(ok, label, detail=""):
    print(f"    {label:<62} {'PASS' if ok else 'FAIL'}   {detail}")
    if not ok:
        fail.append(label)


# ===================================================================
print("=" * 78)
print("WFLOP FIXED-BUDGET REGRESSION TEST")
print("=" * 78)

print("\n[1] Cost model: no overshoot, and no wasted iteration")
# For every algorithm and N, walk the iteration boundaries and probe budgets
# just below, exactly at, and just above each one.
probes = 0
worst_slack = 0
for alg in ALGOS:
    bad = []
    for n in NS:
        mand = R.mandatory_cost(alg, n)
        per = R.per_iteration_cost(alg, n)
        floor_per = 1 if per is None else per
        for k in range(1, 40):
            boundary = mand + k * floor_per
            for b in (boundary - 1, boundary, boundary + 1):
                if b < 1:
                    continue
                probes += 1
                iters, cap, status = R.iterations_for(alg, n, b)
                if status == "infeasible":
                    # Only legitimate when even one iteration cannot be funded.
                    if b >= mand + floor_per:
                        bad.append(f"N={n} B={b}: infeasible but affordable")
                    continue
                if cap is not None:
                    # Data-dependent (GNN family): the cap itself is passed to
                    # the optimizer and layer [3] checks it is honoured.
                    if cap != b:
                        bad.append(f"N={n} B={b}: cap not passed through")
                    continue
                cost = R.budget_cost(alg, n, iters)
                if cost > b:
                    bad.append(f"N={n} B={b}: spends {cost} > {b}")
                if R.budget_cost(alg, n, iters + 1) <= b:
                    bad.append(f"N={n} B={b}: {iters} iters wastes budget")
                worst_slack = max(worst_slack, b - cost)
    report(not bad, f"{alg}", f"{len(bad)} violations" if bad else "")
    for msg in bad[:3]:
        print(f"        {msg}")

print(f"    probed {probes:,} (algorithm, N, budget) combinations; "
      f"worst unavoidable slack {worst_slack} evaluations")
print(f"    -> {'PASS' if not fail else 'FAIL'}")

# ===================================================================
print("\n[2] Feasibility of a common budget")
print("    A budget must clear each algorithm's mandatory start-up cost plus")
print("    one iteration. The GNN family's start-up cost grows with N, so it")
print("    sets the floor for the whole campaign.")
print()
print(f"    {'algorithm':<14}" + "".join(f"{'N=' + str(n):>10}" for n in NS))
for alg in ("GA", "LXSSA", "ACO", "GNNLXSSA", "GNNLXSSA_UQ"):
    row = ""
    for n in NS:
        per = R.per_iteration_cost(alg, n)
        row += f"{R.mandatory_cost(alg, n) + (1 if per is None else per):>10,}"
    print(f"    {alg:<14}{row}")

floor_needed = max(R.mandatory_cost(a, n) + (1 if R.per_iteration_cost(a, n)
                                             is None
                                             else R.per_iteration_cost(a, n))
                   for a in ALGOS for n in NS)
print(f"\n    smallest feasible common budget for N = 2..18 : {floor_needed:,}")
print(f"    (set by the GNN family at N = 18: mandatory "
      f"{R.mandatory_cost('GNNLXSSA', 18):,} = 70 + 64N, plus one iteration)")

parser = argparse.ArgumentParser()
parser.add_argument("--budget", type=int, default=3000)
parser.add_argument("--full", action="store_true")
args = parser.parse_args()

B = args.budget
print(f"\n    candidate common budget: {B:,}")
print(f"    {'algorithm':<14}" + "".join(f"{'N=' + str(n):>9}" for n in NS)
      + "   (iterations at that budget)")
infeasible = []
for alg in ALGOS:
    row = ""
    for n in NS:
        iters, cap, status = R.iterations_for(alg, n, B)
        if status == "infeasible":
            row += f"{'INFEAS':>9}"
            infeasible.append((alg, n))
        else:
            row += f"{iters:>9,}"
    print(f"    {alg:<14}{row}")
report(not infeasible, f"budget {B:,} feasible for every algorithm at every N",
       f"{len(infeasible)} infeasible pairs" if infeasible else "")

# ===================================================================
print("\n[3] Execution: measured evaluations never exceed the budget "
      "(and capped GNN runs spend it exactly)")
run_algos = ALGOS if args.full else [
    "GA", "LXSSA", "ACO", "PF", "GNNLXSSA", "GNNLXSSA_UQ", "GNNQASSA_UQ"]
run_ns = [2, 9, 18] if args.full else [2, 9]
# Budgets chosen to be feasible at every N tested while keeping the GNN family
# to a handful of iterations, so the test finishes in minutes rather than hours.
budgets = [1400, 2000] if not args.full else [1400, 2000, 3000]

print(f"    {'algorithm':<13} {'N':>3} {'budget':>7} {'iters':>6} "
      f"{'evaluations':>16} {'calls':>7}  within cap")
for B_ in budgets:
    for n in run_ns:
        f = make_objective(500, dataset=1)
        for alg in run_algos:
            iters, cap, status = R.iterations_for(alg, n, B_)
            if status == "infeasible":
                print(f"    {alg:<13} {n:3d} {B_:7,} {'-':>6} "
                      f"{'BUDGET-INFEASIBLE (skipped)':>16}")
                continue
            kw = {"dataset": 1} if alg.startswith("GNN") else {}
            if cap is not None:
                kw["max_evals"] = cap
            a = build(alg, **kw)
            *_, ne = a.optimize(f, 2 * n, -500, 500, 4, R.POP, iters, seed=17)
            ev = np.atleast_1d(np.asarray(R.asnumpy(ne))).astype(float)
            oc = np.atleast_1d(np.asarray(R.asnumpy(
                getattr(a, "n_objective_calls", ne)))).astype(float)
            ok = bool(np.all(ev <= B_))
            # a capped (GNN) run must also SPEND its budget: stopping short
            # would make the method look cheap while denying it the search
            if cap is not None:
                ok = ok and bool(np.all(ev == B_))
            span = (f"{ev.min():,.0f}" if ev.min() == ev.max()
                    else f"{ev.min():,.0f}-{ev.max():,.0f}")
            print(f"    {alg:<13} {n:3d} {B_:7,} {iters:6,} {span:>16} "
                  f"{oc.max():7,.0f}  {'PASS' if ok else 'FAIL  <<<'}")
            if not ok:
                fail.append(f"{alg} N={n} B={B_}: spent "
                            f"{ev.min():.0f}-{ev.max():.0f}")
            sys.stdout.flush()

print("\n" + "=" * 78)
if fail:
    print(f"FAILED: {len(fail)} checks")
    for m in fail[:10]:
        print(f"  - {m}")
    sys.exit(1)
print("ALL BUDGET CHECKS PASSED - the budget is a hard cap, never exceeded.")
print("=" * 78)
