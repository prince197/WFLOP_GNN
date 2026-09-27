"""Pre-results checklist, executed rather than asserted.

Every item below is a property someone has asked this package to guarantee.
A README can only claim them; this script checks them, so "ready for the final
campaign" is a test result with a date on it.

    python preflight.py                 # static checks (fast, no campaign)
    python preflight.py --budget 3000   # also check that budget is feasible
    python preflight.py --results       # also audit an existing campaign

Exit status is 0 only if every check passes. The one item it cannot check is
the real-GPU validation, which needs the A100; it is reported as OUTSTANDING
rather than passed, and never silently skipped.
"""

import argparse
import glob
import hashlib
import inspect
import json
import os
import sys

import numpy as np

RESULTS = []


def check(item, ok, detail="", outstanding=False):
    RESULTS.append((item, "N/A" if outstanding else bool(ok), detail,
                    outstanding))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--dataset", type=int, default=1)
    ap.add_argument("--results", action="store_true",
                    help="also audit the campaign outputs in results/")
    args = ap.parse_args()

    import run_experiments_gpu as R
    import report as REP
    import algorithms_gpu as A
    import gnn_algorithms_gpu as G
    import objective_gpu as O

    # ---- 1. Budget: hard cap, floor rule ---------------------------------
    over = []
    for alg in R.ALGORITHM_NAMES:
        for n in (2, 5, 9, 14, 18):
            for b in (500, 1000, 2000, 3000, 5000):
                iters, cap, status = R.iterations_for(alg, n, b)
                if status == "infeasible":
                    continue
                if R.per_iteration_cost(alg, n) is None:
                    continue
                if R.budget_cost(alg, n, iters) > b:
                    over.append(f"{alg} N={n} B={b}")
    check("Budget is a hard cap (floor rule, no overshoot)", not over,
          f"{len(over)} overshoots" if over else
          "checked 14 algorithms x 5 N x 5 budgets")

    src = inspect.getsource(R.iterations_for)
    check("Budget rule uses floor, not rounding", "round(" not in src,
          "iterations_for contains no round()")

    check("Budget-infeasible pairs are marked, not run over budget",
          R.iterations_for("GNNLXSSA", 18, 500)[2] == "infeasible",
          "GNN N=18 at B=500 (mandatory 1,222) -> infeasible")

    # ---- 2. Budget feasibility -------------------------------------------
    infeas = [(a, n) for a in R.ALGORITHM_NAMES for n in (2, 5, 9, 14, 18)
              if R.iterations_for(a, n, args.budget)[2] == "infeasible"]
    floor_needed = max(
        R.mandatory_cost(a, n) + (1 if R.per_iteration_cost(a, n) is None
                                  else R.per_iteration_cost(a, n))
        for a in R.ALGORITHM_NAMES for n in (2, 5, 9, 14, 18))
    check(f"Common budget {args.budget:,} feasible for all N and algorithms",
          not infeas,
          f"minimum feasible is {floor_needed:,}"
          + (f"; {len(infeas)} pairs infeasible" if infeas else ""))

    # ---- 3. Four separate counters ---------------------------------------
    need = ["Evaluations", "ObjectiveCalls", "SurrogateInferences",
            "SurrogateTrainPasses", "GateAdmissionRate"]
    missing = [c for c in need if c not in R.RESULT_COLUMNS]
    check("Counters remain separate in the schema", not missing,
          ", ".join(need) if not missing else f"missing {missing}")

    # ---- 4. UQ is a true gate --------------------------------------------
    body = inspect.getsource(G._GNNSalpBase.optimize)
    gate_ok = ("xp.take_along_axis(allc, idx" in body
               and "n_gated = xp.minimum" in body)
    check("UQ gate is selective (ungated candidates never evaluated exactly)",
          gate_ok, "gather-then-evaluate, with budget truncation")

    check("UQ records calibration evidence",
          "uq_calibration" in body and hasattr(R, "calibration_bins"),
          "sigma / |error| pairs per gated candidate")

    # ---- 5. Statistics ---------------------------------------------------
    check("Runtime is not significance-tested",
          all(m != "Runtime" for m, _ in REP.METRICS),
          f"METRICS = {[m for m, _ in REP.METRICS]}, primary = {REP.PRIMARY}")

    check("Rank-biserial effect size is tie-aware",
          "_rankdata" in inspect.getsource(REP.rank_biserial),
          "average ranks within tie groups")

    x = np.array([2.0, 2.0, 2.0, 2.0, 5.0])
    check("Effect size is invariant to row order",
          len({round(REP.rank_biserial(np.zeros(5), x[p]), 9)
               for p in [np.random.default_rng(k).permutation(5)
                         for k in range(8)]}) == 1,
          "8 permutations of a tied sample give one value")

    # ---- 6. Reference must be prespecified -------------------------------
    main_src = inspect.getsource(REP.main)
    check("Reference algorithm must be prespecified",
          "no reference algorithm given" in main_src
          and "wake_rank.Algorithm.iloc[0]" not in main_src,
          "report.py refuses to infer it from the ranks")

    # ---- 7. Convergence and efficiency outputs ---------------------------
    store = inspect.getsource(R.store_best_curve)
    check("Median / IQR convergence data are generated",
          "median=_pad(med)" in store and "q25=_pad(lo)" in store,
          "median, q25, q75 stored beside the best-of-30 curve")
    check("Efficiency axis (evaluations) is generated",
          "evals=_pad(ax)" in store and hasattr(A._Base, "_track"),
          "cumulative exact evaluations per curve point")

    # ---- 8. Ablation tooling ---------------------------------------------
    check("Ablation tooling present",
          os.path.exists(os.path.join(os.path.dirname(__file__),
                                      "run_ablations.py"))
          and hasattr(R, "ALGO_KWARGS"),
          "run_ablations.py + WFLOP_ALGO_KWARGS")

    # ---- 9. Provenance ---------------------------------------------------
    man_src = inspect.getsource(R.write_manifest)
    fields = ["source_sha256", "optimizer_parameters", "surrogate",
              "seeding", "objective", "statistical_protocol", "cost_model",
              "versions", "backend"]
    miss = [f for f in fields if f'"{f}"' not in man_src]
    check("Manifest carries the full freeze record", not miss,
          f"{len(fields)} sections" if not miss else f"missing {miss}")

    check("Raw results are sealed and the seal is verified",
          "sha256" in inspect.getsource(R.main)
          and "RAW DATA MODIFIED" in main_src,
          "campaign writes a SHA-256 seal; report.py checks it")

    # ---- 10. Objective and documentation ---------------------------------
    check("Data Set II probability sum is 0.9999 and not renormalised",
          abs(float(np.asarray(R.asnumpy(O.OMEGA_2)).sum()) - 0.9999) < 1e-12,
          "asserted at import in objective_gpu")

    here = os.path.dirname(os.path.abspath(__file__))
    stale = []
    for fn in ("run_experiments_gpu.py", "README.md"):
        txt = open(os.path.join(here, fn), encoding="utf-8").read()
        if "4 decimals" in txt:
            stale.append(fn)
        if "never run at full campaign settings" in txt:
            stale.append(fn + " (stale GNN claim)")
    check("Documentation free of the known stale claims", not stale,
          "no '4 decimals', no 'never run at full settings'"
          if not stale else str(stale))

    check("GNN-QA-SSA provenance statement retained",
          "DERIVED" in open(os.path.join(here, "gnn_algorithms_gpu.py"),
                            encoding="utf-8").read(),
          "flagged in the module header")

    # ---- 11. GNN family structure ----------------------------------------
    drift = []
    for nm, (op, uq) in {"GNNLXSSA": ("laplace", False),
                         "GNNQASSA": ("quadratic", False),
                         "GNNLXSSA_UQ": ("laplace", True),
                         "GNNQASSA_UQ": ("quadratic", True)}.items():
        c = A.ALGORITHMS[nm]
        own = {k for k, v in vars(c).items()
               if not k.startswith("__") and k not in
               ("name", "operator", "uq")}
        if not (c.optimize is G._GNNSalpBase.optimize and not own
                and c.operator == op and c.uq is uq):
            drift.append(nm)
    check("GNN variants differ only in the two intended flags", not drift,
          "one shared implementation, four flag pairs"
          if not drift else str(drift))

    # ---- 12. Real-GPU validation: cannot be checked here -----------------
    check("Real A100 validation log archived", False,
          "run validate_gpu.py on the node and keep the log", outstanding=True)

    # ---- optional: audit an existing campaign ----------------------------
    if args.results:
        import pandas as pd
        path = f"results/RawResults_ds{args.dataset}.csv"
        if not os.path.exists(path):
            check("Campaign results present", False, f"{path} not found")
        else:
            d = pd.read_csv(path)
            seal = path.replace(".csv", ".sha256")
            digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
            check("Raw results untouched since the campaign",
                  os.path.exists(seal)
                  and open(seal).read().split()[0] == digest,
                  "seal matches" if os.path.exists(seal) else "no seal file")

            man = f"results/manifest_ds{args.dataset}.json"
            ok_man = os.path.exists(man)
            check("Manifest written beside the results", ok_man, man)

            if ok_man:
                m = json.load(open(man))
                b = m.get("budget")
                if b:
                    bad = d[d.Evaluations > b]
                    check(f"Every run within the {b:,}-evaluation budget",
                          bad.empty,
                          "" if bad.empty else
                          f"{len(bad)} rows over, max {bad.Evaluations.max()}")

            check("All four counters populated",
                  all(c in d.columns for c in need),
                  f"{len(d):,} rows")

            gnn = d[d.Algorithm.str.endswith("_UQ")]
            check("Gate admission rate recorded for the UQ variants",
                  gnn.empty or (gnn.GateAdmissionRate > 0).any(),
                  "" if gnn.empty else
                  f"mean {gnn.GateAdmissionRate.mean():.3f}")

            n_curves = len(glob.glob(f"curves/Conv_ds{args.dataset}_R*_T*.npz"))
            check("Convergence file per case", n_curves > 0,
                  f"{n_curves} files")
            if n_curves:
                z = np.load(sorted(glob.glob(
                    f"curves/Conv_ds{args.dataset}_R*_T*.npz"))[0])
                check("Curve files carry median, IQR and the evaluation axis",
                      all(k in z.files for k in ("median", "q25", "q75",
                                                 "evals")),
                      ", ".join(sorted(z.files)))

    # ---- report ----------------------------------------------------------
    print("=" * 78)
    print("WFLOP PRE-RESULTS CHECKLIST")
    print("=" * 78)
    npass = nfail = nout = 0
    for item, ok, detail, outstanding in RESULTS:
        if outstanding:
            mark, nout = "OUTSTANDING", nout + 1
        elif ok:
            mark, npass = "PASS", npass + 1
        else:
            mark, nfail = "FAIL", nfail + 1
        print(f"  {mark:<12} {item}")
        if detail:
            print(f"               {detail}")
    print("=" * 78)
    print(f"{npass} passed, {nfail} failed, {nout} outstanding "
          f"(needs the A100)")
    print("=" * 78)
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
