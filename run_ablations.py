"""The ablations Papers 2 and 4 need, as one command.

WHY

GNN-QA-SSA and GNN-QA-SSA-UQ are derived methods: the GNN machinery is shared
with the LX variants and only the follower candidate operator differs. That is
a legitimate research design, but it puts the burden of proof on the papers -
a reviewer will ask which component actually earns the result, and "we swapped
the operator and it got better" is not an answer unless the alternatives were
measured too.

Five ablations answer that, and this script runs all five and tabulates them:

  A  SSA -> QA-SSA                   does the quadratic operator help at all,
                                     before any GNN is involved?
  B  GNN-LX-SSA -> GNN-QA-SSA        does it still help once the surrogate is
                                     doing the screening? Identical GNN
                                     settings, so only the operator differs.
  C  GNN-QA-SSA with / without       does the direction-head guidance earn its
     guidance (lambda_g = 0)         place, or is the surrogate's power head
                                     doing all the work?
  D  GNN-QA-SSA -> GNN-QA-SSA-UQ     what does the uncertainty gate change -
                                     in quality, and in evaluations spent?
  E  gate threshold sweep            how sensitive is the saving to sigma? A
                                     result that only holds at one hand-picked
                                     threshold is not a result.

Each ablation is an ordinary campaign with WFLOP_TAG set, so its outputs land
beside the main campaign's rather than on top of them, and each gets its own
manifest. Nothing here touches the frozen raw results.

USAGE

    python run_ablations.py                    # all five, default settings
    python run_ablations.py --only A,C         # a subset
    python run_ablations.py --budget 3000      # under the fixed-budget regime
    python run_ablations.py --cases 500:5,500:9 --runs 10   # cheaper pilot
    python run_ablations.py --dry-run          # print the commands only

Run the LX equivalents for Papers 1 and 3 with --family LX.
"""

import argparse
import json
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def ablations(family):
    """(tag, algorithms, overrides, question) for each ablation."""
    G = f"GNN{family}SSA"                     # GNNQASSA or GNNLXSSA
    base = "QASSA" if family == "QA" else "LXSSA"
    return [
        ("A_operator_plain", ["SSA", base], {},
         f"Does the {family} operator help before any GNN is involved?"),
        ("B_operator_gnn", ["GNNLXSSA", "GNNQASSA"], {},
         "Does it still help once the surrogate screens? Identical GNN "
         "settings, so only the operator differs."),
        ("C_guidance", [G], {G: {"lambda_g": 0.0}},
         "Does the direction-head guidance earn its place? Compare against "
         "the main campaign's run of the same algorithm."),
        ("D_uq", [G, G + "_UQ"], {},
         "What does the uncertainty gate change, in quality and in "
         "evaluations spent?"),
        ("E_threshold_lo", [G + "_UQ"], {G + "_UQ": {"sigma_threshold": 0.02}},
         "Gate threshold sweep, low."),
        ("E_threshold_hi", [G + "_UQ"], {G + "_UQ": {"sigma_threshold": 0.10}},
         "Gate threshold sweep, high."),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="QA", choices=("QA", "LX"),
                    help="QA for Papers 2 and 4 (default); LX for 1 and 3")
    ap.add_argument("--dataset", type=int, default=1, choices=(1, 2))
    ap.add_argument("--budget", type=int, default=None,
                    help="run under the fixed-evaluation-budget regime")
    ap.add_argument("--runs", type=int, default=None,
                    help="override NUM_RUNS for a cheaper pilot")
    ap.add_argument("--cases", default=None,
                    help="restrict to cases, e.g. 500:5,500:9")
    ap.add_argument("--only", default=None,
                    help="comma-separated ablation prefixes, e.g. A,C")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    todo = ablations(args.family)
    if args.only:
        want = tuple(x.strip() for x in args.only.split(","))
        todo = [t for t in todo if t[0].startswith(want)]

    print("=" * 78)
    print(f"ABLATIONS - {args.family} family, data set {args.dataset}"
          + (f", budget {args.budget:,}" if args.budget else
             ", fixed iterations"))
    print("=" * 78)

    for tag, algos, overrides, question in todo:
        env = dict(os.environ)
        env["WFLOP_TAG"] = f"abl_{tag}"
        env["WFLOP_DATASET"] = str(args.dataset)
        env["WFLOP_ALGOS"] = ",".join(algos)
        if overrides:
            env["WFLOP_ALGO_KWARGS"] = json.dumps(overrides)
        else:
            env.pop("WFLOP_ALGO_KWARGS", None)
        if args.budget:
            env["WFLOP_BUDGET"] = str(args.budget)
        if args.runs:
            env["WFLOP_RUNS"] = str(args.runs)
        if args.cases:
            env["WFLOP_CASES"] = args.cases

        print(f"\n--- {tag} " + "-" * (70 - len(tag)))
        print(f"    {question}")
        print(f"    algorithms : {', '.join(algos)}")
        if overrides:
            print(f"    overrides  : {json.dumps(overrides)}")
        if args.dry_run:
            print("    (dry run)")
            continue

        r = subprocess.run([sys.executable, "run_experiments_gpu.py"],
                           cwd=HERE, env=env)
        if r.returncode != 0:
            print(f"    FAILED (exit {r.returncode})")
            sys.exit(r.returncode)

    if args.dry_run:
        return

    # ---- one comparison table across everything that ran -------------------
    print("\n" + "=" * 78)
    print("ABLATION SUMMARY")
    print("=" * 78)
    rows = []
    for tag, algos, overrides, question in todo:
        path = f"results/RawResults_ds{args.dataset}_abl_{tag}.csv"
        p = os.path.join(HERE, path)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        for alg, g in d.groupby("Algorithm"):
            rows.append({
                "Ablation": tag,
                "Algorithm": alg,
                "Override": json.dumps(overrides.get(alg, {})) or "-",
                "MeanWakeLoss": g.WakeLoss.mean(),
                "MedianWakeLoss": g.WakeLoss.median(),
                "BestWakeLoss": g.WakeLoss.min(),
                "MeanEvaluations": g.Evaluations.mean(),
                "MeanObjectiveCalls": g.ObjectiveCalls.mean(),
                "MeanGateRate": g.GateAdmissionRate.mean(),
                "Runs": len(g)})
    if not rows:
        print("no ablation results found")
        return

    out = pd.DataFrame(rows)
    path = os.path.join(HERE, f"results/ablation_summary_ds{args.dataset}.csv")
    out.to_csv(path, index=False)
    with pd.option_context("display.width", 200,
                           "display.max_columns", 20):
        print(out.round(3).to_string(index=False))
    print(f"\nwritten to {path}")
    print("\nRead each block against the one above it: an ablation is evidence "
          "only if\nthe comparison holds at an equal evaluation budget, so "
          "prefer --budget for the\nfinal numbers that go into the papers.")


if __name__ == "__main__":
    main()
