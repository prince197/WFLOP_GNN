"""
============================================================
WFLOP REPORTING  -  statistics and Excel export
============================================================

Reads what the campaign wrote and produces the analysis:

    python report.py                          # all algorithms, data set 1
    python report.py --dataset 2
    python report.py --dataset 1 --paper 1    # one paper's comparison set
    python report.py --algorithms GA,SSA,GNNLXSSA --reference GNNLXSSA

THE FOUR PAPERS
---------------
    --paper 1   GNN-LX-SSA     vs GA PSO DE ACO PF BBO GWO SSA LXSSA
    --paper 2   GNN-QA-SSA     vs GA PSO DE ACO PF BBO GWO SSA QASSA
    --paper 3   GNN-LX-SSA-UQ  vs the paper-1 set + GNN-LX-SSA
    --paper 4   GNN-QA-SSA-UQ  vs the paper-2 set + GNN-QA-SSA

One campaign feeds all four. The statistics are recomputed inside
each paper's subset, because the omnibus test, the average ranks and
the size of the Holm family all depend on how many algorithms are
being compared - slicing a 14-algorithm analysis would be wrong.

Inputs
------
    results/RawResults_ds<D>.csv     one row per run
    curves/Conv_ds<D>_R*_T*.npz      one file per case (optional)

Output
------
    WFLOP_Report_ds<D>.xlsx          multi-sheet workbook
    a summary printed to stdout

WHAT IT COMPUTES, AND WHY IT DIFFERS FROM THE OLD SCRIPT
--------------------------------------------------------
1. Descriptive statistics per (radius, turbines, algorithm) for wake
   loss, energy production, evaluations and runtime: min, max, mean,
   median, std, coefficient of variation and a 95% t confidence
   interval.

2. Friedman omnibus test per case, across all algorithms. Because the
   30 seeds are shared by every algorithm, the samples are paired,
   which is what the test assumes.

3. Wilcoxon signed-rank tests of a reference algorithm against each
   other algorithm - but ONLY for cases where Friedman rejects, and
   with a HOLM correction applied within each family of comparisons.
   The previous pipeline ran roughly 1,600 uncorrected tests, where
   about eighty false positives are expected by chance alone.

4. A rank-biserial correlation effect size for every pairwise test.
   With 30 paired runs even a trivial difference reaches significance,
   so an effect size is what makes the result interpretable.

5. Average ranks per case and overall, for all three metrics.

6. A best-layout table with the winning coordinates, and convergence
   summaries read from the per-case curve files.
============================================================
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, t as student_t

# METRICS carries the SOLUTION-QUALITY measures only. Runtime is deliberately
# absent, and that is a correctness decision, not an oversight:
#
#   The 30 runs of a group are executed as one batched call, so the only
#   runtime that exists is the runtime of the whole group. Writing group_time
#   /30 onto each of the 30 rows produces 30 IDENTICAL numbers. Feeding those
#   into Friedman or Wilcoxon is pseudo-replication - n=1 measurement dressed
#   up as n=30 - and yields p-values that mean nothing.
#
# Runtime is therefore reported DESCRIPTIVELY (the Runtime sheet), never
# tested. WakeLoss and EnergyProduction are two views of the same optimum, so
# WakeLoss is the primary endpoint and EnergyProduction is reported alongside.
METRICS = [("WakeLoss", "min"), ("EnergyProduction", "max")]
PRIMARY = "WakeLoss"
ALPHA = 0.05

# ---------------------------------------------------------------------------
# THE FOUR PAPERS
# ---------------------------------------------------------------------------
# One campaign feeds all four: the runs do not depend on which comparison set
# is chosen afterwards. The STATISTICS, however, must be recomputed inside each
# paper's set - the Friedman omnibus, the average ranks and the size of the
# Holm family all change with the number of algorithms compared. Slicing a
# 14-algorithm analysis would be wrong; `--paper N` re-runs everything on that
# paper's subset.
_CLASSICAL = ["GA", "PSO", "DE", "ACO", "PF", "BBO", "GWO", "SSA"]

PAPERS = {
    1: dict(proposed="GNNLXSSA",
            algorithms=_CLASSICAL + ["LXSSA", "GNNLXSSA"],
            title="GNN-LX-SSA vs classical metaheuristics and LX-SSA"),
    2: dict(proposed="GNNQASSA",
            algorithms=_CLASSICAL + ["QASSA", "GNNQASSA"],
            title="GNN-QA-SSA vs classical metaheuristics and QA-SSA"),
    3: dict(proposed="GNNLXSSA_UQ",
            algorithms=_CLASSICAL + ["LXSSA", "GNNLXSSA", "GNNLXSSA_UQ"],
            title="GNN-LX-SSA-UQ vs the paper-1 set plus GNN-LX-SSA"),
    4: dict(proposed="GNNQASSA_UQ",
            algorithms=_CLASSICAL + ["QASSA", "GNNQASSA", "GNNQASSA_UQ"],
            title="GNN-QA-SSA-UQ vs the paper-2 set plus GNN-QA-SSA"),
}


# ---------------------------------------------------------------------------
def confidence_interval(x, confidence=0.95):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return (float(x.mean()) if n else np.nan,) * 2
    se = x.std(ddof=1) / np.sqrt(n)
    h = se * student_t.ppf((1 + confidence) / 2, n - 1)
    return float(x.mean() - h), float(x.mean() + h)


def describe(df, column):
    rows = []
    for (radius, turb, alg), g in df.groupby(["Radius", "Turbines", "Algorithm"]):
        v = g[column].to_numpy(dtype=float)
        mean, std = v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0
        lo, hi = confidence_interval(v)
        rows.append([radius, turb, alg, len(v), v.min(), v.max(), mean,
                     float(np.median(v)), std,
                     0.0 if abs(mean) < 1e-12 else std / abs(mean) * 100,
                     lo, hi])
    return pd.DataFrame(rows, columns=["Radius", "Turbines", "Algorithm", "N",
                                       "Minimum", "Maximum", "Mean", "Median",
                                       "Std", "CV (%)", "CI Lower", "CI Upper"])


def pivot_case(df, radius, turb, metric):
    """Seeds x algorithms table for one case; None if it is not complete."""
    sub = df[(df.Radius == radius) & (df.Turbines == turb)]
    p = sub.pivot_table(index="Seed", columns="Algorithm", values=metric)
    if p.isna().any().any() or p.shape[1] < 2:
        return None
    return p


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvalues, dtype=float)
    ok = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj, running = np.empty(m), 0.0
    for rank, j in enumerate(idx):
        running = max(running, (m - rank) * p[ok][j])
        adj[j] = min(1.0, running)
    out[ok] = adj
    return out


def rank_biserial(a, b):
    """Effect size for the paired Wilcoxon test, in [-1, 1].

    Positive means `a` tends to be LOWER (better, for a minimised metric).
    """
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    # AVERAGE (mid) ranks for tied absolute differences, matching what the
    # Wilcoxon test itself does. Ordinal ranks would split a tie arbitrarily by
    # array order, so two runs with the same absolute difference could be given
    # different weights and the effect size would depend on row order. Ties are
    # common here: two algorithms often reach the identical layout on an easy
    # case, and rounding to float64 makes exact ties real.
    r = _rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def _rankdata(x):
    """Ranks of x with ties assigned their average rank ('average' method)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.arange(1, x.size + 1, dtype=float)
    xs = x[order]
    i = 0
    while i < xs.size:                       # average within each tie group
        j = i
        while j + 1 < xs.size and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def effect_label(e):
    a = abs(e)
    return "negligible" if a < 0.147 else ("small" if a < 0.33 else
                                           ("medium" if a < 0.474 else "large"))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=int, default=1, choices=(1, 2))
    ap.add_argument("--paper", type=int, default=None, choices=(1, 2, 3, 4),
                    help="restrict the analysis to one paper's comparison set "
                         "and use its proposed method as the reference")
    ap.add_argument("--algorithms", default=None,
                    help="comma-separated subset (overrides --paper)")
    ap.add_argument("--reference", default=None,
                    help="algorithm to compare against (default: the paper's "
                         "proposed method, else the best average rank)")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    args = ap.parse_args()

    path = f"results/RawResults_ds{args.dataset}.csv"
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found - run the campaign first.")

    # Raw results are the shared input to all four paper analyses, so they must
    # be identical for all four. The campaign writes a SHA-256 seal beside them;
    # if the file has changed since, say so loudly rather than quietly
    # analysing edited data.
    seal = path.replace(".csv", ".sha256")
    if os.path.exists(seal):
        import hashlib
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        recorded = open(seal).read().split()[0]
        if digest != recorded:
            raise SystemExit(
                f"RAW DATA MODIFIED since the campaign wrote it.\n"
                f"  {path}\n  recorded {recorded}\n  actual   {digest}\n"
                "The four paper analyses must all derive from the same raw "
                "results. Re-run the campaign, or restore the file. If the "
                "change was intended, delete the seal deliberately - do not "
                "work around this silently.")
        print(f"raw data seal verified   : {recorded[:16]}...")
    df = pd.read_csv(path)

    paper = PAPERS.get(args.paper) if args.paper else None
    wanted = None
    if args.algorithms:
        wanted = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    elif paper:
        wanted = paper["algorithms"]

    tag = f"ds{args.dataset}"
    if wanted:
        missing = [a for a in wanted if a not in set(df.Algorithm)]
        if missing:
            raise SystemExit(
                f"these algorithms are not in {path}: {missing}\n"
                "Run the campaign for them first (WFLOP_ALGOS=...).")
        df = df[df.Algorithm.isin(wanted)].copy()
        if args.paper:
            tag += f"_paper{args.paper}"

    print("=" * 74)
    print(f"WFLOP REPORT - wind data set {args.dataset}")
    if paper:
        print(f"PAPER {args.paper}: {paper['title']}")
        print(f"proposed method  : {paper['proposed']}")
    print(f"{len(df)} runs | {df.Algorithm.nunique()} algorithms | "
          f"{df.groupby(['Radius', 'Turbines']).ngroups} cases | "
          f"{df.Seed.nunique()} seeds")
    print("=" * 74)

    sheets = {"RawResults": df}

    # ---- 1. descriptive statistics ------------------------------------
    for col in ("WakeLoss", "EnergyProduction", "Runtime", "Evaluations"):
        sheets[f"Stats_{col[:22]}"] = describe(df, col)
    print("descriptive statistics    : done")

    # ---- 2. Friedman ---------------------------------------------------
    cases = sorted(df.groupby(["Radius", "Turbines"]).groups)
    fried = []
    for metric, _ in METRICS:
        for radius, turb in cases:
            p = pivot_case(df, radius, turb, metric)
            if p is None:
                continue
            stat, pval = friedmanchisquare(*[p[c] for c in p.columns])
            fried.append([metric, radius, turb, len(p.columns), stat, pval,
                          pval < args.alpha])
    friedman_df = pd.DataFrame(fried, columns=["Metric", "Radius", "Turbines",
                                               "Algorithms", "Statistic",
                                               "PValue", "Reject"])
    sheets["Friedman"] = friedman_df
    rej = int(friedman_df.Reject.sum())
    print(f"Friedman omnibus tests    : {len(friedman_df)} run, {rej} reject "
          f"at alpha={args.alpha}")

    # ---- 3. average ranks ---------------------------------------------
    ranks = []
    for metric, direction in METRICS:
        asc = direction == "min"
        for radius, turb in cases:
            p = pivot_case(df, radius, turb, metric)
            if p is None:
                continue
            r = p.rank(axis=1, ascending=asc).mean()
            for alg, val in r.items():
                ranks.append([metric, radius, turb, alg, val])
    ranks_df = pd.DataFrame(ranks, columns=["Metric", "Radius", "Turbines",
                                            "Algorithm", "AverageRank"])
    overall = (ranks_df.groupby(["Metric", "Algorithm"]).AverageRank.mean()
               .reset_index().sort_values(["Metric", "AverageRank"]))
    sheets["Ranks_PerCase"] = ranks_df
    overall = overall.copy()
    overall["Caveat"] = ("averaged over farm cases; the 39 cases are different "
                         "physical problems, not replications of one - read "
                         "this as a summary, not as evidence with n=39")
    sheets["Ranks_Overall"] = overall
    print("average ranks             : done")

    wake_rank = overall[overall.Metric == PRIMARY].reset_index(drop=True)

    # The reference must be PRE-SPECIFIED. Picking the best-ranked algorithm
    # after seeing the ranks and then testing it against the rest is selection
    # on the outcome: the winner's p-values are optimistic by construction.
    # --paper N sets it from the paper definition; --reference sets it by hand.
    reference = args.reference or (paper["proposed"] if paper else None)
    if reference is None:
        raise SystemExit(
            "no reference algorithm given.\n"
            "Pass --paper 1..4 (uses that paper's proposed method) or "
            "--reference <ALGO>.\n"
            "The reference is deliberately NOT inferred from the ranks: "
            "choosing it after\nseeing which algorithm won would bias every "
            "p-value that follows.")
    if reference not in df.Algorithm.unique():
        raise SystemExit(f"reference algorithm {reference!r} is not in the results")

    # ---- 4. Wilcoxon, gated on Friedman, Holm-corrected ----------------
    pair_rows = []
    for metric, direction in METRICS:
        for radius, turb in cases:
            gate = friedman_df[(friedman_df.Metric == metric)
                               & (friedman_df.Radius == radius)
                               & (friedman_df.Turbines == turb)]
            if gate.empty or not bool(gate.Reject.iloc[0]):
                continue                      # omnibus did not reject: stop here
            p = pivot_case(df, radius, turb, metric)
            if p is None or reference not in p.columns:
                continue
            ref = p[reference].to_numpy(dtype=float)
            block = []
            for alg in p.columns:
                if alg == reference:
                    continue
                other = p[alg].to_numpy(dtype=float)
                try:
                    stat, pval = wilcoxon(ref, other)
                except ValueError:            # all differences zero
                    stat, pval = np.nan, 1.0
                e = rank_biserial(ref, other)
                if direction == "max":
                    e = -e
                block.append([metric, radius, turb, reference, alg, stat, pval,
                              e, effect_label(e)])
            for row, adj in zip(block, holm([b[6] for b in block])):
                row.insert(7, adj)
                row.append("yes" if adj < args.alpha else "no")
                pair_rows.append(row)

    pairs = pd.DataFrame(pair_rows, columns=[
        "Metric", "Radius", "Turbines", "Reference", "Compared", "Statistic",
        "PValue", "PValue_Holm", "EffectSize", "EffectLabel", "Significant"])
    sheets["Wilcoxon"] = pairs
    print(f"pairwise Wilcoxon         : reference = {reference}, "
          f"{len(pairs)} tests, {int((pairs.Significant == 'yes').sum())} "
          f"significant after Holm")

    # ---- 4a. runtime, descriptive only ----------------------------------
    # One measurement per (case, algorithm): the group wall-clock. Reported so
    # the cost of the methods is visible; NOT tested, for the reason in the
    # METRICS comment above.
    rt = (df.groupby(["Radius", "Turbines", "Algorithm"])
            .agg(GroupSeconds=("Runtime", lambda v: float(v.iloc[0]) * len(v)),
                 SecondsPerRunBatched=("Runtime", "first"),
                 SecondsSingleRun=("RuntimeSingleRun", "first"))
            .reset_index())
    rt["Note"] = ("one wall-clock measurement per case; the 30 runs share it, "
                  "so it must not be used in a paired test")
    sheets["Runtime"] = rt
    print("runtime (descriptive)     : done")

    # ---- 4b. evaluation budget ------------------------------------------
    # Central to the papers' claim: the proposed methods reach their results
    # with FEWER exact objective evaluations than the baselines they beat.
    # Evaluations   = exact values the optimizer used  (the algorithmic budget)
    # ObjectiveCalls= exact evaluations actually computed (the compute cost);
    #                 differs from Evaluations only for the UQ variants, whose
    #                 gated batch is padded to a rectangle
    # SurrogateInferences = surrogate forward passes used for screening and
    #                 guidance (training passes are not counted)
    agg = dict(MeanEvaluations=("Evaluations", "mean"),
               MinEvaluations=("Evaluations", "min"),
               MaxEvaluations=("Evaluations", "max"),
               MeanWakeLoss=("WakeLoss", "mean"))
    if "ObjectiveCalls" in df.columns:
        agg["MeanObjectiveCalls"] = ("ObjectiveCalls", "mean")
    scol = ("SurrogateInferences" if "SurrogateInferences" in df.columns
            else "SurrogateEvaluations")
    if scol in df.columns:
        agg["MeanSurrogateInferences"] = (scol, "mean")
    budget = (df.groupby("Algorithm").agg(**agg)
                .reset_index().sort_values("MeanEvaluations"))
    ref_budget = float(budget.loc[budget.Algorithm == reference,
                                  "MeanEvaluations"].iloc[0])
    budget["RelativeToReference"] = budget.MeanEvaluations / ref_budget
    sheets["EvaluationBudget"] = budget
    print("evaluation-budget table   : done")

    # ---- 4c. evaluation efficiency ---------------------------------------
    # Best-so-far wake loss against exact objective evaluations, read from the
    # `evals` axis stored beside each convergence curve. A fixed-iteration
    # campaign cannot show this: the algorithms spend different amounts per
    # iteration, so an iteration-indexed curve compares them at unequal cost.
    eff = []
    for path in sorted(glob.glob(f"curves/Conv_ds{args.dataset}_R*_T*.npz")):
        z = np.load(path, allow_pickle=False)
        if "evals" not in z.files:
            continue
        for name, curve, axis in zip(z["algorithms"], z["curves"], z["evals"]):
            if str(name) not in set(df.Algorithm):
                continue
            for frac in (0.25, 0.50, 0.75, 1.00):
                target = axis[-1] * frac
                j = int(np.searchsorted(axis, target))
                j = min(j, len(curve) - 1)
                eff.append([int(z["radius"]), int(z["turbines"]), str(name),
                            frac, float(axis[j]), float(curve[j])])
    if eff:
        sheets["Efficiency"] = pd.DataFrame(
            eff, columns=["Radius", "Turbines", "Algorithm", "BudgetFraction",
                          "ExactEvaluations", "BestWakeLoss"])
        print(f"efficiency curves         : {len(eff)} points")

    # ---- 5. best layouts ------------------------------------------------
    best = (df.loc[df.groupby(["Radius", "Turbines", "Algorithm"])
                   .WakeLoss.idxmin()]
              .sort_values(["Radius", "Turbines", "WakeLoss"]))
    sheets["BestLayouts"] = best[["Radius", "Turbines", "Algorithm", "Seed",
                                  "WakeLoss", "EnergyProduction", "Evaluations",
                                  "Coordinates"]]
    print("best-layout table         : done")

    # ---- 6. convergence summary ----------------------------------------
    conv = []
    for f in sorted(glob.glob(f"curves/Conv_ds{args.dataset}_R*_T*.npz")):
        d = np.load(f, allow_pickle=False)
        for name, curve, seed, wake in zip(d["algorithms"], d["curves"],
                                           d["best_seed"], d["best_wake"]):
            c = np.asarray(curve, dtype=float)
            start, end = c[0], c[-1]
            improved = np.flatnonzero(c < c[0] - 1e-9)
            conv.append([int(d["radius"]), int(d["turbines"]), str(name),
                         int(seed), float(start), float(end),
                         float(start - end),
                         int(improved[-1]) if improved.size else 0,
                         float(wake)])
    if conv:
        sheets["Convergence"] = pd.DataFrame(conv, columns=[
            "Radius", "Turbines", "Algorithm", "BestSeed", "FirstValue",
            "FinalValue", "TotalImprovement", "LastImprovingIteration",
            "BestWakeLoss"])
        print(f"convergence summary       : {len(conv)} curves")
    else:
        print("convergence summary       : no curve files found, skipped")

    # ---- write -----------------------------------------------------------
    out = f"WFLOP_Report_{tag}.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        for name, frame in sheets.items():
            frame.to_excel(xl, sheet_name=name[:31], index=False)

    print("=" * 74)
    print(f"{out} written ({len(sheets)} sheets)")
    print("\naverage rank on wake loss (lower is better):")
    for _, r in wake_rank.iterrows():
        mark = "  <-- proposed" if r.Algorithm == reference else ""
        print(f"    {r.Algorithm:14} {r.AverageRank:6.2f}{mark}")

    if paper:
        wl = pairs[(pairs.Metric == "WakeLoss")]
        sig = wl[wl.Significant == "yes"]
        beat = sig[sig.EffectSize > 0]
        pos = int(wake_rank.index[wake_rank.Algorithm == reference][0]) + 1
        print(f"\nheadline for paper {args.paper}:")
        print(f"    average-rank position of {reference}: "
              f"{pos} of {len(wake_rank)}")
        print(f"    wake-loss comparisons significant after Holm: "
              f"{len(sig)} of {len(wl)}")
        print(f"    of those, {reference} is the better method in {len(beat)}")
        print(f"    exact evaluations per run, {reference}: "
              f"{ref_budget:,.0f} "
              f"(baselines: {budget[budget.Algorithm != reference].MeanEvaluations.min():,.0f}"
              f"-{budget.MeanEvaluations.max():,.0f})")
    print("=" * 74)


if __name__ == "__main__":
    main()
