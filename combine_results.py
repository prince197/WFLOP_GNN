"""
============================================================
COMBINED RAW RESULTS  -  one workbook, one sheet per data set
============================================================

    python combine_results.py                  # data sets 1 and 2
    python combine_results.py --datasets 1     # just one
    python combine_results.py --tag ABL_A      # an ablation's outputs
    python combine_results.py --budget 3000    # the fixed-budget campaign
    python combine_results.py --out My.xlsx

Reads what the campaign wrote for each wind data set and puts them
side by side in a single workbook:

    WFLOP_RawResults_AllDatasets.xlsx
        RawResults_DS1     every run of wind data set I
        RawResults_DS2     every run of wind data set II

WHY THIS IS A SEPARATE SCRIPT, NOT PART OF report.py
----------------------------------------------------
report.py analyses ONE data set at a time, and that is deliberate: the
Friedman omnibus, the average ranks and the Holm family are all computed
WITHIN a data set, because the two data sets are different wind regimes
and not replications of one another. Pooling them would be wrong.

This script does no statistics at all. It is a transport step: the same
rows the campaign wrote, in one file, so a co-author can open both data
sets without hunting for two CSVs. The per-paper analysis workbooks
(WFLOP_Report_ds<D>_paper<N>.xlsx) are unaffected and are still the place
the statistics live.

THE CSV REMAINS THE SOURCE OF TRUTH
-----------------------------------
results/RawResults_ds<D>.csv and its .sha256 seal are what the papers
derive from. This workbook is a copy, so it is written AFTER the seal is
verified - if a raw file has been edited since the campaign wrote it, this
script stops for exactly the same reason report.py does. No row is
dropped, reordered or edited, and the column order is the campaign's.

ONE PRECISION CAVEAT, STATED RATHER THAN GLOSSED
------------------------------------------------
The xlsx format stores a float at 16 significant digits, one short of the
17 a float64 needs to round-trip exactly. Numeric columns in this workbook
are therefore accurate to about 1 part in 1e16 - the last bit, and nothing
that matters for reading or plotting - but they are NOT bit-identical to
the CSV. Anything that must be exact should be read from the CSV.

The one column where exactness genuinely matters is Coordinates, and it is
safe: it is a STRING, so it is copied verbatim and re-evaluating a layout
from this workbook reproduces its WakeLoss exactly as it does from the CSV.
That matters because the constraint penalty has a coefficient of 1e10 - a
layout rounded even slightly can re-evaluate with a huge spurious penalty.
============================================================
"""

import argparse
import hashlib
import os

import pandas as pd

OUT_DEFAULT = "WFLOP_RawResults_AllDatasets.xlsx"

# Coordinates is a long string (one "x y" pair per turbine at full float
# precision), so it gets a narrow column and everything else a readable one.
# The cell still holds the complete value; only the displayed width changes.
WIDE = {"Coordinates": 40, "Algorithm": 14}
DEFAULT_WIDTH = 13


def verify_seal(path):
    """Stop if the raw file has changed since the campaign sealed it.

    Same rule as report.py: the analyses must all derive from the file the
    campaign actually wrote, so an edit is made loud rather than merely
    forbidden. A missing seal is not an error - a campaign from before the
    seal existed, or an ablation, simply has none.
    """
    seal = path.replace(".csv", ".sha256")
    if not os.path.exists(seal):
        return "no seal found"
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    recorded = open(seal).read().split()[0]
    if digest != recorded:
        raise SystemExit(
            f"RAW DATA MODIFIED since the campaign wrote it.\n"
            f"  {path}\n  recorded {recorded}\n  actual   {digest}\n"
            "Re-run the campaign, or restore the file. If the change was "
            "intended, delete the seal deliberately - do not work around this "
            "silently.")
    return f"seal verified {recorded[:16]}..."


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", type=int, nargs="+", default=[1, 2],
                    choices=(1, 2),
                    help="wind data sets to include (default: 1 2)")
    ap.add_argument("--tag", default="",
                    help="WFLOP_TAG of an ablation, to combine its outputs "
                         "instead of the main campaign's")
    ap.add_argument("--budget", type=int, default=None,
                    help="combine the fixed-budget campaign (WFLOP_BUDGET)")
    ap.add_argument("--smoke", action="store_true",
                    help="combine the WFLOP_SMOKE=1 pipeline check")
    ap.add_argument("--out", default=None, help="output workbook name")
    args = ap.parse_args()

    # same suffix rule as run_experiments_gpu.py
    tag = "".join(f"_{p}" for p in (args.tag,
                                    "smoke" if args.smoke else "",
                                    f"B{args.budget}" if args.budget else "")
                  if p)
    out = args.out or (f"WFLOP_RawResults_AllDatasets{tag}.xlsx")

    print("=" * 74)
    print("WFLOP COMBINED RAW RESULTS")
    print("=" * 74)

    frames, missing = {}, []
    for ds in args.datasets:
        path = f"results/RawResults_ds{ds}{tag}.csv"
        if not os.path.exists(path):
            missing.append(path)
            print(f"data set {ds}: {path} not found - skipped")
            continue
        note = verify_seal(path)
        df = pd.read_csv(path)
        frames[ds] = df
        print(f"data set {ds}: {len(df):,} rows | "
              f"{df.Algorithm.nunique()} algorithms | "
              f"{df.groupby(['Radius', 'Turbines']).ngroups} cases | "
              f"{df.Seed.nunique()} seeds | {note}")

    if not frames:
        raise SystemExit(
            "nothing to combine - no raw results found.\n"
            f"looked for: {', '.join(missing)}\n"
            "Run the campaign first (python run_experiments_gpu.py).")

    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        for ds, df in sorted(frames.items()):
            sheet = f"RawResults_DS{ds}"
            df.to_excel(xl, sheet_name=sheet, index=False)
            ws = xl.sheets[sheet]
            ws.freeze_panes = "A2"                 # keep the header visible
            for i, col in enumerate(df.columns, start=1):
                letter = ws.cell(row=1, column=i).column_letter
                ws.column_dimensions[letter].width = WIDE.get(col,
                                                              DEFAULT_WIDTH)

    print("-" * 74)
    print(f"{out} written ({len(frames)} sheet(s): "
          f"{', '.join(f'RawResults_DS{d}' for d in sorted(frames))})")
    if missing:
        print(f"NOTE: {len(missing)} data set(s) were absent and are not in "
              "the workbook.")
    print("The per-paper statistics remain in WFLOP_Report_ds<D>_paper<N>.xlsx "
          "-\nthey are computed within a data set and are not pooled here.")
    print("Numeric cells are stored at 16 significant digits (the xlsx limit), "
          "so this\nworkbook is for reading and plotting; results/RawResults_"
          "ds<D>.csv stays the\nbit-exact source. Coordinates are strings and "
          "are copied verbatim.")
    print("=" * 74)


if __name__ == "__main__":
    main()
