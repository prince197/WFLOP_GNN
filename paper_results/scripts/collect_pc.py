"""Collect the commercial-power-curve campaign from its 10 result branches.
Writes RawResults_ds<D>_ge15.csv and Efficiency_ds<D>_ge15.csv next to this script."""
import os, io, subprocess, sys
import numpy as np, pandas as pd
REPO = "/home/user/wflop_gnn"; HERE = os.path.dirname(os.path.abspath(__file__))
SLICES = ["gnn-500-750", "gnn-1000a", "gnn-1000b", "base-a", "base-b"]
def git(*a, binary=False):
    out = subprocess.check_output(["git", "-C", REPO, *a])
    return out if binary else out.decode()
for ds in (1, 2):
    raws, eff, missing = [], [], []
    for sl in SLICES:
        br = f"pc-results-ds{ds}-{sl}"
        try: git("fetch", "-q", "origin", br)
        except subprocess.CalledProcessError: missing.append(br); continue
        files = git("ls-tree", "-r", "--name-only", "FETCH_HEAD").split()
        status = git("show", "FETCH_HEAD:run_status.txt") if "run_status.txt" in files else ""
        if "exit 0" not in status: missing.append(br + " (not finished: " + status.strip().replace(chr(10), " ") + ")"); continue
        for f in files:
            if f.startswith(f"results/RawResults_ds{ds}_ge15") and f.endswith(".csv") and "checkpoint" not in f:
                raws.append(pd.read_csv(io.StringIO(git("show", f"FETCH_HEAD:{f}"))))
            if f.startswith(f"curves/Conv_ds{ds}_ge15") and f.endswith(".npz"):
                z = np.load(io.BytesIO(git("show", f"FETCH_HEAD:{f}", binary=True)), allow_pickle=False)
                if "evals" not in z.files: continue
                for name, curve, axis in zip(z["algorithms"], z["curves"], z["evals"]):
                    for frac in (0.25, 0.50, 0.75, 1.00):
                        j = min(int(np.searchsorted(axis, axis[-1] * frac)), len(curve) - 1)
                        eff.append([int(z["radius"]), int(z["turbines"]), str(name), frac, float(axis[j]), float(curve[j])])
    if missing: print(f"DS{ds} incomplete: {missing}")
    if not raws: continue
    raw = pd.concat(raws, ignore_index=True)
    dup = raw.duplicated(["Radius", "Turbines", "Algorithm", "Seed"]).sum()
    cells = raw.groupby(["Radius", "Turbines", "Algorithm"]).size()
    print(f"DS{ds}: {len(raw)} runs, {cells.size} cells, runs/cell {cells.min()}-{cells.max()}, duplicates {dup}, algorithms {sorted(raw.Algorithm.unique())}")
    raw.to_csv(os.path.join(HERE, f"RawResults_ds{ds}_ge15.csv"), index=False)
    pd.DataFrame(eff, columns=["Radius", "Turbines", "Algorithm", "BudgetFraction", "ExactEvaluations", "BestWakeLoss"]).to_csv(
        os.path.join(HERE, f"Efficiency_ds{ds}_ge15.csv"), index=False)
