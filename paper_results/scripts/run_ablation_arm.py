"""
Scoped three-arm ablation, run entirely on one machine so the comparison
is internally consistent:

    LXSSA           penalty constraint handling, no repair, no surrogate
    LXSSA_REPAIR    repair constraint handling, no surrogate
    GNNLXSSA        repair constraint handling, GNWM surrogate

Protocol identical to the main campaign: POP = 30, ITER = 100, 30 seeds
advanced in lockstep, group seed from the campaign's own crc32 rule.

Output: ablation_results.csv with one row per (dataset, radius, turbines,
algorithm, seed).
"""
import csv
import os
import sys
import time
import zlib

import numpy as np

os.environ.setdefault("WFLOP_BACKEND", "cpu")

from backend import xp, asnumpy, device_info                    # noqa: E402
from objective_gpu import (objective_batch, energy_production_batch,
                           min_spacing)                          # noqa: E402
from algorithms_gpu import build                                 # noqa: E402
import gnn_algorithms_gpu                                        # noqa: F401,E402
import ablation_repair                                           # noqa: F401,E402

CASES = [tuple(int(v) for v in c.split(":")) for c in os.environ.get("ABL_CASES", "500:10,750:14,1000:18").split(",")]
DATASETS = [int(x) for x in os.environ.get("ABL_DS", "1,2").split(",")]
ARMS = os.environ.get("ABL_ARMS", "LXSSA,LXSSA_REPAIR,GNNLXSSA").split(",")
NUM_RUNS, POP, ITER = 30, 30, 100

OUT = os.environ.get("ABL_OUT") or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "ablation_results.csv")


def group_seed_of(radius, n_turb, alg_name):
    return zlib.crc32(f"{radius}|{n_turb}|{alg_name}".encode()) % (2 ** 31)


def feasibility(P, radius, min_dist):
    """P : (R,n,2) -> boolean feasibility per run (boundary + spacing)."""
    ok = []
    for lay in P:
        rad = np.sqrt((lay ** 2).sum(axis=1))
        d = np.sqrt(((lay[:, None, :] - lay[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        ok.append(bool(rad.max() <= radius + 1e-6 and d.min() >= min_dist - 1e-6))
    return np.array(ok)


def main():
    print(device_info(), flush=True)
    done = set()
    rows = []
    if os.path.exists(OUT):
        with open(OUT) as fh:
            for row in csv.DictReader(fh):
                rows.append(row)
                done.add((int(row["Dataset"]), int(row["Radius"]),
                          int(row["Turbines"]), row["Algorithm"]))
        print(f"resuming; {len(done)} groups already complete", flush=True)

    fields = ["Dataset", "Radius", "Turbines", "Algorithm", "Seed",
              "Objective", "WakeLoss", "EnergyProduction", "Feasible",
              "Evaluations", "GroupSeconds"]

    for ds in DATASETS:
        md = min_spacing(ds)
        for radius, n_turb in CASES:
            for arm in ARMS:
                key = (ds, radius, n_turb, arm)
                if key in done:
                    continue
                dim = 2 * n_turb
                f = lambda X, _r=radius, _d=ds: objective_batch(X, float(_r),
                                                                dataset=_d)
                kw = dict(dataset=ds) if arm != "LXSSA" else {}
                algo = build(arm, **kw)
                t0 = time.perf_counter()
                best_x, best_f, _curves, n_evals = algo.optimize(
                    f, dim, -float(radius), float(radius),
                    NUM_RUNS, POP, ITER, seed=group_seed_of(radius, n_turb, arm))
                dt = time.perf_counter() - t0

                bx = asnumpy(best_x).reshape(NUM_RUNS, n_turb, 2)
                bf = np.atleast_1d(asnumpy(best_f)).astype(float)
                ep = np.atleast_1d(asnumpy(
                    energy_production_batch(best_x, dataset=ds))).astype(float)
                ev = np.broadcast_to(np.atleast_1d(asnumpy(n_evals)),
                                     (NUM_RUNS,)).astype(float)
                feas = feasibility(bx, radius, md)

                for k in range(NUM_RUNS):
                    rows.append({
                        "Dataset": ds, "Radius": radius, "Turbines": n_turb,
                        "Algorithm": arm, "Seed": k + 1,
                        "Objective": f"{bf[k]:.6f}",
                        "WakeLoss": f"{bf[k]:.6f}" if feas[k] else "",
                        "EnergyProduction": f"{ep[k]:.6f}",
                        "Feasible": int(feas[k]),
                        "Evaluations": int(ev[k]),
                        "GroupSeconds": f"{dt:.3f}"})

                with open(OUT, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=fields)
                    w.writeheader()
                    w.writerows(rows)
                print(f"ds{ds} R{radius} T{n_turb} {arm:13s} "
                      f"{dt:7.1f}s  feasible {int(feas.sum())}/30  "
                      f"evals {int(ev[0])}", flush=True)
    print("ABLATION COMPLETE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
