"""
Objective-robustness re-scoring.

Every layout returned by the campaign is stored in the BestLayouts sheet of the
two paper1 workbooks. This script re-evaluates each of those layouts under
an alternative power curve WITHOUT re-optimizing, and asks whether
the ordering of the algorithms survives the change of scoring function.

Curves
------
  linear   f(s) = lambda' s + eta                       (the campaign objective)
  cubic    f(s) = P_r (s^3 - s_ci^3)/(s_r^3 - s_ci^3)   (Section 3.3)

Both use the identical 24-sector x 21-speed-bin discretization, so the only
thing that changes between them is f evaluated at the bin midpoints. The
linear branch is validated against the workbook's own EnergyProduction column
before anything else is reported.
"""
import sys
import numpy as np

sys.path.insert(0, "/tmp/wfgpu/WFLOP_GPU")
import os
os.environ.setdefault("WFLOP_BACKEND", "cpu")

import objective_gpu as OBJ          # noqa: E402
from backend import asnumpy         # noqa: E402

SPEED = np.asarray(OBJ._SPEED, dtype=float)      # 22 edges -> 21 bins
SMID = 0.5 * (SPEED[:-1] + SPEED[1:])
K_SHAPE = OBJ.K_SHAPE
LAMBDA, ETTA = OBJ.LAMBDA, OBJ.ETTA
P_RATED, CUT_IN, RATED = OBJ.P_RATED, OBJ.CUT_IN, OBJ.RATED
SECTOR_WIDTH = OBJ.SECTOR_WIDTH


def f_linear(s):
    return LAMBDA * s + ETTA


def f_cubic(s):
    return P_RATED * (s ** 3 - CUT_IN ** 3) / (RATED ** 3 - CUT_IN ** 3)


def expected_power(P, dataset, curve, omega=None):
    """P : (B,N,2) -> (B,) expected farm power under `curve`.

    Identical discretization to objective_gpu.expected_power_batch: the
    integral over each speed bin is a midpoint rule weighted by the Weibull
    survival difference, so the two agree exactly when curve is f_linear.
    """
    cfg = OBJ._cfg(dataset)
    psi = np.asarray(asnumpy(cfg["psi"]), dtype=float)
    w = np.asarray(asnumpy(cfg["omega"] if omega is None else omega),
                   dtype=float)
    c = np.asarray(asnumpy(OBJ.waked_speeds(P, cfg["psi"])), dtype=float)
    inv = 1.0 / c                                             # (B,24,N)
    wsec = (SECTOR_WIDTH * w)[None, :, None]                  # (1,24,1)

    E = np.exp(-((SPEED[:, None, None, None] * inv[None]) ** K_SHAPE))
    band = wsec[None] * (E[:-1] - E[1:])                      # (21,B,24,N)
    ramp = np.sum(curve(SMID)[:, None, None, None] * band, axis=(0, 2))  # (B,N)

    e_rated = np.exp(-((RATED * inv) ** K_SHAPE))
    flat = P_RATED * np.sum(wsec * e_rated, axis=1)           # (B,N)
    return np.sum(ramp + flat, axis=1)


def parse_layout(s):
    return np.array([[float(v) for v in pair.split()]
                     for pair in s.strip().split(";")], dtype=float)


def load(path):
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb["BestLayouts"]
    it = ws.iter_rows(min_row=2, values_only=True)
    out = []
    for radius, nt, alg, seed, wake, ep, ev, coords in it:
        if coords is None:
            continue
        out.append(dict(radius=int(radius), n=int(nt), alg=str(alg),
                        seed=int(seed), wake=float(wake), ep=float(ep),
                        P=parse_layout(str(coords))))
    return out


DS = {1: "/root/.claude/uploads/ec2f79c7-655c-56d9-8a95-4b84f717c813/"
         "c9895715-1789104543502_WFLOP_Report_ds1_paper1.xlsx",
      2: "/root/.claude/uploads/ec2f79c7-655c-56d9-8a95-4b84f717c813/"
         "8f88b263-1789104552201_WFLOP_Report_ds2_paper1.xlsx"}


def main():
    import pickle
    res = {}
    for ds, path in DS.items():
        rows = load(path)
        print(f"data set {ds}: {len(rows)} stored layouts")
        # group by turbine count so the (B,N,2) stacks are rectangular
        by_n = {}
        for i, rw in enumerate(rows):
            by_n.setdefault(rw["n"], []).append(i)
        lin = np.zeros(len(rows))
        cub = np.zeros(len(rows))
        for n, idx in sorted(by_n.items()):
            P = np.stack([rows[i]["P"] for i in idx])
            lin[idx] = expected_power(P, ds, f_linear)
            cub[idx] = expected_power(P, ds, f_cubic)
        stored = np.array([rw["ep"] for rw in rows])
        feas = np.array([rw["wake"] < 1e6 for rw in rows])
        err = np.abs(lin[feas] - stored[feas]) / np.maximum(stored[feas], 1e-9)
        print(f"  linear branch vs stored EnergyProduction on {feas.sum()} "
              f"feasible layouts: max relative error {err.max():.3e}")
        res[ds] = dict(rows=[{k: v for k, v in rw.items() if k != "P"}
                             for rw in rows],
                       linear=lin, cubic=cub, feasible=feas)
    with open("/home/claude/wflop/rescore.pkl", "wb") as fh:
        pickle.dump(res, fh)
    print("written rescore.pkl")


if __name__ == "__main__":
    main()
