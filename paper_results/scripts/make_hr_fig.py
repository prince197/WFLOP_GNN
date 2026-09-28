"""Horns Rev layout figure: real layout, best GNN-LX-SSA layout, least-violating baseline layout."""
import os, sys
os.environ.update(WFLOP_SITE="hornsrev", WFLOP_BACKEND="cpu")
sys.path.insert(0, "/home/user/wflop_hr")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import site_hornsrev as S
RES, wake, out = sys.argv[1], sys.argv[2], sys.argv[3]
WARM = sys.argv[4] if len(sys.argv) > 4 else None
poly = np.vstack([S.POLYGON, S.POLYGON[:1]]) / 1000
def load(alg, res=None):
    d = pd.read_csv(f"{res or RES}/RawResults_ds1_{wake}_{alg}_hr.csv")
    return d, [np.array([[float(v) for v in p.split()] for p in c.split(";")]) for c in d.Coordinates]
def close_pairs(P):
    d = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1)); i, j = np.triu_indices(len(P), 1)
    m = d[i, j] < 320 - 1e-6; return i[m], j[m]
def outside(P):
    E = np.roll(S.POLYGON, -1, 0) - S.POLYGON; rel = P[:, None] - S.POLYGON
    return ((E[:, 0] * rel[..., 1] - E[:, 1] * rel[..., 0]) < -1e-6).any(-1)
g, gl = load("GNNLXSSA"); gb = gl[int(np.argmax(g.EnergyProduction.values))]
b, bl = load("BBO"); bb = bl[int(np.argmin(b.WakeLoss.values))]
w, wl = load("GNNLXSSA", WARM); wb = wl[int(np.argmax(w.EnergyProduction.values))]
panels = [("(a) As-built Horns Rev 1", S.REAL_LAYOUT), ("(b) GNN-LX-SSA, random start, best of 30", gb),
          ("(c) GNN-LX-SSA, as-built start, best of 30", wb), ("(d) BBO, least-penalized of 30 runs", bb)]
fig, ax = plt.subplots(2, 2, figsize=(9.5, 7.6), sharex=True, sharey=True); ax = ax.ravel()
for a, (t, P) in zip(ax, panels):
    a.plot(poly[:, 0], poly[:, 1], color="#555555", lw=1.0)
    o = outside(P); i, j = close_pairs(P)
    for u, v in zip(i, j): a.plot(P[[u, v], 0] / 1000, P[[u, v], 1] / 1000, color="#d62728", lw=1.2)
    a.scatter(P[~o, 0] / 1000, P[~o, 1] / 1000, s=14, color="#1f5aa6", zorder=3, label="turbine")
    if o.any(): a.scatter(P[o, 0] / 1000, P[o, 1] / 1000, s=22, marker="x", color="#d62728", zorder=4, label="outside the site")
    a.set_title(f"{t}\n{int(o.sum())} turbines outside, {len(i)} pairs closer than 4D", fontsize=9.5)
    a.set_aspect("equal"); a.grid(color="#e5e5e5", lw=0.6)
for a in ax[2:]: a.set_xlabel("x (km)")
for a in ax[::2]: a.set_ylabel("y (km)")
h = [plt.Line2D([], [], color="#1f5aa6", marker="o", ls="", ms=4, label="turbine"),
     plt.Line2D([], [], color="#d62728", marker="x", ls="", ms=6, label="turbine outside the site"),
     plt.Line2D([], [], color="#d62728", lw=1.2, label="pair closer than 4D (320 m)"),
     plt.Line2D([], [], color="#555555", lw=1.0, label="site boundary (convex hull)")]
fig.legend(handles=h, loc="lower center", ncol=2, fontsize=8.5, frameon=False)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(out, dpi=200)
print("saved", out)
