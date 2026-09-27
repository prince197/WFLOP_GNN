"""Horns Rev analysis: feasibility, violations, AEP and wake loss for every algorithm and wake model."""
import os, sys, glob, json
os.environ.update(WFLOP_SITE="hornsrev", WFLOP_BACKEND="cpu")
sys.path.insert(0, os.environ.get("WFLOP_CODE", "/home/user/wflop_hr"))  # checkout of the hornsrev branch; sys.dont_write_bytecode = True
import numpy as np, pandas as pd
import site_hornsrev as HR
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
POLY = np.asarray(HR.POLYGON); EDGE = np.roll(POLY, -1, 0) - POLY
def boundary(P):
    rel = P[:, None, :] - POLY
    cross = EDGE[:, 0] * rel[..., 1] - EDGE[:, 1] * rel[..., 0]
    out = (cross < 0).any(-1)
    t = np.clip((rel * EDGE).sum(-1) / (EDGE ** 2).sum(-1), 0, 1)
    q = POLY + t[..., None] * EDGE
    d = np.sqrt(((P[:, None, :] - q) ** 2).sum(-1)).min(-1)
    return np.where(out, d, 0.0)
def spacing(P):
    d = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1)); iu = np.triu_indices(len(P), 1)
    return d[iu]
out = {}
for wake in ("jensen", "gaussian"):
    wakemod = {}
    os.environ["WFLOP_WAKE"] = wake
    import importlib, objective_gpu as O; importlib.reload(O)
    ideal = float(O.HR_IDEAL) * 80
    ep_real = float(np.asarray(O.energy_production_batch(HR.REAL_LAYOUT.reshape(1, -1), 0))[0])
    wakemod["_real"] = dict(ep_kW=ep_real, aep_GWh=ep_real * 8760 / 1e6, wake_loss_pct=100 * (1 - ep_real / ideal),
                           ideal_kW=ideal, min_spacing_m=float(spacing(HR.REAL_LAYOUT).min()))
    for f in sorted(glob.glob(os.path.join(RES, f"RawResults_ds1_{wake}_*_hr.csv"))):
        df = pd.read_csv(f); alg = df.Algorithm.iloc[0]; rows = []
        for _, r in df.iterrows():
            P = np.array([[float(v) for v in p.split()] for p in r.Coordinates.split(";")])
            b = boundary(P); s = spacing(P)
            nb = int((b > HR.BOUNDARY_TOL).sum()); ns = int((s < O.MIN_SPACING - 1e-6).sum())
            ep = float(np.asarray(O.energy_production_batch(P.reshape(1, -1), 0))[0])
            rows.append(dict(feasible=(nb == 0 and ns == 0), n_out=nb, max_out_m=float(b.max()), n_close=ns,
                             min_sp_m=float(s.min()), ep=ep, obj=float(r.WakeLoss), evals=int(r.Evaluations)))
        R = pd.DataFrame(rows); feas = R[R.feasible]
        d = dict(runs=len(R), feasible=int(R.feasible.sum()), evals=int(R.evals.median()),
                 med_n_out=float(R.n_out.median()), med_n_close=float(R.n_close.median()),
                 med_min_sp=float(R.min_sp_m.median()), med_max_out=float(R.max_out_m.median()),
                 min_obj=float(R.obj.min()))
        if len(feas):
            wl = 100 * (1 - feas.ep / ideal)
            d.update(best_ep_kW=float(feas.ep.max()), med_ep_kW=float(feas.ep.median()), worst_ep_kW=float(feas.ep.min()),
                     best_wl=float(wl.min()), med_wl=float(wl.median()), worst_wl=float(wl.max()), std_ep=float(feas.ep.std()),
                     best_aep=float(feas.ep.max() * 8760 / 1e6), med_aep=float(feas.ep.median() * 8760 / 1e6),
                     beat_real=int((feas.ep > ep_real).sum()),
                     best_idx=int(feas.ep.idxmax()), med_min_sp_feas=float(feas.min_sp_m.median()))
        wakemod[alg] = d
    out[wake] = wakemod
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hr_summary.json"), "w"), indent=1)
for w, m in out.items():
    print(w, json.dumps(m["_real"]))
    for a, d in m.items():
        if a != "_real": print(f"  {a:9s}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()})
