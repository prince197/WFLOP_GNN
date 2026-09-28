"""Re-score Horns Rev layouts in PyWake: at the campaign's 12 sector centres (like for like)
and at 1-degree directions (PyWake default wd, 0.5 m/s speeds), to check the optimizer did not
exploit the 12-direction discretization. Usage: hr_pywake_rescore.py <wake> <results_csv>"""
import os, sys, json
os.environ.update(WFLOP_SITE="hornsrev", WFLOP_BACKEND="cpu")
sys.path.insert(0, os.environ.get("WFLOP_CODE", "/home/user/wflop_hr"))
import numpy as np, pandas as pd
import site_hornsrev as S
wake, csv = sys.argv[1], sys.argv[2]
os.environ["WFLOP_WAKE"] = wake
import objective_gpu as og
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, wt_x, wt_y
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_farm_models import PropagateDownwind
from py_wake.deficit_models.noj import NOJDeficit
from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit
from py_wake.deficit_models.utils import ct2a_mom1d
from py_wake.superposition_models import SquaredSum
from py_wake.rotor_avg_models import RotorCenter
site = Hornsrev1Site(ti=S.TI)
ws_tab = np.asarray(S.POWER_CURVE_WS)
wt = WindTurbine(name="V80_ct0.8", diameter=S.DIAMETER, hub_height=70.0,
                 powerCtFunction=PowerCtTabular(ws_tab, np.asarray(S.POWER_CURVE_KW), "kW", np.full(ws_tab.shape, S.CT), method="linear"))
dm = (NOJDeficit(k=og.K, ct2a=ct2a_mom1d, rotorAvgModel=RotorCenter()) if wake == "jensen" else
      BastankhahGaussianDeficit(k=og.K_STAR, ceps=0.2, ct2a=ct2a_mom1d, use_effective_ws=False, rotorAvgModel=RotorCenter()))
wfm = PropagateDownwind(site, wt, wake_deficitModel=dm, superpositionModel=SquaredSum(), deflectionModel=None, turbulenceModel=None)
GRIDS = {"sector12": dict(wd=np.arange(0.0, 360.0, 30.0), ws=np.asarray(S.SPEED_MIDS, float)),
         "dir36": dict(wd=np.arange(0.0, 360.0, 10.0), ws=np.asarray(S.SPEED_MIDS, float)),
         "deg1": dict(wd=np.arange(0.0, 360.0, 1.0), ws=np.arange(3.25, 25.0, 0.5))}
def aep(x, y):
    out = {}
    for g, kw in GRIDS.items():
        sim = wfm(x, y, **kw)
        a, f = float(sim.aep().sum()), float(sim.aep(with_wake_loss=False).sum())
        out[g] = dict(aep=a, free=f, wl=100 * (1 - a / f))
    return out
res = {"real": aep(np.asarray(wt_x, float), np.asarray(wt_y, float))}
df = pd.read_csv(csv)
ep = df.EnergyProduction.values
order = np.argsort(-ep)
runs = []
for i in range(len(df)):
    P = np.array([[float(v) for v in p.split()] for p in df.Coordinates.iloc[i].split(";")]) + S.CENTROID_UTM
    r = aep(P[:, 0], P[:, 1]); r["seed"] = int(df.Seed.iloc[i]); r["campaign_aep"] = float(ep[i] * 8760 / 1e6); runs.append(r)
res["runs"] = runs
for g in GRIDS:
    a = np.array([r[g]["aep"] for r in runs]); w = np.array([r[g]["wl"] for r in runs])
    res[f"summary_{g}"] = dict(real_aep=res["real"][g]["aep"], real_wl=res["real"][g]["wl"], best_aep=float(a.max()), median_aep=float(np.median(a)),
                               worst_aep=float(a.min()), median_wl=float(np.median(w)), best_wl=float(w.min()), beat_real=int((a > res["real"][g]["aep"]).sum()),
                               median_gain_pct=float(100 * (np.median(a) / res["real"][g]["aep"] - 1)))
    print(wake, g, json.dumps({k: round(v, 3) for k, v in res[f"summary_{g}"].items()}))
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"hr_pywake_{wake}_{os.environ.get('TAGOUT', 'd12')}.json"), "w"), indent=1)
