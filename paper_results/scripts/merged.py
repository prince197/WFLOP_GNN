"""Load the campaign data used in the paper.
mode "new" (default): the Jensen campaign -- the eight baselines from the original campaign plus
the GNN-LX-SSA rerun (RawResults_ds<D>_merged.csv, Efficiency_ds<D>_merged.csv).
mode "gauss": the Gaussian-wake campaign of Section 10 (RawResults_ds<D>_gaussian.csv,
Efficiency_ds<D>_gaussian.csv).
mode "ge15": the Jensen campaign re-run with the GE 1.5 MW commercial power curve
(RawResults_ds<D>_ge15.csv, Efficiency_ds<D>_ge15.csv)."""
import os, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__))
A9=["GNNLXSSA","PF","BBO","LXSSA","GWO","SSA","PSO","GA","DE"]
_cache={}
def load(ds, mode="new"):
    key=(ds,mode if mode in ("gauss","ge15") else "new")
    if key in _cache: return _cache[key]
    tag={"gauss":"gaussian","ge15":"ge15"}.get(key[1],"merged")
    raw=pd.read_csv(f"{HERE}/RawResults_ds{ds}_{tag}.csv"); raw=raw[raw.Algorithm.isin(A9)]
    eff=pd.read_csv(f"{HERE}/Efficiency_ds{ds}_{tag}.csv")
    ideal=float(np.median((raw.WakeLoss+raw.EnergyProduction)/raw.Turbines))
    raw=raw.assign(feas=raw.WakeLoss<ideal*raw.Turbines).reset_index(drop=True)
    best=raw.loc[raw.groupby(["Radius","Turbines","Algorithm"]).WakeLoss.idxmin()]
    _cache[key]=(raw,eff,best,ideal); return _cache[key]
