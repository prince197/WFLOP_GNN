"""Extra numbers for the Gaussian results section (beyond reanalysis_gauss.json)."""
import json, numpy as np, pandas as pd, merged
from scipy.stats import rankdata, wilcoxon, friedmanchisquare
A9=["GNNLXSSA","PF","BBO","LXSSA","GWO","SSA","PSO","GA","DE"]
J=json.load(open("reanalysis_gauss.json")); JJ=json.load(open("reanalysis_new.json"))  # run reanalyze.py new and reanalyze.py gauss first
MD=308.0
def coords(s): return np.array([[float(t) for t in p.split()] for p in s.split(";")])
def minsp(P):
    d=np.sqrt(((P[:,None]-P[None])**2).sum(-1)); np.fill_diagonal(d,np.inf); return d.min(),d
out={}
for ds in (1,2):
    raw,eff,best,ideal=merged.load(ds,"gauss"); o={}; out[ds]=o
    M=pd.DataFrame(J[str(ds)]["means"]).T[A9].astype(float); M.index=[tuple(map(int,k.split("|"))) for k in M.index]
    # ratio heatmap (same exclusion rule as figure): drop configs where any mean < 1
    keep=~(M.min(axis=1)<1); Mk=M[keep]; o["ratio_n"]=int(keep.sum())
    o["ratio_med"]={a:float(np.nanmedian(Mk[a]/Mk["GNNLXSSA"])) for a in A9[1:]}
    o["ratio_better"]={a:int(((Mk[a]/Mk["GNNLXSSA"])<1).sum()) for a in A9[1:]}
    o["ratio_better_1000"]={a:int(((Mk[a]/Mk["GNNLXSSA"])<1)[[c[0]==1000 for c in Mk.index]].sum()) for a in A9[1:]}
    # GNN vs LX-SSA pointwise (N>=4) on the wake-loss curves
    sub=M[[c[1]>=4 for c in M.index]]
    o["gnn_above_lx"]=[list(c) for c in sub.index if sub.loc[[c],"GNNLXSSA"].item()>sub.loc[[c],"LXSSA"].item()]
    o["gnn_lowest_pts"]=int(sum(1 for c in sub.index if sub.loc[[c]].iloc[0].idxmin()=="GNNLXSSA"))
    o["n_pts"]=len(sub)
    o["lowest_by_rad"]={r:{a:int(sum(1 for c in sub.index if c[0]==r and sub.loc[[c]].iloc[0].idxmin()==a)) for a in A9} for r in (500,750,1000)}
    # SSA vs LX-SSA at matched budget, best-of-30 checkpoints
    vs,vl=[],[]
    for (r_,n_) in sorted(set(zip(eff.Radius,eff.Turbines))):
        g=eff[(eff.Radius==r_)&(eff.Turbines==n_)]
        h=g[g.Algorithm=="SSA"].sort_values("BudgetFraction"); vs.append(h.BestWakeLoss.iloc[-1])
        h=g[(g.Algorithm=="LXSSA")&(g.ExactEvaluations<=3030)].sort_values("BudgetFraction"); vl.append(h.BestWakeLoss.iloc[-1] if len(h) else np.inf)
    vs,vl=np.round(vs,6),np.round(vl,6); dd=vs-vl; nz=dd[dd!=0]; rk=rankdata(np.abs(nz))
    o["ssa_lx_matched"]=dict(lx_better=int((dd>0).sum()),ssa_better=int((dd<0).sum()),n=int(len(nz)),Rp=float(rk[nz>0].sum()),Rm=float(rk[nz<0].sum()),
        p=float(wilcoxon(vs[dd!=0],vl[dd!=0],method="exact").pvalue if len(nz)<=50 and np.isfinite(vl).all() else wilcoxon(vs,vl).pvalue))
    # own-budget LX-SSA vs SSA (from t17 SSA row: R+ = configs where LX-SSA better)
    d=M["SSA"].fillna(1e30)-M["LXSSA"].fillna(1e30); o["ssa_lx_own"]=dict(lx_better=int((d>0).sum()),ssa_better=int((d<0).sum()))
    # matched-budget Friedman
    A18=list(J[str(ds)]["t18"].keys()); mat=[]
    for (r_,n_) in sorted(set(zip(eff.Radius,eff.Turbines))):
        g=eff[(eff.Radius==r_)&(eff.Turbines==n_)]; rm=[]
        for a in A18:
            h=g[(g.Algorithm==a)&(g.ExactEvaluations<=3030)].sort_values("BudgetFraction"); rm.append(h.BestWakeLoss.iloc[-1] if len(h) else np.inf)
        mat.append(rankdata(np.round(rm,6)))
    mat=np.array(mat); fr=friedmanchisquare(*mat.T); n,k=mat.shape
    o["matched_friedman"]=dict(chi=float(fr.statistic),F=float((n-1)*fr.statistic/(n*(k-1)-fr.statistic)),p=float(fr.pvalue))
    # largest configuration stats
    L={}
    for (r_,n_) in [(500,10),(750,14),(1000,18)]:
        for a in A9:
            h=raw[(raw.Radius==r_)&(raw.Turbines==n_)&(raw.Algorithm==a)&raw.feas]
            L[f"{r_}|{a}"]=dict(nf=len(h),mean=float(h.WakeLoss.mean()) if len(h) else None,sd=float(h.WakeLoss.std(ddof=1)) if len(h)>1 else None,best=float(h.WakeLoss.min()) if len(h) else None)
    o["largest"]=L
    # nemenyi group
    ar=J[str(ds)]["avg_rank"]; g=ar["GNNLXSSA"]; o["within_cd"]=[a for a in A9[1:] if ar[a]-g<1.92]
    # feasible/infeasible layout figure numbers (DS1 18@1000)
    if ds==1:
        info={}
        for a in ("GNNLXSSA","DE"):
            r=best[(best.Radius==1000)&(best.Turbines==18)&(best.Algorithm==a)].iloc[0]; P=coords(r.Coordinates); ms,dm=minsp(P)
            info[a]=dict(minsp=float(ms),pairs=int(np.triu(dm<MD,1).sum()),feas=bool(r.WakeLoss<ideal*18))
        o["fig_layout"]=info
    # zero-cells per algorithm
    o["zero_cells"]={a:J[str(ds)]["t10"][a]["zero_cells"] for a in A9}
    o["jensen_rank"]=JJ[str(ds)]["avg_rank"]
json.dump(out,open("gauss_numbers.json","w"),indent=1,default=float)
for ds in (1,2):
    o=out[ds]; print("DS",ds)
    for k,v in o.items():
        if k!="largest": print(" ",k,json.dumps(v,default=float)[:600])
    print("  largest:"); [print("   ",k,{kk:(round(vv,1) if isinstance(vv,float) else vv) for kk,vv in v.items()}) for k,v in o["largest"].items()]
