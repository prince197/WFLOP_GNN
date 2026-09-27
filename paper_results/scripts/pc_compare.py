"""Table 16: the Jensen campaign re-run with the GE 1.5 MW commercial power curve (WFLOP_POWER=ge15),
analysed with the same statistics as the Jensen campaign. Run from this folder."""
import sys; DSS=[int(x) for x in (sys.argv[1:] or ["1","2"])]
import json, glob, numpy as np, pandas as pd, merged
from scipy.stats import rankdata, wilcoxon, friedmanchisquare, kendalltau
A9=["GNNLXSSA","PF","BBO","LXSSA","GWO","SSA","PSO","GA","DE"]
def holm(p):
    p=np.asarray(p,float); o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for k,i in enumerate(o): run=max(run,min(1,(m-k)*p[i])); adj[i]=run
    return adj
def load_g(ds):
    d=pd.read_csv(f"RawResults_ds{ds}_ge15.csv")
    d=d.drop_duplicates(["Radius","Turbines","Algorithm","Seed"])
    ideal=float(np.median((d.WakeLoss+d.EnergyProduction)/d.Turbines))
    return d.assign(feas=d.WakeLoss<ideal*d.Turbines), ideal
def stats(raw):
    cases=sorted(raw.groupby(["Radius","Turbines"]).groups)
    M=raw[raw.feas].groupby(["Radius","Turbines","Algorithm"]).WakeLoss.mean().unstack().reindex(index=pd.MultiIndex.from_tuples(cases),columns=A9).round(6)
    Mw=M.copy()
    for i in range(len(Mw)):
        row=Mw.iloc[i]; Mw.iloc[i]=row.fillna(row.max()*10+1e9)
    Rk=np.array([rankdata(Mw.iloc[i].values) for i in range(len(Mw))]); avg=dict(zip(A9,Rk.mean(0)))
    fr=friedmanchisquare(*Rk.T); n,k=Rk.shape
    W=[]; g=Mw.GNNLXSSA.values
    for a in A9[1:]:
        o=Mw[a].values; d=o-g; nz=d[d!=0]; rr=rankdata(np.abs(nz)); Rp,Rm=rr[nz>0].sum(),rr[nz<0].sum()
        p=wilcoxon(o,g,zero_method="wilcox",method="exact").pvalue
        W.append(dict(alg=a,Rp=float(Rp),Rm=float(Rm),p=float(p),r=float((Rp-Rm)/(Rp+Rm))))
    for w,q in zip(W,holm([w["p"] for w in W])): w["p_holm"]=float(q)
    feas={a:int(raw[raw.Algorithm==a].feas.sum()) for a in A9}
    return dict(avg_rank=avg,chi_t=float(fr.statistic),F_t=float((n-1)*fr.statistic/(n*(k-1)-fr.statistic)),p=float(fr.pvalue),
                wilcoxon=W,feas=feas,by_radius={R:dict(zip(A9,Rk[[i for i,c in enumerate(cases) if c[0]==R]].mean(0))) for R in (500,750,1000)},n=n)
out={}
for ds in DSS:
    g,_=load_g(ds); j=merged.load(ds,"new")[0]
    assert set(g.Algorithm)==set(A9) and len(g)==10530, (sorted(set(g.Algorithm)),len(g))
    sg,sj=stats(g),stats(j)
    tau=kendalltau([sj["avg_rank"][a] for a in A9],[sg["avg_rank"][a] for a in A9])
    # infeasible runs identical to Jensen? (penalty-dominated search is wind/wake independent)
    kk=["Radius","Turbines","Algorithm","Seed"]; mm=g.merge(j,on=kk,suffixes=("_g","_j"))
    out[ds]=dict(ge15=sg,linear_rank=sj["avg_rank"],kendall_tau=float(tau.statistic),
                 feas_same=bool((mm.feas_g==mm.feas_j).all()), n_feas_diff=int((mm.feas_g!=mm.feas_j).sum()))
    print(f"=== DS{ds}  Friedman chi2 {sg['chi_t']:.1f}  Kendall tau(linear, GE1.5 ranks)={tau.statistic:.3f}  feasibility identical to linear: {out[ds]['feas_same']} ({out[ds]['n_feas_diff']} runs differ)")
    for a in sorted(A9,key=lambda a:sg["avg_rank"][a]): print(f"   {a:9s} ge15 {sg['avg_rank'][a]:.2f}  linear {sj['avg_rank'][a]:.2f}  feasible {sg['feas'][a]}")
    print("   wilcoxon:"," | ".join(f"{w['alg']} {w['Rp']:.0f}/{w['Rm']:.0f} p{w['p_holm']:.2g}" for w in sg["wilcoxon"]))
json.dump(out,open("pc_analysis.json","w"),indent=1,default=float)
# ideal (isolated-turbine) power under each curve and the GNN-LX-SSA exact-evaluation budget
for ds in DSS:
    g,ig=load_g(ds); j,_,_,ij=merged.load(ds,"new")
    e=g[g.Algorithm=="GNNLXSSA"].Evaluations
    out[ds].update(ideal_kW_ge15=ig/15, ideal_kW_linear=ij/15, gnn_evals=dict(min=int(e.min()),max=int(e.max()),mean=float(e.mean())),
                   infeasible_ge15=int((~g.feas).sum()), infeasible_linear=int((~j.feas).sum()))
    print(f"DS{ds}: ideal per turbine {ig/15:.2f} kW (linear {ij/15:.2f}); GNN evals {e.min()}-{e.max()} mean {e.mean():.0f}; infeasible runs ge15 {out[ds]['infeasible_ge15']} linear {out[ds]['infeasible_linear']}")
json.dump(out,open("pc_analysis.json","w"),indent=1,default=float)
