import merged, pandas as pd, numpy as np, json, glob
from scipy.stats import rankdata, wilcoxon, friedmanchisquare
def holm(p):
    p=np.asarray(p,float); o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for k,i in enumerate(o): run=max(run,min(1,(m-k)*p[i])); adj[i]=run
    return adj
out={}
for mode,w in (("new","jensen"),("gauss","gaussian")):
  for ds in (1,2):
    raw,_,_,ideal=merged.load(ds,mode)
    n=pd.read_csv(f"mtest/results/RawResults_ds{ds}_t_{w}.csv"); n=n.assign(feas=n.WakeLoss<ideal*n.Turbines)
    allr=pd.concat([raw,n[raw.columns]],ignore_index=True)
    M=allr[allr.feas].groupby(["Radius","Turbines","Algorithm"]).WakeLoss.mean().unstack().round(6)
    Mw=M.apply(lambda r:r.fillna(r.max()*10+1e9),axis=1)
    Rk=np.array([rankdata(Mw.iloc[i]) for i in range(len(Mw))]); ar=dict(zip(Mw.columns,Rk.mean(0)))
    fr=friedmanchisquare(*Rk.T)
    byrad={R:dict(zip(Mw.columns,Rk[[i for i,c in enumerate(Mw.index) if c[0]==R]].mean(0).round(2))) for R in (500,750,1000)}
    g=Mw["GNNLXSSA"].values; tests={}
    others=[a for a in Mw.columns if a!="GNNLXSSA"]; ps=[]
    for a in others:
        o=Mw[a].values; d=o-g; nz=d[d!=0]; r=rankdata(np.abs(nz)); Rp,Rm=r[nz>0].sum(),r[nz<0].sum()
        p=wilcoxon(o,g,method="exact").pvalue if len(nz) else 1.0
        tests[a]=dict(gnn_better=int((d>0).sum()),other_better=int((d<0).sum()),Rp=float(Rp),Rm=float(Rm),p=float(p),r=float((Rp-Rm)/(Rp+Rm)) if Rp+Rm else 0)
        ps.append(p)
    for a,q in zip(others,holm(ps)): tests[a]["p_holm"]=float(q)
    # equal cost: median trace of CMAES/LSHADE read at GNN's final exact-evaluation count, vs GNN final median
    eq={}
    gcur="rerun/curves/Conv_ds{ds}_gnnrerun_R{R}_T{N}.npz" if mode=="new" else "gauss_results/curves/ds{ds}-gnn_Conv_ds{ds}_gauss_R{R}_T{N}.npz"
    for a in ("CMAES","LSHADE"):
        rows=[]
        for (R,N) in Mw.index:
            zg=np.load(gcur.format(ds=ds,R=R,N=N)); zn=np.load(f"mtest/curves/Conv_ds{ds}_t_{w}_R{R}_T{N}.npz")
            k=list(zn["algorithms"]).index(a); b=zg["evals"][0][-1]; ax=zn["evals"][k]; j=int(np.searchsorted(ax,b,side="right"))-1
            rows.append((round(float(zg["median"][0][-1]),6),round(float(zn["median"][k][max(j,0)]),6)))
        A=np.array(rows); d=A[:,1]-A[:,0]
        eq[a]=dict(gnn_better=int((d>0).sum()),other_better=int((d<0).sum()),p=float(wilcoxon(A[:,1],A[:,0]).pvalue),median_ratio=float(np.median(A[A.min(1)>=1,0]/A[A.min(1)>=1,1])))
    out[f"{mode}_ds{ds}"]=dict(avg_rank={k:round(v,2) for k,v in sorted(ar.items(),key=lambda x:x[1])},by_radius=byrad,friedman=float(fr.statistic),
        infeasible={a:int((~n[n.Algorithm==a].feas).sum()) for a in ("LSHADE","CMAES")},
        zero_cells={a:int(n[n.Algorithm==a].groupby(["Radius","Turbines"]).feas.sum().eq(0).sum()) for a in ("LSHADE","CMAES")},
        wilcoxon_vs={a:tests[a] for a in ("CMAES","LSHADE","BBO")},equal_cost=eq,
        largest={f"{R}|{N}":{a:(int(allr[(allr.Radius==R)&(allr.Turbines==N)&(allr.Algorithm==a)].feas.sum()),round(float(M.loc[(R,N),a]),0) if not np.isnan(M.loc[(R,N),a]) else None) for a in ("GNNLXSSA","CMAES","LSHADE","BBO")} for R,N in ((500,10),(750,14),(1000,18))})
json.dump(out,open("modern_analysis.json","w"),indent=1)
for k,v in out.items():
    print("==",k); print(" ranks",v["avg_rank"]); print(" infeasible",v["infeasible"],"zero cells",v["zero_cells"])
    for a,t in v["wilcoxon_vs"].items(): print(f"  GNN vs {a}: GNN better {t['gnn_better']}/39, p_holm {t['p_holm']:.3g}, r {t['r']:+.2f}")
    for a,t in v["equal_cost"].items(): print(f"  equal cost vs {a}: GNN better {t['gnn_better']}, other better {t['other_better']}, p {t['p']:.3g}, median GNN/other {t['median_ratio']:.3f}")
    print("  by radius GNN/CMAES/LSHADE:",{R:(d['GNNLXSSA'],d['CMAES'],d['LSHADE']) for R,d in v['by_radius'].items()})
    print("  largest",v["largest"])
