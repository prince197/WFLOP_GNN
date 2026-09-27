import sys, os, json
HERE=os.path.dirname(os.path.abspath(__file__)); S=os.path.dirname(HERE)
sys.path[:0]=[HERE]  # rescore.py needs the campaign package (objective_gpu, backend) importable
MODE=sys.argv[1]
import numpy as np, merged, rescore
from scipy.stats import rankdata, wilcoxon
A=["GNNLXSSA","GWO","BBO","LXSSA","PF","SSA","PSO","GA","DE"]
def holm(p):
    p=np.asarray(p,float); o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for k,i in enumerate(o): run=max(run,min(1,(m-k)*p[i])); adj[i]=run
    return adj
out={}
for ds in (1,2):
    raw,eff,best,ideal=merged.load(ds,MODE)
    best=best[best.Algorithm.isin(A)].copy()
    P={}; lin={}; cub={}
    for n,g in best.groupby("Turbines"):
        Ps=np.stack([rescore.parse_layout(c) for c in g.Coordinates])
        l=rescore.expected_power(Ps,ds,rescore.f_linear); c=rescore.expected_power(Ps,ds,rescore.f_cubic)
        for i,idx in enumerate(g.index): lin[idx]=l[i]; cub[idx]=c[i]
    il=rescore.expected_power(np.zeros((1,1,2)),ds,rescore.f_linear)[0]; ic=rescore.expected_power(np.zeros((1,1,2)),ds,rescore.f_cubic)[0]
    best["Llin"]=[il*n-lin[i] for i,n in zip(best.index,best.Turbines)]; best["Lcub"]=[ic*n-cub[i] for i,n in zip(best.index,best.Turbines)]
    best["feas"]=best.WakeLoss<ideal*best.Turbines
    fe=best[best.feas]; err=np.max(np.abs(np.array([lin[i] for i in fe.index])-fe.EnergyProduction)/fe.EnergyProduction)
    blocks=[]
    for (r,n),g in best.groupby(["Radius","Turbines"]):
        g=g.set_index("Algorithm").reindex(A)
        if g.feas.all() and np.ptp(g.Llin.values)>1e-9: blocks.append(((r,n),g))
    res={"n_blocks":len(blocks),"max_rel_err":float(err)}
    changed=0
    for sc in ("Llin","Lcub"):
        Rk=np.array([rankdata(np.round(g[sc].values,9)) for _,g in blocks])
        res[sc]={"meanrank":dict(zip(A,Rk.mean(0).tolist()))}
        W=[]
        x=np.array([g.loc["GNNLXSSA",sc] for _,g in blocks])
        for a in A[1:]:
            y=np.array([g.loc[a,sc] for _,g in blocks]); d=y-x; nz=d[np.abs(d)>1e-9]; rk=rankdata(np.abs(nz))
            Rp,Rm=rk[nz>0].sum(),rk[nz<0].sum()
            p=wilcoxon(y[np.abs(d)>1e-9],x[np.abs(d)>1e-9],method="exact").pvalue
            W.append(dict(alg=a,Rp=float(Rp),Rm=float(Rm),p=float(p),r=float((Rp-Rm)/(Rp+Rm))))
        for w,q in zip(W,holm([w["p"] for w in W])): w["p_holm"]=float(q)
        res[sc]["wilcoxon"]=W
    res["rank_changes"]=int(sum(not np.array_equal(rankdata(np.round(g.Llin.values,9)),rankdata(np.round(g.Lcub.values,9))) for _,g in blocks))
    out[ds]=res
    print(f"DS{ds}: blocks {len(blocks)}, lin max rel err {err:.1e}, rank changes {res['rank_changes']}")
    print("  ranks lin/cub:"," ".join(f"{a} {res['Llin']['meanrank'][a]:.2f}/{res['Lcub']['meanrank'][a]:.2f}" for a in A))
    print("  cubic wilcoxon:"," | ".join(f"{w['alg']} {w['Rp']:.0f}/{w['Rm']:.0f} p{w['p_holm']:.2g} r{w['r']:+.3f}" for w in res["Lcub"]["wilcoxon"]))
json.dump(out,open(f"{HERE}/robust_tables.json","w"),indent=1)
