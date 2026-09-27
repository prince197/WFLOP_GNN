"""Recompute every GNN-dependent quantity of the paper from (merged) campaign data.
Usage: reanalyze.py old|new   -> writes reanalysis_<mode>.json and prints a summary."""
import sys, json, glob, numpy as np, pandas as pd
from scipy.stats import rankdata, wilcoxon, friedmanchisquare, kruskal, chi2 as CHI2
MODE=sys.argv[1]
U="/root/.claude/uploads/7dea1081-7422-5053-8471-1aafe20bfbdb"
SRC={1:f"{U}/65c35637-WFLOP_Report_ds1_paper1.xlsx",2:f"{U}/7e81ecf7-WFLOP_Report_ds2_paper1.xlsx"}
NEW="rerun/results/RawResults_ds{ds}_gnnrerun.csv"; NEWEFF="rerun/results/Efficiency_ds{ds}_gnnrerun.csv"
A9=["GNNLXSSA","PF","BBO","LXSSA","GWO","SSA","PSO","GA","DE"]
import merged
def load(ds):
    raw,eff,best,ideal=merged.load(ds,MODE); return raw,eff,ideal
def holm(p):
    p=np.asarray(p,float); o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for k,i in enumerate(o): run=max(run,min(1,(m-k)*p[i])); adj[i]=run
    return adj
def wil(ref,others,M):
    out=[]; g=M[ref].to_numpy()
    for a in others:
        o=M[a].to_numpy(); dd=o-g; nz=dd[dd!=0]; r=rankdata(np.abs(nz))
        Rp,Rm=r[nz>0].sum(),r[nz<0].sum()
        p=wilcoxon(o,g,zero_method="wilcox",method="exact").pvalue if len(nz)>0 else 1.0
        out.append(dict(alg=a,Rp=float(Rp),Rm=float(Rm),p=float(p),r=float((Rp-Rm)/(Rp+Rm)) if Rp+Rm>0 else 0.0))
    ph=holm([x["p"] for x in out])
    for x,q in zip(out,ph): x["p_holm"]=float(q)
    return out
R={}
for ds in (1,2):
    raw,eff,ideal=load(ds); res={}; R[ds]=res
    cases=sorted(raw.groupby(["Radius","Turbines"]).groups)
    # --- Sec 9.1 / Table 10
    t10={}
    for a in A9:
        h=raw[raw.Algorithm==a]; inf=~h.feas
        cell=h.groupby(["Radius","Turbines"]).feas.agg(["sum","count"])
        first_any={};first_all={}
        for rad in (500,750,1000):
            c=cell.loc[rad]; fa=[n for n in c.index if c.loc[n,"sum"]<c.loc[n,"count"]]; fz=[n for n in c.index if c.loc[n,"sum"]==0]
            first_any[rad]=min(fa) if fa else None; first_all[rad]=min(fz) if fz else None
        t10[a]=dict(inf=int(inf.sum()),pct=100*inf.mean(),
            by_rad={rad:100*(~h[h.Radius==rad].feas).mean() for rad in (500,750,1000)},
            zero_cells=int((cell["sum"]==0).sum()),first_any=first_any,first_all=first_all)
    res["t10"]=t10; res["inf_total"]=int((~raw.feas).sum()); res["runs_total"]=len(raw)
    res["zero_cells_total"]=sum(v["zero_cells"] for v in t10.values())
    # --- Table 11
    res["t11"]={a:[int(raw[(raw.Algorithm==a)&(raw.Radius==500)&(raw.Turbines==n)].feas.sum()) for n in (8,9,10)] for a in A9}
    # --- per-config feasible means, worst assignment
    M=raw[raw.feas].groupby(["Radius","Turbines","Algorithm"]).WakeLoss.mean().unstack()
    M=M.reindex(index=pd.MultiIndex.from_tuples(cases),columns=A9).round(6); Mw=M.copy()
    for i in range(len(Mw)):
        row=Mw.iloc[i]; Mw.iloc[i]=row.fillna(row.max()*10+1e9)
    Rk=np.array([rankdata(Mw.iloc[i].to_numpy()) for i in range(len(Mw))]); avg=Rk.mean(0)
    res["avg_rank"]=dict(zip(A9,avg.round(4).tolist()))
    res["t13"]={rad:dict(zip(A9,Rk[[i for i,c in enumerate(cases) if c[0]==rad]].mean(0).round(4).tolist())) for rad in (500,750,1000)}
    n,k=Rk.shape
    chi_u=12*n/(k*(k+1))*(np.sum(avg**2)-k*(k+1)**2/4)
    fr=friedmanchisquare(*Rk.T); ID=lambda c:(n-1)*c/(n*(k-1)-c)
    res["t15"]=dict(chi_u=chi_u,F_u=ID(chi_u),chi_t=fr.statistic,F_t=ID(fr.statistic),p=fr.pvalue,p_u=CHI2.sf(chi_u,k-1))
    # KW per config, infeasible tied worst
    ps=[]
    for c in cases:
        g=raw[(raw.Radius==c[0])&(raw.Turbines==c[1])]
        grp=[np.where(g[g.Algorithm==a].feas,np.round(g[g.Algorithm==a].WakeLoss,6),1e30) for a in A9]
        if np.unique(np.concatenate(grp)).size==1: continue
        ps.append(kruskal(*grp).pvalue)
    res["kw"]=dict(testable=len(ps),rejected=int((holm(ps)<0.05).sum()))
    res["t16"]=wil("GNNLXSSA",[a for a in ["BBO","DE","GA","GWO","PF","PSO","SSA","LXSSA"]],Mw)
    res["t17"]=wil("LXSSA",[a for a in ["BBO","DE","GA","GWO","PF","PSO","SSA","GNNLXSSA"]],Mw)
    # winners count (single best)
    res["single_best"]={a:int(sum(1 for i in range(len(Mw)) if (Mw.iloc[i]==Mw.iloc[i].min()).sum()==1 and Mw.iloc[i].idxmin()==a)) for a in A9}
    # --- Table 14 evaluations
    gg=raw[raw.Algorithm=="GNNLXSSA"]
    res["gnn_evals"]=dict(min=int(gg.Evaluations.min()),max=int(gg.Evaluations.max()),mean_cfg=float(gg.groupby(["Radius","Turbines"]).Evaluations.mean().mean()),
        mean_by_rad={rad:float(gg[gg.Radius==rad].groupby("Turbines").Evaluations.mean().mean()) for rad in (500,750,1000)},
        surr=float(gg.SurrogateInferences.mean()) if "SurrogateInferences" in gg else None)
    # --- Table 18/19 matched budget (penalties ranked by magnitude)
    A18=["GNNLXSSA","BBO","GWO","PF","SSA","LXSSA","PSO","GA","DE"]
    full=[];mat=[];feas={a:0 for a in A18}; MV=[]
    for (r_,n_) in cases:
        g=eff[(eff.Radius==r_)&(eff.Turbines==n_)]; rf=[];rm=[]
        for a in A18:
            h=g[g.Algorithm==a].sort_values("BudgetFraction"); vf=h.BestWakeLoss.iloc[-1]
            hm=h[h.ExactEvaluations<=3030]; vm=hm.BestWakeLoss.iloc[-1] if len(hm) else np.inf
            rf.append(vf); rm.append(vm); feas[a]+=bool(vm<ideal*n_)
        full.append(rankdata(np.round(rf,6))); mat.append(rankdata(np.round(rm,6))); MV.append(rm)
    res["t18"]={a:dict(full=float(np.mean(full,0)[i]),matched=float(np.mean(mat,0)[i]),feas=feas[a]) for i,a in enumerate(A18)}
    MVd=pd.DataFrame(MV,columns=A18).round(6)
    res["t19"]=wil("GNNLXSSA",["BBO","GWO","PF","SSA","LXSSA","PSO","GA","DE"],MVd)
    res["means"]={f"{r_}|{n_}":{a:(None if np.isnan(M.loc[(r_,n_),a]) else float(M.loc[(r_,n_),a])) for a in A9} for (r_,n_) in cases}
json.dump(R,open(f"reanalysis_{MODE}.json","w"),indent=1,default=float)
for ds in (1,2):
    r=R[ds]; print(f"=== DS{ds} infeasible {r['inf_total']}/{r['runs_total']} zero-cells {r['zero_cells_total']}")
    print(" GNN t10:",{k:(round(v,1) if isinstance(v,float) else v) for k,v in r["t10"]["GNNLXSSA"].items() if k!="by_rad"})
    print(" ranks:"," ".join(f"{a} {v:.2f}" for a,v in r["avg_rank"].items()))
    print(" t15: chi_u %.2f F_u %.2f chi_t %.2f F_t %.2f"%(r["t15"]["chi_u"],r["t15"]["F_u"],r["t15"]["chi_t"],r["t15"]["F_t"]),"KW",r["kw"])
    print(" t16:"," | ".join(f"{x['alg']} {x['Rp']:.0f}/{x['Rm']:.0f} p{x['p_holm']:.3g} r{x['r']:+.2f}" for x in r["t16"]))
    print(" t18:"," ".join(f"{a} {v['full']:.2f}/{v['matched']:.2f}/{v['feas']}" for a,v in r["t18"].items()))
    print(" t19:"," | ".join(f"{x['alg']} {x['Rp']:.0f}/{x['Rm']:.0f} p{x['p_holm']:.3g}" for x in r["t19"]))
    print(" gnn evals",r["gnn_evals"]," single best",r["single_best"]["GNNLXSSA"])
