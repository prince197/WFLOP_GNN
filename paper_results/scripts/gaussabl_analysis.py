"""Gaussian-wake ablation: GNN-LX-SSA vs LX-SSA+R (no surrogate) vs LX-SSA, all 39 configurations."""
import numpy as np, pandas as pd, glob, json, merged
from scipy.stats import rankdata, wilcoxon, mannwhitneyu
out={}
TR=pd.read_csv("ablation_median_traces_gaussian.csv")
for ds in (1,2):
    raw,_,_,ideal=merged.load(ds,"gauss"); g=raw[raw.Algorithm=="GNNLXSSA"]; lx=raw[raw.Algorithm=="LXSSA"]
    r=pd.read_csv(f"RawResults_ds{ds}_gaussian_ablation_noSurrogate.csv"); r=r.assign(feas=r.WakeLoss<ideal*r.Turbines)
    mg=g[g.feas].groupby(["Radius","Turbines"]).WakeLoss.mean().round(6); mr=r[r.feas].groupby(["Radius","Turbines"]).WakeLoss.mean().round(6)
    keep=[c for c in mg.index if min(mg[c],mr[c])>=1]
    d=(mr-mg); nz=d[d!=0]; rr=rankdata(np.abs(nz)); Rp,Rm=rr[nz.values>0].sum(),rr[nz.values<0].sum()
    p=wilcoxon(mr.values,mg.values,method="exact").pvalue
    ratio=(mg.loc[keep]/mr.loc[keep])
    mv=[]
    for (R_,N_),t in TR[TR.Dataset==ds].groupby(["Radius","Turbines"]):
        g_=t[t.Algorithm=="GNNLXSSA"].sort_values("Iteration"); r_=t[t.Algorithm=="LXSSA_REPAIR"].sort_values("Iteration")
        b=g_.ExactEvaluations.iloc[-1]; j=int(np.searchsorted(r_.ExactEvaluations.values,b,side="right"))-1
        mv.append((R_,N_,round(g_.MedianBestWakeLoss.iloc[-1],6),round(r_.MedianBestWakeLoss.iloc[max(j,0)],6),float(r_.ExactEvaluations.iloc[max(j,0)]),float(b),round(r_.MedianBestWakeLoss.iloc[-1],6)))
    M=pd.DataFrame(mv,columns=["R","N","gnn","rep","rep_ev","gnn_ev","rep_final"])
    dm=M.rep-M.gnn; nzm=dm[dm!=0]; rm=rankdata(np.abs(nzm)); Rpm,Rmm=rm[nzm.values>0].sum(),rm[nzm.values<0].sum()
    pm=wilcoxon(M.rep,M.gnn,method="exact").pvalue
    three={}
    for R_,N_ in ((500,10),(750,14),(1000,18)):
        cell=lambda h:h[(h.Radius==R_)&(h.Turbines==N_)]
        row={}
        for nm,h in (("LXSSA",lx),("LXSSA_R",r),("GNN",g)):
            c=cell(h); f=c[c.feas]
            row[nm]=dict(nf=int(len(f)),mean=float(f.WakeLoss.mean()) if len(f) else None,best=float(f.WakeLoss.min()) if len(f) else None,ev=float(c.Evaluations.mean()))
        a=cell(g)[cell(g).feas].WakeLoss.values; b=cell(r)[cell(r).feas].WakeLoss.values
        u=mannwhitneyu(a,b,alternative="two-sided"); row["rs_p"]=float(u.pvalue); row["rbis"]=float(1-2*u.statistic/(len(a)*len(b)))
        row["gnn_vs_rep_pct"]=float(100*(row["GNN"]["mean"]/row["LXSSA_R"]["mean"]-1))
        row["lx_to_rep_pct"]=float(100*(1-row["LXSSA_R"]["mean"]/row["LXSSA"]["mean"])) if row["LXSSA"]["mean"] else None
        three[f"{R_}|{N_}"]=row
    res=dict(feas_rep=int(r.feas.sum()),n_rep=len(r),three=three,
        own=dict(gnn_better=int((d>0).sum()),rep_better=int((d<0).sum()),ties=int((d==0).sum()),Rp=float(Rp),Rm=float(Rm),p=float(p),r=float((Rp-Rm)/(Rp+Rm)),
                 median_ratio=float(ratio.median()),ratio_by_radius={R:float(ratio.loc[[c for c in keep if c[0]==R]].median()) for R in (500,750,1000)}),
        matched=dict(gnn_better=int((dm>0).sum()),rep_better=int((dm<0).sum()),ties=int((dm==0).sum()),Rp=float(Rpm),Rm=float(Rmm),p=float(pm),r=float((Rpm-Rmm)/(Rpm+Rmm)),
                 median_ratio=float((M.gnn/M.rep)[(M.gnn>=1)&(M.rep>=1)].median()),rep_ev_mean=float(M.rep_ev.mean())),
        evals=dict(gnn=float(g.Evaluations.mean()),rep=float(r.Evaluations.mean()),gnn_min=float(g.groupby(["Radius","Turbines"]).Evaluations.mean().min()),gnn_max=float(g.groupby(["Radius","Turbines"]).Evaluations.mean().max())))
    out[ds]=res; print(ds,json.dumps(res,indent=0,default=float)[:3000])
json.dump(out,open("gaussabl.json","w"),indent=1,default=float)
