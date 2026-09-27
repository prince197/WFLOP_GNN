import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, LogNorm
from scipy.stats import rankdata
import merged
MODE=sys.argv[1]; OUT=sys.argv[2]
WT=" (Gaussian wake)" if MODE=="gauss" else ""
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False})
INK,MUTED="#1a1a1a","#555555"
LAB={"GNNLXSSA":"GNN-LX-SSA","LXSSA":"LX-SSA"}; L=lambda a:LAB.get(a,a)
COL={"GNNLXSSA":"#C2410C","BBO":"#0F8A6E","GA":"#B45309","DE":"#C2410C","PF":"#4D7C0F","PSO":"#9D174D","SSA":"#7C3AED","GWO":"#0E7490","LXSSA":"#2563A8"}
RC={500:"#2563A8",750:"#C2410C",1000:"#0F8A6E"}
RANGE={500:range(2,11),750:range(2,15),1000:range(2,19)}
D={ds:merged.load(ds,MODE) for ds in (1,2)}
def meanmat(raw):
    cases=sorted(raw.groupby(["Radius","Turbines"]).groups)
    M=raw[raw.feas].groupby(["Radius","Turbines","Algorithm"]).WakeLoss.mean().unstack()
    return M.reindex(index=pd.MultiIndex.from_tuples(cases),columns=merged.A9).round(6),cases
def ranks(raw):
    M,cases=meanmat(raw); Mw=M.copy()
    for i in range(len(Mw)):
        row=Mw.iloc[i]; Mw.iloc[i]=row.fillna(row.max()*10+1e9)
    return dict(zip(merged.A9,np.array([rankdata(Mw.iloc[i].to_numpy()) for i in range(len(Mw))]).mean(0)))
# ---------------- Fig 5 feasibility heatmap ----------------
ORD5=["BBO","DE","GA","GWO","PF","PSO","SSA","LXSSA","GNNLXSSA"]
cmap=LinearSegmentedColormap.from_list("f",["#F3DDD2","#F7F4F2","#9EC1E0","#4A7FB5","#1F3A64"])
F={ds:D[ds][0].groupby(["Algorithm","Radius","Turbines"]).feas.mean()*100 for ds in (1,2)}
same=all(np.allclose(F[1].sort_index().values,F[2].sort_index().values) for _ in [0])
dsl=[1] if same else [1,2]
fig=plt.figure(figsize=(20,4.6*len(dsl)+0.9))
wr=[len(RANGE[r]) for r in (500,750,1000)]
gs=fig.add_gridspec(len(dsl),4,width_ratios=wr+[0.5],wspace=0.08,hspace=0.45,left=0.07,right=0.97,top=0.78 if len(dsl)==1 else 0.86,bottom=0.12 if len(dsl)==1 else 0.07)
for ri,ds in enumerate(dsl):
    for ci,R in enumerate((500,750,1000)):
        ax=fig.add_subplot(gs[ri,ci]); Ns=list(RANGE[R])
        Z=np.array([[F[ds].get((a,R,n),np.nan) for n in Ns] for a in ORD5])
        im=ax.imshow(Z,cmap=cmap,vmin=0,vmax=100,aspect="auto")
        for i in range(len(ORD5)):
            for j in range(len(Ns)):
                if Z[i,j]<100-1e-9: ax.text(j,i,f"{Z[i,j]:.0f}",ha="center",va="center",fontsize=9.5,color="white" if Z[i,j]>60 else "#222")
        ax.set_xticks(range(len(Ns)),Ns,fontsize=11); ax.set_xticks(np.arange(-.5,len(Ns)),minor=True); ax.set_yticks(np.arange(-.5,len(ORD5)),minor=True)
        ax.grid(which="minor",color="white",lw=2); ax.tick_params(which="minor",length=0)
        ax.set_yticks(range(len(ORD5)),[L(a) for a in ORD5] if ci==0 else [],fontsize=12)
        if ci>0: ax.tick_params(axis="y",left=False)
        for s in ax.spines.values(): s.set_visible(False)
        ttl=f"{R} m" if len(dsl)==1 else f"Data Set {ds} — {R} m"
        ax.set_title(ttl,fontsize=15,fontweight="bold",color=RC[R]); ax.set_xlabel("Number of turbines",fontsize=12,color=MUTED)
cax=fig.add_subplot(gs[:,3]); cb=fig.colorbar(im,cax=cax); cb.set_label("% of 30 runs satisfying constraints",fontsize=12,color=MUTED)
fig.suptitle("Constraint-satisfaction rate by algorithm, farm size and turbine count"+WT,fontsize=17,color=INK,y=0.99)
fig.text(0.5,0.895 if len(dsl)==1 else 0.93,"Unlabelled cell = all 30 runs feasible."+(" Identical in Wind Data Set 1 and 2." if same else ""),ha="center",fontsize=12,color=MUTED)
fig.savefig(f"{OUT}/na_fig1_heatmap.png",dpi=130); plt.close(fig)
# ---------------- Fig 9 evaluations vs rank ----------------
Rk={ds:ranks(D[ds][0]) for ds in (1,2)}
ev={a:float(D[1][0][D[1][0].Algorithm==a].groupby(["Radius","Turbines"]).Evaluations.mean().mean()) for a in merged.A9}
ref=ev["GNNLXSSA"]
fig,(a1,a2)=plt.subplots(1,2,figsize=(20,7.6),gridspec_kw=dict(width_ratios=[1.1,1]))
for a in merged.A9:
    x=ev[a]; y1,y2=Rk[1][a],Rk[2][a]
    a1.plot([x,x],[y1,y2],color=COL[a],lw=1.5,ls="--",alpha=.6)
    s=260 if a=="GNNLXSSA" else 150
    a1.scatter(x,y1,marker="o",s=s,color=COL[a],edgecolor="white",lw=1.2,zorder=3); a1.scatter(x,y2,marker="^",s=s,color=COL[a],edgecolor="white",lw=1.2,zorder=3)
    dx={"GNNLXSSA":110,"LXSSA":-110,"BBO":-110,"PSO":-110,"GA":-110,**({"PF":-110} if MODE=="gauss" else {})}.get(a,110); ha="right" if dx<0 else "left"
    a1.text(x+dx,(y1+y2)/2,L(a),color=COL[a],fontsize=13,fontweight="bold" if a=="GNNLXSSA" else None,ha=ha,va="center")
a1.invert_yaxis(); a1.set_ylim(9.6,0.6); a1.set_xlim(min(ev.values())-800,max(ev.values())+900)
a1.set_xlabel("True objective evaluations per run (mean over configurations)",fontsize=13,color=MUTED); a1.set_ylabel("Friedman average rank (lower = better)",fontsize=13,color=MUTED)
a1.grid(True,color="#EAEAEA"); a1.set_title("Cost against quality",loc="left",fontsize=17,fontweight="bold")
a1.scatter([],[],marker="o",color="#666",s=120,label="Data Set 1"); a1.scatter([],[],marker="^",color="#666",s=120,label="Data Set 2"); a1.legend(frameon=False,fontsize=13,loc="lower right")
order=["GNNLXSSA","BBO","GA","DE","PF","PSO","SSA","GWO","LXSSA"]
bars=a2.bar(range(9),[ev[a] for a in order],color=[COL[a] for a in order],width=0.62)
for i,a in enumerate(order):
    a2.text(i,ev[a]+90,f"{ev[a]:,.0f}",ha="center",fontsize=13,fontweight="bold",color=INK)
    a2.text(i,ev[a]-330,f"{ev[a]/ref:.2f}×",ha="center",fontsize=12,fontweight="bold",color="white")
a2.axhline(ref,color=COL["GNNLXSSA"],ls="--",lw=2.2); a2.set_xticks(range(9),[L(a) for a in order],rotation=30,ha="right",fontsize=13)
a2.set_ylabel("True objective evaluations per run",fontsize=13,color=MUTED); a2.set_ylim(0,max(ev.values())*1.15); a2.grid(True,axis="y",color="#EAEAEA")
a2.set_title("Evaluation budget",loc="left",fontsize=17,fontweight="bold"); a2.text(0.02,0.95,"× = multiple of GNN-LX-SSA's budget",transform=a2.transAxes,fontsize=12,color=MUTED)
fig.suptitle("Objective-evaluation budget and what each algorithm achieves with it"+WT,fontsize=17,y=0.99); fig.tight_layout(rect=(0,0,1,0.95))
fig.savefig(f"{OUT}/fig_evals.png",dpi=110); plt.close(fig)
# ---------------- Fig 10 CD diagram ----------------
CD=1.92
fig,axs=plt.subplots(2,1,figsize=(16,11.2))
for ax,ds in zip(axs,(1,2)):
    r=Rk[ds]; srt=sorted(merged.A9,key=lambda a:r[a]); ax.axis("off"); ax.set_xlim(-0.5,11.5); ax.set_ylim(-3.2,6.2)
    X=lambda v:1+(9-v)*1.0   # rank 9 -> x=1, rank 1 -> x=9
    ax.plot([X(9),X(1)],[0,0],color=INK,lw=2)
    for t in range(1,10): ax.plot([X(t)]*2,[0,0.18],color=INK,lw=1.5); ax.text(X(t),0.35,str(t),ha="center",fontsize=14,color=MUTED)
    ax.plot([X(1)-CD,X(1)],[5.3]*2,color=INK,lw=3); ax.plot([X(1)-CD]*2,[5.15,5.45],color=INK,lw=2); ax.plot([X(1)]*2,[5.15,5.45],color=INK,lw=2)
    ax.text(X(1)-CD/2,5.6,f"CD = {CD:.2f}",ha="center",fontsize=14)
    best=srt[:4]; worst=srt[4:][::-1]
    for i,a in enumerate(best):
        y=1.1+i*0.62; c=COL[a] if a in ("GNNLXSSA","BBO") else INK
        ax.plot([X(r[a]),X(r[a]),X(1)+0.15],[0,y,y],color=c,lw=2); ax.text(X(1)+0.3,y,f"{L(a)}  ({r[a]:.2f})",va="center",fontsize=15,color=c,fontweight="bold" if c!=INK else None)
    for i,a in enumerate(worst):
        y=1.1+i*0.62; c=COL[a] if a in ("GNNLXSSA","BBO") else INK
        ax.plot([X(r[a]),X(r[a]),X(9)-0.15],[0,y,y],color=c,lw=2); ax.text(X(9)-0.3,y,f"{L(a)}  ({r[a]:.2f})",va="center",ha="right",fontsize=15,color=c)
    # cliques: maximal groups within CD
    vals=sorted(r[a] for a in merged.A9); cl=[]
    for i in range(len(vals)):
        j=max(k for k in range(len(vals)) if vals[k]-vals[i]<=CD)
        if j>i and not any(c0<=i and j<=c1 for c0,c1 in cl): cl.append((i,j))
    for m,(i,j) in enumerate(cl):
        ax.plot([X(vals[i])+0.05,X(vals[j])-0.05],[-0.35-m*0.28]*2,color="#8C8C8C",lw=6,solid_capstyle="butt")
    ax.text(-0.4,5.4,f"Wind Data Set {ds}",fontsize=18,fontweight="bold")
fig.suptitle("Critical-difference diagram — Friedman average ranks with Nemenyi post-hoc (alpha = 0.05, n = 39, k = 9)"+WT,fontsize=16,y=0.985)
fig.text(0.5,0.02,"Lower rank is better. Grey bars join algorithms that are NOT significantly different.",ha="center",fontsize=13,color=MUTED)
fig.tight_layout(rect=(0,0.04,1,0.96)); fig.savefig(f"{OUT}/na_fig3_cd.png",dpi=100); plt.close(fig)
# ---------------- Fig 11 ratio heatmap ----------------
comp=["DE","PSO","GA","GWO","SSA","LXSSA","PF","BBO"]
cm2=LinearSegmentedColormap.from_list("r",["#1F5A8C","#8FB3D6","#EEEEEE","#EDBB96","#E39A62","#B8410F"])
fig=plt.figure(figsize=(20,12.8)); gs=fig.add_gridspec(2,4,width_ratios=[9,13,17,0.4],hspace=0.35,wspace=0.1,left=0.06,right=0.95,top=0.88,bottom=0.07)
med={}
for ri,ds in enumerate((1,2)):
    M,_=meanmat(D[ds][0])
    Rt={}
    for a in comp:
        Rt[a]=(M[a]/M["GNNLXSSA"]).where(M.min(axis=1)>=1)
    orderc=sorted(comp,key=lambda a:-np.nanmedian(Rt[a]))
    med[ds]={a:{R:float(np.nanmedian(Rt[a].loc[R])) for R in (500,750,1000)} for a in comp}
    for ci,R in enumerate((500,750,1000)):
        ax=fig.add_subplot(gs[ri,ci]); Ns=list(RANGE[R])
        Z=np.clip(np.array([[Rt[a].get((R,n),np.nan) for n in Ns] for a in orderc]),0.4,4.8)
        im=ax.imshow(Z,cmap=cm2,norm=LogNorm(vmin=0.4,vmax=4.8),aspect="auto")
        ax.set_xticks(range(len(Ns)),Ns,fontsize=10.5); ax.set_xticks(np.arange(-.5,len(Ns)),minor=True); ax.set_yticks(np.arange(-.5,len(orderc)),minor=True)
        ax.grid(which="minor",color="white",lw=1.5); ax.tick_params(which="minor",length=0)
        ax.set_yticks(range(len(orderc)),[L(a) for a in orderc] if ci==0 else [],fontsize=13)
        if ci>0: ax.tick_params(axis="y",left=False)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(f"{R} m",fontsize=15,fontweight="bold",color=RC[R])
        if ci==0: ax.text(-0.3,1.22,f"Wind Data Set {ds}",transform=ax.transAxes,fontsize=17,fontweight="bold")
        if ri==1 and ci==1: ax.set_xlabel("Number of turbines",fontsize=14,color=MUTED)
cax=fig.add_subplot(gs[:,3]); cb=fig.colorbar(im,cax=cax); cb.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator()); cb.set_ticks([0.5,1,2,4]); cb.ax.set_yticklabels(["0.5×","1×","2×","4×"]); cb.set_label("wake loss ÷ GNN-LX-SSA",fontsize=13,color=MUTED)
fig.suptitle("Where GNN-LX-SSA wins and loses — wake loss ratio per configuration"+WT,fontsize=18,y=0.985)
fig.text(0.5,0.935,"Orange = competitor worse than GNN-LX-SSA · blue = competitor better · white cell = no feasible run, or a configuration where some algorithm's mean wake loss is below 1/15 kW (optimum ≈ 0)",ha="center",fontsize=12.5,color=MUTED)
fig.savefig(f"{OUT}/optB_heatmap.png",dpi=110); plt.close(fig)
import json; json.dump(dict(ranks=Rk,evals=ev,ratio_median=med,feas_identical=bool(same)),open(f"{OUT}/figs2_{MODE}.json","w"),indent=1)
print("ok",MODE,"identical feas:",same)
