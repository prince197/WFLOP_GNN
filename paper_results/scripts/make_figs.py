import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
from matplotlib.lines import Line2D
U="/root/.claude/uploads/7dea1081-7422-5053-8471-1aafe20bfbdb"
SRC={1:f"{U}/65c35637-WFLOP_Report_ds1_paper1.xlsx",2:f"{U}/7e81ecf7-WFLOP_Report_ds2_paper1.xlsx"}
OUT=sys.argv[2] if len(sys.argv)>2 else "figs_new"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False,
    "axes.edgecolor":"#777","axes.labelcolor":"#555","xtick.color":"#555","ytick.color":"#555"})
INK,MUTED="#1a1a1a","#555555"
C={"GNNLXSSA":"#C2410C","BBO":"#0F8A6E","PF":"#2563A8","LXSSA":"#7C3AED","DE":"#C2410C"}
LAB={"GNNLXSSA":"GNN-LX-SSA","LXSSA":"LX-SSA"}
ALGS=["BBO","DE","GA","GWO","PF","PSO","SSA","LXSSA","GNNLXSSA"]
import merged, sys
MODE=sys.argv[1] if len(sys.argv)>1 else "new"
WT=" (Gaussian wake)" if MODE=="gauss" else ""
raw,best,ideal={}, {}, {}
for ds in (1,2):
    r_,e_,b_,i_=merged.load(ds,MODE); raw[ds]=r_; best[ds]=b_; ideal[ds]=i_
def coords(s): return np.array([[float(v) for v in p.split()] for p in s.split(";")])
def minsp(P):
    d=np.sqrt(((P[:,None]-P[None])**2).sum(-1)); np.fill_diagonal(d,np.inf); return d.min(), d
MD=308.0
KW=15.0   # the CSV/workbook WakeLoss and EnergyProduction columns are in working units (wu); 1 wu = 1/15 kW
# ---------------- Fig 7: mean wake loss over feasible runs ----------------
fig,axs=plt.subplots(2,3,figsize=(15.5,9.2))
report=[]
for i,ds in enumerate((1,2)):
    d=raw[ds]
    for j,R in enumerate((500,750,1000)):
        ax=axs[i,j]; g=d[(d.Radius==R)&(d.Turbines>=4)&d.feas]
        m=g.groupby(["Algorithm","Turbines"]).WakeLoss.mean().unstack(0)/KW   # wu -> kW
        for a in ["DE","GA","GWO","PSO","SSA"]:
            if a in m: ax.plot(m.index,m[a],color="#BDBDBD",lw=1.3,zorder=1)
        for a,mk,lw in [("BBO","^",2.2),("PF","o",2.2),("LXSSA","D",2.2),("GNNLXSSA","s",3.0)]:
            ax.plot(m.index,m[a],color=C[a],marker=mk,ms=7 if a!="GNNLXSSA" else 7.5,lw=lw,mec="white",mew=0.8,label=LAB.get(a,a),zorder=3 if a!="GNNLXSSA" else 4)
        ax.set_yscale("log"); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.grid(True,which="major",color="#EAEAEA",lw=0.8); ax.set_axisbelow(True)
        ax.set_title(f"Data Set {ds} — {R} m",fontsize=14,fontweight="bold",color=INK)
        if j==0: ax.set_ylabel("Mean wake loss (kW, log scale)",fontsize=12)
        if i==1: ax.set_xlabel("Number of turbines",fontsize=12)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        below=(m["GNNLXSSA"]<m["LXSSA"]).all()
        lowest=(m["GNNLXSSA"]<=m.min(axis=1)+1e-9).mean()
        report.append((ds,R,below,round(lowest,2)))
axs[0,0].legend(frameon=False,fontsize=11,loc="upper left")
fig.suptitle("Mean wake loss over feasible runs — nine algorithms"+WT,fontsize=17,color=INK,y=0.995)
fig.text(0.5,0.948,"Grey lines: DE, GA, GWO, PSO, SSA; a line ends where the algorithm has no feasible run. Turbine counts below 4 omitted (all algorithms reach the optimum).\nWake loss in kW of probability-normalized expected power.",ha="center",va="top",fontsize=11,color=MUTED,linespacing=1.4)
fig.tight_layout(rect=(0,0,1,0.9)); fig.savefig(f"{OUT}/fig9A_wakeloss.png",dpi=150); plt.close(fig)
print("Fig7 check (ds,R,GNN below LX-SSA everywhere, frac of N where GNN lowest):",report)
# ---------------- Fig 8 + S1-S5: best layouts GNN vs BBO ----------------
RANGE={500:range(2,11),750:range(6,15),1000:range(10,19)}
def fmt(v): return f"{v:,.1f}" if v>=100 else f"{v:.2f}"
for ds in (1,2):
    b=best[ds]
    for R in (500,750,1000):
        fig,axs=plt.subplots(3,3,figsize=(11.3,13.2))
        for ax,N in zip(axs.flat,RANGE[R]):
            ax.set_aspect("equal"); ax.axis("off")
            ax.add_patch(Circle((0,0),R,fill=False,ls=(0,(4,3)),color="#9E9E9E",lw=1.3))
            txt=[]
            for a,mk,nm in [("BBO","^","BBO"),("GNNLXSSA","s","GNN")]:
                r=b[(b.Radius==R)&(b.Turbines==N)&(b.Algorithm==a)].iloc[0]
                feas=r.WakeLoss<ideal[ds]*N
                if feas:
                    P=coords(r.Coordinates)
                    ax.scatter(P[:,0],P[:,1],marker=mk,s=62,color=C[a],edgecolor="white",lw=0.8,zorder=3 if a=="GNNLXSSA" else 4)
                txt.append((nm, fmt(r.WakeLoss/KW) if feas else "no feasible run"))
            ax.set_xlim(-1.08*R,1.08*R); ax.set_ylim(-1.08*R,1.08*R)
            ax.set_title(f"{N} turbines",fontsize=14,color=INK,pad=4)
            gtxt=dict(txt)
            ax.text(0,-1.22*R,f"GNN {gtxt['GNN']}    BBO {gtxt['BBO']}",ha="center",va="top",fontsize=11,color=MUTED)
        name=f"Wind Data Set {ds} — best layouts of 30 runs, farm radius {R} m"
        fig.suptitle(name+WT,fontsize=17,color=INK,y=0.985)
        h=[Line2D([],[],marker="s",ls="",color=C["GNNLXSSA"],ms=11,label="GNN-LX-SSA"),Line2D([],[],marker="^",ls="",color=C["BBO"],ms=11,label="BBO")]
        fig.legend(handles=h,loc="upper center",ncol=2,frameon=False,fontsize=13,bbox_to_anchor=(0.5,0.955))
        fig.text(0.5,0.012,"Values below each panel are the wake loss of that algorithm's best layout, in kW.",ha="center",fontsize=11,color=MUTED)
        fig.subplots_adjust(left=0.03,right=0.97,top=0.87,bottom=0.075,hspace=0.34,wspace=0.12)
        fig.savefig(f"{OUT}/na_lay_ds{ds}_{R}.png",dpi=150); plt.close(fig)
# ---------------- Fig 6: feasible vs infeasible, 18 turbines 1000 m, DS1 ----------------
b=best[1]; R=1000; N=18
fig,axs=plt.subplots(1,2,figsize=(14.4,8.2))
info={}
for ax,(a,col,ttl) in zip(axs,[("GNNLXSSA","#0F8A6E","GNN-LX-SSA — feasible"),("DE","#C2410C","DE — infeasible")]):
    r=b[(b.Radius==R)&(b.Turbines==N)&(b.Algorithm==a)].iloc[0]; P=coords(r.Coordinates); ms,d=minsp(P)
    ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Circle((0,0),R,fill=False,ls=(0,(4,3)),color="#9E9E9E",lw=1.6))
    for p in P: ax.add_patch(Circle(p,MD/2,color=col,alpha=0.14,lw=0))
    iu=np.argwhere(np.triu(d<MD,1))
    for i,j in iu: ax.plot(*P[[i,j]].T,color=col,lw=2.6,zorder=2)
    ax.scatter(P[:,0],P[:,1],s=60,color=col,edgecolor="white",lw=1,zorder=3)
    ax.set_xlim(-1.2*R,1.2*R); ax.set_ylim(-1.2*R,1.2*R)
    ax.set_title(ttl,fontsize=18,fontweight="bold",color=col)
    sub=f"min spacing {ms:.1f} m" + (f" · {len(iu)} violating pairs" if len(iu) else "")
    ax.text(0,-1.2*R,sub,ha="center",va="top",fontsize=14,color=MUTED)
    info[a]=(int(r.Seed),ms,len(iu),float(np.sqrt((P**2).sum(1)).max()))
fig.suptitle("18 turbines, 1000 m radius, Wind Data Set 1, best of 30 runs — shaded discs are the 154 m exclusion radius (308 m spacing rule)",fontsize=13.5,color=INK)
fig.tight_layout(rect=(0,0.02,1,0.95)); fig.savefig(f"{OUT}/fig3_layouts.png",dpi=150); plt.close(fig)
print("Fig6 info (seed, min spacing, violating pairs, max radius):",info)
