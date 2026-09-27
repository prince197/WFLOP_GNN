import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, FancyArrowPatch
plt.rcParams["font.family"]="DejaVu Sans"
NAVY,GREEN,RED,ORANGE,MUTED="#1F3864","#2E7D32","#B71C1C","#C2560C","#444444"
fig=plt.figure(figsize=(20,8),dpi=200)
bg=fig.add_axes([0,0,1,1]); bg.axis("off"); bg.set_xlim(0,20); bg.set_ylim(0,8)
# ---------- left panel ----------
bg.text(3.0,7.35,"Wind farm layout optimization",ha="center",fontsize=19,fontweight="bold",color=NAVY)
ax=fig.add_axes([0.02,0.16,0.26,0.66]); ax.set_aspect("equal"); ax.axis("off")
R=1.0; ax.set_xlim(-1.25,1.35); ax.set_ylim(-1.15,1.25)
farm=Circle((0,0),R,fc="#DCE7F5",ec=NAVY,lw=2.5); ax.add_patch(farm)
T=np.array([[0.05,0.72],[-0.52,0.42],[0.40,0.36],[0.02,0.02],[-0.47,-0.42],[0.43,-0.47],[0.0,-0.78]])
for x,y in T:
    ax.add_patch(Polygon([[x,y],[x+0.5,y+0.2],[x+0.5,y-0.2]],closed=True,fc="#9DBDE0",ec="none",alpha=0.75,zorder=1,clip_path=farm))
ax.scatter(T[:,0],T[:,1],s=150,color=NAVY,edgecolor="white",lw=1.5,zorder=3)
ax.plot([-0.52,-0.47],[0.42,-0.42],ls=":",color=RED,lw=2.5,zorder=2)
ax.text(-0.66,0.0,"d ≥ 308 m",rotation=90,ha="center",va="center",color=RED,fontsize=15)
ax.add_patch(FancyArrowPatch((-1.2,1.13),(-0.75,1.13),arrowstyle="-|>",mutation_scale=22,color=ORANGE,lw=3))
ax.text(-0.975,1.0,"wind",ha="center",va="top",color=ORANGE,fontsize=15)
bg.text(3.0,1.05,"Jensen wakes · Weibull wind · circular farms\nr = 500, 750, 1000 m · N = 2–18 turbines",ha="center",va="center",fontsize=14,color=MUTED,linespacing=1.5)
# ---------- arrows between panels ----------
for x in (6.05,13.75):
    bg.add_patch(FancyArrowPatch((x,4.1),(x+0.55,4.1),arrowstyle="simple,head_width=0.9,head_length=0.7,tail_width=0.35",mutation_scale=22,color=NAVY))
# ---------- middle panel ----------
bg.text(9.95,7.35,"GNN-LX-SSA",ha="center",fontsize=20,fontweight="bold",color=NAVY)
def box(cx,cy,w,h,txt,ec=NAVY,fc="white",tc=NAVY,fs=14.5):
    bg.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle="round,pad=0.02,rounding_size=0.12",fc=fc,ec=ec,lw=2.4))
    bg.text(cx,cy,txt,ha="center",va="center",fontsize=fs,fontweight="bold",color=tc,linespacing=1.3)
def arr(p,q,c=NAVY):
    bg.add_patch(FancyArrowPatch(p,q,arrowstyle="-|>",mutation_scale=20,color=c,lw=2.6,shrinkA=0,shrinkB=0))
box(8.15,6.05,3.0,1.1,"layout as a\ndirected wake graph")
box(11.85,6.05,3.3,1.1,"GNWM surrogate\n(message-passing GNN)",fc="#DCE7F5")
arr((9.65,6.05),(10.2,6.05))
box(11.0,4.1,1.55,1.05,"power\nhead P̂",fs=14); box(12.75,4.1,1.55,1.05,"guidance\nhead ĝ",fs=14)
arr((11.4,5.5),(11.1,4.63)); arr((12.3,5.5),(12.6,4.63))
box(8.15,4.1,3.0,1.1,"LX-SSA search\n(Laplace operator)",fc="#DCE7F5")
arr((10.22,4.1),(9.65,4.1))
box(8.15,2.15,3.0,1.1,"repair map Π\n(boundary + spacing)",ec=GREEN,fc="#E6F2E7",tc=GREEN)
box(11.85,2.15,3.3,1.1,"exact evaluation of\nscreened candidates only")
arr((8.15,3.55),(8.15,2.7),GREEN); arr((9.65,2.15),(10.2,2.15))
arr((11.0,3.57),(11.5,2.7))
bg.text(9.95,1.05,"feasibility comes from the repair map;\nthe surrogate cuts exact evaluations to about one third of LX-SSA's",ha="center",va="center",fontsize=14,style="italic",color=MUTED,linespacing=1.5)
# ---------- right panel ----------
bg.text(17.0,7.35,"Turbines per farm",ha="center",fontsize=19,fontweight="bold",color=NAVY)
bx=fig.add_axes([0.715,0.37,0.265,0.47])
x=np.arange(3); w=0.36
pub=[8,12,15]; low=[13,26,43]; up=[15,30,49]
b1=bx.bar(x-w/2,pub,w,color="#C8C8C8",ec="#888",label="published limit")
b2=bx.bar(x+w/2,low,w,color=GREEN,label="explicit feasible layout (this work)")
for i in range(3):
    bx.plot([x[i]+w/2-0.2,x[i]+w/2+0.2],[up[i]]*2,color=NAVY,lw=2.6)
    bx.text(x[i]+w/2+0.23,up[i],f"≤{up[i]}",va="center",ha="left",fontsize=13,color=NAVY)
    bx.text(x[i]-w/2,pub[i]+0.8,str(pub[i]),ha="center",fontsize=15,fontweight="bold",color="#333")
    bx.text(x[i]+w/2,low[i]/2,str(low[i]),ha="center",va="center",fontsize=16,fontweight="bold",color="white")
bx.plot([],[],color=NAVY,lw=2.6,label="upper bound")
bx.set_xticks(x,["500 m","750 m","1000 m"],fontsize=15); bx.set_yticks([]); bx.set_ylim(0,54); bx.set_xlim(-0.6,2.75)
for s in ("top","right","left"): bx.spines[s].set_visible(False)
h,l=bx.get_legend_handles_labels(); o=[l.index("published limit"),l.index("explicit feasible layout (this work)"),l.index("upper bound")]; bx.legend([h[i] for i in o],[l[i] for i in o],frameon=False,fontsize=12.5,loc="upper left")
bg.text(17.0,2.25,"published limits are penalty artefacts",ha="center",fontsize=16.5,fontweight="bold",color=RED)
bg.text(17.0,1.3,"feasible in all 2,340 GNN-LX-SSA campaign runs\nbest Friedman rank in both wind regimes",ha="center",va="center",fontsize=14.5,color=NAVY,linespacing=1.5)
fig.savefig("figs_new/graphical_abstract.png",dpi=200,facecolor="white")
