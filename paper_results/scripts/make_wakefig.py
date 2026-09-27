import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R=38.5; D=2*R; K=0.075; CT=0.8; A=1-np.sqrt(1-CT)
TI=0.075; ks=0.3837*TI+0.003678; beta=0.5*(1+np.sqrt(1-CT))/np.sqrt(1-CT); eps=0.2*np.sqrt(beta)
INK="#1F2328"; MUTED="#57606A"; GRID="#E6E8EB"; CJ="#2563A8"; CG="#C2410C"
def jensen_c(x): return A/(1+K*x/R)**2
def gauss(x,r):
    s=ks*x/D+eps; return (1-np.sqrt(np.clip(1-CT/(8*s*s),0,1)))*np.exp(-r**2/(2*(s*D)**2))
plt.rcParams.update({"font.family":"DejaVu Sans","axes.edgecolor":MUTED,"axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":MUTED})
fig,(a1,a2)=plt.subplots(1,2,figsize=(13.2,5.0))
xd=np.linspace(0.5,20,600); x=xd*D
a1.axvspan(0,4,color="#F2F3F5",lw=0); a1.text(2,0.07,"closer than the\n308 m spacing rule\n(infeasible)",ha="center",va="center",fontsize=10.5,color=MUTED)
a1.plot(xd,jensen_c(x),color=CJ,lw=2.4,label="Jensen (top-hat)")
a1.plot(xd,gauss(x,0),color=CG,lw=2.4,label="Gaussian, centre line")
for xx in (4,7,10):
    a1.plot([xx],[gauss(xx*D,0)],"o",color=CG,ms=7,mec="white"); a1.plot([xx],[jensen_c(xx*D)],"o",color=CJ,ms=7,mec="white")
    a1.annotate(f"{gauss(xx*D,0):.2f}",(xx,gauss(xx*D,0)),xytext=(8,4),textcoords="offset points",fontsize=10,color=CG)
    a1.annotate(f"{jensen_c(xx*D):.2f}",(xx,jensen_c(xx*D)),xytext=(8,4),textcoords="offset points",fontsize=10,color=CJ)
a1.set_xlim(0,20); a1.set_ylim(0,0.75); a1.set_xlabel("Downstream distance x / D"); a1.set_ylabel(r"Velocity deficit $\delta s$")
a1.set_title("(a) Deficit on the wake axis",fontsize=13,color=INK,loc="left")
a1.legend(frameon=False,fontsize=11,loc="upper right"); a1.grid(color=GRID,lw=0.8)
yd=np.linspace(-2.5,2.5,801)
for xx,ls in ((4,"-"),(7,"--")):
    rj=(K*xx*D+R)/D
    a2.plot(yd,np.where(np.abs(yd)<rj,jensen_c(xx*D),0),color=CJ,lw=2.2,ls=ls,label=f"Jensen, x = {xx}D")
    a2.plot(yd,gauss(xx*D,yd*D),color=CG,lw=2.2,ls=ls,label=f"Gaussian, x = {xx}D")
a2.set_xlim(-2.5,2.5); a2.set_ylim(0,0.5); a2.set_xlabel("Lateral offset r / D"); a2.set_ylabel(r"Velocity deficit $\delta s$")
a2.set_title("(b) Lateral profile across the wake",fontsize=13,color=INK,loc="left")
a2.legend(frameon=False,fontsize=10.5,loc="upper right"); a2.grid(color=GRID,lw=0.8)
for a in (a1,a2): a.spines[["top","right"]].set_visible(False)
import sys; fig.tight_layout(); fig.savefig((sys.argv[1] if len(sys.argv)>1 else ".")+"/wake_models.png",dpi=160)
print(dict(ks=ks,eps=eps,beta=beta,j=[jensen_c(k*D) for k in (4,7,10)],g=[gauss(k*D,0) for k in (4,7,10)],
  rj=[(K*k*D+R)/D for k in (4,7)], sg=[(ks*k+eps) for k in (4,7)], clip=(np.sqrt(CT/8)-eps)/ks))
