"""
Exact criterion (Theorem: post-aggregation error) evaluated at each relation's own
measured parameters, replacing the noise-dominated window eq:window which is not
valid at the feature separations these networks actually have.

Aggregation beats the attribute-only rule iff  SNR(rho) > r/2, where
  delta = (1-(1+c)rho) * r,   s_B^2 = (1+rho(1-rho) r^2)/D,
  s_H^2 = (1+eta(1-eta) r^2)/D,   SNR = |delta|/(s_B+s_H),  eta = c*rho.
Units: sigma = 1, so ||Delta|| = r.
"""
import json
import numpy as np
from scipy.optimize import brentq

def snr_ratio(rho, c, D, r):
    eta = c*rho
    delta = abs(1-(1+c)*rho)*r
    sB = np.sqrt((1+rho*(1-rho)*r*r)/D)
    sH = np.sqrt((1+eta*(1-eta)*r*r)/D)
    return (delta/(sB+sH))/(r/2)

def exact_window(c, D, r, n=20001):
    xs=np.linspace(1e-9,1-1e-9,n); vs=np.array([snr_ratio(x,c,D,r) for x in xs])-1
    roots=[]
    for i in range(1,n):
        if vs[i-1]==0 or vs[i-1]*vs[i]<0:
            roots.append(brentq(lambda z: snr_ratio(z,c,D,r)-1, xs[i-1], xs[i], xtol=1e-12))
    return roots

# MGTAB, measured
c = 0.368810
rows = [("following",4,0.9693),("friendship",8,0.9889),("mention",4,0.9941),
        ("quote",2,0.9992),("reply",1,0.9910),("union",8,0.9889),
        ("shared URL",12,0.5951),("shared hashtag",53,0.6272)]

for r_label, r in [("paper's r=13.0", 13.0), ("honest r=2.31", 2.31)]:
    print("="*78); print(f"MGTAB exact criterion, {r_label}   (c={c:.4f})"); print("="*78)
    print(f"{'relation':>16} {'D~':>4} {'rho_hat':>8} {'noise-dom window':>20} "
          f"{'EXACT roots':>20} {'SNR/raw at rho':>15} {'verdict':>9}")
    out=[]
    for name,D,rho in rows:
        lo=(1-D**-0.5)/(1+c); hi=(1+D**-0.5)/(1+c)
        roots=exact_window(c,D,r)
        g=snr_ratio(rho,c,D,r)
        v = "helps" if g>1 else "HARMFUL"
        rs = "["+", ".join(f"{x:.3f}" for x in roots)+"]" if roots else "[none]"
        print(f"{name:>16} {D:>4} {rho:>8.4f} "
              f"{'('+format(lo,'.3f')+','+format(hi,'.3f')+')':>20} {rs:>20} "
              f"{g:>15.3f} {v:>9}")
        out.append(dict(relation=name,D=D,rho=rho,noise_dom=[lo,hi],
                        exact_roots=[float(x) for x in roots],
                        snr_ratio_at_rho=float(g),verdict=v))
    print()
    json.dump(out, open(f"exact_window_r{r:g}.json","w"), indent=2)
print("mixture term size at r=2.31, rho=0.99:",
      f"{0.99*0.01*2.31**2:.3f} sigma^2 (vs 1.0) -- noise-dominated assumption is",
      "ok" if 0.99*0.01*2.31**2 < 0.2 else "VIOLATED")
print("mixture term size at r=13,   rho=0.99:",
      f"{0.99*0.01*13**2:.3f} sigma^2 -- clearly VIOLATED")
