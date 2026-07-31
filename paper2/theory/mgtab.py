"""
Produce Table 4: camouflage measured per relation on MGTAB.

MGTAB (Shi et al., Neurocomputing 2025) is openly distributed and labels every one of
its 10,199 accounts, so all ties are between labelled actors. Reads the released
tensors converted to a single .npz (edge_index, edge_type, label_bot) plus the node
attribute matrix; see README for the fetch step.

Usage: python mgtab.py --dir /path/to/mgtab
"""
import argparse, json
import numpy as np
from scipy.stats import norm

REL = {0:"following",1:"friendship",2:"mention",3:"reply",4:"quote",
       5:"shared URL",6:"shared hashtag"}

def undirected(ei, mask):
    s,t = ei[0][mask], ei[1][mask]
    a,b = np.minimum(s,t), np.maximum(s,t)
    p = np.unique(np.stack([a,b],1), axis=0)
    return p[p[:,0]!=p[:,1]]

def per_relation(p, y):
    deg = np.zeros(len(y),int); cross = np.zeros(len(y),int)
    np.add.at(deg, p[:,0], 1); np.add.at(deg, p[:,1], 1)
    diff = y[p[:,0]] != y[p[:,1]]
    np.add.at(cross, p[diff][:,0], 1); np.add.at(cross, p[diff][:,1], 1)
    hasdeg = deg > 0
    rb = (cross/np.maximum(deg,1))[(y==1)&hasdeg]
    rh = (cross/np.maximum(deg,1))[(y==0)&hasdeg]
    # ratio-of-sums estimators, matching the endpoint counting of eq:(balance)
    rho_ros = cross[y==1].sum()/max(deg[y==1].sum(),1)
    eta_ros = cross[y==0].sum()/max(deg[y==0].sum(),1)
    return dict(n_ties=int(len(p)),
        rho_mean=float(rb.mean()), rho_median=float(np.median(rb)),
        eta_mean=float(rh.mean()),
        rho_ratio_of_sums=float(rho_ros), eta_ratio_of_sums=float(eta_ros),
        D_median_bot=float(np.median(deg[(y==1)&hasdeg])),
        deg_mean_bot=float(deg[y==1].mean()), deg_mean_hum=float(deg[y==0].mean()),
        bb=int((y[p[:,0]]+y[p[:,1]]==2).sum()), bh=int(diff.sum()))

def snr_ratio(rho, c, D, r):
    eta=c*rho
    return (abs(1-(1+c)*rho)*r/(np.sqrt((1+rho*(1-rho)*r*r)/D)
            + np.sqrt((1+eta*(1-eta)*r*r)/D)))/(r/2)

def calibrated_r(X, y, seed=0):
    """r from achievable held-out balanced error, not from raw moments."""
    rng=np.random.default_rng(seed); idx=rng.permutation(len(y))
    tr,te=idx[:int(.7*len(y))],idx[int(.7*len(y)):]
    Xs=(X-X[tr].mean(0))/(X[tr].std(0)+1e-9)
    w=np.linalg.pinv(np.c_[Xs[tr],np.ones(len(tr))])@(2*y[tr]-1)
    s=np.c_[Xs[te],np.ones(len(te))]@w
    best=min(.5*((s[y[te]==1]<=t).mean()+(s[y[te]==0]>t).mean())
             for t in np.quantile(s,np.linspace(.001,.999,999)))
    return float(-2*norm.ppf(best)), float(best)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dir",required=True)
    ap.add_argument("--out",default="mgtab_results.json"); a=ap.parse_args()
    d=np.load(f"{a.dir}/MGTAB_graph.npz")
    ei,et,y = d["edge_index"], d["edge_type"], d["label_bot"]
    X=np.load(f"{a.dir}/MGTAB_features.npy")
    c=float(y.sum()/(y==0).sum())
    r,err=calibrated_r(X,y)
    print(f"MGTAB: n={len(y)} bots={int(y.sum())} c={c:.4f}")
    print(f"  calibrated r={r:.3f} (held-out balanced error {err:.4f}); "
          f"naive moment r would be much larger -- see paper Sec. 5.7")
    print(f"\n{'relation':>16} {'ties':>8} {'rho':>7} {'rho~':>5} {'eta':>7} "
          f"{'D~':>4} {'SNR/raw':>8} {'verdict':>9}")
    rows=[]
    for t in sorted(REL):
        st=per_relation(undirected(ei, et==t), y)
        g=snr_ratio(st["rho_mean"], c, st["D_median_bot"], r)
        v="helps" if g>1 else "HARMFUL"
        print(f"{REL[t]:>16} {st['n_ties']:>8d} {st['rho_mean']:>7.4f} "
              f"{st['rho_median']:>5.2f} {st['eta_mean']:>7.4f} "
              f"{st['D_median_bot']:>4.0f} {g:>8.3f} {v:>9}")
        rows.append(dict(relation=REL[t], snr_ratio=float(g), verdict=v, **st))
    st=per_relation(undirected(ei, (et==0)|(et==1)), y)
    g=snr_ratio(st["rho_mean"], c, st["D_median_bot"], r)
    print(f"{'social union':>16} {st['n_ties']:>8d} {st['rho_mean']:>7.4f} "
          f"{st['rho_median']:>5.2f} {st['eta_mean']:>7.4f} "
          f"{st['D_median_bot']:>4.0f} {g:>8.3f} "
          f"{'helps' if g>1 else 'HARMFUL':>9}")
    rows.append(dict(relation="social union", snr_ratio=float(g), **st))
    # edge-balance identity with matched estimators
    pred=c*st["rho_ratio_of_sums"]*(st["deg_mean_bot"]/st["deg_mean_hum"])
    print(f"\nedge-balance (matched, ratio-of-sums): predicted {pred:.8f} "
          f"vs measured {st['eta_ratio_of_sums']:.8f} "
          f"(|diff| {abs(pred-st['eta_ratio_of_sums']):.2e})")
    json.dump(dict(c=c, r_calibrated=r, heldout_balanced_error=err,
                   n=len(y), n_bot=int(y.sum()), relations=rows),
              open(a.out,"w"), indent=2)
    print(f"Wrote {a.out}")

if __name__=="__main__": main()
