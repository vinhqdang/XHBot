"""
Measure the camouflage ratio on real labelled fraud graphs.

TwiBot-20/22 are access-gated (author approval required), so we use the graphs
from CARE-GNN (Dou et al., CIKM 2020) -- the work that introduced the camouflage
framing -- plus the Elliptic bitcoin graph as a low-camouflage contrast case.
All three are openly downloadable.

For each graph and each relation we report the measured camouflage ratio, the
quantities the theory needs (c, D, r, d), and where the graph sits relative to
the predicted harmful window.

Usage:
    python real_data.py --dir /path/containing/YelpChi.mat
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


def window(D, c):
    s = D ** -0.5
    return ((1.0 - s) / (1.0 + c), (1.0 + s) / (1.0 + c))


def kappa_of(r, d):
    return r**2 / (2.0 * np.sqrt(2.0 * d))


def feature_snr(X, y):
    """r = ||Delta|| / sigma, with sigma the pooled per-dimension std."""
    mu1, mu0 = X[y == 1].mean(0), X[y == 0].mean(0)
    delta = np.linalg.norm(mu1 - mu0)
    sigma = np.sqrt(0.5 * (X[y == 1].var(0).mean() + X[y == 0].var(0).mean()))
    return float(delta / sigma) if sigma > 0 else float("nan")


def analyse_relation(A, y, name):
    """Camouflage ratio for one relation, plus degree statistics."""
    A = sp.csr_matrix(A)
    A.data[:] = 1.0
    A.setdiag(0)
    A.eliminate_zeros()
    deg = np.asarray(A.sum(1)).ravel()
    benign_nb = np.asarray(A.dot((y == 0).astype(float))).ravel()
    fraud_nb = np.asarray(A.dot((y == 1).astype(float))).ravel()

    fm = (y == 1) & (deg > 0)
    bm = (y == 0) & (deg > 0)
    rho = benign_nb[fm] / deg[fm]
    eta = fraud_nb[bm] / deg[bm]
    # mean degree per class: the edge-balance identity needs a degree
    # correction when the two classes have different mean degree.
    dB = float(deg[fm].mean()) if fm.any() else float("nan")
    dH = float(deg[bm].mean()) if bm.any() else float("nan")
    return dict(
        relation=name,
        n_edges=int(A.nnz // 2),
        rho_mean=float(rho.mean()),
        rho_median=float(np.median(rho)),
        eta_mean=float(eta.mean()),
        D_median_fraud=float(np.median(deg[fm])),
        D_median_all=float(np.median(deg[deg > 0])),
        deg_mean_fraud=dB,
        deg_mean_benign=dH,
        deg_ratio=dB / dH if dH else float("nan"),
    )


def report(name, X, y, relations):
    print("=" * 78)
    print(f"{name}:  {len(y)} nodes, {int((y==1).sum())} fraud "
          f"({100*(y==1).mean():.1f}%)")
    c = float((y == 1).sum() / max((y == 0).sum(), 1))
    r = feature_snr(X, y) if X is not None else float("nan")
    d = X.shape[1] if X is not None else 0
    print(f"  c = {c:.4f}", end="")
    if X is not None:
        print(f",  feature SNR r = {r:.3f} over d = {d} dims,"
              f"  kappa = {kappa_of(r, d):.3f}")
    else:
        print("  (no features available)")
    print(f"\n  {'relation':<10} {'edges':>10} {'rho':>7} {'eta obs':>8} "
          f"{'c*rho':>7} {'dB/dH':>7} {'c*rho*dB/dH':>12} {'D med':>6} "
          f"{'window':>15} {'verdict':>9}")

    rows = []
    for rel, A in relations.items():
        st = analyse_relation(A, y, rel)
        lo, hi = window(st["D_median_fraud"], c)
        rho = st["rho_mean"]
        eta_corr = c * rho * st["deg_ratio"]
        verdict = ("HELPS" if rho < lo
                   else "flipped" if rho > hi else "HARMFUL")
        if st["D_median_fraud"] <= 1:
            verdict = "degenerate"          # D=1: no averaging benefit exists
        print(f"  {rel:<10} {st['n_edges']:>10,} {rho:>7.4f} "
              f"{st['eta_mean']:>8.4f} {c*rho:>7.4f} {st['deg_ratio']:>7.3f} "
              f"{eta_corr:>12.4f} {st['D_median_fraud']:>6.0f} "
              f"{'(' + format(lo, '.2f') + ',' + format(hi, '.2f') + ')':>15} "
              f"{verdict:>9}")
        st.update(window_lower=lo, window_upper=hi, verdict=verdict,
                  eta_predicted_plain=float(c * rho),
                  eta_predicted_degcorr=float(eta_corr),
                  c=c, r=r, d=int(d))
        rows.append(st)
    return dict(dataset=name, c=c, r=r, d=int(d),
                kappa=float(kappa_of(r, d)) if X is not None else None,
                n_nodes=int(len(y)), n_fraud=int((y == 1).sum()),
                relations=rows)


def load_mat(path, rel_keys):
    d = sio.loadmat(path)
    y = d["label"].ravel().astype(np.int64)
    X = np.asarray(d["features"].todense()
                   if sp.issparse(d["features"]) else d["features"],
                   dtype=float)
    return X, y, {k: d[k] for k in rel_keys if k in d}


def load_elliptic(dirname):
    import csv
    cls_p = os.path.join(dirname, "elliptic_txs_classes.csv")
    edg_p = os.path.join(dirname, "elliptic_txs_edgelist.csv")
    if not (os.path.exists(cls_p) and os.path.exists(edg_p)):
        return None
    idx, lab = {}, []
    with open(cls_p) as fh:
        rd = csv.reader(fh)
        next(rd, None)
        for tid, c in rd:
            if c == "unknown":
                continue
            idx[tid] = len(lab)
            lab.append(1 if c == "1" else 0)      # class 1 = illicit
    y = np.array(lab, dtype=np.int64)
    rows, cols = [], []
    with open(edg_p) as fh:
        rd = csv.reader(fh)
        next(rd, None)
        for u, v in rd:
            if u in idx and v in idx:
                rows.append(idx[u]); cols.append(idx[v])
    n = len(y)
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = ((A + A.T) > 0).astype(float)
    return y, {"tx": A}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="real_data_results.json")
    a = ap.parse_args()
    out = []

    yelp = os.path.join(a.dir, "YelpChi.mat")
    if os.path.exists(yelp):
        X, y, rels = load_mat(yelp, ["homo", "net_rur", "net_rtr", "net_rsr"])
        out.append(report("YelpChi (CARE-GNN)", X, y, rels))

    amz = os.path.join(a.dir, "Amazon.mat")
    if os.path.exists(amz):
        X, y, rels = load_mat(amz, ["homo", "net_upu", "net_usu", "net_uvu"])
        out.append(report("Amazon (CARE-GNN)", X, y, rels))

    ell = load_elliptic(a.dir)
    if ell is not None:
        y, rels = ell
        out.append(report("Elliptic bitcoin", None, y, rels))

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
