"""
Numerical validation of the two theory extensions.

  (E1) Achievable pruning quality, as a genuine bias-variance trade-off.

       A feature-discrepancy edge score cannot separate camouflage from genuine
       edges perfectly; its operating points (e_w, e_c) lie on a ROC governed by
       the edge-discriminability index

            kappa = ||Delta||^2 / (2 sigma^2 sqrt(2 d)) = r^2 / (2 sqrt(2 d)).

       Crucially, down-weighting edges non-uniformly also *reduces the effective
       degree*, because a weighted mean of D terms has the variance of an
       unweighted mean of only

            D_eff = D * (sum w)^2 / (sum w^2)   <=   D

       terms (Cauchy-Schwarz, equality iff the weights are uniform). Pruning
       therefore buys a reduction in bias at the cost of variance, and the
       optimum is interior. This also removes a degenerate solution: deleting
       every edge leaves rho_eff = rho and D_eff = D unchanged, so uniform
       attenuation achieves nothing, as it must.

  (E2) Depth. Under a locally tree-like assumption the class separation
       multiplies by lambda = 1 - (1+c) rho per layer while the noise standard
       deviation divides by sqrt(D), so

            SNR_L / SNR_raw = (lambda sqrt(D))^L,

       making the critical ratio independent of L while the gain or loss is
       exponential in L.

Run with: python extensions.py
"""
import json

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2, ncx2, norm

RNG = np.random.default_rng(7)


# --------------------------------------------------------------------------
# E1: achievable pruning quality
# --------------------------------------------------------------------------

def kappa_of(r, d):
    """Edge-discriminability index for the raw squared-discrepancy score."""
    return r**2 / (2.0 * np.sqrt(2.0 * d))


def roc_gaussian(e_w, kappa):
    """Gaussian (large-d) approximation of the score ROC: e_c given e_w."""
    return norm.cdf(norm.ppf(e_w) + kappa)


def roc_exact(e_w, r, d):
    """
    Exact ROC for the score s = ||x_u - x_v||^2 under the CSBM.

    Within-class: s ~ 2 sigma^2 chi2_d.  Cross-class: s ~ 2 sigma^2 ncx2_d(lam)
    with lam = ||Delta||^2 / (2 sigma^2) = r^2 / 2.
    """
    tau = chi2.isf(e_w, df=d)
    return ncx2.sf(tau, df=d, nc=r**2 / 2.0)


def empirical_roc(e_w, r, d, n=200_000, rng=RNG):
    """Monte-Carlo check of roc_exact by sampling actual feature pairs."""
    delta = np.zeros(d)
    delta[0] = r  # sigma = 1, so ||Delta|| = r
    diff_w = rng.normal(0, 1, (n, d)) - rng.normal(0, 1, (n, d))
    diff_c = (delta + rng.normal(0, 1, (n, d))) - rng.normal(0, 1, (n, d))
    s_w, s_c = (diff_w**2).sum(1), (diff_c**2).sum(1)
    tau = np.quantile(s_w, 1.0 - e_w)
    return float((s_c > tau).mean())


def rho_star(D, c):
    """Critical camouflage ratio with no pruning, noise-dominated regime."""
    return (1.0 - D**-0.5) / (1.0 + c)


def rho_eff(rho, a, b):
    """Effective camouflage ratio: cross-class share of total edge weight."""
    return rho * a / (rho * a + (1.0 - rho) * b)


def D_eff(D, rho, a, b):
    """
    Effective degree of a weighted mean, D * (sum w)^2 / (sum w^2).

    Non-uniform weights strictly reduce it, which is the cost of pruning.
    """
    s1 = (1.0 - rho) * b + rho * a
    s2 = (1.0 - rho) * b**2 + rho * a**2
    if s2 <= 0:
        return 0.0
    return D * s1**2 / s2


def snr_gain(rho, c, D, a, b):
    """
    Ratio of post-aggregation SNR to the graph-free SNR, under pruning
    weights (a, b) = (1 - e_c, 1 - e_w). Aggregation helps iff this exceeds 1.
    """
    re = rho_eff(rho, a, b)
    de = D_eff(D, rho, a, b)
    lam = 1.0 - (1.0 + c) * re
    return lam * np.sqrt(de)


def tolerance_at(c, D, a, b, hi=0.999):
    """Largest rho for which pruned aggregation still beats the raw baseline."""
    f = lambda r: snr_gain(r, c, D, a, b) - 1.0
    if f(1e-9) <= 0:
        return 0.0
    if f(hi) > 0:
        return hi
    return float(brentq(f, 1e-9, hi, xtol=1e-10))


def best_tolerance(c, D, kappa):
    """
    Maximise the tolerated camouflage over the pruner's operating point,
    subject to the ROC induced by kappa. Returns the interior optimum.
    """
    def neg(logit_ew):
        e_w = 1.0 / (1.0 + np.exp(-logit_ew))
        e_c = roc_gaussian(e_w, kappa)
        return -tolerance_at(c, D, 1.0 - e_c, 1.0 - e_w)

    res = minimize_scalar(neg, bounds=(-14.0, 6.0), method="bounded",
                          options={"xatol": 1e-6})
    e_w = 1.0 / (1.0 + np.exp(-res.x))
    e_c = roc_gaussian(e_w, kappa)
    return dict(e_w=float(e_w), e_c=float(e_c),
                ratio=float((1 - e_c) / max(1 - e_w, 1e-300)),
                tolerance=float(-res.fun))


# --------------------------------------------------------------------------
# E2: depth
# --------------------------------------------------------------------------

def simulate_depth(rho, c, D, L, delta=1.0, sigma=1.0, n=4000, rng=RNG):
    """
    Simulate L rounds of mean aggregation on a locally tree-like graph.

    Because every level averages D equally weighted children, the L-fold
    iterated mean equals the flat mean over all D^L leaves; we therefore build
    the leaf class labels by expanding the tree level by level, draw leaf
    features, and take one flat mean.
    """
    eta = c * rho

    def leaves(root_is_bot):
        lab = np.full((n, 1), root_is_bot, dtype=bool)
        for _ in range(L):
            p_opp = np.where(lab, rho, eta)            # per-parent flip prob
            rep = np.repeat(p_opp, D, axis=1)
            flip = rng.random((n, lab.shape[1] * D)) < rep
            lab = np.repeat(lab, D, axis=1) ^ flip
        return lab

    def stat(root_is_bot):
        lab = leaves(root_is_bot)
        mu = np.where(lab, +delta / 2.0, -delta / 2.0)
        return (mu + rng.normal(0, sigma, lab.shape)).mean(axis=1)

    b, h = stat(True), stat(False)
    sd = np.sqrt(0.5 * (b.var() + h.var()))
    return float((b.mean() - h.mean()) / (2.0 * sd))


def predicted_snr_ratio(rho, c, D, L):
    """(lambda sqrt(D))^L."""
    lam = 1.0 - (1.0 + c) * rho
    return (lam * np.sqrt(D)) ** L


# --------------------------------------------------------------------------

def main():
    out = {}
    c = 0.3 / 0.7          # bot prevalence 30%
    D = 16

    print("=" * 74)
    print("E1  ROC of the feature-discrepancy edge score")
    print("=" * 74)
    print(f"{'r':>5} {'d':>5} {'kappa':>8} {'e_w':>6} "
          f"{'e_c exact':>10} {'e_c gauss':>10} {'e_c MC':>9}")
    rows = []
    for r, d in [(3.0, 16), (3.0, 64), (5.0, 64), (2.0, 8)]:
        k = kappa_of(r, d)
        for e_w in (0.05, 0.20):
            ex, ga = roc_exact(e_w, r, d), roc_gaussian(e_w, k)
            mc = empirical_roc(e_w, r, d)
            print(f"{r:5.1f} {d:5d} {k:8.3f} {e_w:6.2f} "
                  f"{ex:10.4f} {ga:10.4f} {mc:9.4f}")
            rows.append(dict(r=r, d=d, kappa=float(k), e_w=e_w,
                             e_c_exact=float(ex), e_c_gauss=float(ga),
                             e_c_mc=mc))
    out["roc"] = rows

    print("\ne_w -> 0 forces e_c -> 0: perfect pruning is unattainable "
          f"(r=3, d=16, kappa={kappa_of(3.0,16):.3f})")
    deg = []
    for e_w in (1e-6, 1e-4, 1e-2, 0.1):
        e_c = roc_gaussian(e_w, kappa_of(3.0, 16))
        print(f"   e_w={e_w:<8g} -> e_c={e_c:.5f}")
        deg.append(dict(e_w=e_w, e_c=float(e_c)))
    out["degenerate_limit"] = deg

    print("\n" + "=" * 74)
    print("E1  Pruning is a bias-variance trade-off: D_eff <= D")
    print("=" * 74)
    rs = rho_star(D, c)
    print(f"D={D}, pi_B=0.3 (c={c:.4f}), unpruned rho*={rs:.4f}")
    print(f"\n{'e_c':>6} {'e_w':>6} {'rho':>6} {'rho_eff':>8} "
          f"{'D_eff':>8} {'SNR gain':>9}")
    tr = []
    for e_c, e_w in [(0.0, 0.0), (0.75, 0.0), (0.75, 0.25), (0.99, 0.0),
                     (0.5, 0.5), (0.999, 0.999)]:
        a, b = 1 - e_c, 1 - e_w
        r0 = 0.5
        print(f"{e_c:6.3f} {e_w:6.3f} {r0:6.2f} {rho_eff(r0,a,b):8.4f} "
              f"{D_eff(D,r0,a,b):8.3f} {snr_gain(r0,c,D,a,b):9.4f}")
        tr.append(dict(e_c=e_c, e_w=e_w, rho=r0,
                       rho_eff=float(rho_eff(r0, a, b)),
                       D_eff=float(D_eff(D, r0, a, b)),
                       snr_gain=float(snr_gain(r0, c, D, a, b))))
    out["tradeoff"] = tr
    print("  (last row: uniform attenuation leaves rho_eff and D_eff "
          "unchanged, as it must)")

    print("\n" + "=" * 74)
    print("E1  Parameter-free optimal tolerance, a function of kappa alone")
    print("=" * 74)
    print(f"{'r':>5} {'d':>5} {'kappa':>8} {'e_w*':>8} {'e_c*':>8} "
          f"{'a/b':>7} {'tol':>8} {'vs rho*':>9}")
    tol = []
    for r, d in [(2.0, 64), (3.0, 64), (3.0, 16), (5.0, 16), (8.0, 16)]:
        k = kappa_of(r, d)
        bt = best_tolerance(c, D, k)
        print(f"{r:5.1f} {d:5d} {k:8.3f} {bt['e_w']:8.4f} {bt['e_c']:8.4f} "
              f"{bt['ratio']:7.3f} {bt['tolerance']:8.4f} "
              f"{bt['tolerance']/rs:9.3f}x")
        tol.append(dict(r=r, d=d, kappa=float(k), **bt))
    out["optimal_tolerance"] = tol
    out["rho_star_base"] = float(rs)

    print("\nProjection: kappa ~ 1/sqrt(dim), so reducing dimension while "
          "retaining ||Delta|| helps")
    proj = []
    for dd in (1, 4, 16, 64, 256):
        k = kappa_of(3.0, dd)
        bt = best_tolerance(c, D, k)
        print(f"   dim={dd:4d}  kappa={k:7.4f}  tolerance={bt['tolerance']:.4f}")
        proj.append(dict(dim=dd, kappa=float(k),
                         tolerance=float(bt["tolerance"])))
    out["projection"] = proj

    print("\n" + "=" * 74)
    print("E2  Depth: SNR_L / SNR_raw = (lambda sqrt(D))^L")
    print("=" * 74)
    print(f"{'rho':>6} {'D':>4} {'L':>3} {'lam*sqrt(D)':>12} "
          f"{'predicted':>11} {'simulated':>11} {'rel err':>9}")
    dep = []
    for Dd in (9,):
        for rho in (0.0, 0.2, 0.35, 0.5):
            for L in (1, 2, 3):
                pred = predicted_snr_ratio(rho, c, Dd, L)
                nn = max(1500, int(3.0e6 / Dd**L))
                sim = simulate_depth(rho, c, Dd, L, n=nn) / 0.5
                err = abs(sim - pred) / max(pred, 1e-12)
                lam_sd = (1.0 - (1.0 + c) * rho) * np.sqrt(Dd)
                print(f"{rho:6.2f} {Dd:4d} {L:3d} {lam_sd:12.4f} "
                      f"{pred:11.4f} {sim:11.4f} {err:9.4f}")
                dep.append(dict(rho=rho, D=Dd, L=L, lam_sqrtD=float(lam_sd),
                                predicted=float(pred), simulated=float(sim),
                                rel_err=float(err)))
    out["depth"] = dep

    print("\nThe critical ratio is depth-independent: lambda*sqrt(D) = 1 "
          "exactly at rho*")
    dind = []
    for Dd in (4, 16, 64):
        r_d = rho_star(Dd, c)
        lam_sd = (1.0 - (1.0 + c) * r_d) * np.sqrt(Dd)
        print(f"   D={Dd:3d}  rho*={r_d:.4f}  lambda*sqrt(D)={lam_sd:.8f}")
        dind.append(dict(D=Dd, rho_star=float(r_d), lam_sqrtD=float(lam_sd)))
    out["depth_independence"] = dind

    with open("extension_results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nWrote extension_results.json")


if __name__ == "__main__":
    main()
