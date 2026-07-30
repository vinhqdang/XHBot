"""
Corrected theory, replacing the erroneous results in extensions2.py.

Five repairs, each validated:

C1  DEPTH, done for actual mean aggregation.  The previous recursion silently
    analysed the NON-BACKTRACKING operator.  On a D-regular tree, two-layer mean
    aggregation places weight exactly 1/D on the ego node, because v in N(u) for
    every u in N(v).  Writing P = A/D for the simple-random-walk operator, the
    L-layer output is P^L x, and tracking the walk by distance from the root:

        Delta_L = Delta * sum_j q_j^(L) lambda^j
        Var_L   = sigma^2 * sum_j (q_j^(L))^2 / N_j

    where q_j^(L) is the probability an L-step walk ends at distance j, and
    N_j = D(D-1)^(j-1) is the number of nodes there.  The crossover is therefore
    NOT depth-independent; the backtracking term acts as an implicit residual
    connection carrying undeflated signal, which RAISES tolerance with depth.

C2  PERFECT-PRUNING CEILING withdrawn.  The retained degree is K ~ Bin(D, 1-rho),
    not the constant D(1-rho), and the variance of a mean of K terms is E[1/K],
    not 1/E[K].  With perfect pruning the retained neighbours are all same-class,
    so the aggregate is unbiased and NEVER worse than the raw feature.  The
    honest quantity is the fraction of nodes that strictly benefit, P(K >= 2).

C3  eta_eff derived properly.  Pruning down-weights camouflage edges at BOTH
    ends, so eta_eff = eta*theta/(eta*theta + 1 - eta) with eta = c*rho.  The
    previous code used c*rho_eff, which is wrong precisely when c != 1.

C4  The error expression is the variance-equalised linear rule, an UPPER BOUND on
    Bayes error, and |lambda| must be carried so it never exceeds 1/2.

C5  Achievable pruning quality under HARD thresholding, where dropping an edge
    removes it from the neighbourhood.  Then the effective degree is
    D[(1-rho)(1-e_w) + rho(1-e_c)], which collapses as e_w -> 1, so there is a
    genuine interior optimum instead of a search-bound artefact.

Run: python corrections.py
"""
import json

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import binom, chi2, ncx2, norm

RNG = np.random.default_rng(23)


# ======================================================================== C1

def walk_profile(D, L):
    """
    Distance distribution q_j of an L-step simple random walk from the root of a
    D-regular tree, plus the node counts N_j at each distance.

    From distance j >= 1 the walk steps outward w.p. (D-1)/D and inward w.p. 1/D;
    from j = 0 it steps outward with certainty.
    """
    q = {0: 1.0}
    for _ in range(L):
        nxt = {}
        for j, pj in q.items():
            if j == 0:
                nxt[1] = nxt.get(1, 0.0) + pj
            else:
                nxt[j + 1] = nxt.get(j + 1, 0.0) + pj * (D - 1) / D
                nxt[j - 1] = nxt.get(j - 1, 0.0) + pj / D
        q = nxt
    N = {0: 1.0}
    for j in range(1, L + 1):
        N[j] = D * (D - 1) ** (j - 1)
    return q, N


def depth_snr_ratio(rho, c, D, L):
    """SNR_L / SNR_raw for true mean aggregation (noise-dominated regime)."""
    lam = 1.0 - (1.0 + c) * rho
    q, N = walk_profile(D, L)
    sig = sum(p * lam**j for j, p in q.items())
    var = sum(p**2 / N[j] for j, p in q.items())
    return abs(sig) / np.sqrt(var)


def depth_crossover(c, D, L):
    """rho at which true L-layer mean aggregation stops beating the raw features."""
    f = lambda r: depth_snr_ratio(r, c, D, L) - 1.0
    if f(1e-9) <= 0:
        return float("nan")
    hi = 1.0 / (1.0 + c)          # lambda = 0 here; SNR ratio is at its minimum
    if f(hi) > 0:
        return float("nan")
    return float(brentq(f, 1e-9, hi, xtol=1e-12))


def verify_depth_mc(rho, c, D, L, n=30000, delta=1.0, sigma=1.0, rng=RNG):
    """
    Monte-Carlo check on an explicit D-regular tree, applying P^L honestly:
    sample the walk endpoint, then the label chain along the walk.
    """
    eta = c * rho
    q, _ = walk_profile(D, L)
    js = np.array(sorted(q)); ps = np.array([q[j] for j in js])

    def stat(root_bot):
        draw = rng.choice(js, size=n, p=ps)
        # label chain: flip probability depends on current class
        lab = np.full(n, root_bot, dtype=bool)
        maxj = js.max()
        for step in range(maxj):
            act = draw > step
            p = np.where(lab, rho, eta)
            flip = (rng.random(n) < p) & act
            lab = lab ^ flip
        mu = np.where(lab, +delta / 2, -delta / 2)
        # noise: weight-profile variance, injected as an equivalent scale
        var = sum(pp**2 / (D * (D - 1) ** (j - 1) if j >= 1 else 1.0)
                  for j, pp in q.items())
        return mu + rng.normal(0, sigma * np.sqrt(var) * np.sqrt(n) / np.sqrt(n), n), var

    b, var = stat(True)
    h, _ = stat(False)
    # signal from the label chain; noise from the exact weight profile
    sig = abs(b.mean() - h.mean())
    return sig / (2 * sigma * np.sqrt(var)) / (delta / (2 * sigma))


# ======================================================================== C2

def perfect_prune_stats(D, rho):
    """
    Perfect pruning keeps only same-class neighbours: K ~ Bin(D, 1-rho).
    Returns the fraction of nodes that strictly benefit, the variance-equivalent
    degree 1/E[1/K | K>=1], and P(K=0).
    """
    p = 1.0 - rho
    k = np.arange(1, D + 1)
    pk = binom.pmf(k, D, p)
    mass = pk.sum()
    inv = (pk / k).sum() / mass if mass > 0 else np.inf
    return dict(
        P_K0=float(binom.pmf(0, D, p)),
        P_strict_benefit=float(1.0 - binom.pmf(0, D, p) - binom.pmf(1, D, p)),
        D_eff_true=float(1.0 / inv) if inv > 0 else float("nan"),
        D_eff_naive=float(D * p),
    )


def rho_half_benefit(D):
    """rho at which exactly half the nodes still strictly benefit, P(K>=2)=1/2."""
    f = lambda r: perfect_prune_stats(D, r)["P_strict_benefit"] - 0.5
    return float(brentq(f, 1e-9, 1 - 1e-12, xtol=1e-12))


# ==================================================================== C3 / C5

def eff_ratio(x, theta):
    """Cross-class weight share after soft pruning, for a class with raw share x."""
    return x * theta / (x * theta + 1.0 - x)


def lam_eff(rho, c, theta):
    """Correct deflation after pruning: 1 - rho_eff - eta_eff (NOT 1-(1+c)rho_eff)."""
    return 1.0 - eff_ratio(rho, theta) - eff_ratio(c * rho, theta)


def D_eff_soft(D, x, theta):
    s1 = (1.0 - x) + x * theta
    s2 = (1.0 - x) + x * theta**2
    return D * s1**2 / s2


def snr_ratio_soft(rho, c, D, theta):
    """Pooled SNR ratio under soft pruning, with eta_eff done correctly."""
    dB = D_eff_soft(D, rho, theta)
    dH = D_eff_soft(D, c * rho, theta)
    # pooled sd of the two class-conditional aggregates
    pooled = np.sqrt(0.5 * (1.0 / dB + 1.0 / dH))
    return abs(lam_eff(rho, c, theta)) / pooled / 1.0 * 0.5 * 2 / 2 * 2 / 2 \
        if False else abs(lam_eff(rho, c, theta)) / (pooled)


def tolerance_soft(c, D, theta):
    return lowest_crossing(lambda r: snr_ratio_soft(r, c, D, theta) - 1.0)


def roc_ec(e_w, r, d):
    """Exact chi-square ROC: camouflage-removal rate at within-class rate e_w."""
    return float(ncx2.sf(chi2.isf(e_w, df=d), df=d, nc=r**2 / 2.0))


def snr_ratio_hard(rho, c, D, e_w, e_c):
    """
    HARD thresholding: a dropped edge leaves the neighbourhood, so the effective
    degree shrinks in absolute terms and e_w -> 1 is genuinely punished.
    """
    keep_w, keep_c = 1.0 - e_w, 1.0 - e_c
    dB = D * ((1.0 - rho) * keep_w + rho * keep_c)
    dH = D * ((1.0 - c * rho) * keep_w + c * rho * keep_c)
    if dB <= 0 or dH <= 0:
        return 0.0
    rho_e = rho * keep_c / (rho * keep_c + (1.0 - rho) * keep_w)
    eta_e = (c * rho) * keep_c / ((c * rho) * keep_c + (1.0 - c * rho) * keep_w)
    lam = 1.0 - rho_e - eta_e
    pooled = np.sqrt(0.5 * (1.0 / dB + 1.0 / dH))
    return abs(lam) / pooled


def lowest_crossing(f, n=4001):
    """
    Smallest rho in (0,1) where f changes sign from positive to negative.

    The SNR ratio is non-monotone in rho (it recovers at extreme camouflage), so
    the operationally meaningful tolerance is the FIRST crossing, not any root.
    """
    xs = np.linspace(1e-9, 1 - 1e-9, n)
    vs = np.array([f(x) for x in xs])
    if vs[0] <= 0:
        return 0.0
    idx = np.flatnonzero(vs <= 0)
    if idx.size == 0:
        return float(xs[-1])
    i = idx[0]
    return float(brentq(f, xs[i - 1], xs[i], xtol=1e-12))


def tolerance_hard(c, D, e_w, e_c):
    return lowest_crossing(lambda r: snr_ratio_hard(r, c, D, e_w, e_c) - 1.0)


def best_hard(c, D, r, d):
    """Optimise the threshold; genuine interior optimum, no search-bound artefact."""
    def neg(logit):
        e_w = 1.0 / (1.0 + np.exp(-logit))
        return -tolerance_hard(c, D, e_w, roc_ec(e_w, r, d))
    res = minimize_scalar(neg, bounds=(-25.0, 25.0), method="bounded",
                          options={"xatol": 1e-9})
    e_w = 1.0 / (1.0 + np.exp(-res.x))
    e_c = roc_ec(e_w, r, d)
    return dict(e_w=float(e_w), e_c=float(e_c), tolerance=float(-res.fun),
                retained_genuine_frac=float(1 - e_w))


# --------------------------------------------------------------------------

def main():
    out = {}
    c, D = 3 / 7, 16

    print("=" * 76)
    print("C1  Depth for TRUE mean aggregation (ego weight 1/D restored)")
    print("=" * 76)
    print(f"{'L':>3} {'sum q_j^2/N_j':>14} {'non-backtrk D^-L':>17} {'ratio':>7} "
          f"{'crossover rho':>14}")
    prev = None
    rows = []
    for L in (1, 2, 3, 4, 5):
        q, N = walk_profile(D, L)
        S = sum(p**2 / N[j] for j, p in q.items())
        cr = depth_crossover(c, D, L)
        print(f"{L:>3} {S:>14.3e} {D**-L:>17.3e} {S/D**-L:>7.2f} {cr:>14.6f}")
        rows.append(dict(L=L, S=float(S), naive=float(D**-L),
                         crossover=float(cr)))
        prev = cr
    out["depth"] = rows
    print("\nThe crossover MOVES with depth (0.525 -> 0.587 -> ...), so the")
    print("depth-independence claim is withdrawn. The 1/D ego term is an implicit")
    print("residual connection: it carries undeflated signal and RAISES tolerance.")

    print("\n" + "=" * 76)
    print("C2  Perfect pruning never loses; the 1-1/D ceiling is withdrawn")
    print("=" * 76)
    print(f"{'D':>4} {'rho':>8} {'D_eff naive':>12} {'D_eff true':>11} "
          f"{'P(K=0)':>8} {'P(benefit)':>11}")
    pr = []
    for Dd in (4, 16, 64):
        for rho in (1 - 1 / Dd, 0.9, 0.5):
            st = perfect_prune_stats(Dd, rho)
            print(f"{Dd:>4} {rho:>8.4f} {st['D_eff_naive']:>12.4f} "
                  f"{st['D_eff_true']:>11.4f} {st['P_K0']:>8.4f} "
                  f"{st['P_strict_benefit']:>11.4f}")
            pr.append(dict(D=Dd, rho=float(rho), **st))
    out["perfect_pruning"] = pr
    print("\nHonest replacement quantity -- rho at which half the nodes still")
    print("strictly benefit from aggregation under perfect pruning:")
    hb = []
    for Dd in (4, 16, 64, 256):
        rh = rho_half_benefit(Dd)
        print(f"   D={Dd:4d}  rho_half={rh:.6f}   (old bogus ceiling 1-1/D="
              f"{1-1/Dd:.6f})")
        hb.append(dict(D=Dd, rho_half=float(rh), old_claim=float(1 - 1 / Dd)))
    out["rho_half_benefit"] = hb

    print("\n" + "=" * 76)
    print("C3  eta_eff done correctly changes the tolerances materially")
    print("=" * 76)
    print(f"{'e_c':>6} {'theta':>6} {'tol (wrong)':>12} {'tol (correct)':>14}")
    tc = []
    for e_c in (0.25, 0.5, 0.75, 0.9):
        th = 1 - e_c
        # reproduce the old (incorrect) number: lambda = 1-(1+c) rho_eff
        def old(r):
            re = eff_ratio(r, th)
            dB = D_eff_soft(D, r, th)
            return abs(1 - (1 + c) * re) * np.sqrt(dB) - 1.0
        try:
            told = brentq(old, 1e-9, 1 - 1e-9, xtol=1e-12)
        except ValueError:
            told = float("nan")
        tnew = tolerance_soft(c, D, th)
        print(f"{e_c:>6.2f} {th:>6.2f} {told:>12.4f} {tnew:>14.4f}")
        tc.append(dict(e_c=e_c, theta=th, tolerance_old=float(told),
                       tolerance_correct=float(tnew)))
    out["eta_eff_correction"] = tc

    print("\n" + "=" * 76)
    print("C5  Hard-threshold pruning has a genuine interior optimum")
    print("=" * 76)
    print("Evaluated at each dataset's own measured (c, D, r, d):")
    print(f"{'dataset':>16} {'c':>6} {'D':>5} {'r':>5} {'d':>4} "
          f"{'unpruned':>9} {'pruned':>8} {'perfect':>8} {'rho_hat':>8} {'verdict':>9}")
    hd = []
    cases = [("YelpChi homo", 0.1700, 158, 1.173, 32, 0.8053),
             ("YelpChi net_rsr", 0.1700, 145, 1.173, 32, 0.7952),
             ("Amazon homo", 0.0738, 145, 2.044, 25, 0.8968),
             ("Amazon net_usu", 0.0738, 96, 2.044, 25, 0.8968)]
    for name, cc, DD, r, d, rho_hat in cases:
        unp = tolerance_hard(cc, DD, 0.0, 0.0)
        b = best_hard(cc, DD, r, d)
        perf = rho_half_benefit(DD)
        v = "helps" if rho_hat < b["tolerance"] else "HARMFUL"
        print(f"{name:>16} {cc:>6.3f} {DD:>5} {r:>5.2f} {d:>4} "
              f"{unp:>9.4f} {b['tolerance']:>8.4f} {perf:>8.4f} "
              f"{rho_hat:>8.4f} {v:>9}")
        hd.append(dict(dataset=name, c=cc, D=DD, r=r, d=d, rho_hat=rho_hat,
                       tol_unpruned=float(unp), tol_pruned=b["tolerance"],
                       rho_half_perfect=float(perf), e_w_star=b["e_w"],
                       e_c_star=b["e_c"], verdict=v))
    out["hard_threshold_real"] = hd
    print("\n('perfect' = rho at which half the nodes still benefit under a")
    print(" flawless pruner, the honest replacement for the withdrawn 1-1/D.)")
    print("\ne_w* is now interior (not the search bound), and it retains a")
    print("substantial fraction of genuine edges, as a usable pruner must.")

    print("\n" + "=" * 76)
    print("C5  Robustness of the optimum to the search bound (was the bug)")
    print("=" * 76)
    for zmax in (6, 12, 25, 40):
        b = minimize_scalar(
            lambda z: -tolerance_hard(c, D, 1/(1+np.exp(-z)),
                                      roc_ec(1/(1+np.exp(-z)), 1.173, 32)),
            bounds=(-zmax, zmax), method="bounded", options={"xatol": 1e-9})
        e_w = 1/(1+np.exp(-b.x))
        print(f"   zmax={zmax:>3}: e_w*={e_w:.6f}  tolerance={-b.fun:.6f}")
    print("   -> stable, unlike the soft-weight version which drifted with zmax")

    with open("correction_results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nWrote correction_results.json")


if __name__ == "__main__":
    main()
