"""
Corrected theory extensions, validated.

E1  Pruning as a bias-variance trade-off.
    Soft pruning multiplies within-class edge weights by b = 1 - e_w and
    camouflage weights by a = 1 - e_c. Both rho_eff and D_eff depend on (a, b)
    only through the retained-weight ratio theta = a/b, so theta is the single
    design parameter:

        rho_eff(theta) = rho theta / (rho theta + 1 - rho)
        D_eff(theta)   = D [(1-rho) + rho theta]^2 / [(1-rho) + rho theta^2]

    Consequence, and a correction to the naive account: a *perfect* pruner
    (theta = 0) does not tolerate arbitrary camouflage. It removes the bias
    entirely but leaves only (1-rho) D neighbours, so it needs (1-rho) D > 1:

        rho_max = 1 - 1/D,      not 1.

    The achievable theta is bounded below by the ROC of the edge score, which
    is governed by kappa = r^2 / (2 sqrt(2 d)); this closes the free parameter.

E2  Depth. The separation obeys Delta_L = lambda^L Delta exactly, with
    lambda = 1 - (1+c) rho, but the variance obeys a recursion that carries a
    label-mixture term:

        V_B^{l+1} = [(1-rho) V_B^l + rho V_H^l + rho(1-rho) Delta_l^2] / D
        V_H^{l+1} = [eta V_B^l + (1-eta) V_H^l + eta(1-eta) Delta_l^2] / D

    The clean form SNR_L/SNR_raw = (lambda sqrt(D))^L is the noise-dominated
    limit of this recursion, not the general case.

Run with: python extensions2.py
"""
import json

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2, ncx2

RNG = np.random.default_rng(11)


# ---------------------------------------------------------------- E1 -------

def kappa_of(r, d):
    return r**2 / (2.0 * np.sqrt(2.0 * d))


def theta_of_threshold(e_w, r, d):
    """
    Retained-weight ratio achieved by a hard threshold whose within-class
    false-positive rate is e_w, using the exact chi-square ROC.
    """
    e_c = ncx2.sf(chi2.isf(e_w, df=d), df=d, nc=r**2 / 2.0)
    return (1.0 - e_c) / (1.0 - e_w)


def theta_min(r, d):
    """Best (smallest) achievable retained-weight ratio over all thresholds."""
    res = minimize_scalar(lambda z: theta_of_threshold(1/(1+np.exp(-z)), r, d),
                          bounds=(-12.0, 12.0), method="bounded",
                          options={"xatol": 1e-8})
    return float(res.fun), float(1/(1+np.exp(-res.x)))


def rho_eff(rho, theta):
    return rho * theta / (rho * theta + 1.0 - rho)


def D_eff(D, rho, theta):
    s1 = (1.0 - rho) + rho * theta
    s2 = (1.0 - rho) + rho * theta**2
    return D * s1**2 / s2


def snr_gain(rho, c, D, theta):
    lam = 1.0 - (1.0 + c) * rho_eff(rho, theta)
    return lam * np.sqrt(D_eff(D, rho, theta))


def tolerance(c, D, theta, hi=1.0 - 1e-9):
    f = lambda r: snr_gain(r, c, D, theta) - 1.0
    if f(1e-9) <= 0:
        return 0.0
    if f(hi) > 0:
        return hi
    return float(brentq(f, 1e-9, hi, xtol=1e-12))


def verify_D_eff(D, rho, theta, n=400_000, rng=RNG):
    """Monte-Carlo check that D_eff is the variance-equivalent degree."""
    is_cross = rng.random((n, D)) < rho
    w = np.where(is_cross, theta, 1.0)
    keep = w.sum(1) > 0            # drop nodes whose every edge was removed
    noise = rng.normal(0, 1.0, (n, D))
    wm = (w * noise).sum(1)[keep] / w.sum(1)[keep]
    return 1.0 / wm.var()          # = D_eff for unit-variance noise


# ---------------------------------------------------------------- E2 -------

def variance_recursion(rho, c, D, L, delta=1.0, sigma=1.0):
    """Exact class-conditional mean/variance recursion; returns SNR at depth L."""
    eta = c * rho
    mB, mH = +delta / 2.0, -delta / 2.0
    vB = vH = sigma**2
    for _ in range(L):
        gap = mB - mH
        nB = (1 - rho) * mB + rho * mH
        nH = eta * mB + (1 - eta) * mH
        nvB = ((1 - rho) * vB + rho * vH + rho * (1 - rho) * gap**2) / D
        nvH = (eta * vB + (1 - eta) * vH + eta * (1 - eta) * gap**2) / D
        mB, mH, vB, vH = nB, nH, nvB, nvH
    sd = np.sqrt(0.5 * (vB + vH))
    return (mB - mH) / (2.0 * sd)


def simulate_depth(rho, c, D, L, delta=1.0, sigma=1.0, n=3000, rng=RNG):
    """Locally tree-like simulation: flat mean over D^L leaves."""
    eta = c * rho

    def stat(root_is_bot):
        lab = np.full((n, 1), root_is_bot, dtype=bool)
        for _ in range(L):
            p = np.where(lab, rho, eta)
            flip = rng.random((n, lab.shape[1] * D)) < np.repeat(p, D, axis=1)
            lab = np.repeat(lab, D, axis=1) ^ flip
        mu = np.where(lab, +delta / 2.0, -delta / 2.0)
        return (mu + rng.normal(0, sigma, lab.shape)).mean(axis=1)

    b, h = stat(True), stat(False)
    return float((b.mean() - h.mean()) / (2 * np.sqrt(0.5 * (b.var() + h.var()))))


# --------------------------------------------------------------------------

def main():
    out = {}
    c, D = 0.3 / 0.7, 16

    print("=" * 76)
    print("E1  D_eff is the variance-equivalent degree (MC check)")
    print("=" * 76)
    print(f"{'rho':>6} {'theta':>7} {'D_eff pred':>11} {'D_eff MC':>10} {'rel err':>9}")
    rows = []
    for rho in (0.3, 0.5):
        for th in (1.0, 0.5, 0.1, 0.0):
            pred = D_eff(D, rho, th)
            mc = verify_D_eff(D, rho, th)
            print(f"{rho:6.2f} {th:7.2f} {pred:11.4f} {mc:10.4f} "
                  f"{abs(mc-pred)/pred:9.5f}")
            rows.append(dict(rho=rho, theta=th, D_eff_pred=float(pred),
                             D_eff_mc=float(mc)))
    out["D_eff"] = rows

    print("\n" + "=" * 76)
    print("E1  A perfect pruner tolerates 1 - 1/D, NOT 1")
    print("=" * 76)
    ceil_rows = []
    for Dd in (4, 16, 64, 256):
        t = tolerance(c, Dd, 0.0)
        print(f"   D={Dd:4d}  tolerance(theta=0)={t:.6f}   "
              f"1-1/D={1-1/Dd:.6f}")
        ceil_rows.append(dict(D=Dd, tolerance=float(t), bound=float(1-1/Dd)))
    out["perfect_ceiling"] = ceil_rows

    print("\n" + "=" * 76)
    print("E1  Tolerance vs retained-weight ratio theta")
    print("=" * 76)
    print(f"{'theta':>7} {'tolerance':>10}   (D=16, unpruned rho*="
          f"{tolerance(c, D, 1.0):.4f})")
    tr = []
    for th in (1.0, 0.75, 0.5, 0.25, 0.1, 0.01, 0.0):
        t = tolerance(c, D, th)
        print(f"{th:7.2f} {t:10.4f}")
        tr.append(dict(theta=th, tolerance=float(t)))
    out["tolerance_curve"] = tr

    print("\n" + "=" * 76)
    print("E1  Achievable theta from the edge-score ROC (no free parameter)")
    print("=" * 76)
    print(f"{'r':>5} {'d':>5} {'kappa':>8} {'theta_min':>10} {'e_w*':>8} "
          f"{'tolerance':>10} {'vs rho*':>9}")
    base = tolerance(c, D, 1.0)
    ach = []
    for r, d in [(2.0, 64), (3.0, 64), (3.0, 16), (5.0, 16), (8.0, 16),
                 (12.0, 16)]:
        tmin, ew = theta_min(r, d)
        t = tolerance(c, D, tmin)
        print(f"{r:5.1f} {d:5d} {kappa_of(r,d):8.3f} {tmin:10.5f} {ew:8.4f} "
              f"{t:10.4f} {t/base:9.3f}x")
        ach.append(dict(r=r, d=d, kappa=float(kappa_of(r, d)),
                        theta_min=tmin, e_w_star=ew, tolerance=float(t)))
    out["achievable"] = ach
    out["rho_star_unpruned"] = float(base)

    print("\n" + "=" * 76)
    print("E2  Exact variance recursion vs simulation")
    print("=" * 76)
    print(f"{'rho':>6} {'L':>3} {'recursion':>10} {'simulated':>10} "
          f"{'rel err':>9} {'(lam sqrtD)^L':>14}")
    dep = []
    Dd = 9
    for rho in (0.0, 0.2, 0.35, 0.5):
        for L in (1, 2, 3):
            rec = variance_recursion(rho, c, Dd, L)
            nn = max(2000, int(4.0e6 / Dd**L))
            sim = simulate_depth(rho, c, Dd, L, n=nn)
            lam = 1.0 - (1.0 + c) * rho
            clean = (lam * np.sqrt(Dd)) ** L * 0.5
            print(f"{rho:6.2f} {L:3d} {rec:10.4f} {sim:10.4f} "
                  f"{abs(sim-rec)/max(abs(rec),1e-12):9.4f} {clean:14.4f}")
            dep.append(dict(rho=rho, L=L, recursion=float(rec),
                            simulated=float(sim), clean_form=float(clean)))
    out["depth"] = dep

    print("\n(the last column is the noise-dominated form, SNR_raw=0.5;")
    print(" it agrees at rho=0 and overestimates once mixture variance bites)")

    print("\n" + "=" * 76)
    print("E2  Noise-dominated regime (sigma=6): clean form is recovered")
    print("=" * 76)
    print(f"{'rho':>6} {'L':>3} {'recursion':>10} {'clean form':>11} {'rel err':>9}")
    nd = []
    for rho in (0.2, 0.35):
        for L in (1, 2, 3):
            rec = variance_recursion(rho, c, Dd, L, sigma=6.0)
            lam = 1.0 - (1.0 + c) * rho
            clean = (lam * np.sqrt(Dd)) ** L * (1.0 / (2 * 6.0))
            print(f"{rho:6.2f} {L:3d} {rec:10.5f} {clean:11.5f} "
                  f"{abs(rec-clean)/clean:9.5f}")
            nd.append(dict(rho=rho, L=L, sigma=6.0, recursion=float(rec),
                           clean=float(clean)))
    out["noise_dominated"] = nd

    print("\n" + "=" * 76)
    print("E2  The harmful window, and why it is bounded above")
    print("=" * 76)
    lo = (1 - D**-0.5) / (1 + c)
    hi_r = (1 + D**-0.5) / (1 + c)
    print("Aggregation helps iff |lambda| sqrt(D) > 1, which has TWO roots:")
    print(f"   rho*      = (1 - 1/sqrt(D))/(1+c) = {lo:.6f}")
    print(f"   rho_upper = (1 + 1/sqrt(D))/(1+c) = {hi_r:.6f}")
    print(f"   harmful window: ({lo:.4f}, {hi_r:.4f})")
    print("Beyond rho_upper the classes are strongly anti-correlated, so the")
    print("neighbourhood is informative again with orientation flipped.")
    out["harmful_window"] = dict(lower=float(lo), upper=float(hi_r),
                                 D=D, c=float(c))

    print("\nCrossovers of the exact recursion (sigma=6, |SNR| vs raw):")
    for L in (1, 2, 3):
        f = lambda r: abs(variance_recursion(r, c, D, L, sigma=6.0)) - 1.0/12.0
        try:
            c1 = brentq(f, 1e-6, lo + 0.12, xtol=1e-10)
        except ValueError:
            c1 = float("nan")
        try:
            c2 = brentq(f, hi_r - 0.12, min(hi_r + 0.12, 0.9999), xtol=1e-10)
        except ValueError:
            c2 = float("nan")
        print(f"   L={L}: lower={c1:.6f}  upper={c2:.6f}")
        out.setdefault("crossover_by_L", []).append(
            dict(L=L, lower=float(c1), upper=float(c2)))

    with open("extension2_results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nWrote extension2_results.json")


if __name__ == "__main__":
    main()
