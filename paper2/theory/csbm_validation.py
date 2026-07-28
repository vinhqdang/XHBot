"""
Numerical validation of the camouflage-CSBM theory.

Every closed form asserted in the paper is checked here against Monte-Carlo
simulation. Run with: python csbm_validation.py
"""
import json
import numpy as np
from scipy.stats import norm

RNG = np.random.default_rng(0)


def eta_of_rho(rho, pi_B):
    """Induced bot-fraction in a human neighbourhood (edge-balance identity)."""
    return pi_B * rho / (1.0 - pi_B)


def sample_projected(n, rho_mix, delta, sigma, D, rng):
    """
    Project onto the discriminant direction and mean-aggregate over D neighbours.

    A node whose neighbourhood is a fraction `rho_mix` of the *opposite* class
    has each neighbour drawn from the opposite class w.p. rho_mix. Returns the
    aggregated projected statistic for `n` such nodes. Own-class mean is at
    +delta/2, opposite-class mean at -delta/2.
    """
    opp = rng.random((n, D)) < rho_mix
    means = np.where(opp, -delta / 2.0, +delta / 2.0)
    return (means + rng.normal(0, sigma, size=(n, D))).mean(axis=1)


def bayes_error_1d(m1, s1, m2, s2, grid=None):
    """Exact min-error threshold for two equal-prior 1-D Gaussians."""
    if grid is None:
        lo = min(m1 - 6 * s1, m2 - 6 * s2)
        hi = max(m1 + 6 * s1, m2 + 6 * s2)
        grid = np.linspace(lo, hi, 200001)
    # class 1 is the lower-mean class
    if m1 > m2:
        m1, s1, m2, s2 = m2, s2, m1, s1
    err = 0.5 * (1 - norm.cdf(grid, m1, s1)) + 0.5 * norm.cdf(grid, m2, s2)
    return err.min()


def theory(rho, pi_B, delta, sigma, D, with_error=True):
    """Closed forms asserted in the paper.

    `with_error=False` skips the (expensive) exact Bayes-error grid search, which
    the SNR-based crossover solver does not need.
    """
    eta = eta_of_rho(rho, pi_B)
    sep = (1.0 - rho - eta) * delta                       # Prop. 1 deflation
    var_B = (sigma**2 + rho * (1 - rho) * delta**2) / D   # mixture-induced var
    var_H = (sigma**2 + eta * (1 - eta) * delta**2) / D
    s_B, s_H = np.sqrt(var_B), np.sqrt(var_H)
    snr = sep / (s_B + s_H)
    res = dict(eta=eta, sep=sep, s_B=s_B, s_H=s_H, snr=snr)
    if with_error:
        res["err"] = bayes_error_1d(-sep / 2, s_H, +sep / 2, s_B)
    return res


def simulate(rho, pi_B, delta, sigma, D, n=400_000, rng=RNG):
    eta = eta_of_rho(rho, pi_B)
    # bots: fraction rho of neighbours are human (opposite class)
    tb = sample_projected(n, rho, delta, sigma, D, rng)
    # humans: fraction eta of neighbours are bots; flip sign convention
    th = -sample_projected(n, eta, delta, sigma, D, rng)
    return dict(sep=tb.mean() - th.mean(), s_B=tb.std(), s_H=th.std(),
                err=empirical_error(tb, th))


def empirical_error(tb, th):
    """Min error over thresholds, equal priors (sort + searchsorted, O(n log n))."""
    tb_s, th_s = np.sort(tb), np.sort(th)
    lo, hi = min(tb_s[0], th_s[0]), max(tb_s[-1], th_s[-1])
    grid = np.linspace(lo, hi, 4001)
    fn = np.searchsorted(tb_s, grid, side="right") / tb_s.size   # P(tb <= t)
    fp = 1.0 - np.searchsorted(th_s, grid, side="right") / th_s.size  # P(th > t)
    return float((0.5 * fn + 0.5 * fp).min())


def critical_rho_simple(D, pi_B):
    """Corollary: sigma-dominated regime, rho* = (1 - 1/sqrt(D)) / (1 + c)."""
    c = pi_B / (1.0 - pi_B)
    return (1.0 - 1.0 / np.sqrt(D)) / (1.0 + c)


def critical_rho_exact(pi_B, delta, sigma, D):
    """Crossover where aggregated SNR equals raw-feature SNR delta/(2 sigma)."""
    raw = delta / (2.0 * sigma)
    lo, hi = 0.0, min(1.0, (1 - pi_B) / pi_B * 0.999999)
    f = lambda r: theory(r, pi_B, delta, sigma, D, with_error=False)["snr"] - raw
    if f(lo) < 0:
        return 0.0
    if f(hi) > 0:
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sgtr_energy_gap(delta, sigma, d, n=200_000, rng=RNG):
    """Lemma: E||x_u - x_v||^2 is 2*d*sigma^2 within-class, +delta^2 across."""
    mu = np.zeros(d)
    mu_b = mu.copy()
    mu_b[0] = delta  # separation along first coordinate, ||Delta|| = delta
    within = rng.normal(0, sigma, (n, d)) - rng.normal(0, sigma, (n, d))
    across = (mu_b + rng.normal(0, sigma, (n, d))) - rng.normal(0, sigma, (n, d))
    return dict(within_emp=(within**2).sum(1).mean(),
                within_thy=2 * d * sigma**2,
                across_emp=(across**2).sum(1).mean(),
                across_thy=delta**2 + 2 * d * sigma**2)


def rho_eff(rho, e_c, e_w=0.0):
    """Effective camouflage ratio after SGTR re-weighting."""
    num = rho * (1 - e_c)
    return num / (num + (1 - rho) * (1 - e_w))


def rho_star_sgtr(rho_star, e_c):
    """Closed form: true-rho tolerance after SGTR."""
    return rho_star / (rho_star + (1 - e_c) * (1 - rho_star))


def main():
    out = {}
    pi_B, delta, sigma, D = 0.30, 2.0, 1.0, 16

    # --- Prop 1 / Thm 1: deflation + Bayes error -------------------------
    rows = []
    for rho in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        t, s = theory(rho, pi_B, delta, sigma, D), simulate(rho, pi_B, delta, sigma, D)
        rows.append(dict(rho=rho, eta=round(t["eta"], 4),
                         sep_thy=round(t["sep"], 4), sep_emp=round(s["sep"], 4),
                         sB_thy=round(t["s_B"], 4), sB_emp=round(s["s_B"], 4),
                         err_thy=round(t["err"], 5), err_emp=round(s["err"], 5)))
    out["deflation_and_error"] = rows

    # --- Corollary: critical camouflage ratio ----------------------------
    crit = []
    for DD in [4, 9, 16, 25, 64]:
        crit.append(dict(D=DD,
                         rho_star_simple=round(critical_rho_simple(DD, pi_B), 4),
                         rho_star_exact=round(critical_rho_exact(pi_B, delta, sigma, DD), 4),
                         rho_star_exact_lowSNR=round(
                             critical_rho_exact(pi_B, 0.2, 1.0, DD), 4)))
    out["critical_ratio"] = crit

    # --- Verify crossover empirically at D=16 ----------------------------
    raw_err = norm.cdf(-delta / (2 * sigma))
    sweep = []
    for rho in np.arange(0.0, 0.71, 0.05):
        s = simulate(float(rho), pi_B, delta, sigma, D, n=200_000)
        sweep.append(dict(rho=round(float(rho), 3), err_agg=round(s["err"], 5),
                          err_raw=round(raw_err, 5), agg_better=bool(s["err"] < raw_err)))
    out["crossover_sweep"] = dict(raw_err=round(raw_err, 5), sweep=sweep)

    # --- SGTR separability lemma -----------------------------------------
    out["sgtr_energy_gap"] = {k: round(float(v), 3)
                              for k, v in sgtr_energy_gap(2.0, 1.0, 8).items()}

    # --- SGTR raises the tolerance ---------------------------------------
    rs = critical_rho_simple(D, pi_B)
    out["sgtr_tolerance"] = [
        dict(e_c=e, rho_eff_at_0p4=round(rho_eff(0.4, e), 4),
             rho_star_base=round(rs, 4), rho_star_sgtr=round(rho_star_sgtr(rs, e), 4))
        for e in [0.0, 0.25, 0.5, 0.75, 0.9]
    ]

    # consistency check: rho_eff(rho_star_sgtr) should equal rho_star_base
    checks = [abs(rho_eff(rho_star_sgtr(rs, e), e) - rs) for e in [0.25, 0.5, 0.75, 0.9]]
    out["sgtr_roundtrip_max_abs_err"] = float(max(checks))

    print(json.dumps(out, indent=2))
    with open("csbm_results.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
