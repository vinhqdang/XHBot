"""Fine crossover localisation + constructive THCA expressivity check."""
import json
import numpy as np
from scipy.stats import norm
from csbm_validation import simulate, critical_rho_exact, critical_rho_simple

RNG = np.random.default_rng(7)
pi_B, delta, sigma, D = 0.30, 2.0, 1.0, 16
raw_err = norm.cdf(-delta / (2 * sigma))

# ---- fine sweep to bracket the empirical crossover -----------------------
fine = []
for rho in np.arange(0.42, 0.521, 0.01):
    e = simulate(float(rho), pi_B, delta, sigma, D, n=1_500_000, rng=RNG)["err"]
    fine.append((round(float(rho), 3), round(e, 5), bool(e < raw_err)))

cross_lo = max(r for r, e, b in fine if b)
cross_hi = min(r for r, e, b in fine if not b)

# ---- THCA: mean-aggregation cannot separate; max-pooling can -------------
# Two multisets with identical mean but different max.
A = np.array([0.0, 2.0])
B = np.array([1.0, 1.0])
thca = dict(
    neigh_A=A.tolist(), neigh_B=B.tolist(),
    mean_A=float(A.mean()), mean_B=float(B.mean()),
    max_A=float(A.max()), max_B=float(B.max()),
    mean_indistinguishable=bool(np.isclose(A.mean(), B.mean())),
    max_distinguishes=bool(not np.isclose(A.max(), B.max())),
)

# Degenerate-weight check: tri-channel reduces exactly to the homophilic
# channel when the fusion attention puts all mass on H.
h_H, h_X, h_S = np.array([1.0, -2.0]), np.array([5.0, 5.0]), np.array([9.0, 9.0])
alpha = np.array([1.0, 0.0, 0.0])
fused = alpha[0] * h_H + alpha[1] * h_X + alpha[2] * h_S
thca["reduces_to_homophilic"] = bool(np.allclose(fused, h_H))

out = dict(
    raw_err=round(float(raw_err), 5),
    rho_star_exact=round(critical_rho_exact(pi_B, delta, sigma, D), 4),
    rho_star_simple=round(critical_rho_simple(D, pi_B), 4),
    empirical_crossover_bracket=[cross_lo, cross_hi],
    fine_sweep=fine,
    thca=thca,
)
print(json.dumps(out, indent=2))
json.dump(out, open("refine_results.json", "w"), indent=2)
