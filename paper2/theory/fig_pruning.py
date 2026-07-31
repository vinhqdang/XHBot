"""Regenerate the pruning-tolerance figure using the CORRECTED theory.

The previous version plotted rho_star_sgtr, the composition-only form that
ignores the effective-degree cost and mis-derives eta_eff. Its curve rose to 1
as pruning improved, which Proposition (perfect pruner) shows is wrong.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from corrections import tolerance_soft, perfect_prune_stats, rho_half_benefit

BLUE, VERM, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#666666"
D, c = 16, 3/7
base = tolerance_soft(c, D, 1.0)

th = np.linspace(1.0, 0.0, 120)
tol = [tolerance_soft(c, D, float(t)) for t in th]
ec = 1.0 - th                                    # attenuation, for the x-axis
half = rho_half_benefit(D)

fig, ax = plt.subplots(figsize=(3.6, 2.7))
ax.axhline(base, color=GREEN, lw=1.6, ls="--", label=rf"no pruning ($\rho^\star={base:.3f}$)")
ax.plot(ec, tol, color=BLUE, lw=2, label="corrected")
ax.fill_between(ec, base, tol, color=BLUE, alpha=0.10)
# the withdrawn curve, shown for contrast
old = [base/(base + (1-x)*(1-base)) for x in ec]
ax.plot(ec, old, color=VERM, lw=1.4, ls=":", label="composition only (withdrawn)")
ax.axhline(half, color=GREY, lw=1.2, ls="-.",
           label=rf"half of strategic actors benefit ($\rho={half:.3f}$)")
ax.set_xlabel(r"attenuation of camouflage ties $\bar{e}_c$")
ax.set_ylabel(r"tolerated camouflage")
ax.set_ylim(0.45, 1.02); ax.set_xlim(0, 1)
ax.grid(True, ls=":", lw=0.6, color="#cccccc"); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left", fontsize=6.5)
fig.tight_layout(); fig.savefig("../figures/theory_pruning_tolerance.png", dpi=200,
                                bbox_inches="tight")
print(f"base={base:.4f}  perfect(theta=0)={tolerance_soft(c,D,0.0):.4f}  half={half:.4f}")
print("wrote figures/theory_pruning_tolerance.png")
