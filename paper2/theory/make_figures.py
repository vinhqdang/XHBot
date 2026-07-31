"""Generate the theory-validation figures. Colours are the Okabe-Ito subset
validated for CVD; every series also carries a distinct marker/linestyle so the
figures survive grayscale printing."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from csbm_validation import (theory, simulate, critical_rho_exact,
                             critical_rho_simple, rho_star_sgtr)

BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
GRID = dict(color="#cccccc", lw=0.6, alpha=0.8)
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9.5, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})

pi_B, delta, sigma, D = 0.30, 2.0, 1.0, 16
OUT = "../figures"


def fig1():
    rho_t = np.linspace(0, 0.7, 200)
    sep_t = [theory(r, pi_B, delta, sigma, D)["sep"] for r in rho_t]
    err_t = [theory(r, pi_B, delta, sigma, D)["err"] for r in rho_t]
    rho_s = np.arange(0.0, 0.71, 0.1)
    sims = [simulate(float(r), pi_B, delta, sigma, D, n=300_000) for r in rho_s]
    raw = norm.cdf(-delta / (2 * sigma))
    rstar = critical_rho_exact(pi_B, delta, sigma, D)

    fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.6))

    ax[0].plot(rho_t, sep_t, color=BLUE, lw=2, label="Theory (Prop. 3)")
    ax[0].plot(rho_s, [s["sep"] for s in sims], ls="none", marker="o", ms=5,
               mfc="white", mec=VERM, mew=1.6, label="Simulation")
    ax[0].set_xlabel(r"camouflage ratio $\rho$")
    ax[0].set_ylabel(r"post-aggregation separation $\delta(\rho)$")
    ax[0].set_title("(a) Signal deflation", loc="left")
    ax[0].grid(True, **GRID); ax[0].set_axisbelow(True)
    ax[0].legend(frameon=False, loc="upper right")

    ax[1].plot(rho_t, err_t, color=BLUE, lw=2, label="Theory (Thm. 4)")
    ax[1].plot(rho_s, [s["err"] for s in sims], ls="none", marker="o", ms=5,
               mfc="white", mec=VERM, mew=1.6, label="Simulation")
    ax[1].axhline(raw, color=GREEN, lw=1.6, ls="--", label="Raw features (no graph)")
    ax[1].axvline(rstar, color="#666666", lw=1.1, ls=":")
    ax[1].annotate(rf"$\rho^\star={rstar:.3f}$", xy=(rstar, 0.42),
                   xytext=(rstar + 0.015, 0.44), fontsize=8, color="#333333")
    ax[1].set_xlabel(r"camouflage ratio $\rho$")
    ax[1].set_ylabel("balanced error")
    ax[1].set_title("(b) Error and the critical ratio", loc="left")
    ax[1].grid(True, **GRID); ax[1].set_axisbelow(True)
    ax[1].legend(frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(f"{OUT}/theory_deflation.png")
    plt.close(fig)


def fig2():
    Ds = np.arange(2, 65)
    exact = [critical_rho_exact(pi_B, delta, sigma, int(d)) for d in Ds]
    simple = [critical_rho_simple(int(d), pi_B) for d in Ds]

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(Ds, simple, color=GREEN, lw=2, ls="--",
            label=r"low-SNR form $\frac{1-D^{-1/2}}{1+c}$")
    ax.plot(Ds, exact, color=BLUE, lw=2, label=r"exact $\rho^\star(D)$")
    c_ = pi_B / (1 - pi_B)
    upper = np.minimum((1 + Ds**-0.5) / (1 + c_), 1.0)
    ax.plot(Ds, upper, color=VERM, lw=1.6, ls="-.",
            label=r"upper edge $\frac{1+D^{-1/2}}{1+c}$")
    ax.fill_between(Ds, 0, exact, color=BLUE, alpha=0.10)
    ax.fill_between(Ds, exact, upper, color=VERM, alpha=0.10)
    ax.text(40, 0.18, "network helps", fontsize=7.5, color="#33556b")
    ax.text(34, 0.62, "network harms", fontsize=7.5, color="#7a4a2a")
    ax.text(28, 0.94, "helps again (sign reversed)", fontsize=6.5, color="#555555")
    ax.set_xlabel("neighbourhood size $D$")
    ax.set_ylabel(r"critical camouflage ratio $\rho^\star$")
    ax.set_ylim(0, 1.0)
    ax.grid(True, **GRID); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/theory_critical_ratio.png")
    plt.close(fig)


def fig3():
    e = np.linspace(0, 0.95, 200)
    base = critical_rho_simple(D, pi_B)
    boosted = [rho_star_sgtr(base, float(x)) for x in e]

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.axhline(base, color=GREEN, lw=1.8, ls="--",
               label=rf"no SGTR ($\rho^\star={base:.3f}$)")
    ax.plot(e, boosted, color=BLUE, lw=2, label=r"with SGTR (Thm. 3)")
    ax.fill_between(e, base, boosted, color=BLUE, alpha=0.10)
    for x in [0.25, 0.5, 0.75, 0.9]:
        ax.plot([x], [rho_star_sgtr(base, x)], marker="o", ms=5, mfc="white",
                mec=VERM, mew=1.6, ls="none")
    ax.set_xlabel(r"mean attenuation of camouflage edges $\bar{e}_c$")
    ax.set_ylabel(r"tolerated camouflage $\rho^\star_{\mathrm{SGTR}}$")
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, **GRID); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/theory_sgtr_tolerance.png")
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    fig1(); fig2(); fig3()
    print("figures written to", os.path.abspath(OUT))
