"""
Generate manuscript figures from the JSON results produced by run_experiments.py.

Run ``python run_experiments.py`` first, then ``python generate_q1_figures.py``.
All numbers come from results/*.json -- nothing is hardcoded.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = "results"
OUT = "manuscript/figures"
os.makedirs(OUT, exist_ok=True)

COLORS = {
    "StandardGCN": "#7f7f7f", "BotRGCN": "#1f77b4", "RGT": "#2ca02c",
    "CARE-GNN": "#9467bd", "NeighborSense": "#8c564b", "HW-GNN": "#ff7f0e",
    "XHBot": "#d62728", "XHBot (Ours)": "#d62728",
}


def _load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def performance_comparison():
    res = _load("main_comparison.json")
    datasets = list(res.keys())
    methods = list(next(iter(res.values())).keys())
    x = np.arange(len(datasets))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    for i, m in enumerate(methods):
        vals = [res[d][m]["f1"] for d in datasets]
        errs = [res[d][m].get("f1_std", 0) for d in datasets]
        ax.bar(x + i * width, vals, width, yerr=errs, capsize=2,
               label=("XHBot (Ours)" if m == "XHBot" else m),
               color=COLORS.get(m, None))
    ax.set_ylabel("F1 Score")
    ax.set_title("Comparative Performance (F1) Across Benchmarks")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.set_ylim([0.3, 1.05])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=4, fontsize=9)
    fig.tight_layout()
    plt.savefig(f"{OUT}/performance_comparison.png", bbox_inches="tight", dpi=300)
    plt.close()


def ablation_figure():
    res = _load("ablation.json")
    labels = list(res.keys())
    f1 = [res[k]["f1"] for k in labels]
    auc = [res[k]["auc"] for k in labels]
    x = np.arange(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    ax.bar(x - width / 2, f1, width, label="F1", color="#9467bd")
    ax.bar(x + width / 2, auc, width, label="AUC-ROC", color="#8c564b")
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study of XHBot Components (TwiBot-20)")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" ", "\n", 1) for l in labels], rotation=20,
                       ha="right", fontsize=8)
    ax.set_ylim([min(f1) - 0.05, 1.02])
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.savefig(f"{OUT}/ablation_study.png", bbox_inches="tight", dpi=300)
    plt.close()


def robustness_figure():
    res = _load("robustness.json")
    ratios = res["camouflage_ratios"]
    markers = {"XHBot": "o", "HW-GNN": "^", "RGT": "s", "BotRGCN": "D"}
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for name, f1s in res["models"].items():
        style = "-" if name == "XHBot" else "--"
        lw = 2.6 if name == "XHBot" else 1.6
        ax.plot(ratios, f1s, marker=markers.get(name, "o"), linestyle=style,
                linewidth=lw, markersize=8, color=COLORS.get(name),
                label=("XHBot (Ours)" if name == "XHBot" else name))
        ax.annotate(f"{f1s[-1]:.2f}", xy=(ratios[-1], f1s[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color=COLORS.get(name), fontweight="bold")
    ax.set_xlabel("Camouflage degree (fraction of bot edges to benign humans)", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Robustness Against Increasing Bot Camouflage (TwiBot-20)", fontsize=13)
    ax.set_xticks(ratios)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(title="Detector", loc="lower left", fontsize=11, title_fontsize=11)
    fig.tight_layout()
    plt.savefig(f"{OUT}/parameter_sensitivity.png", bbox_inches="tight", dpi=300)
    plt.close()


if __name__ == "__main__":
    performance_comparison()
    ablation_figure()
    robustness_figure()
    print("Figures written to", OUT)
