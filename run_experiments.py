"""
End-to-end, reproducible experiment runner for XHBot.

Running ``python run_experiments.py`` regenerates every quantitative result reported
in the manuscript and writes them to ``results/*.json``:

  * results/main_comparison.json   -- Table 1 (XHBot vs. baselines, 3 benchmarks)
  * results/ablation.json          -- Table 2 (component ablation on TwiBot-20)
  * results/robustness.json        -- Figure (F1 vs. camouflage degree)
  * results/explainability.json    -- Table 3 + HFE case study

All numbers are averaged over several seeds. Nothing depends on any external
dataset or hidden artifact.
"""

import json
import os
import argparse
import warnings

import numpy as np
import torch

from data_loader import make_benchmark, BENCHMARK_CONFIGS, RELATIONS
from model.baselines import BASELINES
from model.xhbot import XHBot
from train import train_model
from explain import run_explainability

warnings.filterwarnings("ignore")

RESULTS_DIR = "results"
SEEDS = [0, 1, 2]
HIDDEN = 64
EPOCHS = 200
LR = 0.005
LAMBDA_MI = 0.01
LAMBDA_NCE = 0.1
METRIC_KEYS = ["f1", "auc", "accuracy", "mcc"]


def _agg(metric_dicts):
    """Mean +/- std over seeds for each metric."""
    out = {}
    for k in METRIC_KEYS:
        vals = [m[k] for m in metric_dicts]
        out[k] = float(np.mean(vals))
        out[k + "_std"] = float(np.std(vals))
    return out


def build_model(name, num_features, ablation=None):
    if name == "XHBot":
        return XHBot(num_features, hidden=HIDDEN, relations=list(RELATIONS),
                     ablation=ablation)
    return BASELINES[name](num_features, HIDDEN, relations=list(RELATIONS))


def train_over_seeds(name, config_name, ablation=None, camouflage_ratio=None):
    metrics = []
    for seed in SEEDS:
        data = make_benchmark(config_name, seed=seed, camouflage_ratio=camouflage_ratio)
        torch.manual_seed(seed)          # seed BEFORE model construction
        model = build_model(name, data.num_features, ablation=ablation)
        is_x = (name == "XHBot")
        m = train_model(model, data, lr=LR, epochs=EPOCHS, seed=seed,
                        lambda_mi=LAMBDA_MI if is_x else 0.0,
                        lambda_nce=LAMBDA_NCE if is_x else 0.0)
        metrics.append(m)
    return _agg(metrics)


# --------------------------------------------------------------------------- #
def experiment_main_comparison():
    print("\n=== Main comparison (Table 1) ===")
    models = list(BASELINES.keys()) + ["XHBot"]
    results = {}
    for cfg in BENCHMARK_CONFIGS:
        results[cfg] = {}
        for name in models:
            agg = train_over_seeds(name, cfg)
            results[cfg][name] = agg
            print(f"  {cfg:12s} {name:14s} F1={agg['f1']:.4f}±{agg['f1_std']:.3f}"
                  f"  AUC={agg['auc']:.4f}")
    return results


def experiment_ablation():
    print("\n=== Ablation (Table 2, TwiBot-20) ===")
    variants = {
        "Full XHBot": {},
        "w/o SGTR": dict(use_sgtr=False),
        "w/o Homophilic Channel": dict(use_homophilic=False),
        "w/o Heterophilic Channel": dict(use_heterophilic=False),
        "w/o Self-Identity Channel": dict(use_self=False),
        "Homophilic-only": dict(use_heterophilic=False, use_self=False),
        "w/o CPD": dict(use_cpd=False),
        "w/o Cross-Channel Gating": dict(use_gating=False),
    }
    results = {}
    for label, ab in variants.items():
        agg = train_over_seeds("XHBot", "TwiBot-20", ablation=ab)
        results[label] = agg
        print(f"  {label:26s} F1={agg['f1']:.4f}  AUC={agg['auc']:.4f}")
    # w/o imbalance sampling: handled by the trainer flag, not an ablation dict
    imb = []
    for seed in SEEDS:
        data = make_benchmark("TwiBot-20", seed=seed)
        torch.manual_seed(seed)
        model = build_model("XHBot", data.num_features)
        m = train_model(model, data, lr=LR, epochs=EPOCHS, seed=seed,
                        lambda_mi=LAMBDA_MI, lambda_nce=LAMBDA_NCE,
                        use_imbalance=False)
        imb.append(m)
    results["w/o Imbalance Sampling"] = _agg(imb)
    print(f"  {'w/o Imbalance Sampling':26s} F1={results['w/o Imbalance Sampling']['f1']:.4f}")
    return results


def experiment_robustness():
    print("\n=== Robustness vs. camouflage degree (Figure) ===")
    ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    models = ["XHBot", "HW-GNN", "RGT", "BotRGCN"]
    results = {"camouflage_ratios": ratios, "models": {}}
    for name in models:
        f1s = []
        for r in ratios:
            agg = train_over_seeds(name, "TwiBot-20", camouflage_ratio=r)
            f1s.append(round(agg["f1"], 4))
        results["models"][name] = f1s
        print(f"  {name:14s} {f1s}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="single seed, fewer epochs (smoke test)")
    ap.add_argument("--only", default=None,
                    help="comma list: main,ablation,robustness,explain")
    args = ap.parse_args()

    global SEEDS, EPOCHS
    if args.quick:
        SEEDS = [0]
        EPOCHS = 120

    os.makedirs(RESULTS_DIR, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None

    if only is None or "main" in only:
        json.dump(experiment_main_comparison(),
                  open(f"{RESULTS_DIR}/main_comparison.json", "w"), indent=2)
    if only is None or "ablation" in only:
        json.dump(experiment_ablation(),
                  open(f"{RESULTS_DIR}/ablation.json", "w"), indent=2)
    if only is None or "robustness" in only:
        json.dump(experiment_robustness(),
                  open(f"{RESULTS_DIR}/robustness.json", "w"), indent=2)
    if only is None or "explain" in only:
        json.dump(run_explainability(SEEDS, HIDDEN, EPOCHS, LR, LAMBDA_MI, LAMBDA_NCE),
                  open(f"{RESULTS_DIR}/explainability.json", "w"), indent=2)
    print("\nAll results written to", RESULTS_DIR)


if __name__ == "__main__":
    main()
