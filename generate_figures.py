"""
t-SNE visualization of learned embeddings (manuscript Figure: latent space).

Trains RGT and XHBot on the TwiBot-20 benchmark and projects their test-node
representations with t-SNE. Embeddings are real model outputs, not synthetic blobs.
Run after installing requirements; standalone (does not need results/).
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from data_loader import make_benchmark, RELATIONS
from model.baselines import BASELINES
from model.xhbot import XHBot
from train import train_model
from model.hfe import reset_cache

warnings.filterwarnings("ignore")
OUT = "manuscript/figures"
os.makedirs(OUT, exist_ok=True)


def main():
    data = make_benchmark("TwiBot-20", seed=0)

    torch.manual_seed(0)
    rgt = BASELINES["RGT"](data.num_features, 64, relations=list(RELATIONS))
    train_model(rgt, data, lr=0.01, epochs=200, seed=0)

    torch.manual_seed(0)
    xh = XHBot(data.num_features, hidden=64, relations=list(RELATIONS))
    train_model(xh, data, lr=0.005, epochs=200, seed=0, lambda_mi=0.01, lambda_nce=0.1)

    # embeddings: XHBot aggregated representation; RGT logits as a 2-class proxy
    reset_cache(xh)
    with torch.no_grad():
        h_x = xh(data.x, data.adj)["h"].cpu().numpy()
    captured = {}
    hook = rgt.cls.register_forward_hook(
        lambda m, inp, out: captured.__setitem__("h", inp[0].detach()))
    reset_cache(rgt)
    with torch.no_grad():
        rgt(data.x, data.adj)
    hook.remove()
    h_r = captured["h"].cpu().numpy()

    te = data.test_mask.cpu().numpy()
    y = data.y.cpu().numpy()[te]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    for ax, h, title in [(ax1, h_r[te], "Latent Space: RGT (baseline)"),
                         (ax2, h_x[te], "Latent Space: XHBot (Ours)")]:
        emb = TSNE(n_components=2, random_state=0, init="pca",
                   perplexity=30).fit_transform(h)
        ax.scatter(emb[y == 0, 0], emb[y == 0, 1], s=18, alpha=0.6,
                   color="#1f77b4", label="Humans")
        ax.scatter(emb[y == 1, 0], emb[y == 1, 1], s=18, alpha=0.6,
                   color="#d62728", label="Bots")
        ax.set_title(title)
        ax.legend()
        ax.axis("off")
    fig.tight_layout()
    plt.savefig(f"{OUT}/tsne_visualization.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("t-SNE figure written to", OUT)


if __name__ == "__main__":
    main()
