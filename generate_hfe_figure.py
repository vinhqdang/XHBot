"""
Render the HFE instance-level explanation subgraph (manuscript Figure) for the real
flagged account identified in the case study (results/explainability.json).

Draws the flagged bot's follow ego-network: edges to other bots (botnet, solid red)
and camouflage edges to benign humans (dashed grey), exposing the star-formation
motif. Run after run_experiments.py.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx
import numpy as np
import torch

from data_loader import make_benchmark

OUT = "manuscript/figures"
os.makedirs(OUT, exist_ok=True)


def main():
    with open("results/explainability.json") as f:
        case = json.load(f)["case_study"]
    v = case["node"]
    data = make_benchmark("TwiBot-20", seed=0)
    follow = data.adj["follow"]
    y = data.y

    nbrs = torch.where(follow[v] > 0)[0].tolist()
    # subsample neighbours for legibility while preserving the human/bot ratio
    rng = np.random.default_rng(0)
    humans = [n for n in nbrs if y[n] == 0]
    bots = [n for n in nbrs if y[n] == 1]
    humans = list(rng.choice(humans, size=min(12, len(humans)), replace=False)) if humans else []
    bots = list(rng.choice(bots, size=min(4, len(bots)), replace=False)) if bots else []

    G = nx.DiGraph()
    G.add_node(v, kind="target")
    for h in humans:
        G.add_node(int(h), kind="human"); G.add_edge(v, int(h), kind="camouflage")
    for b in bots:
        G.add_node(int(b), kind="bot"); G.add_edge(int(b), v, kind="botnet")

    pos = nx.spring_layout(G, seed=1, k=0.9)
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    nx.draw_networkx_nodes(G, pos, nodelist=[v], node_color="#d62728",
                           node_size=1300, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=bots, node_color="#ff7f0e",
                           node_size=700, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=humans, node_color="#1f77b4",
                           node_size=600, alpha=0.7, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=[(int(b), v) for b in bots],
                           edge_color="#d62728", width=3.0, ax=ax, arrowsize=15)
    nx.draw_networkx_edges(G, pos, edgelist=[(v, int(h)) for h in humans],
                           edge_color="gray", style="dashed", width=1.2, ax=ax, arrowsize=12)

    red = mlines.Line2D([], [], color="#d62728", lw=3, label="Botnet evidence (retained)")
    gray = mlines.Line2D([], [], color="gray", lw=1.2, ls="--", label="Camouflage edges to humans")
    ax.legend(handles=[red, gray], loc="upper left", fontsize=11)
    ax.set_title(
        f"HFE explanation for flagged account #{v} "
        f"(bot prob {case['bot_probability']:.2f}, "
        f"{int(case['motif']['human_neighbor_ratio']*100)}% human neighbours)",
        fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    plt.savefig(f"{OUT}/hfe_subgraph.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("HFE figure written to", OUT)


if __name__ == "__main__":
    main()
