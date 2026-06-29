"""Hierarchical Forensic Explanation (HFE), plain-PyTorch + networkx.

Provides three levels of forensic evidence for a flagged account (manuscript
Section 3.5 / 4.6):
  * Level 1 (instance): a gradient-based edge attribution + soft mask over the
    target node's incident edges, yielding the causal subgraph.
  * Level 2 (community): community detection over the predicted-bot subgraph.
  * Level 3 (motif): star-formation / camouflage motif statistics.

Fidelity/Sparsity metrics for the instance level live in ``explain.py`` because
they are model-agnostic (applied to both XHBot and baselines).
"""

import torch
import networkx as nx


def reset_cache(model):
    """Clear all lazily-computed adjacency caches so a modified graph re-propagates."""
    for attr in ("_cacheA", "_An", "_mask", "_A", "_L", "_H", "_raw_norm"):
        if hasattr(model, attr):
            setattr(model, attr, None)
    if hasattr(model, "sgtr"):
        model.sgtr._edges = None


def predicted_bot_subgraph(adj_follow, preds, bot_class=1):
    """Build a networkx graph on predicted-bot nodes and their follow edges."""
    bot_nodes = torch.where(preds == bot_class)[0].tolist()
    G = nx.Graph()
    G.add_nodes_from(bot_nodes)
    idx = adj_follow.nonzero(as_tuple=False)
    for dst, src in idx.tolist():
        if preds[dst] == bot_class and preds[src] == bot_class:
            G.add_edge(src, dst)
    return G


def community_evidence(G):
    """Greedy-modularity communities (no external louvain dependency)."""
    if G.number_of_edges() == 0:
        return []
    comms = list(nx.community.greedy_modularity_communities(G))
    return [sorted(int(n) for n in c) for c in comms]


def motif_stats(adj_follow, node, y):
    """Star-formation statistics for a single flagged node on the follow graph."""
    row = adj_follow[node]
    nbrs = torch.where(row > 0)[0]
    deg = int(nbrs.numel())
    if deg == 0:
        return dict(degree=0, human_neighbor_ratio=0.0, neighbor_interlink_density=0.0)
    human_nbrs = nbrs[y[nbrs] == 0]
    human_ratio = float(human_nbrs.numel()) / deg
    # internal clustering among the followed nodes (low => star formation)
    sub = adj_follow[nbrs][:, nbrs]
    possible = deg * (deg - 1)
    interlink = float(sub.sum().item()) / possible if possible > 0 else 0.0
    return dict(degree=deg, human_neighbor_ratio=round(human_ratio, 4),
                neighbor_interlink_density=round(interlink, 6))
