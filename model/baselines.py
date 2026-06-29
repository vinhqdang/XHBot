"""
Baseline bot detectors, re-implemented in plain PyTorch for a fair, self-contained
comparison. Each model consumes the same node features and per-relation dense
adjacencies and outputs class logits, so all models are trained under an identical
protocol (see ``train.py``).

These are faithful but compact re-implementations of the core mechanism of each
published method, adapted to the controlled benchmark; they are not the authors'
original code. The mechanism each one is intended to capture:

  * StandardGCN   -- homophilic low-pass GCN (the naive baseline).
  * BotRGCN       -- relational GCN with per-relation weights (Feng et al., 2021).
  * RGT           -- relation graph transformer: per-relation attention + semantic
                     attention across relations (Feng et al., 2022).
  * CAREGNN       -- similarity-aware neighbour filtering (Dou et al., 2020).
  * HWGNN         -- spectral low/high-pass mixing (Liu et al., 2025).
  * NeighborSense -- node-adaptive low/high-pass gating (Li et al., 2026).
"""

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .layers import normalize_adj, combine_relations, DenseGCNLayer


class StandardGCN(nn.Module):
    def __init__(self, in_dim, hidden, num_classes=2, dropout=0.5, **kw):
        super().__init__()
        self.g1 = DenseGCNLayer(in_dim, hidden)
        self.g2 = DenseGCNLayer(hidden, hidden)
        self.cls = nn.Linear(hidden, num_classes)
        self.dropout = dropout
        self._cacheA = None

    def forward(self, x, adj_dict):
        if self._cacheA is None:
            self._cacheA = normalize_adj(combine_relations(adj_dict))
        A = self._cacheA
        h = Fn.relu(self.g1(x, A))
        h = Fn.dropout(h, self.dropout, training=self.training)
        h = Fn.relu(self.g2(h, A))
        return self.cls(h)


class BotRGCN(nn.Module):
    """Relational GCN: a separate transform per relation, summed."""

    def __init__(self, in_dim, hidden, num_classes=2, dropout=0.5, relations=None, **kw):
        super().__init__()
        self.relations = relations
        self.proj = nn.Linear(in_dim, hidden)
        self.rel1 = nn.ModuleDict({r: nn.Linear(hidden, hidden) for r in relations})
        self.rel2 = nn.ModuleDict({r: nn.Linear(hidden, hidden) for r in relations})
        self.cls = nn.Linear(hidden, num_classes)
        self.dropout = dropout
        self._An = None

    def _norm(self, adj_dict):
        if self._An is None:
            self._An = {r: normalize_adj(adj_dict[r]) for r in self.relations}
        return self._An

    def _layer(self, x, An, rel):
        out = 0
        for r in self.relations:
            out = out + An[r] @ rel[r](x)
        return out / len(self.relations)

    def forward(self, x, adj_dict):
        An = self._norm(adj_dict)
        h = Fn.relu(self.proj(x))
        h = Fn.relu(self._layer(h, An, self.rel1))
        h = Fn.dropout(h, self.dropout, training=self.training)
        h = Fn.relu(self._layer(h, An, self.rel2))
        return self.cls(h)


class RGT(nn.Module):
    """Relation graph transformer (compact): attention over neighbours within each
    relation, then semantic attention across relations."""

    def __init__(self, in_dim, hidden, num_classes=2, dropout=0.5, relations=None, **kw):
        super().__init__()
        self.relations = relations
        self.proj = nn.Linear(in_dim, hidden)
        self.q = nn.ModuleDict({r: nn.Linear(hidden, hidden) for r in relations})
        self.k = nn.ModuleDict({r: nn.Linear(hidden, hidden) for r in relations})
        self.v = nn.ModuleDict({r: nn.Linear(hidden, hidden) for r in relations})
        self.sem = nn.Linear(hidden, 1)
        self.cls = nn.Linear(hidden, num_classes)
        self.dropout = dropout
        self._mask = None
        self.scale = hidden ** 0.5

    def _masks(self, adj_dict):
        if self._mask is None:
            self._mask = {}
            for r in self.relations:
                A = adj_dict[r] + torch.eye(adj_dict[r].shape[0])
                self._mask[r] = (A > 0)
        return self._mask

    def forward(self, x, adj_dict):
        masks = self._masks(adj_dict)
        h = Fn.relu(self.proj(x))
        rel_out = []
        for r in self.relations:
            q, k, v = self.q[r](h), self.k[r](h), self.v[r](h)
            scores = (q @ k.t()) / self.scale
            scores = scores.masked_fill(~masks[r], float("-inf"))
            attn = Fn.softmax(scores, dim=1)
            attn = torch.nan_to_num(attn)
            rel_out.append(attn @ v)
        stack = torch.stack(rel_out, dim=1)              # (N, R, H)
        sem_w = Fn.softmax(self.sem(torch.tanh(stack)), dim=1)  # (N, R, 1)
        h = (stack * sem_w).sum(dim=1)
        h = Fn.dropout(h, self.dropout, training=self.training)
        return self.cls(h)


class CAREGNN(nn.Module):
    """Similarity-aware neighbour filtering: down-weight dissimilar neighbours
    (a differentiable surrogate for CARE-GNN's label-aware RL selector)."""

    def __init__(self, in_dim, hidden, num_classes=2, dropout=0.5, **kw):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.sim = nn.Linear(hidden, hidden)
        self.g = nn.Linear(hidden, hidden)
        self.cls = nn.Linear(hidden, num_classes)
        self.dropout = dropout
        self._A = None

    def forward(self, x, adj_dict):
        if self._A is None:
            self._A = combine_relations(adj_dict)
        A = self._A
        h = Fn.relu(self.proj(x))
        s = self.sim(h)
        # cosine similarity gating on existing edges
        sn = Fn.normalize(s, dim=1)
        gate = (sn @ sn.t()).clamp(min=0)               # keep similar neighbours
        Aw = A * gate
        deg = Aw.sum(1, keepdim=True).clamp(min=1e-9)
        h = Fn.relu(self.g((Aw @ h) / deg))
        h = Fn.dropout(h, self.dropout, training=self.training)
        return self.cls(h)


class HWGNN(nn.Module):
    """Spectral low/high-pass mixing with learnable, class-agnostic weights."""

    def __init__(self, in_dim, hidden, num_classes=2, dropout=0.5, **kw):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.low = nn.Linear(hidden, hidden)
        self.high = nn.Linear(hidden, hidden)
        self.mix = nn.Parameter(torch.tensor(0.5))
        self.cls = nn.Linear(hidden, num_classes)
        self.dropout = dropout
        self._L = None
        self._H = None

    def forward(self, x, adj_dict):
        if self._L is None:
            A = combine_relations(adj_dict)
            self._L = normalize_adj(A)                   # low-pass
            self._H = torch.eye(A.shape[0]) - self._L    # high-pass
        h = Fn.relu(self.proj(x))
        m = torch.sigmoid(self.mix)
        h = Fn.relu(m * self.low(self._L @ h) + (1 - m) * self.high(self._H @ h))
        h = Fn.dropout(h, self.dropout, training=self.training)
        return self.cls(h)


class NeighborSense(nn.Module):
    """Node-adaptive gating between low-pass and high-pass propagation."""

    def __init__(self, in_dim, hidden, num_classes=2, dropout=0.5, **kw):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.low = nn.Linear(hidden, hidden)
        self.high = nn.Linear(hidden, hidden)
        self.gate = nn.Linear(hidden, 1)
        self.cls = nn.Linear(hidden, num_classes)
        self.dropout = dropout
        self._L = None
        self._H = None

    def forward(self, x, adj_dict):
        if self._L is None:
            A = combine_relations(adj_dict)
            self._L = normalize_adj(A)
            self._H = torch.eye(A.shape[0]) - self._L
        h = Fn.relu(self.proj(x))
        g = torch.sigmoid(self.gate(h))                 # per-node gate
        h = Fn.relu(g * self.low(self._L @ h) + (1 - g) * self.high(self._H @ h))
        h = Fn.dropout(h, self.dropout, training=self.training)
        return self.cls(h)


# The reported baselines match the manuscript's Table 1. StandardGCN is retained
# above as a plain-GCN reference for the t-SNE visualization but is not part of the
# published-method comparison.
BASELINES = {
    "BotRGCN": BotRGCN,
    "RGT": RGT,
    "CARE-GNN": CAREGNN,
    "NeighborSense": NeighborSense,
    "HW-GNN": HWGNN,
}
