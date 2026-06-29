"""Tri-Channel Heterophily-Aware Aggregation (THCA), plain-PyTorch implementation.

Three channels operate on the combined multi-relation graph:
  * Homophilic  -- low-pass smoothing over the SGTR-refined adjacency.
  * Heterophilic-- high-pass filtering over the *unrefined* adjacency: each node's
                   deviation from its neighbourhood mean, (I - hat{A}) h, which
                   isolates the anomalous, high-frequency signal that camouflage
                   produces (cf. the Dirichlet-energy view in Section 3.2).
  * Self        -- residual identity pathway.
They are fused by node-specific attention. Channels and gating can be individually
disabled for the ablation study. All operators are deterministic matmuls, so results
are fully reproducible. See manuscript Section 3.3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as Fn


class THCALayer(nn.Module):
    def __init__(self, in_dim, out_dim, relations=None,
                 use_homophilic=True, use_heterophilic=True, use_self=True,
                 use_gating=True):
        super().__init__()
        self.out_dim = out_dim
        self.use_homophilic = use_homophilic
        self.use_heterophilic = use_heterophilic
        self.use_self = use_self
        self.use_gating = use_gating

        self.W_H = nn.Linear(in_dim, out_dim)
        self.W_X = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
        self.W_S = nn.Linear(in_dim, out_dim)
        self.att = nn.Linear(out_dim, out_dim, bias=False)
        self.q = nn.Parameter(torch.randn(out_dim))

    def forward(self, h, A_ref_norm, A_raw_norm):
        """A_ref_norm: normalized SGTR-refined adjacency (low-pass).
        A_raw_norm: normalized unrefined adjacency (for the high-pass channel)."""
        channels = []
        if self.use_homophilic:
            channels.append(A_ref_norm @ self.W_H(h))
        if self.use_heterophilic:
            high_pass = h - A_raw_norm @ h          # (I - hat{A}) h
            channels.append(self.W_X(high_pass))
        if self.use_self:
            channels.append(self.W_S(h))

        stack = torch.stack(channels, dim=1)                  # (N, C, out)
        if self.use_gating and stack.shape[1] > 1:
            scores = (torch.tanh(self.att(stack)) * self.q).sum(-1)   # (N, C)
            alpha = Fn.softmax(scores, dim=1).unsqueeze(-1)
            return (alpha * stack).sum(dim=1)
        return stack.mean(dim=1)
