"""Shared dense-graph layers and utilities (plain PyTorch, no torch_geometric)."""

import torch
import torch.nn as nn
import torch.nn.functional as Fn


def normalize_adj(A, add_self_loops=True):
    """Symmetric normalization: D^{-1/2} (A + I) D^{-1/2} for a dense adjacency."""
    if add_self_loops:
        A = A + torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    deg = A.sum(dim=1)
    dinv = torch.pow(deg.clamp(min=1e-12), -0.5)
    Dinv = torch.diag(dinv)
    return Dinv @ A @ Dinv


def combine_relations(adj_dict):
    """Sum the per-relation adjacencies into a single homogeneous adjacency."""
    mats = list(adj_dict.values())
    out = mats[0].clone()
    for m in mats[1:]:
        out = out + m
    return (out > 0).float()


class DenseGCNLayer(nn.Module):
    """A single graph convolution over a (pre-normalized) dense adjacency."""

    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x, A_norm):
        return A_norm @ self.lin(x)
