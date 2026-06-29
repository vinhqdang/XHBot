"""Spectral-Guided Topology Refinement (SGTR), plain-PyTorch dense implementation.

For each existing edge (u, v) we score a learnable estimate of its per-edge spectral
(Dirichlet) energy from the feature triple [x_u, x_v, (x_u - x_v)^2], and softly
down-weight high-energy (camouflage) edges before aggregation. See manuscript
Section 3.2.
"""

import torch
import torch.nn as nn


class SGTR(nn.Module):
    def __init__(self, in_dim, relations, tau=0.5, hidden=32):
        super().__init__()
        self.relations = relations
        self.tau = tau
        self.scorer = nn.ModuleDict({
            r: nn.Sequential(nn.Linear(3 * in_dim, hidden), nn.ReLU(),
                             nn.Linear(hidden, 1))
            for r in relations
        })
        # Initialise so that the energy score starts near 0 (no pruning), letting
        # SGTR learn to prune gradually instead of destroying ~half the edges at
        # random initialisation.
        for r in relations:
            last = self.scorer[r][-1]
            nn.init.constant_(last.bias, -4.0)
            nn.init.normal_(last.weight, std=0.01)
        self._edges = None  # cached edge indices per relation

    def _edge_index(self, adj_dict):
        if self._edges is None:
            self._edges = {}
            for r in self.relations:
                idx = adj_dict[r].nonzero(as_tuple=False)  # (E, 2): [dst, src]
                self._edges[r] = idx
        return self._edges

    def forward(self, x, adj_dict, enabled=True):
        """Return a dict of refined dense adjacency matrices and the mean energy."""
        if not enabled:
            return {r: adj_dict[r] for r in self.relations}, x.new_zeros(())
        edges = self._edge_index(adj_dict)
        refined = {}
        energies = []
        N = x.shape[0]
        for r in self.relations:
            idx = edges[r]
            if idx.numel() == 0:
                refined[r] = adj_dict[r]
                continue
            dst, src = idx[:, 0], idx[:, 1]
            xu, xv = x[dst], x[src]
            feat = torch.cat([xu, xv, (xu - xv) ** 2], dim=1)
            E = torch.sigmoid(self.scorer[r](feat)).squeeze(-1)  # (E,)
            energies.append(E)
            # soft prune: weight = (1 - E) where E > tau, else 1
            weight = torch.where(E > self.tau, 1.0 - E, torch.ones_like(E))
            Aref = torch.zeros((N, N), device=x.device, dtype=x.dtype)
            Aref[dst, src] = weight
            refined[r] = Aref
        mean_energy = torch.cat(energies).mean() if energies else x.new_zeros(())
        return refined, mean_energy
