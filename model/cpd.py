"""Contrastive Prototype Disentanglement (CPD), plain-PyTorch implementation.

Projects the aggregated representation into a behavioural code z_b and a
social-positioning code z_s, penalises their cross-correlation (an upper bound on
mutual information), and aligns z_b with learnable class prototypes via a supervised
InfoNCE loss. Classification uses z_b only. See manuscript Section 3.4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as Fn


class CPD(nn.Module):
    def __init__(self, in_dim, code_dim=32, num_classes=2, tau_nce=0.1):
        super().__init__()
        self.tau_nce = tau_nce
        self.mlp_b = nn.Sequential(nn.Linear(in_dim, code_dim), nn.ReLU(),
                                   nn.Linear(code_dim, code_dim))
        self.mlp_s = nn.Sequential(nn.Linear(in_dim, code_dim), nn.ReLU(),
                                   nn.Linear(code_dim, code_dim))
        self.prototypes = nn.Parameter(torch.randn(num_classes, code_dim))
        self.classifier = nn.Linear(code_dim, num_classes)

    def forward(self, h):
        z_b = self.mlp_b(h)
        z_s = self.mlp_s(h)
        logits = self.classifier(z_b)
        return logits, z_b, z_s

    def losses(self, z_b, z_s, y, train_mask):
        # disentanglement: squared Frobenius norm of cross-covariance
        zb = z_b - z_b.mean(0)
        zs = z_s - z_s.mean(0)
        cov = (zb.t() @ zs) / (z_b.shape[0] - 1)
        loss_mi = (cov ** 2).sum()
        # supervised InfoNCE to class prototypes
        zbn = Fn.normalize(z_b[train_mask], dim=1)
        pn = Fn.normalize(self.prototypes, dim=1)
        sim = (zbn @ pn.t()) / self.tau_nce
        loss_nce = Fn.cross_entropy(sim, y[train_mask])
        return loss_mi, loss_nce
