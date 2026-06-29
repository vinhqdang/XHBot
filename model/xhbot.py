"""XHBot: SGTR + THCA + CPD, plain-PyTorch implementation.

The ``ablation`` dict toggles individual components for the ablation study:
    use_sgtr, use_homophilic, use_heterophilic, use_self, use_gating, use_cpd
All default to True (the full model).
"""

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .sgtr import SGTR
from .thca import THCALayer
from .cpd import CPD
from .layers import normalize_adj, combine_relations


DEFAULT_ABLATION = dict(use_sgtr=True, use_homophilic=True, use_heterophilic=True,
                        use_self=True, use_gating=True, use_cpd=True)


class XHBot(nn.Module):
    def __init__(self, in_dim, hidden=64, code_dim=32, num_classes=2, num_layers=2,
                 relations=None, dropout=0.5, tau=0.5, ablation=None):
        super().__init__()
        ab = dict(DEFAULT_ABLATION)
        if ablation:
            ab.update(ablation)
        self.ab = ab
        self.relations = relations
        self.dropout = dropout

        self.sgtr = SGTR(in_dim, relations, tau=tau)
        self.proj = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([
            THCALayer(hidden, hidden, relations,
                      use_homophilic=ab["use_homophilic"],
                      use_heterophilic=ab["use_heterophilic"],
                      use_self=ab["use_self"],
                      use_gating=ab["use_gating"])
            for _ in range(num_layers)
        ])
        if ab["use_cpd"]:
            self.cpd = CPD(hidden, code_dim=code_dim, num_classes=num_classes)
        else:
            self.cpd = None
            self.classifier = nn.Linear(hidden, num_classes)
        self._raw_norm = None  # cached normalized combined raw adjacency

    def _raw_adj_norm(self, adj_dict):
        if self._raw_norm is None:
            self._raw_norm = normalize_adj(combine_relations(adj_dict))
        return self._raw_norm

    def forward(self, x, adj_dict):
        refined, mean_energy = self.sgtr(x, adj_dict, enabled=self.ab["use_sgtr"])
        # combine per-relation refined adjacencies, then normalize once
        A_ref = combine_relations(refined) if not self.ab["use_sgtr"] else \
            sum(refined[r] for r in self.relations)
        A_ref_norm = normalize_adj(A_ref)
        A_raw_norm = self._raw_adj_norm(adj_dict)

        h = Fn.relu(self.proj(x))
        for layer in self.layers:
            h = Fn.relu(layer(h, A_ref_norm, A_raw_norm))
            h = Fn.dropout(h, self.dropout, training=self.training)
        if self.cpd is not None:
            logits, z_b, z_s = self.cpd(h)
            return dict(logits=logits, z_b=z_b, z_s=z_s, h=h,
                        refined=refined, mean_energy=mean_energy)
        logits = self.classifier(h)
        return dict(logits=logits, z_b=None, z_s=None, h=h,
                    refined=refined, mean_energy=mean_energy)

    def aux_losses(self, out, y, train_mask):
        if self.cpd is None:
            z = out["logits"].new_zeros(())
            return z, z
        return self.cpd.losses(out["z_b"], out["z_s"], y, train_mask)
