import torch
import torch.nn as nn
from .sgtr import SGTR
from .thca import THCALayer
from .cpd import CPD

class XHBot(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, num_layers=2):
        super(XHBot, self).__init__()
        
        # Module 1: SGTR
        self.sgtr = SGTR(num_relations=num_relations)
        
        # Initial projection for user features
        self.proj = nn.Linear(in_channels, hidden_channels)
        
        # Module 2: THCA Layers
        self.thca_layers = nn.ModuleList([
            THCALayer(hidden_channels, hidden_channels) for _ in range(num_layers)
        ])
        
        # Module 3: CPD
        self.cpd = CPD(
            in_channels=hidden_channels, 
            code_dim=out_channels, 
            original_feat_dim=in_channels,
            num_classes=2
        )
        
    def forward(self, x_dict, edge_index_dict, edge_types):
        # Apply SGTR to get refined edge weights
        edge_weights_dict = self.sgtr(x_dict, edge_index_dict, edge_types)
        
        # Project user features
        if 'user' in x_dict:
            h_dict = {k: v for k, v in x_dict.items()}
            h_dict['user'] = self.proj(x_dict['user'])
        else:
            h_dict = x_dict
            
        # THCA message passing
        for layer in self.thca_layers:
            h_dict = layer(h_dict, edge_index_dict, edge_weights_dict)
            if 'user' in h_dict:
                h_dict['user'] = torch.relu(h_dict['user'])
                
        # CPD Disentanglement
        if 'user' in h_dict:
            z_b, z_b_norm, z_s, logits = self.cpd(h_dict['user'])
            return {
                'z_b': z_b,
                'z_b_norm': z_b_norm,
                'z_s': z_s,
                'logits': logits,
                'h_user': h_dict['user'],
                'edge_weights_dict': edge_weights_dict
            }
        return {}
        
    def get_loss(self, outputs, y_v, x_v_orig, outputs_aug=None):
        loss_proto, loss_disent, loss_recon, loss_infonce = self.cpd.compute_losses(
            outputs['z_b'], outputs['z_b_norm'], outputs['z_s'], outputs['logits'],
            y_v, x_v_orig, outputs_aug['z_b_norm'] if outputs_aug else None
        )
        return loss_proto, loss_disent, loss_recon, loss_infonce
