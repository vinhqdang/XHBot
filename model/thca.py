import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

class THCALayer(nn.Module):
    def __init__(self, in_channels, out_channels, theta_high=0.7, theta_low=0.3):
        super(THCALayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.theta_high = theta_high
        self.theta_low = theta_low
        
        # Self channel projection
        self.W_S = nn.Linear(in_channels, out_channels)
        
        # Homophilic and Heterophilic channel projections
        self.W_H = nn.Linear(in_channels, out_channels)
        self.W_X = nn.Linear(in_channels, out_channels)
        
        # Cross-channel attention
        self.attn_mlp = nn.Sequential(
            nn.Linear(out_channels * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        # Relation-level attention
        self.W_r = nn.Linear(out_channels, out_channels)
        self.a_r = nn.Parameter(torch.randn(out_channels))
        
    def forward(self, x_dict, edge_index_dict, edge_weights_dict=None):
        if 'user' not in x_dict:
            return x_dict
            
        user_x = x_dict['user']
        num_users = user_x.size(0)
        
        h_S = self.W_S(user_x) # (num_users, out_channels)
        
        relation_outputs = []
        
        # Process each relation involving users
        for edge_type in edge_index_dict.keys():
            if edge_type[0] != 'user' or edge_type[2] != 'user':
                continue
                
            edge_index = edge_index_dict[edge_type]
            edge_weights = edge_weights_dict[edge_type] if edge_weights_dict else torch.ones(edge_index.size(1), device=user_x.device)
            
            src, dst = edge_index[0], edge_index[1]
            
            # Compute similarities for thresholding
            sim = F.cosine_similarity(user_x[src], user_x[dst], dim=1)
            
            # Masks for homophilic and heterophilic edges
            homo_mask = sim >= self.theta_high
            hetero_mask = sim < self.theta_low
            
            # Aggregation: Homophilic (Mean pooling)
            if homo_mask.sum() > 0:
                h_H_src = self.W_H(user_x[src[homo_mask]]) * edge_weights[homo_mask].unsqueeze(-1)
                h_H = scatter(h_H_src, dst[homo_mask], dim=0, dim_size=num_users, reduce='mean')
            else:
                h_H = torch.zeros(num_users, self.out_channels, device=user_x.device)
                
            # Aggregation: Heterophilic (Max pooling)
            if hetero_mask.sum() > 0:
                h_X_src = self.W_X(user_x[src[hetero_mask]]) * edge_weights[hetero_mask].unsqueeze(-1)
                h_X = scatter(h_X_src, dst[hetero_mask], dim=0, dim_size=num_users, reduce='max')
            else:
                h_X = torch.zeros(num_users, self.out_channels, device=user_x.device)
                
            # Cross-channel attention
            concat_channels = torch.cat([h_H, h_X, h_S], dim=-1)
            alpha = F.softmax(self.attn_mlp(concat_channels), dim=-1) # (N, 3)
            
            h_v_r = alpha[:, 0:1] * h_H + alpha[:, 1:2] * h_X + alpha[:, 2:3] * h_S
            relation_outputs.append(h_v_r)
            
        if not relation_outputs:
            out_dict = {k: v for k, v in x_dict.items()}
            out_dict['user'] = h_S
            return out_dict
            
        # Relation-level attention
        relation_tensor = torch.stack(relation_outputs, dim=1) # (N, num_relations, out_channels)
        
        transformed = torch.tanh(self.W_r(relation_tensor))
        attn_weights = torch.matmul(transformed, self.a_r)
        beta_r = F.softmax(attn_weights, dim=1)
        
        h_user_final = torch.sum(relation_tensor * beta_r.unsqueeze(-1), dim=1)
        
        out_dict = {k: v for k, v in x_dict.items()}
        out_dict['user'] = h_user_final
        return out_dict
