import torch
import torch.nn as nn
import torch.nn.functional as F

class SGTR(nn.Module):
    def __init__(self, num_relations, beta=1.0, epsilon=0.1):
        super(SGTR, self).__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.num_relations = num_relations
        # Learnable threshold per relation
        self.tau = nn.Parameter(torch.ones(num_relations) * 0.5)
        
    def forward(self, x, edge_index_dict, edge_types):
        """
        x: Dict of node features
        edge_index_dict: Dict of edge_index tensors keyed by edge type tuples
        edge_types: List of edge type tuples
        """
        refined_edge_weights = {}
        
        for i, edge_type in enumerate(edge_types):
            if edge_type not in edge_index_dict:
                continue
                
            edge_index = edge_index_dict[edge_type]
            src_type, rel_type, dst_type = edge_type
            src, dst = edge_index[0], edge_index[1]
            
            # Get node features
            h_src = x[src_type][src]
            h_dst = x[dst_type][dst]
            
            # Ensure same dimension for comparison
            if h_src.shape[1] == h_dst.shape[1]:
                # L2 distance
                l2_dist = torch.norm(h_src - h_dst, p=2, dim=1)
                
                # Cosine similarity
                sim = F.cosine_similarity(h_src, h_dst, dim=1)
                
                # Spectral energy score
                s_r = l2_dist * torch.exp(-self.beta * sim)
                
                # Pruning mask: M_r(u,v) = 1 if s_r(u,v) <= tau_r else epsilon
                mask = torch.where(s_r <= self.tau[i], torch.ones_like(s_r), torch.ones_like(s_r) * self.epsilon)
                refined_edge_weights[edge_type] = mask
            else:
                # If feature dimensions don't match, keep edges
                refined_edge_weights[edge_type] = torch.ones(edge_index.size(1), device=h_src.device)
                
        return refined_edge_weights
