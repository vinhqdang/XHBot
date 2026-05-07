import torch
import networkx as nx
import community as community_louvain # python-louvain
from torch_geometric.utils import to_networkx

class HFE:
    def __init__(self, model, explainer=None):
        self.model = model
        self.explainer = explainer
        
    def level_1_instance_explanation(self, x_dict, edge_index_dict, target_node, target_label):
        """Runs GNNExplainer on a specific user node."""
        if self.explainer is None:
            return None
        explanation = self.explainer(
            x_dict, edge_index_dict, 
            index=target_node, 
            target=target_label
        )
        return explanation
        
    def level_2_community_evidence(self, edge_index, predicted_labels, bot_class=1):
        """Runs Louvain community detection on the subgraph induced by bots."""
        bot_mask = predicted_labels == bot_class
        bot_indices = torch.where(bot_mask)[0].tolist()
        
        src, dst = edge_index
        bot_edges_mask = bot_mask[src] & bot_mask[dst]
        bot_edge_index = edge_index[:, bot_edges_mask]
        
        G = nx.Graph()
        G.add_nodes_from(bot_indices)
        edges = [(src.item(), dst.item()) for src, dst in zip(bot_edge_index[0], bot_edge_index[1])]
        G.add_edges_from(edges)
        
        if len(G.nodes) > 0:
            partition = community_louvain.best_partition(G)
            return partition, G
        return {}, G
        
    def level_3_motif_analysis(self, G_subgraph):
        """Motif counting on the community subgraph."""
        degrees = dict(G_subgraph.degree())
        star_centers = [n for n, d in degrees.items() if d > 5]
        return {
            'star_centers': star_centers,
            'num_edges': G_subgraph.number_of_edges(),
            'num_nodes': G_subgraph.number_of_nodes()
        }
