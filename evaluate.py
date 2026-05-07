import torch
from data_loader import load_dataset
from model.xhbot import XHBot
from model.hfe import HFE

def evaluate_hfe(model, data, edge_types):
    print("Initializing HFE Module...")
    hfe = HFE(model)
    
    model.eval()
    with torch.no_grad():
        outputs = model(data.x_dict, data.edge_index_dict, edge_types)
        logits = outputs['logits']
        preds = logits.argmax(dim=-1)
        
    print("--- Level 2: Community Evidence ---")
    user_follow_edge_type = ('user', 'follow', 'user')
    if user_follow_edge_type in data.edge_index_dict:
        edge_index = data.edge_index_dict[user_follow_edge_type]
        partition, G_bot = hfe.level_2_community_evidence(edge_index, preds)
        print(f"Detected {len(set(partition.values())) if partition else 0} bot communities.")
        print(f"Bot Subgraph has {G_bot.number_of_nodes()} nodes and {G_bot.number_of_edges()} edges.")
        
        print("\n--- Level 3: Motif Analysis ---")
        motifs = hfe.level_3_motif_analysis(G_bot)
        print(f"Found {len(motifs['star_centers'])} potential star-topology centers.")
    else:
        print("No user-follow-user edges found for community analysis.")

def main():
    print("Loading dataset...")
    data = load_dataset()
    edge_types = list(data.edge_index_dict.keys())
    
    model = XHBot(
        in_channels=data['user'].x.shape[1], 
        hidden_channels=128, 
        out_channels=64, 
        num_relations=len(edge_types)
    )
    
    evaluate_hfe(model, data, edge_types)

if __name__ == "__main__":
    main()
