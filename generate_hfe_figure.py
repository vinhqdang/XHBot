import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

os.makedirs('manuscript/figures', exist_ok=True)

def generate_hfe_visualization():
    G = nx.DiGraph()
    
    # Central Bot
    G.add_node("Target\nBot", type='bot', pos=(0, 0))
    
    # Botnet Ring (Homophily)
    G.add_node("Bot 2", type='bot', pos=(-1.5, 1))
    G.add_node("Bot 3", type='bot', pos=(-1.5, -1))
    G.add_edges_from([("Bot 2", "Target\nBot", {'weight': 0.9}), 
                      ("Bot 3", "Target\nBot", {'weight': 0.8}), 
                      ("Bot 2", "Bot 3", {'weight': 0.9})])
    
    # Camouflage Star Formation (Heterophily)
    humans = [f"Human {i}" for i in range(1, 8)]
    np.random.seed(42)
    for i, h in enumerate(humans):
        # Semi-circle on the right
        angle = np.pi/2 - (np.pi / (len(humans)-1)) * i
        r = 2.5
        pos = (r * np.cos(angle), r * np.sin(angle))
        G.add_node(h, type='human', pos=pos)
        # HFE identifies the aggressive outbound following as the motif
        G.add_edge("Target\nBot", h, weight=0.9 if i % 2 == 0 else 0.2)
    
    pos = nx.get_node_attributes(G, 'pos')
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Draw nodes
    bot_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'bot']
    human_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'human']
    
    nx.draw_networkx_nodes(G, pos, nodelist=bot_nodes, node_color='#d62728', node_size=1200, alpha=0.9, label='Malicious Bots')
    nx.draw_networkx_nodes(G, pos, nodelist=human_nodes, node_color='#1f77b4', node_size=800, alpha=0.6, label='Benign Humans')
    
    # Draw edges
    important_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('weight', 1.0) > 0.5]
    pruned_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('weight', 1.0) <= 0.5]
    
    nx.draw_networkx_edges(G, pos, edgelist=important_edges, width=3.5, edge_color='#d62728', alpha=0.8, arrowsize=20)
    nx.draw_networkx_edges(G, pos, edgelist=pruned_edges, width=1.0, edge_color='gray', style='dashed', alpha=0.5, arrowsize=15)
    
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', font_color='white')
    
    # Custom legend
    import matplotlib.lines as mlines
    red_line = mlines.Line2D([], [], color='#d62728', linewidth=3.5, label='HFE Retained Evidence (Causal)')
    gray_line = mlines.Line2D([], [], color='gray', linewidth=1, linestyle='--', label='HFE Pruned Noise')
    
    handles, labels = ax.get_legend_handles_labels()
    # Ensure node legend is combined with edge legend
    handles.extend([red_line, gray_line])
    
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.set_title('Hierarchical Forensic Explanation (HFE):\nSurfacing a "Star-Formation" Camouflage Motif', fontsize=14, pad=10, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('manuscript/figures/hfe_subgraph.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_hfe_visualization()
    print("HFE visualization generated.")
