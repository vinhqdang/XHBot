import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv
import numpy as np
from utils.metrics import compute_classification_metrics
from model.xhbot import XHBot
import warnings
warnings.filterwarnings('ignore')

def generate_planted_heterophily_graph(num_users=2000, bot_ratio=0.15):
    data = HeteroData()
    num_bots = int(num_users * bot_ratio)
    num_humans = num_users - num_bots
    
    # Add massive feature overlap to make the task difficult and rely on graph structure
    bot_feat = torch.randn(num_bots, 128) + 0.3
    human_feat = torch.randn(num_humans, 128) - 0.3
    # Add significant random noise
    bot_feat += torch.randn_like(bot_feat) * 1.5
    human_feat += torch.randn_like(human_feat) * 1.5
    data['user'].x = torch.cat([human_feat, bot_feat], dim=0)
    data['user'].y = torch.cat([torch.zeros(num_humans, dtype=torch.long), torch.ones(num_bots, dtype=torch.long)])
    
    bot_idx = torch.arange(num_humans, num_users)
    human_idx = torch.arange(num_humans)
    
    edges = []
    for i in range(num_humans * 5):
        src = torch.randint(0, num_humans, (1,)).item()
        dst = torch.randint(0, num_humans, (1,)).item()
        edges.append((src, dst))
        
    for i in range(num_bots * 20):
        src = torch.randint(num_humans, num_users, (1,)).item()
        dst = torch.randint(0, num_humans, (1,)).item()
        edges.append((src, dst))
        
    for i in range(num_bots * 2):
        src = torch.randint(num_humans, num_users, (1,)).item()
        dst = torch.randint(num_humans, num_users, (1,)).item()
        edges.append((src, dst))
        
    edge_index = torch.tensor(edges).t().contiguous()
    data['user', 'follow', 'user'].edge_index = edge_index
    
    indices = torch.randperm(num_users)
    train_idx = indices[:int(0.6 * num_users)]
    test_idx = indices[int(0.8 * num_users):]
    
    data['user'].train_mask = torch.zeros(num_users, dtype=torch.bool)
    data['user'].train_mask[train_idx] = True
    data['user'].test_mask = torch.zeros(num_users, dtype=torch.bool)
    data['user'].test_mask[test_idx] = True
    
    return data

class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 2)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

def main():
    torch.manual_seed(42)
    data = generate_planted_heterophily_graph()
    
    print("--- Training Baseline (Standard Homophilic GCN) ---")
    baseline = SimpleGCN(128, 64)
    opt_b = optim.Adam(baseline.parameters(), lr=0.01)
    
    edge_index = data['user', 'follow', 'user'].edge_index
    for epoch in range(200):
        baseline.train()
        opt_b.zero_grad()
        out = baseline(data['user'].x, edge_index)
        loss = nn.CrossEntropyLoss()(out[data['user'].train_mask], data['user'].y[data['user'].train_mask])
        loss.backward()
        opt_b.step()
        
    baseline.eval()
    with torch.no_grad():
        out = baseline(data['user'].x, edge_index)
        pred = out.argmax(dim=-1)
        prob = torch.softmax(out, dim=-1)[:, 1]
        mask = data['user'].test_mask
        base_res = compute_classification_metrics(data['user'].y[mask].numpy(), pred[mask].numpy(), prob[mask].numpy())
        
    print("--- Training Proposed Method (XHBot) ---")
    xhbot = XHBot(in_channels=128, hidden_channels=64, out_channels=32, num_relations=1)
    opt_x = optim.Adam(xhbot.parameters(), lr=0.01)
    
    for epoch in range(200):
        xhbot.train()
        opt_x.zero_grad()
        outputs = xhbot({'user': data['user'].x}, {('user', 'follow', 'user'): edge_index}, [('user', 'follow', 'user')])
        logits = outputs['logits']
        loss_cls = nn.CrossEntropyLoss()(logits[data['user'].train_mask], data['user'].y[data['user'].train_mask])
        l_p, l_d, l_r, l_i = xhbot.get_loss(outputs, data['user'].y, data['user'].x)
        loss = loss_cls + 0.1*(l_p + l_d + l_r + l_i)
        loss.backward()
        opt_x.step()
        
    xhbot.eval()
    with torch.no_grad():
        outputs = xhbot({'user': data['user'].x}, {('user', 'follow', 'user'): edge_index}, [('user', 'follow', 'user')])
        pred = outputs['logits'].argmax(dim=-1)
        prob = torch.softmax(outputs['logits'], dim=-1)[:, 1]
        xhbot_res = compute_classification_metrics(data['user'].y[mask].numpy(), pred[mask].numpy(), prob[mask].numpy())
        
    print("\n================ FINAL COMPARATIVE RESULTS ================")
    print(f"Method       | F1 Score | AUC-ROC | Accuracy | MCC")
    print(f"-------------|----------|---------|----------|-------")
    print(f"Standard GCN | {base_res['f1']:.4f}   | {base_res['auc']:.4f}  | {base_res['accuracy']:.4f}   | {base_res['mcc']:.4f}")
    print(f"XHBot (Ours) | {xhbot_res['f1']:.4f}   | {xhbot_res['auc']:.4f}  | {xhbot_res['accuracy']:.4f}   | {xhbot_res['mcc']:.4f}")
    print("===========================================================")
    print(f"Improvement: +{(xhbot_res['f1'] - base_res['f1'])*100:.2f}% F1, +{(xhbot_res['auc'] - base_res['auc'])*100:.2f}% AUC")

if __name__ == "__main__":
    main()
