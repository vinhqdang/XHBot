import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import load_dataset
from model.xhbot import XHBot
from utils.metrics import compute_classification_metrics

def train_epoch_full(model, data, optimizer, edge_types, lambda_weights, device):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(data.x_dict, data.edge_index_dict, edge_types)
    logits = outputs['logits']
    
    train_mask = data['user'].train_mask
    labels = data['user'].y
    
    criterion = nn.CrossEntropyLoss()
    loss_cls = criterion(logits[train_mask], labels[train_mask])
    
    loss_proto, loss_disent, loss_recon, loss_infonce = model.get_loss(
        outputs, labels, data['user'].x
    )
    
    l1, l2, l3, l4 = lambda_weights
    loss = loss_cls + l1 * loss_proto + l2 * loss_infonce + l3 * loss_recon + l4 * loss_disent
    
    loss.backward()
    optimizer.step()
    return loss.item(), loss_cls.item()

def evaluate_full(model, data, edge_types, mask_name, device):
    model.eval()
    with torch.no_grad():
        outputs = model(data.x_dict, data.edge_index_dict, edge_types)
        logits = outputs['logits']
        
        mask = getattr(data['user'], mask_name)
        probs = torch.softmax(logits[mask], dim=-1)[:, 1]
        preds = logits[mask].argmax(dim=-1)
        
        y_true = data['user'].y[mask].cpu().numpy()
        y_pred = preds.cpu().numpy()
        y_prob = probs.cpu().numpy()
        
        return compute_classification_metrics(y_true, y_pred, y_prob)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading dataset...")
    data = load_dataset()
    data = data.to(device)
    
    edge_types = list(data.edge_index_dict.keys())
    
    in_channels = data['user'].x.shape[1]
    hidden_channels = 128
    out_channels = 64
    num_relations = len(edge_types)
    
    model = XHBot(in_channels, hidden_channels, out_channels, num_relations).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    lambda_weights = (0.1, 0.1, 0.1, 0.1) # proto, infonce, recon, disent
    
    print("Starting full-batch training (NeighborLoader requires pyg-lib which is unavailable on this Py3.13 env)...")
    for epoch in range(1, 6):
        loss, loss_cls = train_epoch_full(model, data, optimizer, edge_types, lambda_weights, device)
        val_metrics = evaluate_full(model, data, edge_types, 'val_mask', device)
        print(f"Epoch {epoch:03d} | Total Loss: {loss:.4f} | Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['auc']:.4f}")
            
    # Final Test
    print("Evaluating on test set...")
    test_metrics = evaluate_full(model, data, edge_types, 'test_mask', device)
    print("\nFinal Test Results:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
