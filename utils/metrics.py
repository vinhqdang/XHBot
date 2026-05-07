import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

def compute_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Compute binary classification metrics for bot detection.
    y_true: Ground truth labels (0 for human, 1 for bot)
    y_pred: Predicted labels
    y_prob: Predicted probabilities for the positive class (bot)
    """
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc'] = 0.5 # If only one class is present in batch
            
    return metrics

def compute_explanation_metrics(original_acc, acc_without_subgraph, acc_only_subgraph, num_edges_subgraph, total_edges):
    """
    Compute explanation quality metrics.
    """
    metrics = {}
    metrics['fidelity_plus'] = original_acc - acc_without_subgraph
    metrics['fidelity_minus'] = acc_only_subgraph
    
    if total_edges > 0:
        metrics['sparsity'] = 1.0 - (num_edges_subgraph / total_edges)
    else:
        metrics['sparsity'] = 0.0
        
    return metrics
