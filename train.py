"""Unified training / evaluation protocol shared by all models.

Every model (baselines and XHBot) is trained identically: same splits, same
optimiser, same imbalance handling, same number of epochs, with model selection on
validation F1. This guarantees a fair comparison (manuscript Section 4.1.4).
"""

import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.metrics import f1_score

from model.xhbot import XHBot
from utils.metrics import compute_classification_metrics


def _class_weights(y, train_mask, use_imbalance):
    if not use_imbalance:
        return None
    yt = y[train_mask]
    counts = torch.bincount(yt, minlength=2).float()
    w = counts.sum() / (2.0 * counts.clamp(min=1))
    return w


def train_model(model, data, lr=0.01, weight_decay=1e-4, epochs=200,
                lambda_mi=0.1, lambda_nce=0.5, use_imbalance=True,
                patience=40, verbose=False, seed=0):
    torch.manual_seed(seed)
    x, y, adj = data.x, data.y, data.adj
    tr, va, te = data.train_mask, data.val_mask, data.test_mask
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    cw = _class_weights(y, tr, use_imbalance)

    is_xhbot = isinstance(model, XHBot)
    best_val, best_state, best_wait = -1.0, None, 0

    def _best_threshold(prob, labels):
        """Pick the decision threshold maximizing F1 on the given split."""
        order = torch.argsort(prob)
        cand = torch.unique(prob[order])
        if cand.numel() > 50:                      # subsample candidate thresholds
            cand = cand[torch.linspace(0, cand.numel() - 1, 50).long()]
        best_t, best_f = 0.5, -1.0
        lab = labels.cpu()
        for t in cand.tolist():
            f = f1_score(lab, (prob >= t).long().cpu(), zero_division=0)
            if f > best_f:
                best_f, best_t = f, t
        return best_t

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x, adj) if is_xhbot else {"logits": model(x, adj)}
        logits = out["logits"]
        loss = Fn.cross_entropy(logits[tr], y[tr], weight=cw)
        if is_xhbot:
            loss_mi, loss_nce = model.aux_losses(out, y, tr)
            loss = loss + lambda_mi * loss_mi + lambda_nce * loss_nce
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        opt.step()

        # validation model selection
        model.eval()
        with torch.no_grad():
            out = model(x, adj) if is_xhbot else {"logits": model(x, adj)}
            pred = out["logits"].argmax(1)
            vf1 = f1_score(y[va].cpu(), pred[va].cpu(), zero_division=0)
        if vf1 > best_val:
            best_val, best_state, best_wait = vf1, \
                {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            best_wait += 1
            if best_wait >= patience:
                break
        if verbose and ep % 25 == 0:
            print(f"  epoch {ep:3d}  loss {loss.item():.4f}  val F1 {vf1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # final test metrics, using a validation-selected decision threshold (applied
    # identically to every model for a fair comparison)
    model.eval()
    with torch.no_grad():
        out = model(x, adj) if is_xhbot else {"logits": model(x, adj)}
        prob = torch.softmax(out["logits"], dim=1)[:, 1]
        thr = _best_threshold(prob[va], y[va])
        pred = (prob >= thr).long()
        m = compute_classification_metrics(
            y[te].cpu().numpy(), pred[te].cpu().numpy(), prob[te].cpu().numpy())
    m["threshold"] = float(thr)
    return m
