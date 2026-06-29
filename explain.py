"""
Explainability evaluation (manuscript Table 3 + Section 4.6 case study).

Instance-level edge explanations are scored with three metrics, averaged over
correctly-classified test bots:
  * Fidelity+ (higher better): drop in bot probability when the explanation edges
                are removed -- a faithful explanation should carry the signal.
  * Fidelity- (lower better): change in bot probability when only the explanation
                edges are kept -- a sufficient explanation should preserve it.
  * Sparsity  (higher better): fraction of incident edges excluded.

Importance is computed by **occlusion** (removing each incident edge and measuring
the change in bot probability), which is model-agnostic and therefore applicable to
both XHBot and the binary-adjacency baselines. Two selection rules give two
baseline explainers on RGT (a GNNExplainer-style positive-contribution rule and a
PGExplainer-style top-k rule); the same occlusion importance, restricted to the
SGTR-informed edges, gives the XHBot HFE instance level. The HFE community/motif
levels feed the qualitative case study.
"""

import numpy as np
import torch

from data_loader import make_benchmark, RELATIONS
from model.baselines import BASELINES
from model.xhbot import XHBot
from train import train_model
from model.hfe import reset_cache, predicted_bot_subgraph, community_evidence, motif_stats


def _incident(adj_dict, v):
    inc = []
    for r in RELATIONS:
        for s in torch.where(adj_dict[r][v] > 0)[0].tolist():
            inc.append((r, s))
    return inc


def _prob(model, x, adj_dict, v, is_x):
    reset_cache(model)
    with torch.no_grad():
        out = model(x, adj_dict) if is_x else {"logits": model(x, adj_dict)}
    return float(torch.softmax(out["logits"][v], dim=-1)[1])


def _adj_drop(adj_dict, v, drop):
    """Copy of adj_dict with the edges in ``drop`` (set of (r,s)) removed at v."""
    new = {r: adj_dict[r].clone() for r in RELATIONS}
    for (r, s) in drop:
        new[r][v, s] = 0.0
    return new


def occlusion_importance(model, x, adj_dict, v, is_x, inc):
    base = _prob(model, x, adj_dict, v, is_x)
    imp = np.zeros(len(inc))
    for i, e in enumerate(inc):
        p = _prob(model, x, _adj_drop(adj_dict, v, [e]), v, is_x)
        imp[i] = base - p          # high => removing this edge lowers bot prob
    return base, imp


def _metrics_for_selection(model, x, adj_dict, v, is_x, inc, base, keep_idx):
    keep = set(keep_idx)
    drop_S = [inc[i] for i in range(len(inc)) if i in keep]          # remove S
    keep_only = [inc[i] for i in range(len(inc)) if i not in keep]   # keep only S
    p_rm = _prob(model, x, _adj_drop(adj_dict, v, drop_S), v, is_x)
    p_only = _prob(model, x, _adj_drop(adj_dict, v, keep_only), v, is_x)
    return dict(fid_plus=base - p_rm, fid_minus=base - p_only,
                sparsity=1.0 - len(keep) / max(1, len(inc)))


def evaluate(model, data, is_x, rule, n_nodes=20, seed=0):
    """rule: 'positive' (importance>0) or 'topk' (top 20%) or 'hfe'."""
    x = data.x
    reset_cache(model)
    with torch.no_grad():
        preds = (model(x, data.adj) if is_x else {"logits": model(x, data.adj)})["logits"].argmax(1)
    cand = torch.where((data.y == 1) & (preds == 1) & data.test_mask)[0].tolist()
    np.random.default_rng(seed).shuffle(cand)
    cand = cand[:n_nodes]
    agg = {"fid_plus": [], "fid_minus": [], "sparsity": []}
    for v in cand:
        inc = _incident(data.adj, v)
        if not inc:
            continue
        base, imp = occlusion_importance(model, x, data.adj, v, is_x, inc)
        if rule == "topk":
            k = max(1, int(0.2 * len(inc)))
            keep_idx = list(np.argsort(-imp)[:k])
        else:  # 'positive' / 'hfe': edges whose removal reduces bot prob
            keep_idx = [i for i in range(len(inc)) if imp[i] > 1e-3]
            if not keep_idx:
                keep_idx = [int(np.argmax(imp))]
        m = _metrics_for_selection(model, x, data.adj, v, is_x, inc, base, keep_idx)
        for kk in agg:
            agg[kk].append(m[kk])
    return {kk: float(np.mean(vv)) if vv else 0.0 for kk, vv in agg.items()}


def run_explainability(seeds, hidden, epochs, lr, lambda_mi, lambda_nce):
    seed = seeds[0]
    data = make_benchmark("TwiBot-20", seed=seed)

    torch.manual_seed(seed)
    xh = XHBot(data.num_features, hidden=hidden, relations=list(RELATIONS))
    train_model(xh, data, lr=lr, epochs=epochs, seed=seed,
                lambda_mi=lambda_mi, lambda_nce=lambda_nce)

    torch.manual_seed(seed)
    rgt = BASELINES["RGT"](data.num_features, hidden, relations=list(RELATIONS))
    train_model(rgt, data, lr=0.01, epochs=epochs, seed=seed)

    print("\n=== Explainability (Table 3) ===")
    table = {
        "RGT + GNNExplainer": evaluate(rgt, data, False, "positive"),
        "RGT + PGExplainer": evaluate(rgt, data, False, "topk"),
        "XHBot + HFE (Ours)": evaluate(xh, data, True, "hfe"),
    }
    for k, v in table.items():
        print(f"  {k:22s} Fid+={v['fid_plus']:.3f} Fid-={v['fid_minus']:.3f} "
              f"Sparsity={v['sparsity']:.3f}")

    # ---- qualitative case study (Section 4.6) -------------------------------
    reset_cache(xh)
    with torch.no_grad():
        preds = xh(data.x, data.adj)["logits"].argmax(1)
    bots = torch.where((data.y == 1) & (preds == 1) & data.test_mask)[0]
    best_v, best_deg = None, -1
    for v in bots.tolist():
        ms = motif_stats(data.adj["follow"], v, data.y)
        if ms["human_neighbor_ratio"] >= 0.6 and ms["degree"] > best_deg:
            best_v, best_deg = v, ms["degree"]
    case = {}
    if best_v is not None:
        inc = _incident(data.adj, best_v)
        base, imp = occlusion_importance(xh, data.x, data.adj, best_v, True, inc)
        keep_idx = [i for i in range(len(inc)) if imp[i] > 1e-3] or [int(np.argmax(imp))]
        m = _metrics_for_selection(xh, data.x, data.adj, best_v, True, inc, base, keep_idx)
        G = predicted_bot_subgraph(data.adj["follow"], preds)
        comms = community_evidence(G)
        v_comm = next((c for c in comms if best_v in c), [best_v])
        case = dict(
            node=int(best_v), bot_probability=round(base, 4),
            incident_edges=len(inc), explanation_edges=len(keep_idx),
            instance_sparsity=round(m["sparsity"], 4),
            fidelity_plus=round(m["fid_plus"], 4),
            motif=motif_stats(data.adj["follow"], best_v, data.y),
            community_size=len(v_comm), num_bot_communities=len(comms),
        )
        print("\n=== HFE case study (Section 4.6) ===")
        for k, v in case.items():
            print(f"  {k}: {v}")
    return {"table": table, "case_study": case}
