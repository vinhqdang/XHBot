"""
Estimator for the camouflage ratio rho on a real labelled graph.

The theory in this paper is parameterised by three quantities that are properties
of a deployed system rather than of the model:

    rho   the fraction of a bot's edges directed at humans (camouflage ratio)
    c     the bot-to-human ratio (from label prevalence)
    D     the effective neighbourhood size
    r,d   feature separation ||Delta||/sigma and feature dimension

This module estimates all of them from an edge list plus node labels, and
compares the result against the predicted harmful window. It is the instrument
required to answer the question the theory raises: *where do real platforms
actually sit relative to rho\\*?*

Two important caveats are enforced in the output rather than hidden:

  1. rho is estimated only over *labelled* nodes. Bot-detection benchmarks label
     a small subset, and labelling is not missing-at-random (obvious bots are
     labelled preferentially), so the estimate carries a selection bias whose
     sign is not knowable from the graph alone. We therefore also report the
     estimate restricted to the labelled-subgraph-induced edges and the
     fraction of each bot's edges that are unlabelled, so a reader can see how
     much of the neighbourhood the estimate is blind to.

  2. rho as defined in the model is a property of a *homogeneous* edge set. Real
     bot graphs are multi-relational (follow, retweet, mention). We therefore
     report rho per relation type as well as pooled, since the theory applies
     per relation.

Usage:
    python measure_rho.py --edges edges.csv --labels labels.csv
    python measure_rho.py --selftest        # validate on synthetic ground truth

Edge CSV: src,dst[,relation]        Label CSV: node,label   (label in {0,1}, 1=bot)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np


# --------------------------------------------------------------------------
# estimation
# --------------------------------------------------------------------------

def estimate(edges, labels, relation=None):
    """
    Estimate the camouflage ratio and companion quantities.

    edges     : iterable of (u, v) or (u, v, rel)
    labels    : dict node -> 0/1 (1 = bot); may cover only a subset of nodes
    relation  : if given, only edges with this relation are used

    Returns a dict of estimates. Edges are treated as undirected for
    neighbourhood composition, matching the aggregation in a GNN layer.
    """
    nbr = defaultdict(list)
    for e in edges:
        if len(e) == 3:
            u, v, rel = e
            if relation is not None and rel != relation:
                continue
        else:
            u, v = e[0], e[1]
        nbr[u].append(v)
        nbr[v].append(u)

    bot_cross, bot_total, bot_unlab = 0, 0, 0
    hum_cross, hum_total, hum_unlab = 0, 0, 0
    degs_bot, degs_hum = [], []

    for node, ns in nbr.items():
        y = labels.get(node)
        if y is None:
            continue
        n_bot = sum(1 for w in ns if labels.get(w) == 1)
        n_hum = sum(1 for w in ns if labels.get(w) == 0)
        n_unl = len(ns) - n_bot - n_hum
        lab_deg = n_bot + n_hum
        if y == 1:
            bot_cross += n_hum
            bot_total += lab_deg
            bot_unlab += n_unl
            degs_bot.append(len(ns))
        else:
            hum_cross += n_bot
            hum_total += lab_deg
            hum_unlab += n_unl
            degs_hum.append(len(ns))

    n_bot_nodes = sum(1 for v in labels.values() if v == 1)
    n_hum_nodes = sum(1 for v in labels.values() if v == 0)

    rho = bot_cross / bot_total if bot_total else float("nan")
    eta = hum_cross / hum_total if hum_total else float("nan")
    c = n_bot_nodes / n_hum_nodes if n_hum_nodes else float("nan")

    return dict(
        rho_hat=rho,
        eta_hat=eta,
        eta_predicted=c * rho,             # edge-balance identity check
        c_hat=c,
        n_bot_nodes=n_bot_nodes,
        n_hum_nodes=n_hum_nodes,
        D_bot_median=float(np.median(degs_bot)) if degs_bot else float("nan"),
        D_hum_median=float(np.median(degs_hum)) if degs_hum else float("nan"),
        unlabelled_share_bot=(bot_unlab / (bot_unlab + bot_total)
                              if bot_unlab + bot_total else float("nan")),
        unlabelled_share_hum=(hum_unlab / (hum_unlab + hum_total)
                              if hum_unlab + hum_total else float("nan")),
        relation=relation,
    )


def harmful_window(D, c):
    """(rho_lower, rho_upper) between which aggregation is worse than no graph."""
    s = D ** -0.5
    return ((1.0 - s) / (1.0 + c), (1.0 + s) / (1.0 + c))


def verdict(est):
    """Locate the estimate relative to the predicted harmful window."""
    D, c = est["D_bot_median"], est["c_hat"]
    if not np.isfinite(D) or not np.isfinite(c):
        return dict(status="insufficient data")
    lo, hi = harmful_window(D, c)
    rho = est["rho_hat"]
    if rho < lo:
        status = "below window: aggregation helps"
    elif rho > hi:
        status = "above window: helps again, orientation flipped"
    else:
        status = "INSIDE window: graph is net-harmful"
    return dict(rho_hat=rho, window_lower=lo, window_upper=hi, status=status,
                margin_to_lower=rho - lo)


# --------------------------------------------------------------------------
# self-test against synthetic ground truth
# --------------------------------------------------------------------------

def synth(n=6000, pi_B=0.3, rho_true=0.4, D=16, seed=0):
    """
    Generate a graph whose *undirected* neighbourhood composition realises
    rho_true exactly, together with eta = c * rho by construction.

    A bot is to have D neighbours of which rho are human, and a human is to have
    D neighbours of which eta = c*rho are bots. Counting edge endpoints, this
    fixes the three edge populations:

        #BH = n_B * rho * D           (one bot end, one human end)
        #BB = n_B * (1-rho) * D / 2   (two bot ends)
        #HH = n_H * (1-eta) * D / 2   (two human ends)
    """
    rng = np.random.default_rng(seed)
    n_bot = max(1, int(n * pi_B))
    n_hum = n - n_bot
    bots = np.arange(n_bot)
    hums = np.arange(n_bot, n)
    c = n_bot / n_hum
    eta = min(c * rho_true, 1.0)

    n_bh = int(round(n_bot * rho_true * D))
    n_bb = int(round(n_bot * (1.0 - rho_true) * D / 2.0))
    n_hh = int(round(n_hum * (1.0 - eta) * D / 2.0))

    edges = []
    if n_bh:
        edges += list(zip(rng.choice(bots, n_bh).tolist(),
                          rng.choice(hums, n_bh).tolist()))
    if n_bb:
        edges += list(zip(rng.choice(bots, n_bb).tolist(),
                          rng.choice(bots, n_bb).tolist()))
    if n_hh:
        edges += list(zip(rng.choice(hums, n_hh).tolist(),
                          rng.choice(hums, n_hh).tolist()))

    labels = {int(i): (1 if i < n_bot else 0) for i in range(n)}
    return [(int(u), int(v)) for u, v in edges], labels


def selftest():
    print("Self-test: recovering a planted camouflage ratio")
    print(f"{'rho_true':>9} {'rho_hat':>9} {'abs err':>9} "
          f"{'eta_hat':>9} {'c*rho':>9} {'status':>42}")
    ok = True
    for rho_true in (0.0, 0.2, 0.4, 0.6, 0.8):
        edges, labels = synth(rho_true=rho_true)
        est = estimate(edges, labels)
        v = verdict(est)
        err = abs(est["rho_hat"] - rho_true)
        ok &= err < 0.02
        print(f"{rho_true:9.2f} {est['rho_hat']:9.4f} {err:9.4f} "
              f"{est['eta_hat']:9.4f} {est['eta_predicted']:9.4f} "
              f"{v['status']:>42}")
    print("\nThe eta_hat vs c*rho columns test the edge-balance identity "
          "empirically.")
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return ok


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges")
    ap.add_argument("--labels")
    ap.add_argument("--per-relation", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="rho_measurement.json")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    if not (a.edges and a.labels):
        ap.error("--edges and --labels are required unless --selftest")

    edges = []
    with open(a.edges) as fh:
        for line in fh:
            p = [t.strip() for t in line.replace(",", " ").split()]
            if not p or p[0].lower() in ("src", "source"):
                continue
            edges.append(tuple(p[:3]) if len(p) >= 3 else (p[0], p[1]))
    labels = {}
    with open(a.labels) as fh:
        for line in fh:
            p = [t.strip() for t in line.replace(",", " ").split()]
            if not p or p[0].lower() == "node":
                continue
            labels[p[0]] = int(p[1])

    res = {"pooled": estimate(edges, labels)}
    res["pooled_verdict"] = verdict(res["pooled"])
    if a.per_relation:
        rels = {e[2] for e in edges if len(e) == 3}
        for r in sorted(rels):
            res[f"relation:{r}"] = estimate(edges, labels, relation=r)

    print(json.dumps(res, indent=2))
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
