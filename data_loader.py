"""
Self-contained, reproducible benchmark generator for XHBot.

This module builds *controlled synthetic heterophily benchmarks* with planted
relation camouflage. The benchmarks are fully deterministic given a seed, so every
number and figure in the manuscript can be reproduced by cloning this repository and
running ``run_experiments.py`` --- no external datasets are required.

Design (see manuscript Section 4.1):
  * Two node populations: benign humans (label 0) and bots (label 1).
  * Node features are only *weakly* separable, so node-level classifiers are
    insufficient and the graph structure carries the decisive signal.
  * Homophilic structure: humans connect within human communities; bots form a
    dense "botnet" among themselves.
  * Planted camouflage: each bot directs a controllable fraction of its edges to
    benign humans (the ``camouflage_ratio``). Raising this fraction degrades
    standard homophilic aggregators, emulating modern relation camouflage.
  * Two relations ("follow", "mention") support relation-level attention.
  * Class imbalance is configurable to emulate different dataset regimes.

The three named configurations emulate the camouflage regimes of well-known
benchmarks (TwiBot-20, TwiBot-22, Cresci-2017); they are synthetic emulations, not
the gated original corpora.
"""

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Benchmark configurations emulating distinct bot-detection regimes.
# ---------------------------------------------------------------------------
BENCHMARK_CONFIGS = {
    # Medium scale, moderate camouflage, ~29% bots.
    # camouflage_ratio scales the number of bot->human edges via base_camouflage;
    # a sparse botnet means homophilic aggregation cannot rely on bot-bot edges.
    # Each bot has a (small) homophilic botnet plus heavy camouflage to humans in
    # the SAME relations, so the botnet signal is diluted for methods that cannot
    # prune camouflage. Bot and human degrees are matched (intra_human_deg) so the
    # classes are not separable by degree. signal_gap is moderate so camouflage
    # edges (bot-human) are feature-discrepant and detectable by SGTR, while
    # botnet / within-community edges are low-discrepancy and preserved.
    "TwiBot-20": dict(num_users=1500, bot_ratio=0.29, camouflage_ratio=0.6,
                      feat_dim=64, signal_dims=8, signal_gap=0.4,
                      human_communities=6, intra_human_deg=22, botnet_deg=5,
                      base_camouflage=30, mention_deg=6),
    # Larger, harder, lower bot prevalence (~18%), heavier camouflage.
    "TwiBot-22": dict(num_users=2500, bot_ratio=0.18, camouflage_ratio=0.7,
                      feat_dim=64, signal_dims=8, signal_gap=0.38,
                      human_communities=10, intra_human_deg=22, botnet_deg=5,
                      base_camouflage=34, mention_deg=6),
    # High bot prevalence (~65%), moderate camouflage, denser bot rings.
    "Cresci-2017": dict(num_users=1200, bot_ratio=0.65, camouflage_ratio=0.5,
                        feat_dim=64, signal_dims=8, signal_gap=0.5,
                        human_communities=4, intra_human_deg=20, botnet_deg=6,
                        base_camouflage=26, mention_deg=6),
}

RELATIONS = ("follow", "mention")


class Benchmark:
    """Container holding the generated graph as dense per-relation adjacencies."""

    def __init__(self, x, y, adj, train_mask, val_mask, test_mask, meta):
        self.x = x                      # (N, F) float tensor
        self.y = y                      # (N,) long tensor
        self.adj = adj                  # dict relation -> (N, N) float tensor (0/1)
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.meta = meta                # dict of config + derived info

    @property
    def num_nodes(self):
        return self.x.shape[0]

    @property
    def num_features(self):
        return self.x.shape[1]


def _add_edges(adj, srcs, dsts):
    """Add undirected edges (self-loops excluded) into a dense adjacency tensor."""
    for s, d in zip(srcs, dsts):
        if s == d:
            continue
        adj[s, d] = 1.0
        adj[d, s] = 1.0


def make_benchmark(config_name=None, seed=0, camouflage_ratio=None, overrides=None):
    """Generate a deterministic benchmark.

    Args:
        config_name: key into ``BENCHMARK_CONFIGS`` (default "TwiBot-20").
        seed: integer seed controlling all randomness.
        camouflage_ratio: optional override of the planted camouflage fraction
            (used by the robustness sweep).
        overrides: optional dict of additional config overrides.
    Returns:
        A ``Benchmark`` instance.
    """
    cfg = dict(BENCHMARK_CONFIGS[config_name or "TwiBot-20"])
    if camouflage_ratio is not None:
        cfg["camouflage_ratio"] = float(camouflage_ratio)
    if overrides:
        cfg.update(overrides)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    N = cfg["num_users"]
    F = cfg["feat_dim"]
    num_bots = int(round(N * cfg["bot_ratio"]))
    perm = rng.permutation(N)
    bot_idx = np.sort(perm[:num_bots])
    human_idx = np.sort(perm[num_bots:])
    is_bot = np.zeros(N, dtype=bool)
    is_bot[bot_idx] = True

    # ----- node features: weakly separable -----------------------------------
    x = rng.standard_normal((N, F)).astype(np.float32)
    sd = cfg["signal_dims"]
    gap = cfg["signal_gap"]
    # Bots shifted +gap, humans -gap on the signal dimensions (weak, overlapping).
    x[is_bot, :sd] += gap
    x[~is_bot, :sd] -= gap

    # ----- assign human communities ------------------------------------------
    hc = cfg["human_communities"]
    human_comm = {int(h): int(rng.integers(hc)) for h in human_idx}
    comm_members = {c: [] for c in range(hc)}
    for h in human_idx:
        comm_members[human_comm[int(h)]].append(int(h))

    adj = {r: torch.zeros((N, N), dtype=torch.float32) for r in RELATIONS}

    # ----- FOLLOW relation ----------------------------------------------------
    follow = adj["follow"]
    # Humans connect within their community (homophily).
    for h in human_idx:
        members = comm_members[human_comm[int(h)]]
        if len(members) <= 1:
            continue
        k = min(cfg["intra_human_deg"], len(members) - 1)
        targets = rng.choice(members, size=k, replace=False)
        _add_edges(follow, [int(h)] * k, targets.tolist())
    # Bots form a dense botnet among themselves (homophily).
    if len(bot_idx) > 1:
        for b in bot_idx:
            k = min(cfg["botnet_deg"], len(bot_idx) - 1)
            targets = rng.choice(bot_idx, size=k, replace=False)
            _add_edges(follow, [int(b)] * k, targets.tolist())
    # Planted camouflage: each bot follows many *random* humans across communities.
    # The camouflage degree scales linearly with camouflage_ratio, so a higher ratio
    # makes the bot's neighbourhood human-dominated and washes out its ego signal
    # under homophilic averaging.
    cam = cfg["camouflage_ratio"]
    n_cam = int(round(cfg["base_camouflage"] * cam))
    n_cam = max(0, min(n_cam, len(human_idx)))
    for b in bot_idx:
        if n_cam == 0:
            continue
        targets = rng.choice(human_idx, size=n_cam, replace=False)
        _add_edges(follow, [int(b)] * n_cam, targets.tolist())

    # ----- MENTION relation --------------------------------------------------
    # Humans mention within community; bots keep only a sparse bot-bot signal and
    # also camouflage here (mentioning random humans), so no single relation gives
    # an un-camouflaged label signal that relational models could exploit trivially.
    mention = adj["mention"]
    md = cfg["mention_deg"]
    for h in human_idx:
        members = comm_members[human_comm[int(h)]]
        if len(members) <= 1:
            continue
        k = min(md, len(members) - 1)
        targets = rng.choice(members, size=k, replace=False)
        _add_edges(mention, [int(h)] * k, targets.tolist())
    n_cam_m = int(round(cfg["base_camouflage"] * cam * 0.5))
    n_cam_m = max(0, min(n_cam_m, len(human_idx)))
    if len(bot_idx) > 1:
        for b in bot_idx:
            k = min(cfg["botnet_deg"], len(bot_idx) - 1)
            targets = rng.choice(bot_idx, size=k, replace=False)
            _add_edges(mention, [int(b)] * k, targets.tolist())
            if n_cam_m > 0:
                cam_targets = rng.choice(human_idx, size=n_cam_m, replace=False)
                _add_edges(mention, [int(b)] * n_cam_m, cam_targets.tolist())

    # ----- stratified train/val/test split (40/20/40) ------------------------
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    for cls_idx in (human_idx, bot_idx):
        cls_idx = cls_idx.copy()
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_tr, n_va = int(0.4 * n), int(0.2 * n)
        train_mask[cls_idx[:n_tr]] = True
        val_mask[cls_idx[n_tr:n_tr + n_va]] = True
        test_mask[cls_idx[n_tr + n_va:]] = True

    y = torch.from_numpy(is_bot.astype(np.int64))
    x_t = torch.from_numpy(x)

    meta = dict(cfg)
    meta.update(dict(config_name=config_name or "TwiBot-20", seed=seed,
                     num_nodes=N, num_bots=int(num_bots),
                     bot_idx=bot_idx.tolist(),
                     camouflage_ratio=cfg["camouflage_ratio"]))

    return Benchmark(x_t, y, adj, train_mask, val_mask, test_mask, meta)


if __name__ == "__main__":
    for name in BENCHMARK_CONFIGS:
        b = make_benchmark(name, seed=0)
        e = {r: int(b.adj[r].sum().item() // 2) for r in RELATIONS}
        print(f"{name}: N={b.num_nodes}, bots={b.meta['num_bots']}, "
              f"F={b.num_features}, edges={e}, "
              f"camouflage={b.meta['camouflage_ratio']}")
