# Research Plan: XHBot — Explainable Heterophily-Aware GNN for Social Robot (Bot) Fraud Detection

**Target venue:** EAI Endorsed Transactions on AI and Robotics (Scopus, CiteScore 7.6)

---

## 1. Problem Statement

Social bots — automated software-controlled accounts on online social platforms — are a form of social robot that commits coordinated fraud: opinion manipulation, disinformation amplification, fake-follower markets, and election interference. The term "bot" derives directly from "robot," and framing these agents as malicious automated robots operating within a social graph positions this problem squarely within the AI and Robotics research agenda.

Graph Neural Networks (GNNs) have become the dominant paradigm for bot detection because modern bots disguise themselves effectively at the individual level but betray themselves through their relational patterns in the social graph. However, two critical problems remain unsolved simultaneously:

**Problem 1 — Heterophily blindness.** Standard GNN aggregation assumes homophily (similar nodes connect together). In bot networks, bots strategically connect to many benign human users to blend in — a *heterophilic* camouflage strategy. Conventional message passing averages over these mixed neighborhoods, diluting the bot signal. Recent work (e.g., NeighborSense, HW-GNN 2025) confirms heterophily as the dominant structural property of bot networks, yet most deployed detectors still use homophily-assuming aggregation.

**Problem 2 — Explainability gap.** No existing work applies post-hoc subgraph explainability to a graph-based bot detection model evaluated on TwiBot-20 or TwiBot-22. Platform operators, moderators, and regulators need forensic evidence — *which subgraph of accounts and which relational interactions* triggered a classification — not just a binary label. Without this, automated bot detection systems remain black boxes that cannot support transparent moderation decisions.

These two problems compound: a model that is robust against heterophilic camouflage AND produces interpretable subgraph evidence for each classification decision does not yet exist.

**Research question:** Can we design a GNN-based social bot detector that (1) explicitly models the heterophilic structure of bot-human interaction graphs, (2) resists camouflage-by-dilution, and (3) produces post-hoc subgraph explanations that are faithful, concise, and actionable for platform moderation?

---

## 2. Literature Review

### 2.1 Foundations: GNN-Based Bot Detection

**BotRGCN: Twitter Bot Detection with Relational Graph Convolutional Networks**
Shangbin Feng, Herun Wan, Ningnan Wang, Minnan Luo.
*Proceedings of the 2021 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM 2021)*, pp. 236–239. ACM, 2021.
The first prominent application of Relational GCN (R-GCN) to Twitter bot detection. Constructs a heterogeneous follow graph and fuses three modalities (semantic, property, neighborhood). Establishes the community-aware aggregation paradigm but uses shallow concatenation without heterophily awareness. Key baseline for all subsequent work.

**Heterogeneity-Aware Twitter Bot Detection with Relational Graph Transformers (RGT)**
Shangbin Feng, Zhaoxuan Tan, Rui Li, Minnan Luo.
*Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 36, no. 4, pp. 3977–3985, 2022.
Replaces R-GCN's single-relation aggregation with multi-relation transformer attention plus a semantic attention network across relations. Demonstrates that modeling relation heterogeneity improves over BotRGCN on TwiBot-20. Still does not address heterophily or explainability.

**TwiBot-20: A Comprehensive Twitter Bot Detection Benchmark**
Shangbin Feng, Herun Wan, Ningnan Wang, Jundong Li, Minnan Luo.
*Proceedings of the 30th ACM International Conference on Information & Knowledge Management (CIKM 2021)*, pp. 4485–4494. ACM, 2021.
The first large-scale graph-based bot detection benchmark (229,573 users, 455,958 follow relationships, three user modalities). Reveals that prior methods do not generalize across user domains. Primary evaluation dataset for this work.

**TwiBot-22: Towards Graph-Based Twitter Bot Detection**
Shangbin Feng, Zhaoxuan Tan, Herun Wan, Ningnan Wang, Zilong Chen, Binchi Zhang, Qinghua Zheng, Wenqian Zhang, Zhenyu Lei, Shujie Yang, Xinshun Feng, Qingyue Zhang, Hongrui Wang, Yuhan Liu, Yuyang Bai, Heng Wang, Zijian Cai, Yanbo Wang, Lijing Zheng, Zihan Ma, Jundong Li, Minnan Luo.
*Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*, Datasets and Benchmarks Track.
The largest graph-based bot benchmark (~1M users, 88M tweets, multiple node and edge types: user, tweet, list, hashtag; follow, post, mention, retweet, like, contain). Re-implements 35 baselines on 9 datasets. Shows that no single method dominates across dataset domains. Secondary evaluation dataset for this work.

### 2.2 Recent Bot Detection Advances (2025–2026)

**Neighborhood Perceivable Graph Neural Network for Relational Heterogeneous Twitter Bot Detection (NeighborSense)**
Yan Li, Haoyu Lu, Wanying Chen.
*PLOS One*, vol. 21, Feb. 2026. DOI: 10.1371/journal.pone.0342686
Directly addresses heterogeneous relational aggregation through a dynamically maintained shortcut module that gates message passing based on local neighborhood feature statistics. Confirms that adaptive, locally-aware aggregation is critical for bot detection. Does not address class imbalance or provide explanations.

**CB-MTE: Social Bot Detection via Multi-Source Heterogeneous Feature Fusion**
Meng Cheng, Yuzhi Xiao, Tao Huang, Chao Lei, Chuang Zhang.
*Sensors*, vol. 25, no. 11, article 3549, June 2025. DOI: 10.3390/s25113549
Fuses DistilBERT text embeddings, graph embeddings, and manifold-learned structural features via a CatBoost-based collaborative reasoning mechanism. Validated on TwiBot-22 with cross-topic migration testing. No explainability module.

**HW-GNN: Homophily-Aware Gaussian-Window Constrained Graph Spectral Network for Social Network Bot Detection**
arXiv:2511.22493, November 2025.
Spectral approach observing that homophilic bot communities and heterophilic bot-human mixed communities exhibit distinct frequency distributions. Gaussian-window polynomial basis functions construct adaptive graph filters achieving +4.3% F1 gain. Demonstrates plugin compatibility with other spectral GNN backbones. No post-hoc explanation.

**Social Bot Detection via Heterogeneous Graph Learning and Sample Balancing Strategies**
Chenyi Fu, Kun Chen, Xiaoying Pan, Sheng Yu, Jian Ni, Yanyan Min.
*BDSC 2025 / Springer Communications in Computer and Information Science*, vol. 2622, 2026. DOI: 10.1007/978-981-95-0880-8_18
Combines multi-dimensional interaction graphs with similarity-based affinity graphs and explicit class-imbalance mitigation. One of the few 2025–2026 papers jointly addressing heterogeneity and imbalance. No explainability.

**An Integrated Pre-Trained Auto-Encoder and Graph Neural Network for General Social Bot Detection (AEG4BD)**
Pham Phuoc.
*Annals of Data Science*, 2025. DOI: 10.1007/s40745-025-00633-9
Platform-independent architecture using self-supervised auto-encoder pre-training followed by GNN fine-tuning, targeting generalization across different social platforms beyond Twitter.

**MGDIL: Multi-Granularity Summarization and Domain-Invariant Learning for Cross-Domain Social Bot Detection**
arXiv:2603.27928, March 2026.
Uses LLM-based multi-granularity summarization to unify heterogeneous platform signals, combined with domain-adversarial learning and cross-domain contrastive learning to handle distribution shifts across datasets and platforms. State-of-the-art for cross-platform generalization.

### 2.3 GNN Fraud Detection: Methodologically Transferable Work

**Enhancing Graph Neural Network-Based Fraud Detectors against Camouflaged Fraudsters (CARE-GNN)**
Yingtong Dou, Zhiwei Liu, Li Sun, Yutong Deng, Hao Peng, Philip S. Yu.
*Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM 2020)*, Virtual Event, Ireland.
Foundational camouflage-resistant fraud detection framework. Introduces label-aware similarity measure, RL-based neighbor selector, and relation-aware aggregation on YelpChi and Amazon datasets. The core idea — filtering out camouflaging neighbors before aggregation — directly transfers to bot graph heterophily mitigation.

**Efficient Relation-Aware Heterogeneous Graph Neural Network for Fraud Detection (DRHGNN)**
Enwei Li, Jia Ouyang, Shuai Xiang, Menglin Yang, Yuxiang Chen.
*World Wide Web*, vol. 28, article 55, 2025. DOI: 10.1007/s11280-025-01369-5
Relation-aware computation graph preprocessing + stochastic projection dimensionality reduction for efficient heterogeneous fraud detection. Knowledge distillation variant enables compact representations. Relevant for efficient heterogeneous aggregation design.

**Dual-Channel Heterophilic Message Passing for Graph Fraud Detection**
arXiv:2504.14205, April 2025.
Addresses the coexistence of heterophily and homophily in fraud graphs with separate message passing channels for each, controlled by a learned gating mechanism. The closest methodological precursor to our proposed approach — applied to financial fraud (YelpChi, Amazon) but not to bot/social graphs.

### 2.4 GNN Explainability

**GNNExplainer: Generating Explanations for Graph Neural Networks**
Rex Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik, Jure Leskovec.
*Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*.
The foundational instance-level GNN explainer. Formulates explanation as optimization maximizing mutual information between GNN predictions and a subgraph structure mask. Our work integrates this as the post-hoc explanation module applied, for the first time, to a heterophily-aware bot detection GNN.

---

## 3. Proposed Algorithm: XHBot

**XHBot** (eXplainable Heterophily-aware Bot detector) is a four-module framework with stacked novel contributions. Each module addresses a distinct open problem identified in the literature gap analysis.

### Preliminary — Heterogeneous Graph Construction

Following TwiBot-22 schema, construct G = (V, E, R, X) where:
- **Node types V:** users, tweets, hashtags, lists
- **Edge types R:** follow, post, mention, retweet, like, contain (6 relation types)
- **Node features X:** for user nodes — RoBERTa-base embeddings of profile description concatenated with mean-pooled tweet embeddings (768-dim), plus numerical metadata (account age, follower/following counts, verified status, posting frequency, average engagement rate) projected to 64-dim via MLP, plus categorical embeddings (location, language). For tweet/hashtag/list nodes — text embeddings only.

---

### Module 1 — Spectral-Guided Topology Refinement (SGTR)

**Motivation.** Modern bots follow large numbers of benign human accounts to blend into neighborhood aggregations — a *relation camouflage* strategy (Dou et al., CIKM 2020). Standard GNNs aggregate over all edges uniformly, so these camouflage edges pollute bot representations with human signal. SGTR removes them *before* any aggregation using a spectral edge scoring approach, inspired by HW-GNN (arXiv:2511.22493) but extended to a learned, trainable pruning mechanism.

**Method.** For each relation type r, compute the spectral energy of each edge (u, v) ∈ E_r:

```
s_r(u,v) = ||h_u^(0) - h_v^(0)||_2 · exp(−β · sim(h_u^(0), h_v^(0)))
```

High s_r(u,v) indicates a cross-class (heterophilic) connection — potentially camouflage. Apply a learned RL-based threshold τ_r per relation, trained via REINFORCE to maximize validation F1:

```
M_r(u,v) = 1  if s_r(u,v) ≤ τ_r     [retain — informative edge]
           ε  if s_r(u,v) > τ_r     [soft-prune — camouflage candidate, ε=0.1]
```

The soft mask (rather than hard removal) retains gradient flow for camouflage edges with weight ε, preventing information loss for edges that are heterophilic but genuinely meaningful (e.g., a journalist following many politicians). The refined graph G̃ = (V, Ẽ, R, X) uses masked adjacency Ã_r = M_r ⊙ A_r.

**Novel contribution.** No existing bot detection paper applies spectral edge scoring as a *pre-aggregation topology refinement* step — HW-GNN uses spectral filters as aggregation operators; CARE-GNN uses label-aware similarity but at feature level. SGTR operates at the graph structure level before any message passing, enabling cleaner neighborhood signals for all subsequent modules.

---

### Module 2 — Tri-Channel Heterophily-Aware Aggregation (THCA)

**Motivation.** Even after SGTR, the refined graph G̃ contains genuinely mixed neighborhoods — bots in bot communities (homophilic) and bots with human followers (heterophilic). A single aggregation channel cannot capture both signals. The dual-channel approach (arXiv:2504.14205) uses two channels but ignores the node's own identity, and applies to homogeneous financial fraud graphs, not multi-relational social graphs.

**Method.** For each relation r and each node v, partition N_r(v) in G̃ into three groups using label-aware cosine similarity:

```
N_r^H(v) = {u ∈ N_r(v) : sim(h_u, h_v) ≥ θ_high}    [homophilic — same-class community]
N_r^X(v) = {u ∈ N_r(v) : sim(h_u, h_v) < θ_low}     [heterophilic — cross-class exposure]
N_r^S(v) = {v}                                         [self-channel — identity preservation]
```

Three parallel aggregation channels:
```
h_v^{H,r} = AGG_H({h_u : u ∈ N_r^H(v)})    [low-pass: mean pooling within community]
h_v^{X,r} = AGG_X({h_u : u ∈ N_r^X(v)})    [high-pass: max pooling of cross-class signal]
h_v^{S,r} = W_S · h_v                        [self: linear projection]
```

Cross-channel attention learns how much each channel contributes per node:
```
α_v^{H,r}, α_v^{X,r}, α_v^{S,r} = softmax(MLP([h_v^{H,r} ∥ h_v^{X,r} ∥ h_v^{S,r}]))

h_v^r = α_v^{H,r} · h_v^{H,r} + α_v^{X,r} · h_v^{X,r} + α_v^{S,r} · h_v^{S,r}
```

Relation-level attention then aggregates across all 6 edge types (following RGT, Feng et al., AAAI 2022):
```
h_v = Σ_{r∈R} β_r · h_v^r,    β_r = softmax(a^T · tanh(W_r · h_v^r))
```

**Novel contribution.** The tri-channel design (H + X + S) is novel: existing papers use either dual-channel (no self-channel, leads to over-smoothing of own identity) or single-channel with spectral separation. The self-channel prevents bots with many human connections from being "washed out" to human representations. The cross-channel attention is per-node (not global), allowing the model to learn that some bots rely primarily on their bot-community signal while others primarily expose themselves through heterophilic connections.

---

### Module 3 — Contrastive Prototype Disentanglement (CPD)

**Motivation.** Even with good aggregation, a GNN may learn spurious correlations — e.g., classifying accounts as bots because they are in a country with many bots, not because of their actual behavioral patterns. CPD forces the model to separate *why* an account is a bot (its behavioral anomaly signature) from *where* it sits socially (its structural position), producing more robust and interpretable representations.

**Method.** The THCA output h_v is passed through two projection heads:

```
z_v^b = MLP_b(h_v)    [bot-signature code: used for classification]
z_v^s = MLP_s(h_v)    [social-behavior code: used for reconstruction regularisation]
```

**Disentanglement loss** — minimize mutual information between the two codes via MINE estimator:
```
L_disent = MI(z_v^b, z_v^s)    ← minimized
```

**Prototype contrastive loss** — maintain a learnable prototype per class c ∈ {bot, human}:
```
L_proto = -log [exp(sim(z_v^b, p_{y_v}) / τ) / Σ_c exp(sim(z_v^b, p_c) / τ)]
```

**Graph augmentation InfoNCE** — two stochastic augmented views of G̃ (random edge dropout + feature masking) generate a second representation z̃_v^b; the InfoNCE loss brings same-node representations close:
```
L_InfoNCE = -log [exp(sim(z_v^b, z̃_v^b) / τ) / Σ_{u≠v} exp(sim(z_v^b, z̃_u^b) / τ)]
```

**Social reconstruction loss** — z_v^s should reconstruct the node's own features:
```
L_recon = ||Decoder(z_v^s) - x_v||_2^2
```

**Combined training objective:**
```
L = L_focal + λ_1 · L_proto + λ_2 · L_InfoNCE + λ_3 · L_recon + λ_4 · L_disent
```

where L_focal is the imbalance-weighted focal loss for classification (γ=2, α=inverse class frequency).

**Novel contribution.** No existing bot detection paper applies prototype-contrastive disentanglement to social graphs. The disentanglement ensures that z_v^b captures *only* bot-anomalous behavior (e.g., posting patterns, account age inconsistencies), not social position, making explanations generated in Module 4 causally meaningful rather than position-correlated.

---

### Module 4 — Hierarchical Forensic Explanation (HFE)

**Motivation.** GNNExplainer (Ying et al., NeurIPS 2019) produces instance-level edge masks — a minimal subgraph for a single prediction. This is useful for individual accounts but insufficient for platform moderation, which requires understanding *coordinated bot networks*, not just individual nodes. HFE produces three scales of evidence simultaneously.

**Level 1 — Instance-level edge explanation.** Apply GNNExplainer to z_v^b (not h_v) — i.e., explain the disentangled bot-signature code rather than the raw representation. This ensures the explanation reflects genuinely bot-diagnostic patterns, not social position artifacts. Optimize edge mask M^(1) ∈ [0,1]^|Ẽ|:
```
M_v^(1) = argmax_{M} MI(ŷ_v, GNN(G̃ ⊙ M, X))
Output: top-k_1 edges forming the minimal explanatory subgraph G_s^(1)
```

**Level 2 — Community-level coordination evidence.** After classification, run Louvain community detection on the subgraph induced by all predicted bot nodes. For each detected bot community C:
- Compute community bot density d(C) = |predicted bots in C| / |C|
- Compute edge homogeneity h(C) = fraction of intra-community edges among bot nodes
- Generate community-level explanation: the densest subgraph of bot-bot edges in C

```
G_C^(2) = densest_subgraph({v ∈ C : ŷ_v = bot})
```

This surfaces *coordinated bot networks* — groups of bots acting together — which single-node explanations cannot reveal.

**Level 3 — Motif-level diagnostic patterns.** Pre-define a library of suspicious structural motifs (star topology: one bot follows many humans; triangle: three bots with shared human follower; chain: linear bot retweet chains). For each flagged account, compute motif participation scores:
```
m_v^{(k)} = count of motif k in k-hop neighborhood of v
```

Report the top-contributing motif as operator-readable evidence: "This account participates in 47 star-topology bot coordination patterns."

**Counterfactual edge flip evaluation** — for each L1 explanation, generate a counterfactual by flipping the top-k explanatory edges (removing bot-bot edges, adding random human edges) and verify the prediction flips to human. This provides formal validity evidence for the explanation.

**Novel contribution.** HFE is the first three-level forensic explanation framework for bot detection, going from individual account evidence → coordinated network evidence → structural motif patterns. No existing paper combines instance-level GNNExplainer with community detection and motif analysis in a unified explanation pipeline.

---

### Full Algorithm

```
Input:  Social graph G=(V,E,R,X), partial labels Y
Output: ŷ ∈ {bot, human}^|V|, forensic evidence {G_s^(1), G_C^(2), motifs^(3)}

=== TRAINING PHASE ===

Step 1: Heterogeneous graph construction
  - Encode user features: RoBERTa(text) ∥ MLP(metadata) → x_v ∈ ℝ^832
  - Construct multi-relational adjacency {A_r}_{r∈R}

Step 2: SGTR — Spectral-guided topology refinement
  - For each r: compute edge spectral scores s_r(u,v)
  - Learn pruning threshold τ_r via REINFORCE on validation F1
  - Produce refined masked adjacency {Ã_r}_{r∈R}

Step 3: THCA — Tri-channel aggregation (L=2 layers)
  - For each layer l, each node v, each relation r:
    a. Partition N_r(v) → {N^H, N^X, N^S} via similarity thresholds
    b. Compute h^H, h^X, h^S via mean/max/linear aggregation
    c. Cross-channel attention → h_v^r
  - Relation-level attention → h_v ∈ ℝ^256

Step 4: CPD — Contrastive prototype disentanglement
  - Project h_v → z_v^b (bot code) and z_v^s (social code)
  - Update class prototypes {p_bot, p_human} via EMA
  - Compute L = L_focal + λ₁·L_proto + λ₂·L_InfoNCE + λ₃·L_recon + λ₄·L_disent
  - Backpropagate; alternate with SGTR threshold update

=== INFERENCE PHASE ===

Step 5: Classify → ŷ_v = argmax MLP_cls(z_v^b)

Step 6: HFE — Hierarchical forensic explanation
  - L1: GNNExplainer on z_v^b → instance subgraph G_s^(1)
  - L2: Louvain on bot subgraph → community clusters G_C^(2)
  - L3: Motif counting → top diagnostic pattern per account
  - Counterfactual validation: flip top-k edges → verify prediction flip
```

---

## 4. Evaluation Datasets

### Dataset 1 — TwiBot-20
- **Name:** TwiBot-20
- **Link:** https://github.com/BunsenFeng/TwiBot-20
- **Scale:** 229,573 users (11,826 labeled), 33,488,192 tweets, 455,958 follow relationships
- **Bot ratio:** ~29% bots in labeled set
- **Graph structure:** Follow edges between user nodes; user profile + tweet text + metadata features
- **Split:** Official train/dev/test split provided
- **Why used:** The primary large-scale graph-based bot detection benchmark; used by BotRGCN, RGT, NeighborSense, MGDIL. Necessary for comparison with all major baselines.

### Dataset 2 — TwiBot-22
- **Name:** TwiBot-22
- **Link:** https://twibot22.github.io/ and https://github.com/LuoUndergradXJTU/TwiBot-22
- **Scale:** ~1,000,000 users, 88,650,948 tweets, 170,185,937 follow relationships; multiple node types (user, tweet, list, hashtag) and edge types (follow, post, mention, retweet, like, contain)
- **Bot ratio:** ~13.9% bots (139,943 bots / 860,057 humans)
- **Graph structure:** Full heterogeneous information network with 4 node types and 6 edge types
- **Split:** Official train/dev/test split provided
- **Why used:** The largest and most comprehensive graph-based bot benchmark; tests scalability and multi-relational heterogeneous modeling. Used by CB-MTE, MGDIL, and 35 re-implemented baselines.

### Dataset 3 — Cresci-2017
- **Name:** Cresci-2017 (Social Spambots)
- **Link:** https://botometer.osome.iu.edu/bot-repository/datasets.html
- **Scale:** 3,474 genuine users + 7,543 social spambots (3 families + traditional spambots)
- **Bot ratio:** ~68%
- **Graph structure:** Profile + tweet level; no native follow graph (graph constructed from retweet/mention structure)
- **Why used:** Legacy benchmark for comparison with pre-GNN baselines; widely cited for third-generation spambot detection. Allows validation of our heterophily hypothesis (spambot families show distinct structural camouflage patterns).

---

## 5. Evaluation Metrics

### 5.1 Bot Detection Performance Metrics

| Metric | Formula | Rationale |
|--------|---------|-----------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall classification correctness; reported for fair comparison with baselines |
| **Precision** | TP/(TP+FP) | Fraction of flagged accounts that are true bots; critical for operator trust |
| **Recall** | TP/(TP+FN) | Fraction of actual bots caught; critical for platform safety |
| **F1 Score** | 2·(P·R)/(P+R) | Harmonic mean; primary metric under class imbalance |
| **AUC-ROC** | Area under ROC curve | Threshold-independent discriminative ability; standard for imbalanced detection |
| **MCC** | Matthews Correlation Coefficient | Most informative single metric under severe imbalance; ranges [−1, 1] |

Primary ranking metric: **F1 Score** (consistent with BotRGCN, RGT, NeighborSense, CB-MTE papers).

### 5.2 Explanation Quality Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| **Fidelity+** | Accuracy drop when the explanation subgraph is *removed* from input | Higher = explanation is necessary |
| **Fidelity−** | Accuracy when *only* the explanation subgraph is retained | Higher = explanation is sufficient |
| **Sparsity** | 1 − (|explanation edges| / |total edges|) | Higher = more concise explanation |
| **Explanation Accuracy** | On synthetic injected bot subgraphs with ground-truth motifs: fraction correctly identified | Measures faithfulness to true causal structure |

### 5.3 Ablation Study Metrics
Report F1 and AUC for the following ablations to validate each component:
- XHBot without dual-channel (standard homophilic aggregation only)
- XHBot without gating (equal weight on both channels)
- XHBot without imbalance sampling (standard cross-entropy only)
- XHBot without relation-level attention (uniform relation weighting)

### 5.4 Computational Efficiency
- Training time per epoch (seconds)
- Inference time per node (milliseconds)
- GPU memory footprint (GB)
Reported on a standard GPU (e.g., NVIDIA A100 or V100) to demonstrate practical deployability.

---

## 6. References

[1] Shangbin Feng, Herun Wan, Ningnan Wang, Minnan Luo. "BotRGCN: Twitter Bot Detection with Relational Graph Convolutional Networks." *Proceedings of the 2021 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM 2021)*, pp. 236–239. ACM, 2021. https://doi.org/10.1145/3487351.3488336

[2] Shangbin Feng, Zhaoxuan Tan, Rui Li, Minnan Luo. "Heterogeneity-Aware Twitter Bot Detection with Relational Graph Transformers." *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 36, no. 4, pp. 3977–3985, 2022. https://doi.org/10.1609/aaai.v36i4.20314

[3] Shangbin Feng, Herun Wan, Ningnan Wang, Jundong Li, Minnan Luo. "TwiBot-20: A Comprehensive Twitter Bot Detection Benchmark." *Proceedings of the 30th ACM International Conference on Information & Knowledge Management (CIKM 2021)*, pp. 4485–4494. ACM, 2021. https://doi.org/10.1145/3459637.3482019

[4] Shangbin Feng, Zhaoxuan Tan, Herun Wan, Ningnan Wang, Zilong Chen, Binchi Zhang, Qinghua Zheng, Wenqian Zhang, Zhenyu Lei, Shujie Yang, Xinshun Feng, Qingyue Zhang, Hongrui Wang, Yuhan Liu, Yuyang Bai, Heng Wang, Zijian Cai, Yanbo Wang, Lijing Zheng, Zihan Ma, Jundong Li, Minnan Luo. "TwiBot-22: Towards Graph-Based Twitter Bot Detection." *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*, Datasets and Benchmarks Track. https://proceedings.neurips.cc/paper_files/paper/2022/hash/e4fd610b1d77699a02df07ae97de992a-Abstract-Datasets_and_Benchmarks.html

[5] Yan Li, Haoyu Lu, Wanying Chen. "Neighborhood Perceivable Graph Neural Network for Relational Heterogeneous Twitter Bot Detection." *PLOS One*, vol. 21, 2026. https://doi.org/10.1371/journal.pone.0342686

[6] Meng Cheng, Yuzhi Xiao, Tao Huang, Chao Lei, Chuang Zhang. "CB-MTE: Social Bot Detection via Multi-Source Heterogeneous Feature Fusion." *Sensors*, vol. 25, no. 11, article 3549, 2025. https://doi.org/10.3390/s25113549

[7] Yingtong Dou, Zhiwei Liu, Li Sun, Yutong Deng, Hao Peng, Philip S. Yu. "Enhancing Graph Neural Network-Based Fraud Detectors against Camouflaged Fraudsters." *Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM 2020)*. ACM, 2020. https://doi.org/10.1145/3340531.3411903

[8] Rex Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik, Jure Leskovec. "GNNExplainer: Generating Explanations for Graph Neural Networks." *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*. https://papers.nips.cc/paper/9123-gnnexplainer-generating-explanations-for-graph-neural-networks

[9] Enwei Li, Jia Ouyang, Shuai Xiang, Menglin Yang, Yuxiang Chen. "Efficient Relation-Aware Heterogeneous Graph Neural Network for Fraud Detection." *World Wide Web*, vol. 28, article 55, 2025. https://doi.org/10.1007/s11280-025-01369-5

[10] Chenyi Fu, Kun Chen, Xiaoying Pan, Sheng Yu, Jian Ni, Yanyan Min. "Social Bot Detection via Heterogeneous Graph Learning and Sample Balancing Strategies." In *Big Data and Social Computing — BDSC 2025*, Springer Communications in Computer and Information Science, vol. 2622, 2026. https://doi.org/10.1007/978-981-95-0880-8_18

[11] Stefano Cresci, Roberto Di Pietro, Marinella Petrocchi, Angelo Spognardi, Maurizio Tesconi. "The Paradigm-Shift of Social Spambots: Evidence, Theories, and Tools for the Arms Race." In *Proceedings of the 26th International Conference on World Wide Web Companion (WWW 2017 Companion)*, pp. 963–972, 2017. https://doi.org/10.1145/3041021.3055135