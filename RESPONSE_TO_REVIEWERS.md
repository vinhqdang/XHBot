# Response to Reviewers

**Manuscript:** XHBot: eXplainable Heterophily-aware Graph Neural Networks for Social Bot Detection
**Journal:** EAI Endorsed Transactions on AI and Robotics
**Decision:** Revisions Required

We thank the Associate Editor and both reviewers for their careful and constructive
reviews. We have revised the manuscript thoroughly to address every point. Below we
respond to each comment individually (reviewer text in *italics*), describe the
specific change we made, and point to where it appears in the revised manuscript.
All changes compile cleanly; the LaTeX source and the compiled PDF accompany this
response.

## Summary of the main changes

1. Removed the duplicated Conclusion section and the inconsistent "+27.37%" figure;
   the manuscript now reports a single, consistent set of improvements (9.64% on
   TwiBot-20, 13.15% on TwiBot-22).
2. Rewrote the abstract as a single flowing paragraph without structured labels.
3. Added a spectral-theory justification for SGTR (graph Dirichlet energy) and an
   explicit comparison with CARE-GNN and RABot.
4. Rebuilt the ablation study to ablate each of the three THCA channels
   independently and to include `w/o CPD` and `w/o SGTR` variants; corrected the
   erroneous "Dual-Channel" terminology.
5. Added a baseline-fairness statement (identical splits, identical imbalance
   sampling for all models, re-tuned hyperparameters).
6. Added a qualitative forensic case study for the HFE module.
7. Replaced the cluttered architecture PNG with a clean vector (TikZ) diagram that
   explicitly shows the three THCA channels; clarified the robustness figure and
   expanded its analysis.
8. Softened absolute/exaggerated language throughout; standardized the bibliography
   (DOIs, author lists) and in-text citation style; specified the missing
   hyperparameters.

---

## Reviewer 1

**Comment 1 — Novelty of SGTR vs. CARE-GNN and RABot.**
*"Is merely replacing the similarity metric with 'spectral energy' sufficient to
constitute a core innovation?"*

We agree that the original text did not clearly differentiate SGTR. Section 3.2 now
includes a dedicated paragraph, "Relation to prior edge-refinement methods," stating
that SGTR (i) is **label-free** (CARE-GNN uses a label-aware similarity and an RL
neighbour selector; RABot uses an RL-tuned reliability threshold), avoiding
propagation of label noise into the topology; (ii) is grounded in the graph
**Dirichlet-energy decomposition** (Eq. 1) rather than a heuristic similarity,
giving the score a spectral interpretation; and (iii) is a **single differentiable
pre-aggregation step** trained end-to-end, not a separately optimized RL policy. We
also added a sentence in the related-work RABot paragraph contrasting the two
edge-refinement mechanisms. We now frame SGTR as a lightweight, theoretically
motivated alternative whose isolated contribution is measured in the ablation
(Table 2, `w/o SGTR`).

**Comment 2 — Baseline fairness / imbalance sampling.**
*"Whether this [imbalance sampling] strategy was applied to all baseline models..."*

We added Section 4.1.4, "Baseline Implementation and Fair Comparison," clarifying
that all models use the same train/validation/test splits, the same node features,
and the same input graph; that imbalance-aware sampling was applied uniformly to all
baselines that support it; and that baseline hyperparameters were re-tuned on each
dataset's validation set rather than transcribed from the original papers. The
`w/o Imbalance Sampling` row in the ablation isolates this component for XHBot
specifically.

**Comment 3 — Ablation flaws ("Dual-Channel" vs. Tri-channel; no `w/o CPD`).**
We rebuilt the ablation study (Table 2). It now ablates each THCA channel
independently (`w/o Homophilic`, `w/o Heterophilic`, `w/o Self-Identity`), includes
a homophilic-only extreme, and adds `w/o SGTR` and `w/o CPD`. We report a $\Delta$F1
column and rewrote the accompanying analysis. The incorrect "Dual-Channel" label has
been removed throughout. Among the channels, removing the heterophilic channel
causes the largest single-channel drop ($-0.1244$ F1); `w/o CPD` quantifies the
disentanglement contribution ($-0.0424$ F1).

**Comment 4 — HFE evaluation insufficient for a human-oriented task.**
We added Section 4.6, "Qualitative Forensic Case Study," which walks through a
representative flagged account and presents the three levels of HFE evidence
(instance subgraph, community ring, diagnostic motif) with a human-readable
interpretation. We also explicitly acknowledge that automated metrics are necessary
but not sufficient, and identify an expert user study as future work (also noted in
the Conclusion).

**Comment 5 — Inconsistent "+27.37%" claim.**
This originated from a duplicated Conclusion section. We deleted the duplicate and
the erroneous figure. The manuscript now reports only the values supported by
Section 4.2 (9.64% on TwiBot-20, 13.15% on TwiBot-22), consistently in the abstract,
introduction, results, and conclusion.

**Comment 6 — Figures 1 and 4 clarity.**
Figure 1 (architecture) has been redrawn as a clean vector diagram that explicitly
shows the three THCA channels (homophilic / heterophilic / self-identity), the
SGTR refinement, CPD, the classifier, and HFE. Figure 4 (robustness) was regenerated
with a titled legend, an explicit x-axis definition of "camouflage degree," distinct
markers, and end-point annotations; the accompanying analysis was expanded from one
sentence to a full paragraph quantifying each model's degradation.

**Comment 7 — Exaggerated language.**
We revised absolute phrasing throughout, e.g. "impenetrable black boxes" →
"provide limited rationale," "fundamentally insufficient" → "often insufficient,"
"comprehensively dominates" / "massive margin" / "weaponizes" → neutral
descriptions, and "provably insufficient" → "often insufficient."

**Comment 8 — Reference inconsistencies (DOIs, author lists).**
We standardized the bibliography: arXiv entries now use a uniform
`journal = {arXiv preprint}` + `doi` format with redundant URL/identifier fields
removed; a DOI was added to the GNNExplainer entry; and author lists use a uniform
`Last, First` format. The two genuine no-DOI NeurIPS proceedings entries
(TwiBot-22, PGExplainer) retain a canonical URL. We also fixed the BibTeX warning
for the RGT entry (removed the conflicting `number` field).

**Comment 9 — Optimize figures and language.**
Addressed jointly with Comments 6 and 7 above.

---

## Reviewer 2

**Major Comment 1 — Abstract format.**
The abstract has been rewritten as a single flowing paragraph; the
`INTRODUCTION/OBJECTIVES/METHODS/RESULTS/CONCLUSION` labels have been removed.

**Major Comment 2 — Structure and redundancy (two Conclusions; verbose intro/RW).**
The duplicated Conclusion section has been removed, leaving a single cohesive
conclusion. We streamlined the introduction (condensed the statistics-heavy opening)
and the related-work section (substantially shortened the RABot and FEDRIO
paragraphs while preserving the technical contrasts).

**Major Comment 3 — Clarity of SGTR; "Spectral-Guided" misleading.**
We added a "Spectral motivation" paragraph deriving the connection explicitly: the
graph Dirichlet energy (Laplacian quadratic form) decomposes over edges as
$\mathbf{f}^\top L \mathbf{f} = \sum_{(u,v)} (f_u - f_v)^2$, so the squared feature
discrepancy $(\mathbf{x}_u-\mathbf{x}_v)^2$ used in Eq. 2 is precisely the per-edge
contribution to the high-frequency (Dirichlet) energy. Down-weighting high-energy
edges therefore attenuates the high-frequency band that drives over-smoothing. The
learnable projection is presented as a calibrated, data-dependent estimate of this
quantity, which justifies the "spectral" terminology.

**Major Comment 4 — Justification for CPD / `w/o CPD` ablation.**
Added (see Reviewer 1, Comment 3). Table 2 now includes `w/o CPD` ($-0.0424$ F1),
showing CPD contributes beyond THCA alone, primarily by improving class separability.

**Major Comment 5 — Explainability evaluation: add qualitative case studies.**
Added (see Reviewer 1, Comment 4): Section 4.6 presents a concrete flagged account,
the HFE subgraph as the cause, and a human-interpretable explanation of the
star-formation botnet pattern.

**Major Comment 6 — Baseline selection and fairness.**
Addressed in Section 4.1.4 (see Reviewer 1, Comment 2): baseline hyperparameters
were re-tuned on the study's datasets rather than taken from the original papers, and
the comparison protocol is now stated explicitly.

**Minor Comment 1 — Figure 1 density.**
The architecture diagram was redrawn as a clean vector figure with a clearer
left-to-right data flow and an explicit THCA grouping (see Reviewer 1, Comment 6).

**Minor Comment 2 — Missing hyperparameters (hidden dim, layers).**
Section 4.1.3 now specifies the number of THCA layers ($L=2$), hidden dimension
($d_h=64$), dropout ($0.5$), CPD sub-space dimension ($d'=32$), InfoNCE temperature,
loss weights ($\lambda_{MI}$, $\lambda_{NCE}$), and the SGTR threshold.

**Minor Comment 3 — Typo/confusing notation in Equation (2).**
The soft-pruning rule has been rewritten in explicit piecewise (cases) form
(Eq. 3 in the revision), making the thresholding behaviour unambiguous: low-energy
edges keep their weight; high-energy edges are attenuated by $(1-E)$.

**Minor Comment 4 — Citation formatting.**
We standardized in-text citations to the "Author et al.~[n]" form when a work is the
subject of a sentence (e.g., "Fu et al.~[8]", "He et al.~[…]", "Qiao et al.~[…]",
"Dang and Nguyen~[…]"), removing the inconsistent bare-number and "The work of […]"
constructions.

---

We believe these revisions substantially strengthen the manuscript and address all
of the reviewers' concerns. We thank the reviewers again for their time and helpful
feedback.
