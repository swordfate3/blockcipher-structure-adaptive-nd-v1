# Innovation 1 Runtime-SPN Differentiated Architecture Review

Date: 2026-07-25

## Research Question

Can a mixture-of-experts or another conditional architecture preserve one
cipher-name-free SPN backbone while allowing different SPN diffusion and S-box
primitives to receive different learned computations?

The target is not a bank of PRESENT, GIFT, SKINNY, RECTANGLE, uKNIT and Dialga
networks. The target remains one runtime-parameterized model whose parameter
geometry does not depend on the cipher name, block width, cell count or linear-
layer shape. Routing must be derived from the supplied cell/S-box/GF(2)
descriptor and must remain meaningful on an unseen cipher.

## Evidence That Constrains The Design

Two completed results point in different but compatible directions.

1. Runtime-E4 is worth preserving. On SKINNY-64/64 r7 at `1000000/class`
   training and `500000/class` validation, correct general-GF(2) topology
   reached AUC `0.653191631/0.653623658` on seeds 0/1. Corrupted topology reached
   `0.607162433/0.606953760`, and no topology reached
   `0.511826119/0.515817818`. The two-seed formal project gate passed.
2. One shared recurrent transition processor is not enough. On uKNIT-BC
   prefix-r5 at `2048/class`, the correct two-transition recurrent window
   reached `0.501017094/0.527083397`. Seed1 passed every attribution gate, but
   seed0 was below the last-transition anchor, corrupted topology and no
   topology. Candidate final train AUCs `0.644991/0.626094` versus validation
   AUCs `0.501017/0.527083` also show a substantial generalization gap.
3. Dialga supplies a real 128-bit heterogeneous-width stress case. Its
   prefix-r4 Runtime-E4 candidates reached `0.958417/0.958679` on seeds 0/1 and
   passed the local correct-versus-corrupted/no-topology training gate, while
   the later GIFT-to-Dialga frozen transfer remained attribution-incomplete.
   This makes Dialga useful in joint end-to-end training, not as evidence that
   an existing 64-bit source checkpoint already transfers universally.

The conclusion is therefore not "replace Runtime-E4 with a larger model." The
supported exact inverse-linear/state-view path should stay shared, while the
failed assumption that all transition types need the same residual update
should be tested with a small conditional module.

## Audit Of The Existing Repository MoE

The repository already registers `moe_v4_uniform`, `moe_v4_hard`,
`moe_v4_soft` and later v5 variants in
`src/blockcipher_nd/models/structure/moe.py`. They are not the Runtime-SPN MoE
needed by the current method objective.

| Property | Existing v4/v5 implementation | Current requirement |
| --- | --- | --- |
| Routing input | 19 global profile values such as `is_spn`, block/key width and rounds | Runtime cell membership, per-cell S-box truth table and actual GF(2) edges |
| SPN distinction | Hard routing gives every SPN the same fixed expert weights | Distinguish permutation, multi-source GF(2), non-contiguous cells and changing transitions |
| SPN adapter | Reshapes contiguous groups of four bits and mixes each group mean | Must obey external `cell_membership`; RECTANGLE cells are non-contiguous |
| Granularity | One constant gate vector for the whole cipher/run | Per-transition or per-cell structural routing |
| Experts | Six complete CNN/MLP/ResNet-style models; all logits are computed | Shared backbone plus small primitive modules; matched active compute |
| Sparsity | Output mixture is dense even in `soft` mode | Sparse Top-k only if a later gate justifies it |
| Unseen-cipher test | No runtime-topology holdout contract | Whole-cipher holdout with no cipher ID |

No current result/config artifact containing `moe_v4_*` was found in the
workspace result corpus. The code proves that an earlier family-level MoE was
implemented; it does not provide current evidence that this MoE passed or
failed. Reusing its claim language would therefore be unsafe.

## Literature Findings

### Direct neural-cryptanalysis boundary

The targeted exact search for `"mixture of experts" "neural distinguisher"
block cipher cryptanalysis SPN` returned SPN neural-distinguisher papers,
quantum distinguishers and unrelated MoE work, but no paper that routes an SPN
neural distinguisher from a runtime S-box/GF(2) descriptor. This is a bounded
search result, not proof of nonexistence.

The closest verified cryptanalysis works still use other forms of structural
adaptation:

- Liu, Li, Ren and Chen (2026), *A Highly Efficient Neural Distinguisher
  Framework for IoT-Friendly Lightweight SPN Block Ciphers*, uses inverse
  round operations and a `3 x 4 x n/4` Conv2D state representation for SKINNY
  and MIDORI. It is SPN-aware but the preprocessing remains cipher-adjusted.
- Ge and Wang (2026), IACR ePrint 2026/535, uses invertible SPN components for
  related-key feature enhancement on SKINNY and PRESENT. It is a unified
  protocol, not a runtime-topology processor.
- Bellini et al. (2023), *A Cipher-Agnostic Neural Training Pipeline*, obtains
  generality with DBitNet and automated difference search while deliberately
  avoiding cipher topology.
- Wu and Guo (2024) build PRESENT-specific `invP`/`invS` integral features.
  This is evidence for structure-derived views, not cross-SPN shared weights.

### Architecture evidence from adjacent fields

| Work | Verified mechanism | Relevance to Runtime-SPN |
| --- | --- | --- |
| Schlichtkrull et al., R-GCN (2018) | Relation-specific message transforms on multi-relational graphs | Different linear/S-box relation types can use small shared bases instead of cipher-specific networks |
| Perez et al., FiLM (AAAI 2018) | Conditioning data produces feature-wise affine modulation | Cheap way to let a runtime descriptor modify a shared transition block |
| Ha, Dai and Le, HyperNetworks (ICLR 2017) | One network generates another network's weights | Expressive, but full weight generation is high risk with few cipher structures |
| Yang et al., CondConv (NeurIPS 2019) | Per-example kernel is a mixture of learned expert kernels | A dense basis mixture is a simpler first test than sparse whole-model MoE |
| Brockschmidt, GNN-FiLM (ICML 2020) | Target-node state modulates incoming relation-specific messages | Closest established pattern for per-cell/per-edge structure-conditioned diffusion |
| Hu et al., HGT (WWW 2020) | Node/edge-type-dependent attention and transformations | Supports typed transitions, but the full transformer is unnecessary for the first gate |
| Fedus, Zoph and Shazeer, Switch Transformer (JMLR 2022) | Top-1 sparse routing with constant active compute | Shows the capacity benefit, but also documents routing/communication/training complexity |
| Zoph et al., ST-MoE (2022) | Router stability and transfer design, including router z-loss | Sparse MoE needs its own stability protocol; it is not a free architecture swap |
| Wang et al., Graph MoE (NeurIPS 2023) | Nodes select hop-1/hop-2 aggregation experts for diverse graph structures | Strong precedent for local structural experts rather than one expert per graph/domain |
| Pfeiffer et al., *Modular Deep Learning* (TMLR 2023) | Separates modules, routing, aggregation and shared parameters | When expert structural metadata is known, deterministic routing is a legitimate and lower-risk first design |

Verified metadata and abstracts are stored in:

```text
sources/research_runtime_spn_architecture_core_arxiv_20260725.xml
sources/research_gnn_film_20260725.json
sources/research_graph_mixture_experts_20260725.json
sources/research_modular_deep_learning_20260725.json
sources/research_moe_stability_routing_20260725.json
sources/research_spn_neural_distinguisher_moe_exact_20260725.json
```

## Technology Ranking

| Rank | Route | Fit | Main reason | Main risk |
| ---: | --- | ---: | --- | --- |
| 1 | Shared Runtime-E4 plus primitive-conditioned low-rank adapters and FiLM | `9/10` | Smallest change; preserves the proven exact topology path; router can use local primitive descriptors | Adapter gain may be too small or may still overfit |
| 2 | Dense CondConv-style basis mixture inside the transition update | `8.5/10` | Differentiated kernels without hard Top-k, capacity drops or expert starvation | Dense mixture adds active compute and still needs parameter matching |
| 3 | Typed R-GCN/GNN-FiLM message path around the exact GF(2) view | `7.5/10` | Natural handling of relation types and per-cell context | Ordinary real-valued aggregation can lose XOR semantics; previous coarse graph routes were weak |
| 4 | Sparse Top-2 structural MoE of small adapters | `6.5/10` | Can specialize while keeping active compute bounded | Too few structures, router collapse, cipher identification and unstable training |
| 5 | Low-rank hypernetwork that generates adapter deltas | `5.5/10` | Can interpolate to unseen descriptors | Harder attribution and higher sample demand |
| 6 | Full-kernel hypernetwork or full HGT/GraphGPS replacement | `3.5/10` | Maximum expressivity | Too much mechanism and capacity for the current evidence scale |
| 7 | Larger recurrent/Transformer processor without structural modules | `2.5/10` | More sequence capacity | U3 already shows that recurrence alone does not produce stable attribution |

## MoE Suitability Verdict

MoE is conditionally suitable, but only at the primitive-module level.

The appropriate unit of specialization is not the cipher and not a complete
neural distinguisher. It is the transition operation applied inside a shared
Runtime-E4 processor. A useful first candidate is:

```text
ciphertext pairs + exact runtime inverse-linear/state views
  -> shared cell encoder
  -> shared Runtime-E4 transition update
  -> structure-routed low-rank residual adapters
       permutation/fan-in-1 adapter
       multi-source GF(2) adapter
  -> invariant cell/pair pooling
  -> shared classifier
```

For cell or transition descriptor `z` and hidden token `h`, the added update
can be kept small:

```text
h_next = Shared(h, z) + sum_e alpha_e(z) * A_e(B_e(h))
```

`A_e(B_e(h))` is a low-rank residual adapter. In the first version,
`alpha_e(z)` should be deterministic from local linear-row fan-in and relation
type. It must not read cipher name, cipher ID, key width, block width or a
global round-count fingerprint. The shared path stays active for every input.
Only after deterministic primitive routing passes should a learned soft or
Top-2 router be tested.

The exact GF(2) inverse/state view must remain outside the learned mixture. A
real-valued GNN sum is not an algebraically faithful substitute for XOR. The
adapter should learn how to interpret the exact view, not relearn the cipher's
linear layer from scratch.

## Required Controls

Any differentiated architecture must include:

1. A same-budget, parameter-matched dense Runtime-E4 anchor.
2. The candidate with correct primitive routing.
3. An equal-parameter uniform-mixture control that disables structural
   selection while retaining every adapter.
4. A shuffled-descriptor routing control that keeps data and topology views
   fixed but assigns the wrong primitive to the update.
5. Per-cipher metrics and macro metrics; an average gain may not hide a severe
   regression on one SPN.
6. Router/expert utilization and gradient norms. A nominal expert with no
   traffic or gradients is not evidence of specialization.
7. Cell-relabeling invariance and a whole-cipher holdout before any
   unseen-cipher claim.

A cipher-ID router may be used only as a clearly labeled diagnostic upper
bound. It is prohibited from the candidate and cannot support the method
claim.

## Recommended First Experiment

Do not launch a full sparse MoE or a remote job yet. The first executable gate
should ask one narrow question:

```text
question       = does deterministic structure-primitive routing of two small
                 adapters improve a jointly trained Runtime-E4 model?
anchor         = parameter-matched dense Runtime-E4
one variable   = primitive-conditioned adapter update
core tasks     = GIFT-64 r6, SKINNY-64/64 r7, RECTANGLE-80 r6
stress tasks   = uKNIT-BC prefix-r5, Dialga-128 prefix-r4
train scale    = 2048/class/cipher
validation     = 1024/class/cipher
pairs/sample   = 4
seeds          = 0, 1
epochs         = 10
execution      = local sub-medium diagnostic
negatives      = encrypted random plaintext pairs
sampling       = one batch per cipher per optimizer step; mean the five task
                 losses before one shared-parameter update
models         = dense anchor, correct routing, uniform mixture,
                 shuffled primitive routing
```

All five tasks use one shared parameter state per model role and seed. They do
not produce five independently optimized backbones. Separate per-cipher
protocol adapters are allowed only to bind different block widths and runtime
descriptors to that same shared backbone. No cipher-specific trainable head is
allowed in this first gate.

The active trainable-parameter difference between anchor and candidate should
be at most one percent. If necessary, shrink the shared candidate update or
width-match the anchor; do not credit extra parameters as structural gain.

The gate is stratified so Dialga's high absolute AUC cannot hide a weak uKNIT
or core task. Advance only if both seeds satisfy all of the following
preregistered checks:

```text
candidate macro-AUC - dense anchor     >= +0.005
candidate macro-AUC - uniform mixture  >= +0.005
candidate macro-AUC - shuffled routing >= +0.005
core three-cipher candidate-anchor       >= -0.005 per cipher
uKNIT/Dialga candidate-anchor            >= -0.005 per cipher
new-cipher stress macro versus all three >= +0.005
both primitive adapters receive traffic and nonzero gradients
```

Report three separate aggregates: the core three-cipher macro AUC, the two-new-
cipher stress macro AUC and the five-cipher macro AUC. The five-cipher average
is descriptive only; it cannot override a failed core or stress check.

## Executed Evidence Update

The recommended deterministic two-Adapter experiment and its two post-hold
audits have now completed.

1. The additive `fan_in_1`/`multi_source` joint candidate did not pass either
   seed's core or stress gate. Its most severe individual regression was uKNIT
   seed0 at `-0.017501` versus the dense anchor.
2. Descriptor collisions are real, but same-bucket Adapter gradients were
   positively aligned: uKNIT/Dialga mean cosine `0.562826` and
   GIFT/RECTANGLE `0.177383`. Shared-backbone gradients were also positive on
   average. Immediate descriptor splitting or multi-task conflict correction
   is therefore not the strongest next explanation.
3. Frozen counterfactuals found that the additive Adapter was functionally
   weak on seed1 despite route sensitivity and nearly full effective rank.
   Simply amplifying its scale reduced training macro AUC.
4. A parameter-matched multiplicative gate passed all readiness checks but
   failed the same five-cipher two-seed research gate. It did not consistently
   beat its dense/uniform/shuffled controls or the additive source.
5. A 4096-parameter True FiLM conditioner used a 128-dimensional local
   descriptor containing S-box truth bits and GF(2) diffusion statistics. It
   exactly matched the old additive model at 446562 total parameters, passed
   11/11 readiness checks, distinguished both known descriptor-collision
   pairs, preserved cell relabeling and received nonzero gradients. The real
   two-seed joint diagnostic still held: core macro deltas versus dense were
   only `+0.001635` and `+0.002331`, while correct-versus-shuffled core deltas
   were `-0.003714` and `-0.000405`. It also failed the old additive source on
   the core macro in both seeds.
6. The final typed GNN-FiLM audit preserved all 16
   `target_bit_role x source_bit_role` GF(2) relation channels around the exact
   inverse-linear view. All four roles exactly matched `446562` parameters,
   reconstructed the five inverse matrices, preserved cell-relabeling
   equivariance and received nonzero finite relation gradients. The completed
   two-seed joint diagnostic still held. Seed0 correct-minus-control core macro
   deltas were `-0.000871/-0.002684/-0.000339` versus
   dense/uniform/shuffled; seed1 deltas were
   `+0.001596/+0.004312/+0.003563`. Neither seed reached `+0.005`, and the
   stress macro failed every control on both seeds except seed1 dense at only
   `+0.000211`. uKNIT also regressed by `-0.015956` versus shuffled on seed1.

This evidence closes deterministic low-rank additive, multiplicative, local
True FiLM and typed GNN-FiLM residual effects under the current five-cipher
joint protocol. It does not reject the method-level runtime-parameterized SPN
objective: the shared
Runtime-E4 topology path remains supported on SKINNY at formal project scale,
while these local joint gates reject the tested differentiated side paths.

### Updated Technology Ranking

| Rank | Route after executed evidence | Status | Reason |
| ---: | --- | --- | --- |
| 1 | Consolidate the supported shared Runtime-E4 method without differentiated side modules | active | Formal SKINNY topology attribution and local cross-cipher representation evidence are stronger than every tested differentiated branch |
| 2 | Typed R-GCN/GNN-FiLM path around the exact GF(2) view | closed | Exact 16-relation implementation was active and valid but failed the two-seed joint control gate |
| 3 | Refined deterministic primitive descriptor | closed | True FiLM already used a collision-distinguishing local descriptor without stable gain |
| 4 | Sparse Top-2 Adapter MoE | prohibited | Additive, multiplicative, True FiLM and typed message paths all failed; learned routing would add an unidentifiable degree of freedom |
| 5 | Larger generic recurrent/Transformer processor | closed | Does not target the observed attribution failure |

The typed relation-specific comparison is now complete. It confirms that the
exact Runtime-E4 GF(2) path should be retained, but does not justify another
learned residual around that path. No whole-cipher holdout is authorized for
the differentiated candidate because the required joint gate did not pass.
The next work item is evidence consolidation and a requirement-by-requirement
audit of the supported Runtime-E4 method before any new method hypothesis is
preregistered.

## New-Algorithm Family Boundary

The three uploaded algorithm papers do not all belong to this Runtime-SPN
experiment:

| Paper algorithm | Structural family | Decision |
| --- | --- | --- |
| uKNIT | non-round-aligned 4-bit-cell SPN | include in the five-cipher stress panel |
| Dialga | heterogeneous 4-bit-cell tweakable SPN, 128-bit state | include in the five-cipher stress panel with fixed zero tweak |
| MSX | generalized Feistel with addition, rotation, XOR and 32-bit integer multiplication; no S box | exclude from the SPN adapter gate |

MSX is not evidence against Runtime-SPN and must not be coerced into a GF(2)
linear descriptor. A later cross-structure method may reuse the shared
multi-task shell but needs Feistel-branch, word-rotation, carry and
multiplication descriptors plus separate primitive modules. Its metrics must
remain a distinct family panel.

## Final Recommendation

MoE and typed residual message passing are closed for the current
differentiated branch. The evidence-backed priority is now:

```text
1. freeze the supported Runtime-E4 architecture and exact runtime descriptor contract
2. consolidate formal SKINNY, local GIFT/RECTANGLE/Dialga and frozen-representation evidence
3. retain additive, gate, True FiLM and typed GNN-FiLM as closed matched ablations
4. audit the remaining gap between current evidence and a legitimate whole-cipher holdout
5. preregister a new method hypothesis only if it addresses that exact gap
6. do not reopen learned soft/Top-2 MoE or mechanically scale a closed residual branch
```

This route preserves what the evidence actually supports: one shared,
cipher-name-free runtime-parameterized SPN backbone can exploit exact GF(2)
topology, while the tested differentiated residual modules have not yet made
that backbone uniformly effective across heterogeneous structures. The
unseen-cipher claim remains open and must not be replaced by a lookup table of
cipher-specific networks or inferred from within-cipher controls.

## Verified Web Sources

- Ha, Dai and Le, *HyperNetworks*: https://arxiv.org/abs/1609.09106
- Perez et al., *FiLM*: https://arxiv.org/abs/1709.07871
- Schlichtkrull et al., *Modeling Relational Data with Graph Convolutional Networks*: https://arxiv.org/abs/1703.06103
- Yang et al., *CondConv*: https://arxiv.org/abs/1904.04971
- Brockschmidt, *GNN-FiLM*: https://proceedings.mlr.press/v119/brockschmidt20a.html
- Hu et al., *Heterogeneous Graph Transformer*: https://arxiv.org/abs/2003.01332
- Fedus, Zoph and Shazeer, *Switch Transformers*: https://arxiv.org/abs/2101.03961
- Zoph et al., *ST-MoE*: https://arxiv.org/abs/2202.08906
- Wang et al., *Graph Mixture of Experts*: https://neurips.cc/virtual/2023/poster/72025
- Pfeiffer et al., *Modular Deep Learning*: https://arxiv.org/abs/2302.11529
- Ge and Wang, *Improved Related-Key Differential Neural Distinguishers for SPN Block Ciphers*: https://eprint.iacr.org/2026/535
