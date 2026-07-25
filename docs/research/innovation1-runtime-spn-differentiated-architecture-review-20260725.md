# Innovation 1 Runtime-SPN Differentiated Architecture Review

Date: 2026-07-25

## Research Question

Can a mixture-of-experts or another conditional architecture preserve one
cipher-name-free SPN backbone while allowing different SPN diffusion and S-box
primitives to receive different learned computations?

The target is not a bank of PRESENT, SKINNY, GIFT and uKNIT networks. The target
remains one runtime-parameterized model whose parameter geometry does not depend
on the cipher name, block width, cell count or linear-layer shape. Routing must
be derived from the supplied cell/S-box/GF(2) descriptor and must remain
meaningful on an unseen cipher.

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
source tasks   = GIFT-64 r6, SKINNY-64/64 r7, RECTANGLE-80 r6
train scale    = 2048/class/cipher
validation     = 1024/class/cipher
pairs/sample   = 4
seeds          = 0, 1
epochs         = 10
execution      = local sub-medium diagnostic
negatives      = encrypted random plaintext pairs
sampling       = balanced by cipher and class within joint training
models         = dense anchor, correct routing, uniform mixture,
                 shuffled primitive routing
```

The active trainable-parameter difference between anchor and candidate should
be at most one percent. If necessary, shrink the shared candidate update or
width-match the anchor; do not credit extra parameters as structural gain.

Advance only if both seeds satisfy all of the following preregistered checks:

```text
candidate macro-AUC - dense anchor     >= +0.005
candidate macro-AUC - uniform mixture  >= +0.005
candidate macro-AUC - shuffled routing >= +0.005
no per-cipher candidate-anchor margin  <  -0.005
both primitive adapters receive traffic and nonzero gradients
```

If this gate passes, the next experiment is a whole-cipher holdout with a
frozen shared backbone and the already established low-capacity target-head
protocol. RECTANGLE is the preferred first holdout because it has
non-contiguous cells and existing Runtime-E4 transfer evidence. If the gate
holds, stop adapter/MoE scaling and inspect whether the primitive descriptor,
parameter matching or joint sampler failed. Do not add experts, samples,
epochs or remote compute as a rescue.

Only after a joint-training pass and a whole-cipher holdout pass should the
route add a third heterogeneous-S-box/round-transition adapter and return to
uKNIT. Sparse learned Top-2 routing is a later ablation, not the method's
starting point.

## Final Recommendation

Keep MoE as a constrained component of the design, not as the architecture
name or the immediate experiment. The evidence-backed priority is:

```text
1. deterministic primitive routing + low-rank FiLM/adapters
2. same-capacity dense and routing controls
3. joint multi-SPN local gate
4. whole-cipher holdout
5. only then learned soft/Top-2 structural MoE
```

This route directly addresses the observed failure mode: the shared backbone
can exploit general GF(2) topology, but a single recurrent update does not
stably interpret every heterogeneous transition. It also preserves the thesis
claim boundary: one shared runtime-parameterized SPN model with differentiated
structure primitives, rather than a lookup table of cipher-specific networks.

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
