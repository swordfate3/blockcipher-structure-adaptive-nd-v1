# Innovation 1 Runtime-SPN Typed-Relation GNN-FiLM Five-Cipher Plan

Date: 2026-07-26

```text
status = completed-hold
whole_cipher_holdout = prohibited
remote_scale = no
```

## Research Question

Can one shared Runtime-E4 model use the externally supplied GF(2) relation
types during message propagation, rather than only conditioning a hidden state
before or after a generic mixer, and thereby obtain stable attributable gains
across GIFT, SKINNY, RECTANGLE, uKNIT and Dialga?

The completed additive Adapter, multiplicative gate and 128-dimensional True
FiLM candidates all received traffic and gradients but did not consistently
beat parameter-matched controls. This experiment changes the location and
granularity of structural computation, not its sample budget or benchmark.

## Design Audit

Runtime-E4 already computes the previous state with the exact inverse linear
map over GF(2). That exact path remains unchanged. Its existing S-box edge gate
then collapses each source-cell/target-cell `4 x 4` GF(2) block to one scalar
cell adjacency value. The proposed typed path preserves the 16
`target_bit_role x source_bit_role` channels around that exact view.

Observed relation coverage in the frozen five-cipher runtime descriptors is:

| Cipher | Inverse GF(2) edges per transition | Nonempty role-pair types |
| --- | ---: | ---: |
| GIFT-64 | 64 | 4/16 |
| SKINNY-64/64 | 128 | 4/16 |
| RECTANGLE-80 | 64 | 4/16 |
| uKNIT-BC | 192 | 16/16 |
| Dialga-128 | 384 | 16/16 |

This candidate therefore adds information that the current cell-level graph
gate discards, particularly on the two heterogeneous stress ciphers. It does
not replace XOR with real-valued aggregation: exact GF(2) inversion still
constructs the predecessor inputs, while typed messages are a bounded neural
residual that interprets those exact inputs.

## One Variable

Only the pre-mixer relation-specific message residual changes:

```text
exact_previous = inverse_linear_GF2(ciphertext_state)
transition = shared_typed_fusion(current_state, exact_previous)
transition = existing_Sbox_edge_gate(transition)
transition += 0.1 * typed_relation_message(transition, inverse_GF2_edges)
transition = existing_shared_mixer(transition)
```

There are 16 relation types. Every type has one feature-wise `gamma` and
`beta`, each of width 128:

```text
16 * 128 + 16 * 128 = 4096 trainable parameters
```

The complete candidate therefore has `446562` parameters, exactly matching
the completed additive Adapter and True FiLM panels. There is no cipher name,
cipher ID, explicit block width, global fingerprint, task-specific head or
task-specific trainable state.

## Frozen Controls

All four roles use the same parameter tensors and active module geometry:

| Role | Edge support | Relation labels |
| --- | --- | --- |
| `dense` | fixed complete cell graph | all 16 transforms averaged |
| `correct` | exact inverse-GF(2) edge support | true bit-role pair |
| `uniform` | exact inverse-GF(2) edge support | relation-agnostic average |
| `shuffled` | exact inverse-GF(2) edge support | fixed cyclically wrong type |

The `uniform` role name is retained by the shared experiment shell; in this
plan it means the parameter-matched relation-agnostic graph control.

## Readiness Gate

Before any result-producing diagnostic, all checks must pass:

1. the 16 relation channels reconstruct every bit of the exact inverse GF(2)
   matrix for all five structures;
2. correct, shuffled and relation-agnostic controls preserve identical edge
   support where required;
3. the four roles have identical state geometry and exactly `446562`
   parameters;
4. one state dict handles both 64-bit and 128-bit structures;
5. all typed `gamma` and `beta` parameters receive finite nonzero gradients;
6. correct relation messages preserve cell-relabeling invariance;
7. correct, dense, relation-agnostic and shuffled controls produce distinct
   logits under shared weights on a heterogeneous structure;
8. existing Runtime-E4, recurrent-window, Dialga, Adapter and True FiLM
   regressions remain green;
9. the five-task one-epoch smoke uses equal task weights, one shared optimizer
   state and strict encrypted-random-plaintext negatives.

Any failure stops the experiment and authorizes only a focused repair.

## Joint Diagnostic Protocol

```text
core = GIFT-64 r6, SKINNY-64/64 r7, RECTANGLE-80 r6
stress = uKNIT-BC prefix-r5, Dialga-128 prefix-r4
train = 2048/class/cipher
validation = 1024/class/cipher
pairs_per_sample = 4
seeds = 0, 1
epochs = 10
batch_size = 256
loss = MSE
optimizer = Adam, lr 1e-4, weight_decay 1e-5
checkpoint = restored best validation macro AUC
negative = encrypted random plaintexts
execution = local diagnostic only, reusing the frozen disk cache
```

## Advance And Stop Gates

For each seed independently, `correct` must satisfy all of:

```text
core macro AUC >= every matched control + 0.005
stress macro AUC >= every matched control + 0.005
every individual cipher delta versus every matched control >= -0.005
typed relation traffic > 0
typed gamma and beta gradients > 0 and finite
```

Only a full two-seed pass authorizes preregistration of a whole-cipher holdout.
A hold closes this typed residual candidate at the current protocol. It does
not authorize more samples, epochs, relation ranks, relation layers, learned
MoE, cipher-ID routing or remote scale as a rescue.

## Evidence-Backed Next Action

If readiness passes, run exactly the frozen four-role, five-cipher, two-seed
local diagnostic above. If the joint gate passes, preregister an entire-cipher
holdout in which the candidate cipher is absent from all scratch training. If
the joint gate holds, consolidate the already supported Runtime-E4 method for
the thesis instead of continuing architecture enumeration.

## Readiness Result

The implementation and protocol gate completed locally:

```text
run_id = i1_runtime_spn_typed_relation_gnn_film_five_cipher_readiness_20260726
status = pass
decision = innovation1_runtime_spn_typed_relation_readiness_passed
checks = 11/11
regression = 289 passed
```

The gate reconstructed all five inverse GF(2) matrices from the 16 typed
relations, parameter-matched all four roles at `446562`, reused one state dict
for 64-bit and 128-bit states, observed finite nonzero `gamma`/`beta`
gradients, preserved cell-relabeling equivariance and confirmed that the four
control roles produce distinct logits under shared weights.

Artifacts:

```text
outputs/local_readiness/i1_runtime_spn_typed_relation_gnn_film_five_cipher_readiness_20260726/
```

## Completed Joint Result

The frozen five-cipher, four-role, two-seed diagnostic completed with 40 result
rows and eight shared-role checkpoints:

```text
run_id = i1_runtime_spn_typed_relation_gnn_film_five_cipher_joint_2048_seed0_seed1_20260726
status = hold
decision = innovation1_runtime_spn_typed_relation_not_supported
protocol_valid = true
source_anchor_valid = additive:true, true_film:true
core_pass = false
full_pass = false
```

Correct-relation validation AUCs were:

| Seed | GIFT-64 r6 | SKINNY r7 | RECTANGLE r6 | uKNIT prefix-r5 | Dialga prefix-r4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `0.533156` | `0.508939` | `0.718196` | `0.504588` | `0.943933` |
| 1 | `0.532015` | `0.540340` | `0.696939` | `0.510626` | `0.947201` |

The gate depends on matched deltas rather than these absolute AUCs:

| Seed | Control | Core macro delta | Stress macro delta |
| ---: | --- | ---: | ---: |
| 0 | dense | `-0.000871` | `-0.004002` |
| 0 | uniform | `-0.002684` | `-0.000242` |
| 0 | shuffled | `-0.000339` | `-0.001464` |
| 1 | dense | `+0.001596` | `+0.000211` |
| 1 | uniform | `+0.004312` | `-0.003523` |
| 1 | shuffled | `+0.003563` | `-0.007594` |

No seed passed the required `+0.005` core and stress margins against every
control. Individual-cipher floor failures included uKNIT seed0 versus dense
at `-0.006647`, RECTANGLE seed0 versus uniform/shuffled at
`-0.005424/-0.005510`, and uKNIT seed1 versus uniform/shuffled at
`-0.007715/-0.015956`.

The typed candidate also failed to improve consistently over the completed
source anchors:

| Seed | Historical anchor | Core macro delta | Stress macro delta |
| ---: | --- | ---: | ---: |
| 0 | additive | `-0.000201` | `+0.000087` |
| 0 | True FiLM | `+0.001908` | `-0.007108` |
| 1 | additive | `-0.000463` | `-0.001718` |
| 1 | True FiLM | `+0.000989` | `-0.002582` |

Both seeds recorded large positive typed-message traffic and finite nonzero
`gamma`/`beta` gradient totals. The hold therefore rejects the tested
relation-specific residual hypothesis; it is not explained by an inactive
module, broken gradient path or invalid protocol.

Artifacts:

```text
outputs/local_diagnostic/i1_runtime_spn_typed_relation_gnn_film_five_cipher_joint_2048_seed0_seed1_20260726/
```

The final `curves.svg` passed `visual-qa-redraw` after rendered-pixel
inspection at `1800 x 1037` and `1280 x 737`. Chinese glyphs, the three-line
title, seed panels, zero and `+0.005` reference lines, axis bounds, three
control markers and the bottom legend were all readable without overlap,
clipping, occlusion or ambiguous scaling. The inspection record is
`visual_qa_passed.marker` in the run directory.

### Evidence-Backed Final Action

Close the additive Adapter, multiplicative gate, True FiLM and typed GNN-FiLM
residual branch under this joint protocol. Do not rescue it with extra relation
layers, experts, samples, epochs or remote compute, and do not open a
whole-cipher holdout for this failed candidate. Consolidate the already
supported shared Runtime-E4 exact-topology method and use this complete
parameter-matched panel as its differentiated-architecture ablation evidence.
