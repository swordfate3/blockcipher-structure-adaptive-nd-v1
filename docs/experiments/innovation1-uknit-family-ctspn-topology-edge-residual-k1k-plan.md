# Innovation 1 uKNIT-Family CT-SPN Topology Edge Residual K1-K

**Date:** 2026-07-28
**Status:** completed / held after valid local diagnostic
**Execution:** local CPU readiness followed by a local fixed-budget diagnostic

## 1. Research Question

K1-I restored a strong exact-GF(2) signal on Dialga-128 r4 but failed
correct-operator attribution and remained near chance on uKNIT-BC r5. K1-J
then showed that cross-cell regrouping explains only `0.0%-0.8%` of the
Dialga native-minus-no-topology gap: the current invariant path uses two
complementary pooled statistics, not correct native-cell membership.

K1-K tests one change only:

> Can a bounded, topology-equivariant and position-preserving edge residual
> make the exact K1-I representation use the correct two runtime GF(2)
> operators, while retaining the same shared parameter shapes on 64- and
> 128-bit SPNs?

K1-K does not change the labels, negative definition, plaintext pairs,
differences, keys, splits, samples, epochs, loss, optimizer or K1-I base path.

## 2. Frozen Source Authorities

The sources are:

```text
K1-I root = outputs/local_diagnostic/
  i1_uknit_family_ctspn_gf2_boolean_view_k1i_2048_seed0_seed1_20260728

K1-J root = outputs/local_audit/
  i1_uknit_family_ctspn_position_cell_attribution_k1j_20260728
```

Required SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| K1-I gate | `e1823155149ce6146358650ae711269b617c93f4f7d48aaaa3e231348bfd675d` |
| K1-I checkpoint manifest | `4def7bc0019d7a258d962c622cfc79db1b69e0f85dc0b491a17bf081683e465f` |
| K1-I dataset manifest | `ecc990e4d724ec35fdce8bd52d947c78280db2140853feddee07189ade4341f0` |
| K1-J gate | `e77ab3811837fca0e9e7536df4ab2c58e0ea8f82222529e5c0c7c3903bb1d9dc` |

K1-I must retain its exact hold decision:

```text
innovation1_uknit_family_ctspn_k1i_dialga_signal_recovered_operator_attribution_not_supported
```

K1-J must retain its exact completed attribution decision:

```text
innovation1_uknit_family_ctspn_k1j_joint_pool_branch_signal_supported
```

Every protocol check in both authorities must remain true.

## 3. Single Architecture Change

The K1-I Boolean-view base remains byte-for-byte behaviorally unchanged:

```text
12 exact Boolean channels per bit
-> shared bit encoder
-> global-bit and cell-summary invariant pools
-> pair projection and pair aggregation
-> K1-I sample embedding and classifier
```

K1-K adds one residual beside that base. For every ciphertext pair:

1. Gather each native cell's four bit embeddings in runtime `bit_role=0..3`
   order and pass the flattened tuple through one shared cell encoder.
2. For each of the two transition slots, enumerate every nonzero entry of the
   corresponding inverse GF(2) matrix as a directed `(target_bit, source_bit)`
   edge.
3. Build each edge message from the ordered source/target cell tokens, the
   source/target bit embeddings, one-hot source/target bit roles and a two-way
   transition-slot code.
4. Use one shared edge function for both transitions and every state width.
   Aggregate messages at target cells and apply one shared cell update.
5. Pool only the topology-induced cell update, project it to a fixed-width
   sample residual and combine it with the K1-I sample embedding as:

```text
k1k_embedding = k1i_embedding
              + tanh(residual_gate) * tanh(edge_residual_embedding)
```

`residual_gate` is a learned scalar initialized to exact zero. At zero, K1-K
must reproduce K1-I logits exactly after the K1-I base state is loaded. The
residual is bounded and cannot become an unrestricted raw-input bypass.

The residual may use:

- the existing twelve deterministic K1-I Boolean channels;
- ordered cell membership and four bit roles;
- explicit nonzero source/target edges from both runtime inverse matrices;
- the two transition-slot identities.

It may not use:

- absolute cell or bit indices;
- cipher identity, descriptor name or block width as a learned feature;
- key bits, key identity or plaintext identity;
- S-box truth tables, DDTs, trails or partial decryption;
- a raw ciphertext bypass, path token ID or unrestricted concatenation into the
  classifier.

## 4. Equivariance Contract

Jointly relabeling complete cells must relabel the internal cell and edge sets
without changing the final logit. The intervention must:

- relabel both ciphertext endpoints with the same cell permutation;
- conjugate both runtime matrices with the induced bit permutation;
- preserve each bit's role `0..3` inside its relabeled cell;
- leave learned parameters and transition-slot order unchanged.

With a nonzero diagnostic residual gate, maximum absolute logit difference
must be at most `1e-6` on both uKNIT-64 and Dialga-128 fixtures. Testing only
the zero gate is insufficient because it would exercise only K1-I.

## 5. Same-Checkpoint Controls

Train only the correct candidate. Strict-load the selected candidate state into
three geometry-identical controls and evaluate every split:

| Condition | Runtime intervention | Purpose |
|---|---|---|
| `exact_ordered` | both native matrices in native slot order | candidate |
| `operator_reversed` | exchange the two transition slots | test schedule semantics |
| `operator_corrupted` | fixed seed-`20260728` source-column permutation in both matrices | test edge/operator semantics |
| `no_topology` | replace both matrices by identity | remove runtime diffusion topology |

All four models must have identical state-dict shapes and hashes after strict
loading. Their operator, edge and Boolean-view fingerprints must be distinct.
No separately trained control may satisfy the attribution gate.

## 6. Zero-Training Readiness Gate

Before generating a training row, require all of the following:

1. the plan contains exactly uKNIT-BC r5 and Dialga-128 r4, seeds `0/1`, one
   candidate row each;
2. all frozen K1-I/K1-J digests and decisions match;
3. all twelve existing train/validation/fresh caches are present, digest-bound
   and reusable without regeneration;
4. candidate and all controls have identical parameter geometry on 64- and
   128-bit structures and strict-load the same state dict;
5. total trainable parameters are at most `200000` and independent of state
   width;
6. loading a K1-I base state and setting the residual gate to zero reproduces
   K1-I logits within `1e-7` on both widths;
7. both transition slots contain nonempty explicit edge sets, and masking
   either slot changes a nonzero-gate residual embedding by more than `1e-7`;
8. reversed, corrupted and no-topology structures each change a nonzero-gate
   residual logit by more than `1e-7`;
9. joint cell relabeling passes the nonzero-gate `1e-6` equivariance bound on
   both widths;
10. the implementation reports no absolute identity, cipher identity, S-box
    semantics, path token or raw bypass;
11. readiness consumes zero training rows and zero optimizer steps.

Any failure authorizes only repair of the failed implementation or source
binding. It does not authorize training, a weaker gate or a substitute model.

## 7. Fixed-Budget Local Diagnostic

After readiness passes, run exactly:

| Field | Frozen value |
|---|---|
| Ciphers / rounds | uKNIT-BC r5; Dialga-128 r4 |
| Candidate | K1-K exact GF(2) base plus topology edge residual |
| Anchor | completed K1-I exact Boolean-view rows |
| Seeds | `0`, `1` |
| Samples | `2048/class` train; `1024/class` fresh same-key; `1024/class` cross-key |
| Pairs per sample | `4` |
| Positive difference | uKNIT `0xD1`; Dialga `0x40` through the frozen profile |
| Negative definition | encrypted random plaintexts |
| Keys | frozen K1-I train and cross-key keys |
| Epochs | `10` |
| Batch size | `64` |
| Loss / optimizer | MSE / Adam |
| Learning rate / weight decay | `1e-4` / `1e-5` |
| Checkpoint | best validation AUC, restored before all controls |
| Device | local CPU |

The K1-I caches must be reused. This is a small local diagnostic, not formal
training, paper-scale evidence, an attack or a cipher-family conclusion.

## 8. Frozen Advance Gate

Apply every threshold separately to both seeds and both fresh splits. No mean
may hide a failed row.

For uKNIT-BC:

```text
candidate AUC                         >= 0.520
candidate AUC - same-row K1-I AUC    >= +0.005
candidate AUC - every topology control >= +0.005
```

For Dialga-128:

```text
candidate AUC - same-row K1-I AUC    >= -0.005
candidate AUC - every topology control >= +0.005
```

Training-split rows are descriptive only. The gate must also require exact
checkpoint reuse, identical datasets per seed/split, finite metrics and every
readiness/protocol check.

## 9. Decisions And Next Actions

- **All gates pass:** keep K1-K and create a separate remote
  `65536/class` disk-cached diagnostic plan. Do not call that formal evidence.
- **Dialga retained and operator controls pass, but uKNIT fails:** keep the
  residual only as correct-operator calibration evidence; do not scale it.
  The next single-variable route must add exact heterogeneous S-box/operator
  composition while retaining the K1-K controls.
- **Dialga retained but operator controls fail:** hold K1-K as another
  topology-insensitive route and audit residual-gate magnitude plus edge
  attribution before redesign.
- **Dialga anchor is lost:** discard the residual implementation and return to
  K1-I; do not compensate with data, width or epochs.
- **Protocol invalid:** repair only the failed binding or implementation and
  rerun unchanged.

Until every local gate passes, do not launch remote training, add samples,
epochs, width, pairs, seeds, MoE experts, S-box/DDT/trail features, keys, cipher
IDs, partial decryption or another raw branch.

## 10. Run IDs And Required Artifacts

```text
readiness_run_id = i1_uknit_family_ctspn_topology_edge_residual_k1k_readiness_20260728
training_run_id  = i1_uknit_family_ctspn_topology_edge_residual_k1k_2048_seed0_seed1_20260728
```

Readiness output:

```text
outputs/local_readiness/<readiness_run_id>/
  preflight.json
  results.jsonl
  validation.json
  gate.json
  progress.jsonl
```

Training output:

```text
outputs/local_diagnostic/<training_run_id>/
  preflight.json
  dataset_manifest.jsonl
  results.jsonl
  controls.jsonl
  split_attribution.csv
  checkpoint_manifest.json
  history.csv
  summary.json
  validation.json
  gate.json
  progress.jsonl
  curves.svg
  plot_report.json
  visual_qa_passed.marker
  checkpoints/
```

After each completed result-producing phase, refresh
`outputs/00_RECENT_RESULTS.md` and JSON and verify the expected run ID is the
newest indexed result. The completed training record must append exact metrics,
claim scope and an evidence-backed next action to this document.

## 11. Completed Readiness And Diagnostic

The zero-training readiness gate completed before training:

```text
run_id   = i1_uknit_family_ctspn_topology_edge_residual_k1k_readiness_20260728
status   = pass
decision = innovation1_uknit_family_ctspn_k1k_execution_authorized
protocol checks = all pass
training rows   = 0
optimizer steps = 0
```

The candidate had exactly `128707` learned parameters on both 64- and 128-bit
states. A loaded K1-I base with a zero residual gate reproduced K1-I logits
exactly on both widths. Nonzero-gate joint cell relabeling changed logits by at
most `1.35e-7`; both transition slots and all three wrong-topology controls
changed the residual output. Readiness therefore authorized the frozen local
training protocol without weakening a gate.

The training run then completed four candidate rows and sixty strict-loaded,
zero-step evaluation rows:

```text
run_id   = i1_uknit_family_ctspn_topology_edge_residual_k1k_2048_seed0_seed1_20260728
status   = hold
decision = innovation1_uknit_family_ctspn_k1k_dialga_retained_operator_attribution_not_supported
training rows   = 4 / 4
evaluation rows = 60 / 60
validation errors = []
```

All twelve frozen data caches were reused by exact digest. No training,
validation or fresh-split cache was regenerated.

## 12. Fresh-Split Results

| Cipher | Seed | Split | K1-K AUC | K1-I AUC | Delta vs K1-I | Weakest correct-vs-control margin |
|---|---:|---|---:|---:|---:|---:|
| uKNIT-BC r5 | 0 | same-key fresh | `0.506373` | `0.508171` | `-0.001797` | `-0.019545` |
| uKNIT-BC r5 | 0 | cross-key | `0.514879` | `0.513726` | `+0.001153` | `+0.008578` |
| uKNIT-BC r5 | 1 | same-key fresh | `0.484057` | `0.495883` | `-0.011827` | `-0.028407` |
| uKNIT-BC r5 | 1 | cross-key | `0.505805` | `0.504108` | `+0.001697` | `+0.009770` |
| Dialga-128 r4 | 0 | same-key fresh | `0.965747` | `0.965361` | `+0.000386` | `+0.002356` |
| Dialga-128 r4 | 0 | cross-key | `0.957891` | `0.958006` | `-0.000114` | `-0.002029` |
| Dialga-128 r4 | 1 | same-key fresh | `0.958886` | `0.958613` | `+0.000273` | `+0.004170` |
| Dialga-128 r4 | 1 | cross-key | `0.954312` | `0.954252` | `+0.000060` | `+0.002865` |

The uKNIT training-split AUC increased from K1-I's `0.529811/0.545691` to
`0.557716/0.584099`, and both cross-key rows beat the same-checkpoint controls.
Neither cross-key row reached the frozen `0.520` floor or the required
`+0.005` K1-I improvement. More importantly, both same-key-fresh rows remained
near chance and failed at least one control. The apparent training improvement
is therefore split-specific fitting, not uKNIT generalization.

Every Dialga fresh row retained K1-I within `0.005`, and candidate AUC remained
`0.954312-0.965747`. However, no fresh row achieved the required `+0.005`
margin over every reversed, corrupted and no-topology control. The weakest
margin ranged from `-0.002029` to `+0.004170`. K1-K therefore preserved the
strong global Dialga signal without establishing correct-operator attribution.

The rendered Chinese chart passed `visual-qa-redraw` at its native
`1536x979` size and an enlarged `1920x1224` inspection size. It has no text
overlap, clipping, missing glyph, ambiguous title or curve-separation failure.
The completed run is entry `001` in both recent-result indexes.

## 13. Claim Scope And Decision

K1-K establishes only that a bounded position-preserving edge residual can be
added without destroying K1-I's Dialga signal. It does not establish that the
learned residual uses the correct runtime operators, and it provides no uKNIT
generalization evidence.

This remains a local two-seed `2048/class` mechanism diagnostic with
`1024/class` fresh same-key and cross-key splits. It is not formal scale,
paper-scale evidence, an attack, a SOTA result, arbitrary-SPN transfer evidence
or an uKNIT architecture ceiling.

The frozen decision tree selects:

```text
Dialga anchor retained + operator controls failed
-> hold K1-K
-> no remote scale
-> audit learned residual-gate magnitude and exact edge attribution
```

## 14. Executable Next Action: K1-L Residual Attribution Audit

K1-L must be a zero-training audit over the four frozen K1-K checkpoints. Its
question is whether K1-K failed because the learned residual gate remained too
small, because the edge branch carried no label-associated information, or
because it carried information that was insensitive to the correct operator.

- **Same-budget anchor:** the exact four K1-K candidate checkpoints and their
  K1-I base embeddings on the existing train-seen, same-key-fresh and cross-key
  caches.
- **Required controls:** exact, reversed, corrupted and no-topology structures;
  residual gate forced to zero; transition slot 0 masked; transition slot 1
  masked; deterministic label-blind edge-message row permutation.
- **One variable:** evaluation-time exposure and intervention of the trained
  residual path. No parameter, data, label, key, metric or checkpoint changes.
- **Scale:** zero new samples, zero epochs and zero optimizer steps; local CPU.
- **Readiness gate:** exact checkpoint/cache digests, strict state equality,
  deterministic interventions, finite outputs and exact replay of all K1-K
  source AUCs within `1e-7`.
- **Advance decision:** if the gate magnitude and residual norm are nontrivial
  and exact edges uniquely carry fresh-split label association on Dialga,
  repair only residual fusion/optimization before reconsidering architecture.
  If the residual is active but wrong operators retain the same contribution,
  stop pure linear-edge redesign and make the next single variable exact
  heterogeneous S-box/operator composition for uKNIT. If the residual remains
  effectively closed, test a bounded gate-opening schedule locally while
  retaining all K1-K controls.
- **Stop rules:** no remote scale, more samples, epochs, pairs, seeds, width,
  MoE, cipher IDs, keys, DDTs, trails, partial decryption or raw bypass before
  K1-L identifies which mechanism failed.
