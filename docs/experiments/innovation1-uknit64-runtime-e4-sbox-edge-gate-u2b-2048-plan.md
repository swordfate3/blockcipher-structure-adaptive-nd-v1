# Innovation 1 uKNIT U2-B S-box/Topology Edge Gate Plan

Date: 2026-07-24

## Status

```text
stage    = completed
run_id   = i1_rtg1_uknit64_runtime_e4_sbox_edge_gate_u2b_2048_seed0_seed1_20260724
training = local diagnostic
decision = hold / close parameter-free edge gate
```

## Question

Can one parameter-geometry-preserving interaction between per-cell S-box
identity and the exact runtime linear graph make correct uKNIT ownership a
stable advantage over both the U1 anchor and shuffled ownership?

U1 and U2-A reject only additive post-mixer `late_cell` injection. U2-A proved
that its seed1 checkpoint used ownership (`+0.019989` AUC) while seed0 did not
(`+0.000217`). U2-B changes one mechanism: move S-box information before the
E4 mixer and multiply it with messages defined by the runtime linear topology.

## Frozen Protocol

```text
cipher          = uKNIT-BC prefix r4
runtime window  = round_start 2, processor_steps 2
train           = 2048/class
validation      = 1024/class
seeds           = 0,1
epochs          = 10
pairs/sample    = 4
feature         = ciphertext_pair_bits
negative        = encrypted_random_plaintexts
train key       = 0x00000000000000000000000000000000
validation key  = 0x11111111111111111111111111111111
loss            = MSE
optimizer       = Adam, lr 1e-4, weight decay 1e-5
checkpoint      = best validation AUC
device          = local CPU
```

The data, labels, validation set, metric, checkpoint rule, keys and uKNIT
window are unchanged from U1. This is a local `2048/class` diagnostic, not
formal, paper-scale, attack, cross-cipher or breakthrough evidence.

## One Architectural Variable

Add `sbox_context_mode=edge_gate` to the existing runtime E4 backbone:

1. Aggregate the active round's exact bit-level inverse GF(2) matrix into a
   normalized cell-to-cell adjacency using runtime cell membership.
2. Propagate the current cell tokens along this adjacency.
3. Propagate the encoded source-cell S boxes along the same adjacency.
4. Form an elementwise sigmoid gate from source-neighbor and target-cell S-box
   encodings, then add `gate * graph_message` before the unchanged E4 mixer.

Use only the existing S-box encoder and runtime tensors. Add no trainable layer,
position table or cipher-sized parameter, so `edge_gate`, `late_pair`, correct
and shuffled rows retain identical parameter geometry across SPNs. The same
operation must accept both permutation matrices and general reversible GF(2)
linear matrices.

## Six Rows

For each seed:

| Role | Runtime structure | Context mode |
| --- | --- | --- |
| candidate | correct S2/L2, S3/L3 | `edge_gate` |
| anchor | correct S2/L2, S3/L3 | `late_pair` |
| assignment control | exact L2/L3, shuffled S-box ownership | `edge_gate` |

Matrix:

```text
configs/experiment/innovation1/innovation1_spn_uknit64_runtime_e4_sbox_edge_gate_u2b_2048_seed0_seed1.csv
```

## Readiness Gate

Before the six-row run require:

1. `edge_gate` forward/backward is finite on a tiny batch.
2. Correct and shuffled ownership produce different logits.
3. Joint cell relabeling of inputs, topology and S boxes preserves logits.
4. Candidate, anchor and shuffled control have identical parameter geometry.
5. Existing PRESENT, GIFT, SKINNY and uKNIT runtime tests remain green.
6. The six-row matrix passes plan parsing and model-construction validation.

Any failure blocks training and must be repaired without changing this matrix.

## Research Gate

For each seed require all three:

```text
candidate AUC >= 0.520
candidate AUC - late_pair anchor AUC >= +0.005
candidate AUC - shuffled edge_gate AUC >= +0.005
```

Advance only if both seeds pass. A pass opens a same-checkpoint ownership swap
audit of both best candidate checkpoints before any scale increase. A miss on
either seed closes this exact parameter-free edge-gate design; do not tune its
scale, add layers, increase samples/epochs or launch remote GPU in response.

## Execution And Outputs

```text
output root = outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_sbox_edge_gate_u2b_2048_seed0_seed1_20260724
cache root  = outputs/local_cache/i1_rtg1_uknit64_runtime_e4_sbox_edge_gate_u2b_2048_seed0_seed1_20260724
```

The result must contain `results.jsonl`, `progress.jsonl`, checkpoints,
`validation.json`, `gate.json`, `summary.json`, `history.csv`, `curves.svg` and
`visual_qa_passed.marker`. After completion refresh both recent-result indexes.

## Blocked Routes

Do not add DDT, trails, partial decryption, extra S-box losses, paired training,
another window, more seeds, more epochs, larger data, remote GPU, or a second
new architecture mechanism in U2-B. Those are separate hypotheses and would
make the result unattributable.

## Completed Result

The six-row run completed locally from pushed commit `69b99aec`. Plan
alignment passed with six expected and six observed rows, no missing or
unexpected rows, no duplicate keys and no field mismatches. All protocol
checks passed, including exact descriptor window, strict encrypted-random-
plaintext negatives, disk-backed datasets, equal parameter geometry and the
three required roles for both seeds.

| Seed | Correct edge gate | `late_pair` anchor | Shuffled edge gate | Candidate - anchor | Candidate - shuffled |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.532396317 | 0.518405437 | 0.526177406 | +0.013990879 | +0.006218910 |
| 1 | 0.521487236 | 0.538745403 | 0.530868053 | -0.017258167 | -0.009380817 |

Seed0 passed all three preregistered research checks. Seed1 reached the
absolute `0.520` AUC floor but lost to both the unchanged anchor and shuffled
ownership control. The joint decision is therefore:

```text
status   = hold
decision = innovation1_uknit_sbox_edge_gate_hold
claim    = local 2048/class mechanism diagnostic only
```

The result does not show that the runtime SPN backbone is ineffective. The
same backbone already has positive correct-versus-corrupted topology evidence
on PRESENT, GIFT and general-GF(2) SKINNY. It shows specifically that this
parameter-free use of truth-table embeddings as a linear-edge gate does not
make uKNIT cell-specific S-box ownership stable across two seeds. Do not tune,
scale or launch this exact edge-gate design remotely.

Artifacts:

```text
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_sbox_edge_gate_u2b_2048_seed0_seed1_20260724/results.jsonl
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_sbox_edge_gate_u2b_2048_seed0_seed1_20260724/progress.jsonl
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_sbox_edge_gate_u2b_2048_seed0_seed1_20260724/validation.json
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_sbox_edge_gate_u2b_2048_seed0_seed1_20260724/gate.json
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_sbox_edge_gate_u2b_2048_seed0_seed1_20260724/summary.json
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_sbox_edge_gate_u2b_2048_seed0_seed1_20260724/history.csv
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_sbox_edge_gate_u2b_2048_seed0_seed1_20260724/curves.svg
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_sbox_edge_gate_u2b_2048_seed0_seed1_20260724/visual_qa_passed.marker
```

`visual-qa-redraw` first found that training-set dashed curves competed with
the close validation controls. The final SVG uses validation-only curves and
passed a `2312 x 1646` rendered-pixel inspection: no overlap, clipping,
missing glyphs or ambiguous legend; the AUC range exposes the close controls.

## Evidence-Backed Next Action: U2-C State-Triplet Tokens

The next question is whether the ownership instability comes from E4
discarding the two ciphertext values and retaining only their XOR difference.
Liu et al. 2026 explicitly use the SPN state format
`(C_bar, C_bar_prime, delta_C_bar)` and a `3 x 4 x n/4` Conv2D tensor so the
network sees both state values and the penultimate-round-aligned difference.
U2-C should preserve that information while retaining runtime cell and linear
topology instead of introducing a cipher-sized Conv2D layout.

Freeze U2-C as follows before implementation:

```text
question          = do runtime state-triplet cell tokens stabilize true uKNIT S-box ownership?
one variable      = difference-only cell token -> shared (C, C', C xor C') cell token
candidate         = correct topology + state-triplet token + edge gate
same-budget anchor= correct topology + difference-only edge gate
required control  = shuffled S-box ownership + state-triplet token + edge gate
cipher/window     = uKNIT-BC prefix r4, round_start 2, processor_steps 2
scale             = 2048/class train, 1024/class validation
seeds/epochs      = 0,1 / 10
pairs/sample      = 4
negative          = encrypted_random_plaintexts
execution         = local CPU diagnostic
```

Reuse the same `cell_encoder`, `typed_fusion`, E4 blocks, pooling and
classifier so all U2-C roles retain the same parameter geometry. Apply the
runtime inverse linear operator separately to the two ciphertext values and
their XOR difference, encode the three cell values with the shared encoder,
and combine them symmetrically so swapping the pair order cannot change the
logit. Reuse the exact U2-B cache and validation arrays.

For each seed require candidate AUC `>= 0.520`, candidate-minus-anchor
`>= +0.005`, and candidate-minus-shuffled `>= +0.005`. Both seeds must pass
before any same-checkpoint audit. A miss closes this state-triplet-plus-edge-
gate combination. Do not add Conv2D, DDT, trails, partial decryption, more
pairs, more epochs, a new difference, another window, larger data or remote
GPU inside U2-C.
