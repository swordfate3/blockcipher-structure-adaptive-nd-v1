# Innovation 1 uKNIT U2-B S-box/Topology Edge Gate Plan

Date: 2026-07-24

## Status

```text
stage    = preregistered
run_id   = i1_rtg1_uknit64_runtime_e4_sbox_edge_gate_u2b_2048_seed0_seed1_20260724
training = local diagnostic
decision = pending
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
