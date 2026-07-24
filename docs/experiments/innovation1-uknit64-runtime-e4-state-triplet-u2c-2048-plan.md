# Innovation 1 uKNIT U2-C Runtime State-Triplet Token Plan

Date: 2026-07-24

## Status

```text
stage    = preregistered
run_id   = i1_rtg1_uknit64_runtime_e4_state_triplet_u2c_2048_seed0_seed1_20260724
training = local diagnostic
decision = pending
```

## Question

Can a runtime, cipher-width-independent version of the SPN state-triplet input
stabilize cell-specific uKNIT S-box ownership across two seeds?

U2-B showed a split result: the parameter-free edge gate beat its anchor and
shuffled ownership at seed0, but lost to both at seed1. The current E4 frontend
discards the two ciphertext values and retains only their XOR difference.
Liu et al. 2026 instead preserve `(C_bar, C_bar_prime, delta_C_bar)` in a
three-channel SPN state tensor. U2-C changes only that information bottleneck,
while retaining the runtime cell graph rather than introducing a fixed-size
Conv2D layout.

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

Reuse the U2-B disk cache exactly. The data, labels, keys, difference,
validation arrays, metric, checkpoint selection and training budget are
frozen. This is a local diagnostic, not formal, paper-scale, attack,
cross-cipher, SOTA or breakthrough evidence.

## One Representation Variable

Add `cell_input_mode=state_triplet` to the existing runtime E4 model:

1. Keep the ordered pair values `C` and `C'` instead of immediately
   discarding them after computing `C xor C'`.
2. Apply the exact runtime inverse linear matrix separately to `C`, `C'` and
   `C xor C'`.
3. Split all six states into runtime-defined 4-bit cells.
4. Reuse the same `cell_encoder` on each cell value.
5. Mean the two endpoint embeddings and then mean that result with the
   difference embedding. This preserves pair-swap invariance and introduces
   no trainable parameter.
6. Feed the resulting current/previous tokens into the unchanged
   `typed_fusion`, edge gate, E4 mixers, pooling and classifier.

The `difference_only` mode must remain byte-for-byte compatible with U2-B.
Both modes must have identical state-dict keys and tensor shapes. The new mode
must remain jointly cell-relabel equivariant and accept permutation or general
reversible GF(2) matrices.

## Six Rows

For each seed:

| Role | Runtime structure | S-box context | Cell input |
| --- | --- | --- | --- |
| candidate | correct | `edge_gate` | `state_triplet` |
| same-budget anchor | correct | `edge_gate` | `difference_only` |
| ownership control | S-box assignment shuffled only | `edge_gate` | `state_triplet` |

Matrix:

```text
configs/experiment/innovation1/innovation1_spn_uknit64_runtime_e4_state_triplet_u2c_2048_seed0_seed1.csv
```

## Readiness Gate

Before training require:

1. Six rows parse and build with equal parameter geometry.
2. `difference_only` logits remain identical to the pre-U2-C path under a
   fixed seed and input.
3. `state_triplet` forward/backward is finite.
4. Swapping `C` and `C'` leaves state-triplet logits unchanged.
5. Joint relabeling of input cells, topology and S boxes leaves logits
   unchanged.
6. State-triplet logits change when either the endpoint values or uKNIT S-box
   ownership changes while the XOR difference is held fixed.
7. Existing PRESENT, GIFT, SKINNY and uKNIT runtime regression tests pass.

Any failure blocks training and must be repaired without changing this matrix.

## Research Gate

For each seed require all three:

```text
candidate AUC >= 0.520
candidate AUC - difference-only edge-gate anchor AUC >= +0.005
candidate AUC - shuffled state-triplet edge-gate AUC >= +0.005
```

Both seeds must pass. A pass opens a same-checkpoint correct-versus-shuffled
ownership audit before any new scale. A miss closes this exact
state-triplet-plus-edge-gate combination; do not tune scale or architecture
after reveal.

## Execution And Outputs

```text
output root = outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_state_triplet_u2c_2048_seed0_seed1_20260724
cache root  = outputs/local_cache/i1_rtg1_uknit64_runtime_e4_sbox_edge_gate_u2b_2048_seed0_seed1_20260724
```

Require `results.jsonl`, `progress.jsonl`, checkpoints, `validation.json`,
`gate.json`, `summary.json`, `history.csv`, `curves.svg` and
`visual_qa_passed.marker`, then refresh both recent-result indexes.

## Blocked Routes

Do not add Conv2D, DDT, trails, partial decryption, auxiliary losses, more
pairs, another difference/window, more seeds/epochs, larger data or remote GPU
inside U2-C. Do not compare its AUC directly with Liu et al.; the cipher,
difference, rounds, negative definition, data size and architecture differ.
