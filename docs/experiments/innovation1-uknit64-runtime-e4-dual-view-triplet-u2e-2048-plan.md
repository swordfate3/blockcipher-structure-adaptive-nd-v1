# Innovation 1 uKNIT U2-E Runtime Dual-View Triplet Plan

Date: 2026-07-24

## Status

```text
stage    = preregistered
run_id   = i1_rtg1_uknit64_runtime_e4_dual_view_triplet_u2e_2048_seed0_seed1_20260724
training = local diagnostic
decision = pending
```

## Question

Can a parameter-free fusion of the raw state triplet and deterministic inverse
S-box triplet retain U2-C's stronger signal while preserving U2-D's stable
correct-ownership attribution?

U2-C's state-triplet candidate beat the difference-only anchor numerically on
both seeds, but shuffled S-box ownership was slightly better. U2-D reversed
that attribution failure: correct inverse-S-box ownership beat shuffled on both
seeds by `+0.009091854` and `+0.040328026`. Replacing the state view lost
`0.012248039` against the anchor at seed0 and gained only `0.002607822` at
seed1. U2-E retains both views and changes no learned parameter.

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

Reuse the exact U2-B through U2-D disk cache. This is a local mechanism
diagnostic, not formal, paper-scale, an attack, SOTA, cross-cipher evidence or
a breakthrough.

## One Representation Variable

Add `cell_input_mode=dual_view_triplet`:

1. Preserve the unchanged U2-C current-state `(C, C', C xor C')` token.
2. Build the unchanged U2-C previous-state triplet after exact inverse linear
   transformation.
3. Build the unchanged U2-D previous-state triplet after exact inverse linear
   and per-cell inverse S-box transformation, recomputing the endpoint XOR.
4. Encode both previous triplets with the same shared `cell_encoder`.
5. Average the two previous-view embeddings with fixed weight `0.5`; do not add
   a learned gate, projection, scalar or auxiliary loss.
6. Leave `typed_fusion`, S-box edge gate, E4 mixers, pooling and classifier
   unchanged.

The `difference_only`, `state_triplet` and `inverse_sbox_triplet` paths must
remain numerically unchanged under fixed weights and inputs. All four modes
must have identical state-dict geometry. The dual view must preserve pair-swap
invariance and joint cell-relabel equivariance, and independent mode must
bypass both runtime inverse operations.

## Six Rows

For each seed:

| Role | Runtime structure | S-box context | Cell input |
| --- | --- | --- | --- |
| candidate | correct | `edge_gate` | `dual_view_triplet` |
| same-budget anchor | correct | `edge_gate` | `state_triplet` |
| ownership control | S-box assignment shuffled only | `edge_gate` | `dual_view_triplet` |

Matrix:

```text
configs/experiment/innovation1/innovation1_spn_uknit64_runtime_e4_dual_view_triplet_u2e_2048_seed0_seed1.csv
```

## Readiness Gate

Before training require:

1. Fixed-input logits for the three existing modes remain unchanged.
2. Dual-view previous tokens equal the exact mean of state and inverse-S-box
   previous tokens under shared weights.
3. Candidate, anchor and control have identical state-dict geometry and
   `442466` parameters.
4. Candidate forward/backward is finite, pair-swap invariant and jointly
   cell-relabel equivariant.
5. Correct and shuffled S-box ownership produce different dual-view previous
   tokens and logits under the same weights.
6. Independent mode executes neither inverse linear nor inverse S-box
   operations.
7. The six-row plan parses/builds and all PRESENT, GIFT, SKINNY and uKNIT
   runtime regressions pass.

Any failure blocks training without changing the frozen matrix.

## Research Gate

For each seed require:

```text
candidate AUC >= 0.520
candidate AUC - state-triplet anchor AUC >= +0.005
candidate AUC - shuffled dual-view candidate AUC >= +0.005
```

Both seeds must pass. A pass opens a same-checkpoint correct-versus-shuffled
view audit before scale. A miss closes this exact dual-view fusion design.

## Execution And Outputs

```text
output root = outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_dual_view_triplet_u2e_2048_seed0_seed1_20260724
cache root  = outputs/local_cache/i1_rtg1_uknit64_runtime_e4_sbox_edge_gate_u2b_2048_seed0_seed1_20260724
```

Require `results.jsonl`, `progress.jsonl`, checkpoints, `validation.json`,
`gate.json`, `summary.json`, `history.csv`, `curves.svg` and
`visual_qa_passed.marker`, then refresh both recent-result indexes.

## Blocked Routes

Do not tune the fusion weight, add learned fusion parameters, DDT/trail
features, partial-decryption guesses, more pairs, another difference/window,
extra seeds/epochs, larger data or remote GPU inside U2-E. Do not compare its
AUC directly with Liu et al.
