# Innovation 1 uKNIT U2-D Runtime Inverse-S-box Triplet Plan

Date: 2026-07-24

## Status

```text
stage    = completed
run_id   = i1_rtg1_uknit64_runtime_e4_inverse_sbox_triplet_u2d_2048_seed0_seed1_20260724
training = local diagnostic
decision = hold / correct ownership identified, no anchor improvement
```

## Question

Does executing the externally supplied runtime S-box as a deterministic
inverse operator make correct uKNIT cell ownership useful and stable across
two seeds?

U2-C retained `(C, C', C xor C')` and improved numerically over the
difference-only anchor on both seeds, but shuffled S-box ownership remained
slightly better. U1 through U2-C encode S-box truth tables as neural metadata;
they never apply those tables to the observed cell values. U2-D changes that
single mechanism.

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

Reuse the exact U2-B/U2-C disk cache. This is a local mechanism diagnostic,
not formal, paper-scale, attack, SOTA, cross-cipher or breakthrough evidence.

## One Operator Variable

Add `cell_input_mode=inverse_sbox_triplet`:

1. Preserve the unchanged current-state triplet token from U2-C.
2. Apply the exact last runtime inverse linear matrix separately to `C` and
   `C'`.
3. Reconstruct each runtime 4-bit S-box table from `sbox_truth_bits`, invert
   it exactly, and apply the inverse table to the corresponding 4-bit cell.
4. Recompute the previous-state difference from the two inverse-S-box
   endpoints rather than applying an S-box to an XOR difference.
5. Encode the two endpoints and recomputed difference with the unchanged
   shared cell encoder and symmetric triplet aggregation.
6. Leave `typed_fusion`, edge gate, E4 mixers, pooling and classifier
   unchanged.

For uKNIT prefix rounds, `C = L(S(X xor K))`; therefore
`S_inverse(L_inverse(C)) = X xor K`, and XORing the two recovered endpoints
cancels the unknown round key. No guessed subkey or secret value enters the
model.

## Six Rows

For each seed:

| Role | Runtime structure | S-box context | Cell input |
| --- | --- | --- | --- |
| candidate | correct | `edge_gate` | `inverse_sbox_triplet` |
| same-budget anchor | correct | `edge_gate` | `state_triplet` |
| ownership control | S-box assignment shuffled only | `edge_gate` | `inverse_sbox_triplet` |

Matrix:

```text
configs/experiment/innovation1/innovation1_spn_uknit64_runtime_e4_inverse_sbox_triplet_u2d_2048_seed0_seed1.csv
```

## Readiness Gate

Before training require:

1. Runtime inverse S-box lookup exactly reverses every table for every loaded
   round and cell.
2. Applying forward S-box after the inverse operator reconstructs all tested
   cell values.
3. Correct inverse-S-box triplet matches a direct cipher-side round replay on
   deterministic uKNIT states.
4. Candidate, anchor and control have identical state-dict geometry.
5. Candidate forward/backward is finite, pair-swap invariant and jointly
   cell-relabel equivariant.
6. Correct and shuffled S-box ownership produce different previous-state
   triplet tokens and logits under the same weights.
7. The six-row plan parses/builds and all PRESENT, GIFT, SKINNY and uKNIT
   runtime regressions pass.

Any failure blocks training without changing the frozen matrix.

## Research Gate

For each seed require:

```text
candidate AUC >= 0.520
candidate AUC - state-triplet anchor AUC >= +0.005
candidate AUC - shuffled inverse-S-box candidate AUC >= +0.005
```

Both seeds must pass. A pass opens a same-checkpoint correct-versus-shuffled
operator audit before scale. A miss closes this exact inverse-S-box triplet
design.

## Execution And Outputs

```text
output root = outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_inverse_sbox_triplet_u2d_2048_seed0_seed1_20260724
cache root  = outputs/local_cache/i1_rtg1_uknit64_runtime_e4_sbox_edge_gate_u2b_2048_seed0_seed1_20260724
```

Require `results.jsonl`, `progress.jsonl`, checkpoints, `validation.json`,
`gate.json`, `summary.json`, `history.csv`, `curves.svg` and
`visual_qa_passed.marker`, then refresh both recent-result indexes.

## Blocked Routes

Do not add DDT, trail features, guessed partial decryption, Conv2D, auxiliary
losses, more pairs, another difference/window, more seeds/epochs, larger data
or remote GPU inside U2-D. Do not compare its AUC directly with Liu et al.

## Completed Result

The six-row local run completed from local commit `6a309dd7`. The normal
GitHub push first failed because the sandbox could not reach the server; the
required elevated push was then rejected by the platform reviewer because the
external repository ownership/privacy could not be established. No alternate
transfer route or remote execution was used.

Plan alignment passed with six expected and observed rows, no missing,
unexpected or duplicate keys, and no field mismatches. Every protocol check
passed, including disk-backed cache reuse, strict encrypted-random-plaintext
negatives, the exact uKNIT descriptor window, and equal `442466`-parameter
geometry.

| Seed | Correct inverse-S-box triplet | State-triplet anchor | Shuffled inverse-S-box triplet | Candidate - anchor | Candidate - shuffled |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.523654938 | 0.535902977 | 0.514563084 | -0.012248039 | +0.009091854 |
| 1 | 0.540867805 | 0.538259983 | 0.500539780 | +0.002607822 | +0.040328026 |

Both candidates exceeded the absolute `0.520` floor, and correct ownership
beat shuffled ownership by more than `+0.005` on both seeds. This is the first
uKNIT local panel in this route where executing the correct runtime S box made
the assignment direction stable across both seeds. However, neither seed met
the required `+0.005` improvement over the frozen state-triplet anchor: seed0
lost `0.012248039`, while seed1 gained only `0.002607822`. The joint decision
is therefore:

```text
status   = hold
decision = innovation1_uknit_inverse_sbox_triplet_hold
keep     = runtime inverse-S-box execution as a real ownership-sensitive mechanism
reject   = replacing the state-triplet previous token with this inverse view
```

This local `2048/class` mechanism diagnostic supports correct S-box ownership
attribution only inside the frozen uKNIT prefix-r4 panel. It is not formal,
paper-scale, an attack, cross-cipher evidence, SOTA, or a breakthrough. Close
this exact replacement design; do not increase samples, epochs, seeds or use a
remote GPU.

Artifacts:

```text
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_inverse_sbox_triplet_u2d_2048_seed0_seed1_20260724/results.jsonl
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_inverse_sbox_triplet_u2d_2048_seed0_seed1_20260724/progress.jsonl
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_inverse_sbox_triplet_u2d_2048_seed0_seed1_20260724/validation.json
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_inverse_sbox_triplet_u2d_2048_seed0_seed1_20260724/gate.json
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_inverse_sbox_triplet_u2d_2048_seed0_seed1_20260724/summary.json
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_inverse_sbox_triplet_u2d_2048_seed0_seed1_20260724/history.csv
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_inverse_sbox_triplet_u2d_2048_seed0_seed1_20260724/curves.svg
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_inverse_sbox_triplet_u2d_2048_seed0_seed1_20260724/visual_qa_passed.marker
```

The validation-only SVG passed `visual-qa-redraw` after an exact
`2167 x 986` pixel render. The Chinese title and six role labels are readable;
the legend, AUC curves, 50% reference line and summary table have no overlap,
clipping, missing glyphs or ambiguous axes.

## Evidence-Backed Next Action: U2-E Dual-View Triplet Fusion

U2-C retained stronger raw state-triplet performance, while U2-D made correct
S-box ownership consistently identifiable. The next bounded hypothesis is to
retain both signals rather than replacing one with the other:

```text
question          = can a parameter-free dual view retain U2-C signal and U2-D ownership attribution?
one variable      = previous state triplet -> symmetric mean of state and inverse-S-box triplets
candidate         = correct topology + dual-view triplet + unchanged edge gate
same-budget anchor= correct topology + U2-C state triplet + unchanged edge gate
required control  = shuffled S-box ownership + dual-view triplet + unchanged edge gate
cipher/window     = uKNIT-BC prefix r4, round_start 2, processor_steps 2
scale             = 2048/class train, 1024/class validation
seeds/epochs      = 0,1 / 10
pairs/sample      = 4
negative          = encrypted_random_plaintexts
execution         = local CPU diagnostic with the existing disk cache
```

Use the same shared cell encoder for both previous views and average their
embeddings before the unchanged typed fusion, so candidate, anchor and control
retain identical parameter geometry. Before training, require exact U2-C and
U2-D branch compatibility, finite forward/backward, pair-swap invariance,
joint cell-relabel equivariance, correct-versus-shuffled token sensitivity and
a six-row plan-alignment check.

For each seed require candidate AUC `>= 0.520`, candidate-minus-state-triplet
anchor `>= +0.005`, and candidate-minus-shuffled dual-view `>= +0.005`. Both
seeds must pass before any checkpoint audit or scale discussion. A miss closes
the dual-view design. Do not add learned fusion parameters, DDT/trail features,
partial-decryption guesses, more pairs, another window, additional epochs,
larger data or remote GPU inside U2-E.
