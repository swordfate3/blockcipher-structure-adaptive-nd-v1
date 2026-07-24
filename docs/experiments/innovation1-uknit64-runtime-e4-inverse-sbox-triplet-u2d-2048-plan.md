# Innovation 1 uKNIT U2-D Runtime Inverse-S-box Triplet Plan

Date: 2026-07-24

## Status

```text
stage    = preregistered
run_id   = i1_rtg1_uknit64_runtime_e4_inverse_sbox_triplet_u2d_2048_seed0_seed1_20260724
training = local diagnostic
decision = pending
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
