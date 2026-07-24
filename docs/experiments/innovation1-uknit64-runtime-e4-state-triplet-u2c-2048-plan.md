# Innovation 1 uKNIT U2-C Runtime State-Triplet Token Plan

Date: 2026-07-24

## Status

```text
stage    = completed
run_id   = i1_rtg1_uknit64_runtime_e4_state_triplet_u2c_2048_seed0_seed1_20260724
training = local diagnostic
decision = hold / retain representation lead, reject S-box ownership claim
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

## Completed Result

The six-row local run completed from pushed commit `2b107caf`. Plan alignment
passed with six expected and observed rows, no missing, unexpected or
duplicate keys, and no field mismatches. Every protocol check passed,
including disk-backed cache reuse, strict encrypted-random-plaintext
negatives, exact uKNIT descriptor window, equal `442466`-parameter geometry
and the three frozen roles for both seeds.

| Seed | Correct state triplet | Difference-only anchor | Shuffled state triplet | Candidate - anchor | Candidate - shuffled |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.535902977 | 0.532396317 | 0.537521839 | +0.003506660 | -0.001618862 |
| 1 | 0.538259983 | 0.521487236 | 0.540617943 | +0.016772747 | -0.002357960 |

Both state-triplet candidates exceeded the absolute `0.520` floor and both
improved numerically over their difference-only anchors. Seed0 missed the
pre-registered `+0.005` anchor margin, while seed1 passed it. More
importantly, shuffled ownership was slightly better than correct ownership on
both seeds. The joint decision is:

```text
status   = hold
decision = innovation1_uknit_state_triplet_hold
keep     = state-triplet representation as a promising generic input lead
reject   = stable correct S-box ownership attribution
```

This result supports the literature-motivated observation that retaining the
two state values can be more informative than an XOR-only bottleneck. It does
not support the current truth-table-embedding edge gate: the gain is not tied
to correct uKNIT S-box ownership. Do not scale, tune or remotely launch this
exact state-triplet-plus-edge-gate combination.

Artifacts:

```text
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_state_triplet_u2c_2048_seed0_seed1_20260724/results.jsonl
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_state_triplet_u2c_2048_seed0_seed1_20260724/progress.jsonl
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_state_triplet_u2c_2048_seed0_seed1_20260724/validation.json
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_state_triplet_u2c_2048_seed0_seed1_20260724/gate.json
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_state_triplet_u2c_2048_seed0_seed1_20260724/summary.json
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_state_triplet_u2c_2048_seed0_seed1_20260724/history.csv
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_state_triplet_u2c_2048_seed0_seed1_20260724/curves.svg
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_state_triplet_u2c_2048_seed0_seed1_20260724/visual_qa_passed.marker
```

The final validation-only SVG passed `visual-qa-redraw` at `2312 x 1646`
rendered pixels: no text overlap, clipping, missing glyphs, ambiguous labels
or unreadable table; the AUC scale exposes the close controls.

## Evidence-Backed Next Action: U2-D Runtime Inverse-S-box Operator

U1 through U2-C encode each S-box truth table as metadata, but never execute
the S-box as the known cipher operation. The closest SPN literature obtains
penultimate-round-aligned values through deterministic inverse linear and
inverse substitution operations. uKNIT prefix rounds have the public form
`C = L(S(X xor K))`, so the exact runtime descriptor permits:

```text
S_inverse(L_inverse(C))  = X xor K
S_inverse(L_inverse(C')) = X' xor K
their XOR                = X xor X'
```

No secret key or guessed subkey is required for this transform.

Freeze U2-D before implementation:

```text
question          = does executing the runtime inverse S-box stabilize true uKNIT ownership?
one variable      = state-triplet previous token -> inverse-S-box state-triplet token
candidate         = correct topology + inverse-S-box triplet + unchanged edge gate
same-budget anchor= correct topology + U2-C state triplet + unchanged edge gate
required control  = shuffled S-box ownership + inverse-S-box triplet + unchanged edge gate
cipher/window     = uKNIT-BC prefix r4, round_start 2, processor_steps 2
scale             = 2048/class train, 1024/class validation
seeds/epochs      = 0,1 / 10
pairs/sample      = 4
negative          = encrypted_random_plaintexts
execution         = local CPU diagnostic
```

Implement the inverse lookup as a parameter-free runtime operator derived
from `sbox_truth_bits`; do not add a learned layer. Keep the current-state
triplet branch unchanged, and replace only the previous-state triplet with
the two inverse-S-box endpoint values and their recomputed XOR. Candidate,
anchor and shuffled control must retain identical state-dict geometry, pair
swap invariance and cell-relabel equivariance.

For each seed require candidate AUC `>= 0.520`, candidate-minus-anchor
`>= +0.005`, and candidate-minus-shuffled `>= +0.005`. Both seeds must pass
before a same-checkpoint operator audit. A miss closes this inverse-S-box
operator design. Do not add DDT, trails, partial decryption guesses, Conv2D,
more pairs, another difference/window, extra epochs, larger data or remote GPU
inside U2-D.
