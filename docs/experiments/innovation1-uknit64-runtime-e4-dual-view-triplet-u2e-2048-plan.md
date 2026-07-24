# Innovation 1 uKNIT U2-E Runtime Dual-View Triplet Plan

Date: 2026-07-24

## Status

```text
stage    = completed local diagnostic
run_id   = i1_rtg1_uknit64_runtime_e4_dual_view_triplet_u2e_2048_seed0_seed1_20260724
training = local diagnostic
decision = innovation1_uknit_dual_view_triplet_hold
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

## Completed Result

The six-row local CPU diagnostic completed from local commit `d7b6fe69`.
Plan/result validation passed with six expected and observed rows, no missing,
unexpected or duplicate keys, and no field mismatches. Every frozen protocol
check passed, including exact descriptor-window selection, strict encrypted
random plaintext negatives, disk-cache reuse, equal model geometry and the
three unique roles for both seeds.

| Seed | Correct dual view | State-triplet anchor | Shuffled dual view | Candidate - anchor | Candidate - shuffled |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.523506165 | 0.535902977 | 0.514215469 | -0.012396812 | +0.009290695 |
| 1 | 0.533703327 | 0.538259983 | 0.530916691 | -0.004556656 | +0.002786636 |

Both candidates exceeded the absolute `0.520` floor, but neither retained the
state-triplet anchor: seed0 lost `0.012396812` AUC and seed1 lost `0.004556656`.
Only seed0 met the `+0.005` correct-versus-shuffled margin; seed1 reached only
`+0.002786636`. The joint decision is therefore:

```text
status   = hold
decision = innovation1_uknit_dual_view_triplet_hold
keep     = the separate U2-C state-triplet anchor and U2-D ownership-sensitive inverse operator evidence
reject   = fixed 0.5/0.5 embedding-space fusion of those two views
```

This result does not show that runtime S-box tables are useless. U2-D remains
the relevant evidence that actually executing the correct per-cell inverse
S box can expose ownership. U2-E shows only that averaging that view into the
stronger state representation both dilutes the anchor and fails to retain a
stable attribution margin. Close this exact fusion; do not tune its weight,
increase samples or epochs, add seeds, or launch it remotely.

Artifacts:

```text
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_dual_view_triplet_u2e_2048_seed0_seed1_20260724/results.jsonl
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_dual_view_triplet_u2e_2048_seed0_seed1_20260724/progress.jsonl
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_dual_view_triplet_u2e_2048_seed0_seed1_20260724/validation.json
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_dual_view_triplet_u2e_2048_seed0_seed1_20260724/gate.json
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_dual_view_triplet_u2e_2048_seed0_seed1_20260724/summary.json
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_dual_view_triplet_u2e_2048_seed0_seed1_20260724/history.csv
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_dual_view_triplet_u2e_2048_seed0_seed1_20260724/curves.svg
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_dual_view_triplet_u2e_2048_seed0_seed1_20260724/visual_qa_passed.marker
```

The validation-only SVG passed `visual-qa-redraw` after rendering the exact
artifact at `2167 x 986` pixels. The Chinese title, subtitle, six-series legend,
zoomed AUC axis and validation summary table are readable with no overlap,
clipping, missing glyphs or ambiguous scale.

## Evidence-Backed Next Action: U2-F Runtime S-box Query Token

U2-C shows that the raw previous-state triplet is the stronger representation.
U2-D shows that querying the correct runtime inverse S box gives a real
assignment-sensitive signal. U2-E shows that compressing those views by a
fixed mean loses both properties. The next distinct representation should keep
the complete U2-C path unchanged and expose only the key-cancelling inverse
input difference as a separate sample-conditioned query token:

```text
V       = L_inverse(C)
V'      = L_inverse(C')
deltaV  = V xor V'
deltaU  = S_inverse(V) xor S_inverse(V')
```

Freeze U2-F before implementation:

```text
question          = does an explicit runtime deltaU query token add stable information without replacing the state anchor?
one variable      = capacity-matched third query token: deltaV duplicate -> runtime inverse-S-box deltaU
candidate         = correct topology + unchanged state triplets + correct deltaU query
same-budget anchor= correct topology + unchanged state triplets + deltaV identity query
required control  = shuffled S-box ownership + unchanged state triplets + shuffled deltaU query
cipher/window     = uKNIT-BC prefix r4, round_start 2, processor_steps 2
scale             = 2048/class train, 1024/class validation
seeds/epochs      = 0,1 / 10
pairs/sample      = 4
negative          = encrypted_random_plaintexts
execution         = local CPU diagnostic with the existing disk cache
```

All three rows must use the same three-input fusion geometry and parameter
count. The candidate may not average, replace or otherwise change the U2-C
current/previous state-triplet inputs. Before training, require fixed-input
proof that the identity anchor exactly bypasses S-box lookup, the candidate
changes under ownership shuffle, old modes remain numerically unchanged, and
pair swap plus joint cell relabeling invariance still hold.

For each seed require candidate AUC `>= 0.520`, candidate-minus-identity-anchor
`>= +0.005`, and candidate-minus-shuffled-query `>= +0.005`. Both seeds must
pass before any checkpoint audit or scale discussion. A miss closes the query
token representation. Do not add DDT/trail probabilities, guessed subkeys,
more pairs, another difference/window, extra epochs, larger data or remote GPU
inside U2-F.
