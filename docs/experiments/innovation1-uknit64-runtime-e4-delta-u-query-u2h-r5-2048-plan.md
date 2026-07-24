# Innovation 1 uKNIT U2-H Runtime Delta-U Cross-Window Replication Plan

Date: 2026-07-24

## Status

```text
stage    = completed local diagnostic
run_id   = i1_rtg1_uknit64_runtime_e4_delta_u_query_u2h_r5_2048_seed0_seed1_20260724
training = completed from local source commit 9c73e2a9
decision = innovation1_uknit_delta_u_cross_window_not_replicated
```

## Research Question

Does the U2-F/U2-G runtime delta-U query mechanism replicate on the next
aligned uKNIT transition window, or is its support specific to the prefix-r4
window?

U2-F showed that a separately trained correct delta-U query beat a
capacity-matched delta-V query and shuffled S-box-ownership query on both
seeds. U2-G then showed that the same frozen checkpoints materially depend on
the correct query. U2-H changes only the cipher prefix and runtime window.

## Frozen Same-Budget Anchor

```text
source anchor    = U2-F prefix-r4, round_start 2, processor_steps 2
cipher           = uKNIT-BC prefix r5
runtime window   = round_start 3, processor_steps 2
train            = 2048/class
validation       = 1024/class
seeds            = 0,1
epochs           = 10
pairs/sample     = 4
input difference = 0x40
feature          = ciphertext_pair_bits
negative         = encrypted_random_plaintexts
train key        = 0x00000000000000000000000000000000
validation key   = 0x11111111111111111111111111111111
loss             = MSE
optimizer        = Adam, lr 1e-4, weight decay 1e-5
checkpoint       = best validation AUC
device           = local CPU
```

This is a local mechanism replication. It is not formal or paper-scale
evidence, an attack, SOTA, cross-cipher evidence, or a breakthrough.

## One Variable

Relative to U2-F, change only:

```text
cipher prefix        r4 -> r5
runtime_round_start  2  -> 3
```

The two-round runtime processor therefore remains aligned to the last two
available uKNIT transitions. The architecture and three roles remain:

| Role | Runtime structure | Third query token |
| --- | --- | --- |
| candidate | correct | `deltaU = S_inverse(V) xor S_inverse(V')` |
| same-budget anchor | correct | `deltaV = V xor V'` |
| ownership control | S-box assignment shuffled only | shuffled-ownership `deltaU` |

All six rows must have identical `458850`-parameter geometry. No U2-F cache is
reused because prefix-r5 ciphertexts are a different dataset; U2-H receives a
new parameter-matched disk cache.

Matrix:

```text
configs/experiment/innovation1/innovation1_spn_uknit64_runtime_e4_delta_u_query_u2h_r5_2048_seed0_seed1.csv
```

## Readiness Gate

Before training require:

1. The six-row matrix parses as seeds `0,0,0,1,1,1` and rounds `5`.
2. Every model loads descriptor rounds `3` and `4` with two processor steps.
3. Candidate, anchor and shuffled control have identical state-dict geometry
   and `458850` parameters.
4. Candidate and anchor preserve the two U2-C state-triplet inputs and differ
   only in the third query value.
5. The shuffled control changes only per-cell S-box ownership.
6. Strict encrypted-random-plaintext negatives, keys, input difference,
   samples, pairs, seeds, optimizer and checkpoint rule match U2-F.
7. A new disk cache records both train and validation progress.
8. Existing uKNIT, delta-query and U2-F gate regressions pass.

Any failure blocks training without changing the matrix.

## Research Gate

For each seed require:

```text
candidate AUC >= 0.520
candidate AUC - deltaV identity anchor AUC >= +0.005
candidate AUC - shuffled deltaU query AUC >= +0.005
```

Both seeds must pass.

## Execution And Outputs

```text
output root = outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_delta_u_query_u2h_r5_2048_seed0_seed1_20260724
cache root  = outputs/local_cache/i1_rtg1_uknit64_runtime_e4_delta_u_query_u2h_r5_2048_seed0_seed1_20260724
```

Require `results.jsonl`, `progress.jsonl`, checkpoints, `plan_validation.json`,
`validation.json`, `gate.json`, `summary.json`, `history.csv`, `curves.svg` and
a visual-QA marker. Refresh both recent-result indexes after completion.

## Evidence-Dependent Next Action

- Pass: freeze both r5 candidate checkpoints and run the same U2-G query-only
  counterfactual on their exact validation caches before scale discussion.
- Hold: classify the current delta-U mechanism as window-specific and stop
  cross-window or scale advancement; retain U2-F/U2-G as r4-only evidence.
- Fail: repair only the protocol/readiness mismatch and rerun this unchanged
  matrix; do not interpret its AUC values.

## Blocked Routes

Do not change samples, epochs, pairs, difference, architecture, loss,
optimizer, keys, controls, or thresholds inside U2-H. Do not add DDT/trail
features, guessed subkeys, partial decryption, extra seeds, or remote execution.

## Completed Result

The six-row local CPU replication completed from source commit `9c73e2a9`.
The exact plan/result validator found six planned and six observed rows with no
missing, unexpected, duplicate, or mismatched keys. Every protocol check
passed, including prefix r5, descriptor `round_start=3`, two loaded rounds,
strict encrypted-random-plaintext negatives, new disk-backed train/validation
caches, same within-seed data/training protocols, and equal `458850` parameter
geometry.

| Seed | Correct delta-U query | Delta-V identity anchor | Shuffled delta-U query | Candidate - anchor | Candidate - shuffled |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.490056992 | 0.520309448 | 0.504012585 | -0.030252457 | -0.013955593 |
| 1 | 0.500100136 | 0.511524200 | 0.505267620 | -0.011424065 | -0.005167484 |

Both seeds missed all three preregistered research checks. The correct delta-U
candidate stayed below `0.520` and underperformed both the capacity-matched
delta-V anchor and shuffled S-box-ownership query. This is a valid negative
result rather than a protocol or implementation failure.

```text
status   = hold
decision = innovation1_uknit_delta_u_cross_window_not_replicated
keep     = U2-F/U2-G as prefix-r4 mechanism evidence only
discard  = cross-window or scaled advancement of this exact delta-U query design
claim    = the current delta-U support is window-specific at local diagnostic scale
```

This result does not invalidate U2-G's same-checkpoint finding on prefix r4,
but it prevents promoting that finding to a stable uKNIT representation. Do
not run an r5 same-checkpoint U2-I audit because the trained candidate itself
did not pass the prerequisite performance gates.

Artifacts:

```text
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_delta_u_query_u2h_r5_2048_seed0_seed1_20260724/results.jsonl
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_delta_u_query_u2h_r5_2048_seed0_seed1_20260724/progress.jsonl
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_delta_u_query_u2h_r5_2048_seed0_seed1_20260724/plan_validation.json
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_delta_u_query_u2h_r5_2048_seed0_seed1_20260724/validation.json
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_delta_u_query_u2h_r5_2048_seed0_seed1_20260724/gate.json
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_delta_u_query_u2h_r5_2048_seed0_seed1_20260724/summary.json
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_delta_u_query_u2h_r5_2048_seed0_seed1_20260724/history.csv
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_delta_u_query_u2h_r5_2048_seed0_seed1_20260724/curves.svg
outputs/local_diagnostic/i1_rtg1_uknit64_runtime_e4_delta_u_query_u2h_r5_2048_seed0_seed1_20260724/visual_qa_passed.marker
```

The exact SVG was rendered to `1156 x 526` pixels and passed
`visual-qa-redraw`. Its Chinese title and protocol subtitle, six-series legend,
zoomed AUC axis, validation curves, and final/best AUC summary table are
readable without overlap, clipping, missing glyphs, or ambiguous scale.

## Evidence-Backed Next Action

Close the uKNIT delta-U query branch at U2-H. Do not increase its samples,
epochs, pairs, seeds, rounds, or move it to the remote GPU. Innovation 1's
next training priority remains the separately supported SKINNY RTG2-B scale
replication:

```text
cipher          = SKINNY-64/64 r7
candidate       = correct general-GF(2) runtime topology
anchor/control  = no topology / corrupted topology
train           = 262144/class
validation      = 131072/class
seed/epochs     = 0 / 5
pairs/sample    = 4
execution       = remote A6000 GPU0 only
```

RTG2-B is already plan- and readiness-aligned, but launch remains forbidden
until source commit `4a44f5fc` and its dependent local commits are published to
GitHub. SSH access to `lxy-a6000` is healthy; the current blocker is local
GitHub push connectivity, not the remote workstation. While publication is
blocked, do not substitute another unmotivated uKNIT training run or launch an
unpublished dirty overlay.
