# Innovation 1 PRESENT r8 Cell-Value Histogram Screen Plan

**Date:** 2026-07-06

**Status:** completed local diagnostic / held, no remote launch

## Result

The planned `2048/class`, seeds `0,1` local diagnostic completed with all four
rows and a complete gate artifact:

```text
results = outputs/local_smoke/i1_present_r8_cell_value_histogram_2048/results.jsonl
gate = outputs/local_smoke/i1_present_r8_cell_value_histogram_2048/gate.json
decision = hold_cell_value_histogram_local_screen
action = do_not_promote_to_diverse_expert_pool
```

Per-seed AUC:

| Seed | Global-stat control AUC | Histogram candidate AUC | Candidate margin |
|---:|---:|---:|---:|
| 0 | `0.8532476425170898` | `0.5871686935424805` | `-0.2660789489746094` |
| 1 | `0.874751091003418` | `0.6144542694091797` | `-0.2602968215942383` |

Aggregate:

```text
mean_global_stat_control_auc = 0.8639993667602539
mean_histogram_candidate_auc = 0.6008114814758301
mean_candidate_margin_vs_control_auc = -0.26318788528442383
min_histogram_candidate_auc = 0.5871686935424805
```

Interpretation:

The histogram candidate is weak-positive above random on both seeds, but it
does not clear the same-input global-statistics control. This means it is not a
good enough non-neighbor expert for the diverse multi-network pool yet. Treat
this as a held local screen, not a remote-launch basis, not formal SPN/PRESENT
evidence, and not a breakthrough claim.

Next action:

Keep the trail-position residual route as the current strongest controlled
local SPN candidate. For multi-network work, search for a genuinely different
expert family that can beat or complement the global-stat control before
spending ensemble effort. Do not promote the current cell-value histogram model
to frozen-score diversity/error-overlap evaluation unless it is redesigned or a
new same-protocol screen changes this result.

## Why This Plan Exists

The r8 trail-position beamstats route is currently the strongest local
SPN-aware diagnostic route, but it should not be turned into a wider ensemble
until a genuinely different weak-positive expert exists. Near-neighbor
aggregation is not enough.

This plan tests one different representation already present in the codebase:

```text
present_pairset_histogram_hybrid
```

Instead of only using active/position statistics, this model summarizes
cross-pair 4-bit PRESENT cell-value histograms. The intent is not to beat the
trail-position candidate at the same scale. The immediate question is whether a
value-distribution expert has its own positive same-protocol signal over the
same-input global-statistics control.

## Question

Holding data construction and feature encoding fixed:

```text
Does a cross-pair 4-bit cell-value histogram model beat the same-input
global-statistics control on the r8 matched-negative integral setting?
```

## Fixed Protocol

```text
cipher = PRESENT-80
rounds = 8
samples_per_class = 2048
pairs_per_sample = 16
feature_encoding = present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits
sample_structure = plaintext_integral_nibble_difference_matched_negative
integral_active_nibble = 0
difference_profile = present_zhang_wang2022_mcnd
negative_mode = encrypted_random_plaintexts
train_key = 0x00000000000000000000
validation_key = 0x11111111111111111111
seeds = 0, 1
```

Config:

```text
configs/experiment/innovation1/innovation1_spn_present_r8_cell_value_histogram_2048_local.csv
```

Rows:

| Row | Model | Role |
|---:|---|---|
| 0 | `present_pairset_global_stats` | same-input global-statistics control, seed0 |
| 1 | `present_pairset_histogram_hybrid` | cell-value histogram candidate, seed0 |
| 2 | `present_pairset_global_stats` | same-input global-statistics control, seed1 |
| 3 | `present_pairset_histogram_hybrid` | cell-value histogram candidate, seed1 |

## Gate

This is a local diagnostic. It can justify a follow-up diversity/error-overlap
design only if it is clearly positive. It does not justify remote launch by
itself.

Advance only if:

```text
histogram AUC > global-stat control AUC on both seeds
and mean histogram AUC >= mean global-stat control AUC + 0.01
and histogram AUC >= 0.55 on both seeds
```

Hold if:

```text
histogram loses to the global-stat control on either seed
or the result is a single-seed spike
or both rows are close to chance
```

If the gate passes, the next step is not immediate ensembling. The next step is
to export compatible frozen scores with:

```text
expert_family = cell_value_histogram
candidate_status = local_weak_positive
```

and then compare error overlap against the r7 InvP anchor and the r8
trail-position route after their own controls pass.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_cell_value_histogram_2048_local.csv \
  --epochs 3 \
  --batch-size 64 \
  --hidden-bits 16 \
  --device cpu \
  --learning-rate 0.0001 \
  --optimizer adam \
  --weight-decay 0.00001 \
  --loss mse \
  --checkpoint-metric val_auc \
  --restore-best-checkpoint \
  --train-eval-interval 1 \
  --dataset-cache-root outputs/local_cache/i1_present_r8_cell_value_histogram_2048 \
  --dataset-cache-chunk-size 512 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_cell_value_histogram_2048/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_cell_value_histogram_2048/progress.jsonl
```

Postprocess gate:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/gate-cell-value-histogram \
  --results outputs/local_smoke/i1_present_r8_cell_value_histogram_2048/results.jsonl \
  --output outputs/local_smoke/i1_present_r8_cell_value_histogram_2048/gate.json
```

## Claim Scope

Allowed:

```text
local diagnostic for a value-distribution SPN expert
possible future diverse-expert candidate if the gate passes
```

Not allowed:

```text
remote-launch basis
formal PRESENT result
breakthrough claim
diverse expert pool evidence by itself
replacement for the trail-position residual gate
```
