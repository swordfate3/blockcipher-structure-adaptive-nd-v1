# Innovation 1 PRESENT r8 Raw Matrix Inception Pair-Stack Screen Plan

**Date:** 2026-07-06

**Status:** planned local diagnostic only / no remote launch

## Why This Plan Exists

The r8 trail-position route is currently the strongest controlled local SPN
candidate, but frozen-score aggregation of the trail-position expert with its
same-input global-stat control did not improve over the best single
trail-position model. The cell-value histogram screen was weak-positive but
lost badly to the same-input global-stat control.

The next useful step is therefore not to average more near-neighbor models. It
is to screen a genuinely different expert family that could later supply
compatible frozen scores for a diverse pool.

This plan tests a raw PRESENT MCND-style matrix representation already present
in the codebase:

```text
feature_encoding = present_mcnd_cell_matrix_bits
candidate = present_inception_mcnd_pair_stack_matrix
control = present_pairset_global_stats
```

The raw pair-stack Inception candidate is structurally different from the
trail-position route: it consumes raw pair cell matrices and uses multi-kernel
2D convolution over the stacked pair/cell grid, rather than public
trail-position summaries or cross-pair value histograms.

## Question

Holding the r8 matched-negative integral sampling protocol fixed:

```text
Can a raw MCND pair-stack Inception matrix model beat the same-input
global-statistics control and become a viable non-neighbor expert candidate?
```

## Fixed Protocol

```text
cipher = PRESENT-80
rounds = 8
samples_per_class = 2048
pairs_per_sample = 16
feature_encoding = present_mcnd_cell_matrix_bits
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
configs/experiment/innovation1/innovation1_spn_present_r8_raw_matrix_inception_pair_stack_2048_local.csv
```

Rows:

| Row | Model | Role |
|---:|---|---|
| 0 | `present_pairset_global_stats` | raw matrix same-input global-statistics control, seed0 |
| 1 | `present_inception_mcnd_pair_stack_matrix` | raw pair-stack Inception candidate, seed0 |
| 2 | `present_pairset_global_stats` | raw matrix same-input global-statistics control, seed1 |
| 3 | `present_inception_mcnd_pair_stack_matrix` | raw pair-stack Inception candidate, seed1 |

## Gate

Advance only if:

```text
candidate AUC > global-stat control AUC on both seeds
and mean candidate AUC >= mean global-stat control AUC + 0.01
and candidate AUC >= 0.55 on both seeds
```

If the gate passes, the next step is not an immediate claim. The next step is
to export compatible frozen scores with:

```text
expert_family = raw_matrix_inception
candidate_status = local_weak_positive
```

Then compare error overlap against the trail-position frozen score artifacts
inside a protocol-aligned diversity gate.

Hold if:

```text
candidate loses to the same-input global-stat control on either seed
or candidate AUC is a single-seed spike
or candidate is near chance on both seeds
```

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_raw_matrix_inception_pair_stack_2048_local.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_raw_matrix_inception_pair_stack_2048 \
  --dataset-cache-chunk-size 512 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_raw_matrix_inception_pair_stack_2048/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_raw_matrix_inception_pair_stack_2048/progress.jsonl
```

## Claim Scope

Allowed:

```text
local diagnostic screen for a raw matrix Inception non-neighbor expert
possible future frozen-score candidate only if the gate passes
```

Not allowed:

```text
remote-launch basis
formal PRESENT result
breakthrough claim
raw single-sample SOTA claim
diverse expert pool evidence by itself
replacement for the trail-position residual gate
```
