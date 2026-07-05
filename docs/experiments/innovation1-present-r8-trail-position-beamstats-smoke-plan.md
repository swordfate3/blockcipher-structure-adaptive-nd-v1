# Innovation 1 PRESENT r8 Trail-Position Beamstats Smoke Plan

**Date:** 2026-07-06

**Status:** planned local smoke only / no remote launch

## Why This Plan Exists

The current GPD-style `beamstats4/deep3` route failed the higher-sample
semantic attribution gate and should not be used as the next diverse expert.
However, that failure was specifically about globally compressed semantic
scalars. The repository already contains a more position-aware model:

```text
present_trail_position_stats_pairset
```

This model preserves per-depth, per-word, and per-cell activity statistics over
parameterized DDT beamstats inputs. It is different enough from raw/InvP/DDT
near-neighbor models to deserve one small local smoke, but not enough evidence
for remote training.

## Question

Holding data construction and feature encoding fixed:

```text
Does preserving DDT trail position statistics beat a same-input global-statistic
control on the r8 matched-negative integral setting?
```

## Fixed Protocol

```text
cipher = PRESENT-80
rounds = 8
samples_per_class = 128
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
configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_smoke.csv
```

Rows:

| Row | Model | Role |
|---:|---|---|
| 0 | `present_pairset_global_stats` | same-input global-statistics control |
| 1 | `present_trail_position_stats_pairset` | candidate preserving depth/word/cell trail-position statistics |
| 2 | `present_pairset_global_stats` | seed1 control |
| 3 | `present_trail_position_stats_pairset` | seed1 candidate |

## Gate

This is only a route-discovery smoke. It can justify a larger local diagnostic,
not a remote run.

Advance only if:

```text
candidate AUC > global-stat control AUC on both seeds
and mean candidate AUC >= mean global-stat control AUC + 0.01
and candidate AUC is above 0.52 on both seeds
```

Hold/stop if:

```text
candidate loses to the global-stat control on either seed
or the result is only a single-seed spike
or both rows are close to chance
```

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_smoke.csv \
  --epochs 3 \
  --batch-size 32 \
  --hidden-bits 16 \
  --device cpu \
  --learning-rate 0.0001 \
  --optimizer adam \
  --weight-decay 0.00001 \
  --loss mse \
  --checkpoint-metric val_auc \
  --restore-best-checkpoint \
  --train-eval-interval 1 \
  --output outputs/local_smoke/i1_present_r8_trail_position_beamstats_smoke/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_trail_position_beamstats_smoke/progress.jsonl
```

## Claim Scope

Allowed:

```text
local smoke on position-aware DDT beamstats statistics
route-discovery evidence only
```

Not allowed:

```text
remote-launch basis
PRESENT r8 result claim
diverse expert pool evidence
neural architecture breakthrough
```

## 128/Class Smoke Result

Run:

```text
outputs/local_smoke/i1_present_r8_trail_position_beamstats_smoke/results.jsonl
```

Validation:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_smoke.csv \
  --results outputs/local_smoke/i1_present_r8_trail_position_beamstats_smoke/results.jsonl \
  --expected-rows 4

status = pass
```

Results:

| Seed | Model | AUC | Calibrated accuracy |
|---:|---|---:|---:|
| 0 | `present_pairset_global_stats` | `0.608642578125` | `0.6328125` |
| 0 | `present_trail_position_stats_pairset` | `0.9423828125` | `0.9140625` |
| 1 | `present_pairset_global_stats` | `0.62890625` | `0.625` |
| 1 | `present_trail_position_stats_pairset` | `0.9541015625` | `0.90625` |

Smoke decision:

```text
support_512_class_local_confirmation
```

Interpretation:

```text
The position-aware trail-statistics candidate beat the same-input global-stat
control on both seeds by a large margin. This is still only a 128/class local
smoke with validation_samples_per_class = 64, so it is not route evidence for
remote launch. It does justify a 512/class local diagnostic.
```

## 512/Class Local Diagnostic

Config:

```text
configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_512_local.csv
```

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_512_local.csv \
  --epochs 3 \
  --batch-size 32 \
  --hidden-bits 16 \
  --device cpu \
  --learning-rate 0.0001 \
  --optimizer adam \
  --weight-decay 0.00001 \
  --loss mse \
  --checkpoint-metric val_auc \
  --restore-best-checkpoint \
  --train-eval-interval 1 \
  --output outputs/local_smoke/i1_present_r8_trail_position_beamstats_512/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_trail_position_beamstats_512/progress.jsonl
```

Advance after 512/class only if:

```text
candidate AUC > global-stat control AUC on both seeds
and mean candidate AUC >= mean global-stat control AUC + 0.01
and candidate AUC is above 0.55 on both seeds
```

If this passes, the next action is still a controlled local attribution or
larger local diagnostic, not an immediate remote launch.

### 512/Class Local Diagnostic Result

Run:

```text
outputs/local_smoke/i1_present_r8_trail_position_beamstats_512/results.jsonl
```

Validation:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_512_local.csv \
  --results outputs/local_smoke/i1_present_r8_trail_position_beamstats_512/results.jsonl \
  --expected-rows 4

status = pass
```

Artifacts:

```text
outputs/local_smoke/i1_present_r8_trail_position_beamstats_512/results.jsonl
outputs/local_smoke/i1_present_r8_trail_position_beamstats_512/progress.jsonl
outputs/local_smoke/i1_present_r8_trail_position_beamstats_512/curves.svg
outputs/local_smoke/i1_present_r8_trail_position_beamstats_512/history.csv
```

Results:

| Seed | Model | AUC | Calibrated accuracy | Accuracy |
|---:|---|---:|---:|---:|
| 0 | `present_pairset_global_stats` | `0.813568115234375` | `0.744140625` | `0.71484375` |
| 0 | `present_trail_position_stats_pairset` | `0.98883056640625` | `0.95703125` | `0.896484375` |
| 1 | `present_pairset_global_stats` | `0.7928619384765625` | `0.751953125` | `0.5078125` |
| 1 | `present_trail_position_stats_pairset` | `0.9859771728515625` | `0.9453125` | `0.744140625` |

Aggregate:

```text
mean_global_control_auc = 0.8032150268554688
mean_position_candidate_auc = 0.9874038696289062
mean_auc_delta = +0.1841888427734375
candidate_min_auc = 0.9859771728515625
```

Gate decision:

```text
support_trail_position_beamstats_local_confirmation
```

Interpretation:

```text
The 512/class local diagnostic preserves the 128/class smoke result:
position-aware trail statistics strongly beat the same-input global-statistics
control on both seeds. This is the strongest local non-neighbor SPN
representation signal found in this branch so far.
```

Claim boundary:

```text
This is not a PRESENT r8 breakthrough claim and not a remote-launch basis yet.
The sample structure is an r8 matched-negative integral setting, not the
standard r7 Zhang/Wang Case2 MCND benchmark. The signal may still be a
deterministic data-construction/statistic effect rather than a neural
architecture gain.
```

Next action:

```text
1. Add or run a local attribution/control audit for the position-statistics
   vector itself.
2. Test whether a deterministic position-statistics baseline can match the
   neural candidate.
3. Add pair-order / active-nibble / difference controls before any remote job.
4. Treat the route as a promising SPN representation/data candidate, not as a
   qualified diverse ensemble expert yet.
```
