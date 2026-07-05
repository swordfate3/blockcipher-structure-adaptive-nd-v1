# Innovation 1 PRESENT r8 GPD-Style Beamstats Plan

**Date:** 2026-07-06

**Status:** 512/class seed0+seed1 diagnostics completed / beam family unstable / no remote launch

## Why This Plan Exists

The user asked for independent route selection rather than simply following the
latest suggested ensemble direction. Re-checking the current evidence and local
literature notes points away from a wider near-neighbor neural ensemble and
toward SPN-aware input or feature representation.

The important correction for this route is:

```text
do not duplicate the existing single-step Sinv feature
```

The repository already has a PRESENT zero-key one-round structural inverse
feature:

```text
present_pair_xor_paligned_sinv_cell_matrix_bits
```

That feature computes the structural form of:

```text
S^{-1}(P^{-1}(C)) xor S^{-1}(P^{-1}(C'))
```

So this plan does not add a new alias for the same idea. Instead, it tests
existing multi-round DDT/partial-inverse candidate-path features that are closer
to the Generic Partial Decryption direction while preserving the current
benchmark protocol.

## External And Local Motivation

Generic Partial Decryption treats partial inverse states as feature engineering
for neural distinguishers. The project-local paper note is:

```text
papers/innovation_one/text/2025_gpd_feature_engineering_nd.txt
```

The relevant local extract says GPD adds differences from previous rounds to
neural distinguisher inputs and restricts the tested pipeline to two rounds of
partial decryption for controlled evaluation. The project should treat that as
a feature-engineering hypothesis, not as permission to change labels or
negative-sample definitions.

Current local evidence also says to be careful:

```text
r8 InvP/Sinv integral screen: Sinv matrix AUC near random
r8 aligned neural follow-up: below fixed pair_xor_column_sum_variance baseline
projection v2: unstable at local smoke scale
near-neighbor r7 ensemble: weak positive but below gate
```

Therefore this plan is only a small local screen: can multi-round DDT beam or
beam-statistics features show a cleaner signal than the existing InvP/Sinv
controls under the same matched-negative r8 setting?

## Fixed Protocol

The plan keeps the benchmark stable:

```text
cipher = PRESENT-80
rounds = 8
seed = 0
samples_per_class = 128
pairs_per_sample = 16
sample_structure = plaintext_integral_nibble_difference_matched_negative
integral_active_nibble = 0
difference_profile = present_zhang_wang2022_mcnd
negative_mode = encrypted_random_plaintexts
train_key = 0x00000000000000000000
validation_key = 0x11111111111111111111
checkpoint_metric = val_auc
restore_best_checkpoint = true
```

The seed1 repeat keeps the same protocol and changes only:

```text
seed = 1
```

## Matrix

Config:

```text
configs/experiment/innovation1/innovation1_spn_present_r8_gpd_style_beamstats_smoke.csv
configs/experiment/innovation1/innovation1_spn_present_r8_gpd_style_beamstats_smoke_seed1.csv
configs/experiment/innovation1/innovation1_spn_present_r8_gpd_style_beamstats_512_seed0.csv
configs/experiment/innovation1/innovation1_spn_present_r8_gpd_style_beamstats_512_seed1.csv
```

Rows:

| Row | Feature encoding | Role |
|---:|---|---|
| 0 | `present_pair_xor_paligned_cell_matrix_bits` | InvP control |
| 1 | `present_pair_xor_paligned_sinv_cell_matrix_bits` | existing zero-key one-round Sinv control |
| 2 | `present_delta_paligned_sinv_sboxddt_beam4deep3_cell_matrix_bits` | DDT beam candidate path feature |
| 3 | `present_delta_paligned_sinv_sboxddt_beamstats4deep3_cell_matrix_bits` | compressed DDT beam-statistics candidate |

## Gate

This is a local smoke and route-sanity screen only.

Do not launch a remote scale-up unless the local result shows a stable positive
direction and beats the relevant controls:

```text
candidate AUC > InvP control AUC
candidate AUC > Sinv control AUC
candidate is not merely rediscovering the fixed deterministic baseline
result is repeated on at least one additional local seed before remote planning
```

Even if positive, the next step should be a planned `65536/class` diagnostic,
not a claim of breakthrough or formal route evidence.

## Claim Scope

Allowed:

```text
local GPD-style partial inverse / DDT beamstats feature smoke
SPN representation-route diagnostic
```

Not allowed:

```text
PRESENT r8 breakthrough
formal training
SOTA
evidence that ensemble is unnecessary forever
evidence that GPD works for PRESENT at scale
```

## Local Smoke Result

Run:

```text
outputs/local_smoke/i1_present_r8_gpd_style_beamstats_smoke/results.jsonl
```

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_gpd_style_beamstats_smoke.csv \
  --epochs 3 \
  --batch-size 64 \
  --hidden-bits 32 \
  --device cpu \
  --learning-rate 0.0001 \
  --optimizer adam \
  --weight-decay 0.00001 \
  --loss mse \
  --checkpoint-metric val_auc \
  --restore-best-checkpoint \
  --train-eval-interval 1 \
  --output outputs/local_smoke/i1_present_r8_gpd_style_beamstats_smoke/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_gpd_style_beamstats_smoke/progress.jsonl
```

Artifacts:

```text
outputs/local_smoke/i1_present_r8_gpd_style_beamstats_smoke/results.jsonl
outputs/local_smoke/i1_present_r8_gpd_style_beamstats_smoke/progress.jsonl
outputs/local_smoke/i1_present_r8_gpd_style_beamstats_smoke/curves.svg
outputs/local_smoke/i1_present_r8_gpd_style_beamstats_smoke/history.csv
```

Results:

| Feature encoding | AUC | Calibrated accuracy | Interpretation |
|---|---:|---:|---|
| `present_pair_xor_paligned_cell_matrix_bits` | `0.496337890625` | `0.5546875` | InvP control, near random by AUC |
| `present_pair_xor_paligned_sinv_cell_matrix_bits` | `0.44287109375` | `0.5078125` | existing single-step Sinv control, negative at this smoke |
| `present_delta_paligned_sinv_sboxddt_beam4deep3_cell_matrix_bits` | `0.5145263671875` | `0.5625` | tiny weak-positive local smoke candidate |
| `present_delta_paligned_sinv_sboxddt_beamstats4deep3_cell_matrix_bits` | `0.462158203125` | `0.5234375` | compressed beamstats did not preserve the small beam signal |

Seed0 decision:

```text
diagnostic_weak_beam_candidate_no_remote_launch
```

The DDT beam path is the only positive row in this tiny smoke, and it is still
too small and too under-sampled to justify a remote job. It is useful as route
selection evidence only:

```text
expanded DDT beam path > compressed beamstats path for the next local seed check
```

Next action:

```text
Run a seed1 local repeat only if continuing the GPD-style branch.
Do not launch remote from this seed0 smoke.
Do not claim a neural architecture gain.
```

## Local Seed1 Repeat Result

Run:

```text
outputs/local_smoke/i1_present_r8_gpd_style_beamstats_smoke_seed1/results.jsonl
```

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_gpd_style_beamstats_smoke_seed1.csv \
  --epochs 3 \
  --batch-size 64 \
  --hidden-bits 32 \
  --device cpu \
  --learning-rate 0.0001 \
  --optimizer adam \
  --weight-decay 0.00001 \
  --loss mse \
  --checkpoint-metric val_auc \
  --restore-best-checkpoint \
  --train-eval-interval 1 \
  --output outputs/local_smoke/i1_present_r8_gpd_style_beamstats_smoke_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_gpd_style_beamstats_smoke_seed1/progress.jsonl
```

Artifacts:

```text
outputs/local_smoke/i1_present_r8_gpd_style_beamstats_smoke_seed1/results.jsonl
outputs/local_smoke/i1_present_r8_gpd_style_beamstats_smoke_seed1/progress.jsonl
outputs/local_smoke/i1_present_r8_gpd_style_beamstats_smoke_seed1/curves.svg
outputs/local_smoke/i1_present_r8_gpd_style_beamstats_smoke_seed1/history.csv
```

Results:

| Feature encoding | Seed0 AUC | Seed1 AUC | Interpretation |
|---|---:|---:|---|
| `present_pair_xor_paligned_cell_matrix_bits` | `0.496337890625` | `0.54296875` | control moved from near-random to weak positive |
| `present_pair_xor_paligned_sinv_cell_matrix_bits` | `0.44287109375` | `0.5361328125` | single-step Sinv was not stable |
| `present_delta_paligned_sinv_sboxddt_beam4deep3_cell_matrix_bits` | `0.5145263671875` | `0.527587890625` | only row weak-positive in both seeds, but does not beat seed1 controls |
| `present_delta_paligned_sinv_sboxddt_beamstats4deep3_cell_matrix_bits` | `0.462158203125` | `0.606689453125` | seed1 spike, not reproduced from seed0 |

Updated decision:

```text
unstable_gpd_style_candidate_hold_no_remote_launch
```

The seed1 result keeps the GPD-style branch alive as a local representation
candidate, but it does not pass a scale-up gate. The largest seed1 AUC belongs
to the compressed beamstats row, yet that same row was below random on seed0.
The expanded DDT beam row is weak-positive in both seeds, but it does not
consistently beat InvP/Sinv controls. The correct interpretation is:

```text
beam-family route has a signal/noise hint;
128/class validation is too noisy to select a winner;
do not launch remote from these smokes.
```

Next action if continuing this branch:

```text
prepare a 512/class local diagnostic with the same four rows, or a narrower
beam-vs-controls diagnostic with larger validation, before any 65536/class
remote plan.
```

## Local 512/Class Diagnostic Result

Run:

```text
outputs/local_smoke/i1_present_r8_gpd_style_beamstats_512_seed0/results.jsonl
```

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_gpd_style_beamstats_512_seed0.csv \
  --epochs 3 \
  --batch-size 64 \
  --hidden-bits 32 \
  --device cpu \
  --learning-rate 0.0001 \
  --optimizer adam \
  --weight-decay 0.00001 \
  --loss mse \
  --checkpoint-metric val_auc \
  --restore-best-checkpoint \
  --train-eval-interval 1 \
  --output outputs/local_smoke/i1_present_r8_gpd_style_beamstats_512_seed0/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_gpd_style_beamstats_512_seed0/progress.jsonl
```

Artifacts:

```text
outputs/local_smoke/i1_present_r8_gpd_style_beamstats_512_seed0/results.jsonl
outputs/local_smoke/i1_present_r8_gpd_style_beamstats_512_seed0/progress.jsonl
outputs/local_smoke/i1_present_r8_gpd_style_beamstats_512_seed0/curves.svg
outputs/local_smoke/i1_present_r8_gpd_style_beamstats_512_seed0/history.csv
```

Results:

| Feature encoding | 128 seed0 AUC | 128 seed1 AUC | 512 seed0 AUC | Interpretation |
|---|---:|---:|---:|---|
| `present_pair_xor_paligned_cell_matrix_bits` | `0.496337890625` | `0.54296875` | `0.540496826171875` | InvP control remains weak-positive at 512 |
| `present_pair_xor_paligned_sinv_cell_matrix_bits` | `0.44287109375` | `0.5361328125` | `0.5286407470703125` | single-step Sinv is below InvP at 512 |
| `present_delta_paligned_sinv_sboxddt_beam4deep3_cell_matrix_bits` | `0.5145263671875` | `0.527587890625` | `0.562957763671875` | best 512 row; expanded DDT beam now beats controls |
| `present_delta_paligned_sinv_sboxddt_beamstats4deep3_cell_matrix_bits` | `0.462158203125` | `0.606689453125` | `0.5418472290039062` | seed1 spike collapses to control-level at 512 |

512 diagnostic decision:

```text
keep_expanded_ddt_beam_candidate_prepare_512_seed1_repeat
```

This is a meaningful local refinement. The 128/class seed1 beamstats spike did
not survive the larger local diagnostic, while the expanded DDT beam row moved
ahead of both controls:

```text
DDT beam AUC - InvP control AUC = +0.022460937500
DDT beam AUC - Sinv control AUC = +0.034317016602
DDT beam AUC - beamstats AUC = +0.021110534668
```

This still does not justify a remote launch. It does justify a second
`512/class` local repeat on seed1. Only if the expanded DDT beam remains
control-beating under that repeat should this branch advance to a planned
`65536/class` diagnostic.

Updated next action:

```text
prepare and run configs/experiment/innovation1/innovation1_spn_present_r8_gpd_style_beamstats_512_seed1.csv
```

## Local 512/Class Seed1 Repeat Result

Run:

```text
outputs/local_smoke/i1_present_r8_gpd_style_beamstats_512_seed1/results.jsonl
```

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_gpd_style_beamstats_512_seed1.csv \
  --epochs 3 \
  --batch-size 64 \
  --hidden-bits 32 \
  --device cpu \
  --learning-rate 0.0001 \
  --optimizer adam \
  --weight-decay 0.00001 \
  --loss mse \
  --checkpoint-metric val_auc \
  --restore-best-checkpoint \
  --train-eval-interval 1 \
  --output outputs/local_smoke/i1_present_r8_gpd_style_beamstats_512_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_gpd_style_beamstats_512_seed1/progress.jsonl
```

Artifacts:

```text
outputs/local_smoke/i1_present_r8_gpd_style_beamstats_512_seed1/results.jsonl
outputs/local_smoke/i1_present_r8_gpd_style_beamstats_512_seed1/progress.jsonl
outputs/local_smoke/i1_present_r8_gpd_style_beamstats_512_seed1/curves.svg
outputs/local_smoke/i1_present_r8_gpd_style_beamstats_512_seed1/history.csv
```

Results:

| Feature encoding | 512 seed0 AUC | 512 seed1 AUC | Mean AUC | Interpretation |
|---|---:|---:|---:|---|
| `present_pair_xor_paligned_cell_matrix_bits` | `0.540496826171875` | `0.5263595581054688` | `0.5334281921386719` | InvP control remains weak-positive |
| `present_pair_xor_paligned_sinv_cell_matrix_bits` | `0.5286407470703125` | `0.56329345703125` | `0.5459671020507812` | Sinv control is itself volatile and wins over expanded beam on seed1 |
| `present_delta_paligned_sinv_sboxddt_beam4deep3_cell_matrix_bits` | `0.562957763671875` | `0.51806640625` | `0.5405120849609375` | expanded DDT beam seed0 improvement did not reproduce |
| `present_delta_paligned_sinv_sboxddt_beamstats4deep3_cell_matrix_bits` | `0.5418472290039062` | `0.5724639892578125` | `0.5571556091308594` | best two-seed mean, but margin is still local diagnostic only |

Seed1 calibrated accuracies:

```text
InvP control = 0.5390625
Sinv control = 0.560546875
DDT beam = 0.548828125
DDT beamstats = 0.583984375
```

Updated decision:

```text
hold_gpd_style_beam_family_no_remote_launch
```

This repeat falsifies the narrow seed0 reading that the expanded DDT beam is
the stable winner. The compressed beamstats row now has the best two-seed mean
and the best seed1 result, but the 128/class beamstats row was below random on
seed0 and the 512 seed0 margin over InvP was only about `+0.00135` AUC. Treat
beamstats as a lightweight local candidate, not as scale-up evidence.

Current branch action:

```text
do not launch 65536/class from this GPD-style branch yet
prefer a lean local confirmation or attribution check over a remote run
candidate to keep locally = beamstats, not expanded DDT beam
candidate to demote = expanded DDT beam unless a new attribution explains seed1 failure
```

Relationship to diverse experts:

```text
The GPD-style beamstats row is a possible future non-neighbor expert source,
but only after compatible weak-positive score artifacts and low-overlap checks
exist. Do not use this result to justify a wider near-neighbor ensemble now.
```
