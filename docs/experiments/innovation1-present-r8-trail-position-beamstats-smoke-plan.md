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

## Position-Statistics Attribution Audit

Implementation:

```text
CLI = scripts/audit-spn-features --trail-position-attribution-plan ...
API = trail_position_attribution_from_task
stat_vector = PresentTrailPositionStatsPairSetDistinguisher._position_statistics(...)
combiner = top_position_stat_oriented_zscore_mean
```

The audit extracts the deterministic position-statistics vector used by
`present_trail_position_stats_pairset`, scores each scalar against the labels,
and builds a fixed top-k oriented z-score composite. This asks whether the
neural route is mostly rediscovering a small deterministic statistic.

Commands:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/audit-spn-features \
  --trail-position-attribution-plan configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_512_local.csv \
  --row-index 1 \
  --samples-per-class 2048 \
  --seed 0 \
  --key-split validation \
  --top-k 16 \
  --output outputs/local_audits/i1_present_r8_trail_position_attribution_seed0_2048.json

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/audit-spn-features \
  --trail-position-attribution-plan configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_512_local.csv \
  --row-index 3 \
  --samples-per-class 2048 \
  --seed 1 \
  --key-split validation \
  --top-k 16 \
  --output outputs/local_audits/i1_present_r8_trail_position_attribution_seed1_2048.json
```

Artifacts:

```text
outputs/local_audits/i1_present_r8_trail_position_attribution_seed0_512.json
outputs/local_audits/i1_present_r8_trail_position_attribution_seed1_512.json
outputs/local_audits/i1_present_r8_trail_position_attribution_seed0_2048.json
outputs/local_audits/i1_present_r8_trail_position_attribution_seed1_2048.json
```

Results:

| Scale | Seed | Best scalar | Directional best-scalar AUC | Top-16 composite AUC | Composite best accuracy |
|---:|---:|---|---:|---:|---:|
| 512/class | 0 | `cell_span_cell6` | `0.6741142272949219` | `0.9032249450683594` | `0.8388671875` |
| 512/class | 1 | `depth_word_span_depth2_trailword3` | `0.6828956604003906` | `0.8659286499023438` | `0.810546875` |
| 2048/class | 0 | `depth_word_span_depth2_trailword1` | `0.6629594564437866` | `0.8734362125396729` | `0.791748046875` |
| 2048/class | 1 | `depth_word_span_depth1_trailword1` | `0.6627322435379028` | `0.8486461639404297` | `0.769287109375` |

Attribution decision:

```text
position_stats_deterministic_baseline_required
```

Interpretation:

```text
The deterministic position-statistics vector explains a large fraction of the
trail-position neural signal. A single span statistic already reaches about
0.663-0.683 directional AUC, while a fixed top-16 oriented composite reaches
about 0.849-0.903 AUC.

However, the composite remains below the 512/class neural candidate AUC
0.985977-0.988831. This leaves possible residual value in the learned nonlinear
combination, but the route can no longer be judged without an explicit
deterministic position-statistics baseline.
```

Updated next action:

```text
1. Add deterministic position-statistics baseline rows or a postprocess gate
   before any larger neural diagnostic.
2. Run pair-order, active-nibble, and difference controls against both the
   deterministic composite and the neural candidate.
3. Do not launch remote training until the route survives those controls.
4. Do not use this as a diverse expert until compatible frozen scores and
   diversity/error-overlap checks exist.
```

## Train-Selected Position-Statistics Split Baseline

Implementation:

```text
CLI = scripts/audit-spn-features --trail-position-split-baseline-plan ...
API = trail_position_split_baseline_from_task
selection_split = train
evaluation_split = validation
combiner = train_selected_position_stat_oriented_zscore_mean
```

This audit closes the label-selection gap in the earlier attribution result.
It selects the top-k position-statistics axes on the train key, fits each axis
orientation plus z-score normalization on the train split only, and then applies
that fixed composite to the validation key. Validation labels are used only for
reporting the final score.

Commands:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/audit-spn-features \
  --trail-position-split-baseline-plan configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_512_local.csv \
  --row-index 1 \
  --samples-per-class 2048 \
  --seed 0 \
  --top-k 16 \
  --output outputs/local_audits/i1_present_r8_trail_position_split_baseline_seed0_2048.json

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/audit-spn-features \
  --trail-position-split-baseline-plan configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_512_local.csv \
  --row-index 3 \
  --samples-per-class 2048 \
  --seed 1 \
  --top-k 16 \
  --output outputs/local_audits/i1_present_r8_trail_position_split_baseline_seed1_2048.json
```

Artifacts:

```text
outputs/local_audits/i1_present_r8_trail_position_split_baseline_seed0_512.json
outputs/local_audits/i1_present_r8_trail_position_split_baseline_seed1_512.json
outputs/local_audits/i1_present_r8_trail_position_split_baseline_seed0_2048.json
outputs/local_audits/i1_present_r8_trail_position_split_baseline_seed1_2048.json
```

Results:

| Scale | Seed | Train composite AUC | Validation composite AUC | Validation best accuracy | Best train-selected statistic |
|---:|---:|---:|---:|---:|---|
| 512/class | 0 | `0.8651924133300781` | `0.7695465087890625` | `0.703125` | `depth_word_span_depth1_trailword6` |
| 512/class | 1 | `0.9015998840332031` | `0.8455047607421875` | `0.7734375` | `depth_word_span_depth1_trailword7` |
| 2048/class | 0 | `0.8498256206512451` | `0.8056130409240723` | `0.735595703125` | `depth_word_span_depth2_trailword3` |
| 2048/class | 1 | `0.8753311634063721` | `0.8421728610992432` | `0.766845703125` | `depth_word_span_depth2_trailword2` |

Split-baseline decision:

```text
support_trail_position_signal_but_require_neural_residual_gate
```

Interpretation:

```text
The train-selected deterministic position-statistics composite remains strong
on the validation key, so the route is not merely a validation-label
feature-selection artifact. The strongest selected families are mostly
depth/word/cell span statistics, consistent with the hypothesis that this
r8 matched-negative integral setting exposes SPN trail-position distribution
structure.

The deterministic split baseline is still below the 512/class neural candidate
AUC 0.985977-0.988831, so a nonlinear neural residual may exist. But the
baseline is strong enough that future neural trail-position claims must beat
this train-selected deterministic composite under the same split and controls.
```

Updated route gate:

```text
1. Do not remote-launch this route from the current evidence.
2. Before larger neural training, add a same-protocol deterministic
   position-stat baseline row or postprocess gate.
3. Add active-nibble, pair-order, and difference controls for both deterministic
   and neural routes.
4. Only treat this as a diverse expert candidate after it produces compatible
   frozen scores and passes an error-overlap/diversity check against the r7
   InvP/P-layer anchor and near-neighbor controls.
```

## Trail-Position Control Baseline

Implementation:

```text
CLI = scripts/audit-spn-features --trail-position-control-baseline-plan ...
API = trail_position_control_baseline_from_task
baseline = train-selected position-statistics split baseline
controls = active_nibble_1, input_difference_0x90, pair_order_reverse
scale = 512/class
```

This audit asks whether the deterministic trail-position signal is specific to
the active-nibble/difference alignment or whether it survives obvious control
perturbations. Each variant fits its own train-selected deterministic composite
and evaluates on the validation key.

Commands:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/audit-spn-features \
  --trail-position-control-baseline-plan configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_512_local.csv \
  --row-index 1 \
  --samples-per-class 512 \
  --seed 0 \
  --top-k 16 \
  --control-active-nibbles 1 \
  --control-input-differences 0x90 \
  --control-pair-orders reverse \
  --output outputs/local_audits/i1_present_r8_trail_position_control_baseline_seed0_512.json

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/audit-spn-features \
  --trail-position-control-baseline-plan configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_512_local.csv \
  --row-index 3 \
  --samples-per-class 512 \
  --seed 1 \
  --top-k 16 \
  --control-active-nibbles 1 \
  --control-input-differences 0x90 \
  --control-pair-orders reverse \
  --output outputs/local_audits/i1_present_r8_trail_position_control_baseline_seed1_512.json
```

Results:

| Seed | Variant | Validation composite AUC | Validation best accuracy |
|---:|---|---:|---:|
| 0 | baseline | `0.7695465087890625` | `0.703125` |
| 0 | active_nibble_1 | `0.49724578857421875` | `0.515625` |
| 0 | input_difference_0x90 | `0.5139007568359375` | `0.5322265625` |
| 0 | pair_order_reverse | `0.7695465087890625` | `0.703125` |
| 1 | baseline | `0.8455047607421875` | `0.7734375` |
| 1 | active_nibble_1 | `0.49993133544921875` | `0.521484375` |
| 1 | input_difference_0x90 | `0.5224685668945312` | `0.5302734375` |
| 1 | pair_order_reverse | `0.8455047607421875` | `0.7734375` |

Control decision:

```text
trail_position_signal_requires_active_difference_alignment
pair_order_not_current_bottleneck
```

Interpretation:

```text
The active-nibble and input-difference controls collapse near chance across
both seeds, so the deterministic signal is not a generic artifact of the
matched-negative integral sample structure. It depends on the alignment between
the active plaintext nibble and the chosen input difference.

The pair-order reversal control exactly matches the baseline because the
selected features are dominated by order-invariant span/range statistics. This
is useful negative evidence against spending the next model slot on pair-order
sequence modeling for this route.
```

Updated next action:

```text
1. Keep trail-position as a controlled local SPN/integral representation
   candidate.
2. Do not remote-launch yet.
3. Do not prioritize pair-order models for this route.
4. If training another neural candidate, require it to beat the deterministic
   split baseline and include active-nibble/difference mismatch controls.
```

## Trail-Position Neural Residual Gate

Implementation:

```text
CLI = scripts/gate-trail-position-residual
API = gate_trail_position_residual
results = outputs/local_smoke/i1_present_r8_trail_position_beamstats_512/results.jsonl
baseline_audits =
  outputs/local_audits/i1_present_r8_trail_position_control_baseline_seed0_512.json
  outputs/local_audits/i1_present_r8_trail_position_control_baseline_seed1_512.json
margin = 0.01
output = outputs/local_audits/i1_present_r8_trail_position_residual_gate_512.json
```

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/gate-trail-position-residual \
  --results outputs/local_smoke/i1_present_r8_trail_position_beamstats_512/results.jsonl \
  --baseline-audit outputs/local_audits/i1_present_r8_trail_position_control_baseline_seed0_512.json \
  --baseline-audit outputs/local_audits/i1_present_r8_trail_position_control_baseline_seed1_512.json \
  --margin 0.01 \
  --output outputs/local_audits/i1_present_r8_trail_position_residual_gate_512.json
```

Gate result:

```text
status = pass
decision = support_trail_position_neural_residual_local
action = run_controlled_local_medium_diagnostic_before_remote_launch
pair_order_assessment = pair_order_not_bottleneck
min_candidate_margin_vs_deterministic_auc = 0.140472412109375
min_candidate_margin_vs_global_auc = 0.175262451171875
min_deterministic_margin_vs_mismatch_auc = 0.255645751953125
```

Per-seed summary:

| Seed | Neural candidate AUC | Global-stat control AUC | Deterministic baseline AUC | Max mismatch control AUC | Candidate vs deterministic | Candidate vs global |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | `0.98883056640625` | `0.813568115234375` | `0.7695465087890625` | `0.5139007568359375` | `+0.2192840576171875` | `+0.175262451171875` |
| 1 | `0.9859771728515625` | `0.7928619384765625` | `0.8455047607421875` | `0.5224685668945312` | `+0.140472412109375` | `+0.193115234375` |

Interpretation:

```text
The 512/class local diagnostic supports a possible neural residual over the
train-selected deterministic position-statistics baseline. The candidate clears
both the deterministic baseline and the same-input global-stat control on both
seeds, while the deterministic baseline stays well above active-nibble and
input-difference mismatch controls.

This is still local diagnostic evidence only. It justifies another controlled
local or medium diagnostic, not a remote launch, not a PRESENT r8 breakthrough,
and not a Zhang/Wang r7 Case2 claim.
```

Updated next action:

```text
1. Treat trail-position residual as the current best local SPN/integral
   architecture-representation candidate.
2. Before any remote launch, design a medium diagnostic with the same residual
   gate and disk-backed feature/cache readiness if scale reaches remote size.
3. Keep diverse neural ensemble as a later validator only after this route has
   compatible frozen score artifacts and low error-overlap evidence.
```
