# Innovation 1 PRESENT r8 Residual Bucket Axis Spectrum Plan

**Date:** 2026-07-07

**Status:** local axis-spectrum diagnostic complete; aux-depth-word global probe held

## Question

The current best local third-family candidate is:

```text
trail-position neural expert
+ matched raw117 compressed SPN structural expert
+ bucket-conditioned residual feature expert
```

The open question is not whether to add more similar neural networks. The open
question is:

```text
Inside each trail+raw117 residual reliability bucket, which SPN depth/word/cell
feature groups explain the remaining threshold errors or soft residual loss?
```

If those groups are stable, they can become a better next candidate source than
generic near-neighbor score averaging. The current 262144/class trail-position
result is strong but mixed: the candidate is nearly perfect, while the same-input
global control is also nearly perfect. That means the next useful question is
not "can another similar model average help?", but "which SPN axes still carry
independent residual signal after the current strong scores are frozen?"

## Diagnostic Tool

New local CLI:

```text
scripts/analyze-residual-bucket-axis-spectrum
```

Inputs:

```text
feature_dir = existing bit-sensitivity / compressed SPN feature artifact
bucket_artifacts = exactly two aligned frozen score artifacts
bucket_feature = logit_gap_abs by default
bucket_count = train/local diagnostic bucket count
target = residual_error_at_0_5 by default
```

Output:

```text
outputs/local_audits/i1_present_r8_residual_bucket_axis_spectrum.json
```

The tool:

```text
1. validates labels and sample_ids against the frozen score artifacts;
2. computes the same residual bucket values used by V16/V17;
3. groups feature columns by semantic axis names such as
   primary_depth_mean or aux_word_mean;
4. reports per-bucket label AUC, hard residual-error AUC, and selected target
   AUC for each group;
5. ranks groups by AUC distance from 0.5;
6. also reports `global_top_groups` so sparse buckets do not hide useful soft
   residual axes.
```

Supported targets:

```text
residual_error_at_0_5 = hard mistakes from mean frozen probability at threshold 0.5
residual_loss = abs(label - mean frozen probability)
signed_margin = signed distance from the correct side of threshold
global_candidate_gap = absolute probability gap between the two frozen artifacts
```

Continuous targets are binarized at the median for AUC ranking. This is a
diagnostic ranking device only; it is not a replacement for held-out validation
or a publication-style metric.

## Why Continuous Targets Matter

On very strong local validation artifacts, hard 0.5-threshold errors can be too
sparse to rank feature axes reliably. A bucket may contain zero hard mistakes or
only one residual class, which makes per-bucket residual-error AUC undefined.

`residual_loss` keeps the same frozen labels and probabilities but asks a softer
question:

```text
Which SPN feature groups correlate with samples that the current frozen scores
still find less comfortable, even when they remain on the correct side of 0.5?
```

This is useful for source selection. A group that consistently explains soft
residual loss is a better candidate for a non-neighbor residual expert than a
new model that only re-learns the same trail-position/global-stat signal.

## Guardrails

This diagnostic must not change:

```text
labels
negative_mode
sample_structure
validation keys
metric computation for existing gates
```

It is not a new trained expert yet. It is a source-selection diagnostic for a
future residual structural expert.

## Intended Use After 262k Retrieval

After the 262144/class trail-position score artifacts are complete and the
V16/V17 bucket residual planner is ready:

```text
1. export train/validation compressed SPN span features;
2. run trail+raw117 frozen score exports;
3. run `scripts/analyze-residual-bucket-axis-spectrum` on train artifacts with
   both `residual_error_at_0_5` and `residual_loss`;
4. freeze any candidate group selection before validation scoring;
5. require shuffle-label, train-bucket-shuffle, validation-bucket-shuffle, and
   no-bucket controls before any remote-scale claim.
```

## 2048/Class Train/Validation Axis Spectrum

The first same-protocol local diagnostic used existing 2048/class train and
held-out validation artifacts:

```text
feature_view = compressed_span_summary
bucket_artifacts =
  trail_stats_logistic_scores
  span_raw117_matched_scores
bucket_feature = logit_gap_abs
bucket_count = 5
targets =
  residual_error_at_0_5
  residual_loss
```

Artifacts:

```text
seed0 train residual_loss =
  outputs/local_audits/i1_present_r8_seed0_train_residual_loss_axis_spectrum.json
seed0 validation residual_loss =
  outputs/local_audits/i1_present_r8_seed0_validation_residual_loss_axis_spectrum.json
seed1 train residual_loss =
  outputs/local_audits/i1_present_r8_seed1_train_residual_loss_axis_spectrum.json
seed1 validation residual_loss =
  outputs/local_audits/i1_present_r8_seed1_validation_residual_loss_axis_spectrum.json

seed0 train hard error =
  outputs/local_audits/i1_present_r8_seed0_train_hard_error_axis_spectrum.json
seed1 train hard error =
  outputs/local_audits/i1_present_r8_seed1_train_hard_error_axis_spectrum.json
```

Hard 0.5-threshold residual errors are too sparse for source selection:

| Split | Rows | Hard Error Rate | Soft Residual Mean |
|---|---:|---:|---:|
| seed0 train | `4096` | `0.0` | `0.006508632420401186` |
| seed0 validation | `2048` | `0.0009765625` | `0.009336364924479985` |
| seed1 train | `4096` | `0.000244140625` | `0.007914455329723208` |
| seed1 validation | `2048` | `0.0009765625` | `0.009156557486921957` |

The recurring `residual_loss` axis families are:

| Axis Family | Evidence | Interpretation |
|---|---|---|
| `aux_word_global_mean` / `aux_word_mean` | appears in seed0 validation and both seed1 splits | stable weak auxiliary residual signal |
| `aux_depth_word_*` | dominates seed1 train and validation, especially global/depth/trailword aggregates | strongest non-neighbor residual candidate family, but not stable enough at exact depth index |
| `primary_depth_cell/trailword depth1-depth2` | appears across seed0 validation and seed1 train | strong but likely overlaps the already dominant primary/trail signal |
| `aux_cell_*` | appears mainly in seed0 train/validation | seed-dependent auxiliary cell residual; keep as secondary context, not first next expert |

Decision:

```text
decision = aux_depth_word_aux_word_residual_family_selected_for_probe
status = local_source_selection_diagnostic_only
action =
  do not freeze a single depth index;
  use the family-level aux_depth_word_ + aux_word_ scope as a sanity probe;
  treat primary_depth_* recurrence as evidence of strong existing signal, not
  as a new non-neighbor expert by itself
```

## 2048/Class Aux-Depth-Word/Aux-Word Probe

The next local probe tested whether the axis-spectrum-selected auxiliary
families should be added as an ordinary global frozen-score expert:

```text
cli = scripts/fit-compressed-feature-expert
selected_prefixes =
  aux_depth_word_
  aux_word_
feature_count = 96
fit_split = train
score_split = held-out validation
steps = 2000
learning_rate = 0.05
l2 = 0.001
standardize = true
```

Artifacts:

```text
seed0 probe report =
  outputs/local_audits/i1_present_r8_seed0_aux_depth_word_aux_word_probe_report.json
seed1 probe report =
  outputs/local_audits/i1_present_r8_seed1_aux_depth_word_aux_word_probe_report.json

seed0 trail+raw117+aux probe ensemble =
  outputs/local_audits/i1_present_r8_seed0_trail_raw117_auxword_probe_three_score_ensemble.json
seed1 trail+raw117+aux probe ensemble =
  outputs/local_audits/i1_present_r8_seed1_trail_raw117_auxword_probe_three_score_ensemble.json
```

Metrics:

| Seed | Aux Probe AUC | Trail+Raw117 AUC | Trail+Raw117+Aux Probe AUC | Three-vs-Two Delta |
|---:|---:|---:|---:|---:|
| 0 | `0.960240364074707` | `0.9999990463256836` | `0.9999790191650391` | `-0.00002002716064453125` |
| 1 | `0.9550857543945312` | `0.999995231628418` | `0.9999704360961914` | `-0.0000247955322265625` |

The probe is meaningfully non-random and less correlated with the strong
structural experts, but direct global score averaging is worse:

```text
seed0 aux-pair probability correlation ~= 0.84, disagreement ~= 0.098
seed1 aux-pair probability correlation ~= 0.83, disagreement ~= 0.111
```

Decision:

```text
decision = hold_aux_depth_word_aux_word_as_global_ensemble_member
status = useful_residual_source_but_not_global_vote_expert
action =
  do not add aux_depth_word_ + aux_word_ as a plain fourth/global ensemble
  score;
  use this family only through bucket-conditioned or residual-conditioned
  selection;
  next implementation candidate should combine the axis-spectrum family with
  train-derived residual buckets rather than average it globally
```

Interpretation:

```text
The auxiliary depth-word/word family has real information, but it is weaker
than the existing trail-position and raw117 anchors. Its value is not as a
standalone or globally averaged expert. The correct next architecture is a
conditional residual expert: use the current strong scores to identify where
the sample is uncertain or where experts disagree, then apply the auxiliary
family only where it explains residual loss.
```

## 2048/Class Bucket-Conditioned Aux-Depth-Word/Aux-Word Probe

The follow-up probe tested whether the same auxiliary family becomes useful
when trained as a bucket-conditioned classifier over train-derived
`logit_gap_abs` reliability buckets:

```text
cli = scripts/fit-bucket-conditioned-feature-expert
selected_prefixes =
  aux_depth_word_
  aux_word_
feature_count = 96
bucket_artifacts =
  trail_stats_logistic_scores
  span_raw117_matched_scores
bucket_feature = logit_gap_abs
bucket_count = 5
fit_split = train
score_split = held-out validation
steps = 1000
learning_rate = 0.05
l2 = 0.0003
standardize = true
```

Artifacts:

```text
seed0 bucket probe report =
  outputs/local_audits/i1_present_r8_seed0_bucket_aux_depth_word_aux_word_probe_report.json
seed1 bucket probe report =
  outputs/local_audits/i1_present_r8_seed1_bucket_aux_depth_word_aux_word_probe_report.json

seed0 trail+raw117+bucket aux probe ensemble =
  outputs/local_audits/i1_present_r8_seed0_trail_raw117_bucket_auxword_probe_three_score_ensemble.json
seed1 trail+raw117+bucket aux probe ensemble =
  outputs/local_audits/i1_present_r8_seed1_trail_raw117_bucket_auxword_probe_three_score_ensemble.json
```

Metrics:

| Seed | Bucket Aux Probe AUC | Trail+Raw117 AUC | Trail+Raw117+Bucket Aux AUC | Three-vs-Two Delta |
|---:|---:|---:|---:|---:|
| 0 | `0.9528608322143555` | `0.9999990463256836` | `0.9999780654907227` | `-0.0000209808349609375` |
| 1 | `0.9466629028320312` | `0.999995231628418` | `0.9999561309814453` | `-0.00003910064697265625` |

Decision:

```text
decision = hold_bucket_conditioned_aux_depth_word_aux_word_classifier
status = useful_axis_family_but_wrong_training_objective
action =
  do not promote this bucket-conditioned aux-depth-word/word classifier;
  do not remote-scale this route;
  preserve the aux-depth-word/word family as a residual-source candidate;
  next architecture should predict residual_loss, signed correction, or a
  learned gate over frozen strong scores rather than reclassifying labels
  globally
```

Interpretation:

```text
Bucket conditioning alone is not enough. The auxiliary family still contains
non-random SPN signal, but training it as another ordinary label classifier
does not improve the strong trail+raw117 anchor. The failure mode points to the
objective, not necessarily the feature family: the next probe should learn a
small residual/gated correction on top of frozen strong scores.
```

## Claim Scope

This is a local diagnostic plan and tooling record only. It does not report a
formal PRESENT r8 result, does not claim a breakthrough, does not replace the
262144/class trail-position postprocess gate, and does not provide formal
SPN/PRESENT evidence.
