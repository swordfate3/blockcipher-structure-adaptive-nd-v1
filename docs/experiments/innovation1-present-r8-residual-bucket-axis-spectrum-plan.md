# Innovation 1 PRESENT r8 Residual Bucket Axis Spectrum Plan

**Date:** 2026-07-07

**Status:** local axis-spectrum diagnostic complete; aux-depth-word global/bucket probes held; residual correction probe held pending stronger attribution

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

## 2048/Class Residual Logit Correction Probe

The next implementation changed the objective from ordinary label
classification to frozen-score correction:

```text
base_score =
  logit_mean(trail_stats_logistic_scores, span_raw117_matched_scores)

corrected_score =
  base_score + residual_correction(aux_depth_word_, aux_word_, reliability views, optional buckets)
```

New local CLI:

```text
scripts/fit-residual-correction-feature-expert
```

The base score artifacts stay frozen. The fitted model only learns an additive
correction from train split features. Validation remains final-only. This tests
the hypothesis that the auxiliary family should repair residual uncertainty
rather than vote globally as another classifier.

Primary probe settings:

```text
selected_prefixes =
  aux_depth_word_
  aux_word_
selected_feature_count = 96
base_fusion = logit_mean
reliability_features =
  min_confidence
  confidence_gap_abs
  logit_gap_abs
  signed_logit_delta_model1_minus_model0
bucket_feature = logit_gap_abs
bucket_count = 5
correction_feature_count = 585
steps = 1000
learning_rate = 0.05
l2 = 0.001
```

Artifacts:

```text
seed0 primary correction report =
  outputs/local_audits/i1_present_r8_seed0_aux_residual_correction_report.json
seed1 primary correction report =
  outputs/local_audits/i1_present_r8_seed1_aux_residual_correction_report.json

seed0 no-bucket control =
  outputs/local_audits/i1_present_r8_seed0_aux_residual_correction_nobucket_report.json
seed1 no-bucket control =
  outputs/local_audits/i1_present_r8_seed1_aux_residual_correction_nobucket_report.json

seed0 label-shuffle control =
  outputs/local_audits/i1_present_r8_seed0_aux_residual_correction_labelshuffle_report.json
seed1 label-shuffle control =
  outputs/local_audits/i1_present_r8_seed1_aux_residual_correction_labelshuffle_report.json

seed0 train-bucket-shuffle control =
  outputs/local_audits/i1_present_r8_seed0_aux_residual_correction_trainbucketshuffle_report.json
seed1 train-bucket-shuffle control =
  outputs/local_audits/i1_present_r8_seed1_aux_residual_correction_trainbucketshuffle_report.json

seed0 validation-bucket-shuffle control =
  outputs/local_audits/i1_present_r8_seed0_aux_residual_correction_valbucketshuffle_report.json
seed1 validation-bucket-shuffle control =
  outputs/local_audits/i1_present_r8_seed1_aux_residual_correction_valbucketshuffle_report.json
```

Metrics:

| Seed | Route | Base Logit Mean AUC | Corrected AUC | Delta |
|---:|---|---:|---:|---:|
| 0 | bucketed correction | `0.9999990463256836` | `1.0` | `+0.00000095367431640625` |
| 0 | no-bucket control | `0.9999990463256836` | `0.9999990463256836` | `0.0` |
| 0 | label-shuffle control | `0.9999990463256836` | `0.7959432601928711` | `-0.2040557861328125` |
| 0 | train-bucket-shuffle control | `0.9999990463256836` | `1.0` | `+0.00000095367431640625` |
| 0 | validation-bucket-shuffle control | `0.9999990463256836` | `0.9999990463256836` | `0.0` |
| 1 | bucketed correction | `0.999995231628418` | `0.9999961853027344` | `+0.00000095367431640625` |
| 1 | no-bucket control | `0.999995231628418` | `0.9999971389770508` | `+0.0000019073486328125` |
| 1 | label-shuffle control | `0.999995231628418` | `0.8183321952819824` | `-0.18166303634643555` |
| 1 | train-bucket-shuffle control | `0.999995231628418` | `0.9999971389770508` | `+0.0000019073486328125` |
| 1 | validation-bucket-shuffle control | `0.999995231628418` | `0.9999961853027344` | `+0.00000095367431640625` |

Decision:

```text
decision = hold_aux_residual_correction_bucket_gate
status = residual_objective_tool_ready_but_current_bucket_gate_not_attributed
action =
  do not remote-scale this exact bucketed correction route;
  keep the residual-correction objective and CLI as useful infrastructure;
  do not claim that logit_gap_abs bucket gating adds independent signal here;
  next residual architecture should target a harder residual slice, use a
  train-holdout selected correction family, or optimize soft residual loss more
  directly before medium-scale promotion
```

Interpretation:

```text
This probe is better aligned with the research direction than plain multi-score
averaging because it freezes the strong trail+raw117 score and learns only a
small correction. However, the measured AUC gains are at the 1e-6 level on an
already saturated 2048/class local split. The label-shuffle controls collapse,
which argues against a simple leakage bug, but no-bucket and bucket-shuffle
controls match or exceed the primary bucketed route. Therefore the current
evidence supports the residual-correction objective as tooling, not this
specific bucket gate as a candidate for remote scale-up.
```

## 2048/Class Residual-Focused Correction Follow-up

The next local follow-up kept the same frozen base score and auxiliary feature
family, but changed the fitting objective to focus on rows that the frozen base
score finds least comfortable:

```text
base_score = logit_mean(trail_stats_logistic_scores, span_raw117_matched_scores)
selected_prefixes = aux_depth_word_ + aux_word_
bucket_count = 0
reliability_features = enabled
residual_focus =
  rank train rows by abs(label - sigmoid(base_score))
  give the top residual-loss rows full weight
  give background rows weight 0.1
validation = full held-out split, not residual-sliced
```

This directly tests the hypothesis from the previous section:

```text
The auxiliary depth-word/word family should be used to repair the frozen strong
score's soft residual cases, not as a global classifier and not as a
logit_gap_abs bucket gate.
```

New CLI options:

```text
--residual-focus-fraction
--residual-focus-background-weight
```

Artifacts:

```text
seed0 focus 5% =
  outputs/local_audits/i1_present_r8_seed0_aux_residual_focus05_nobucket_report.json
seed1 focus 5% =
  outputs/local_audits/i1_present_r8_seed1_aux_residual_focus05_nobucket_report.json

seed0 focus 10% =
  outputs/local_audits/i1_present_r8_seed0_aux_residual_focus10_nobucket_report.json
seed1 focus 10% =
  outputs/local_audits/i1_present_r8_seed1_aux_residual_focus10_nobucket_report.json

seed0 focus 20% =
  outputs/local_audits/i1_present_r8_seed0_aux_residual_focus20_nobucket_report.json
seed1 focus 20% =
  outputs/local_audits/i1_present_r8_seed1_aux_residual_focus20_nobucket_report.json

seed0 focus 10% label-shuffle control =
  outputs/local_audits/i1_present_r8_seed0_aux_residual_focus10_labelshuffle_report.json
seed1 focus 10% label-shuffle control =
  outputs/local_audits/i1_present_r8_seed1_aux_residual_focus10_labelshuffle_report.json
```

Metrics:

| Seed | Route | Focused Train Rows | Base Logit Mean AUC | Corrected AUC | Delta | Validation Accuracy |
|---:|---|---:|---:|---:|---:|---:|
| 0 | uniform no-bucket | n/a | `0.9999990463256836` | `0.9999990463256836` | `0.0` | `0.99951171875` |
| 0 | focus 5% | `205` | `0.9999990463256836` | `1.0` | `+0.00000095367431640625` | `0.99951171875` |
| 0 | focus 10% | `410` | `0.9999990463256836` | `1.0` | `+0.00000095367431640625` | `0.99951171875` |
| 0 | focus 20% | `820` | `0.9999990463256836` | `1.0` | `+0.00000095367431640625` | `0.99951171875` |
| 0 | focus 10% label-shuffle | `410` | `0.9999990463256836` | `0.8786773681640625` | `-0.1213216781616211` | `0.79345703125` |
| 1 | uniform no-bucket | n/a | `0.999995231628418` | `0.9999971389770508` | `+0.0000019073486328125` | `0.9990234375` |
| 1 | focus 5% | `205` | `0.999995231628418` | `1.0` | `+0.00000476837158203125` | `0.9990234375` |
| 1 | focus 10% | `410` | `0.999995231628418` | `1.0` | `+0.00000476837158203125` | `0.9990234375` |
| 1 | focus 20% | `820` | `0.999995231628418` | `0.9999990463256836` | `+0.000003814697265625` | `0.99951171875` |
| 1 | focus 10% label-shuffle | `410` | `0.999995231628418` | `0.9135026931762695` | `-0.08649253845214844` | `0.830078125` |

Decision:

```text
decision = keep_residual_focused_aux_correction_as_local_candidate
status = local_2048class_candidate_needs_medium_scale_gate
action =
  keep focus 5% and focus 10% as the current best residual-correction variants;
  do not claim a breakthrough from 2048/class saturated validation;
  do not use the older logit_gap_abs bucketed gate as the promoted route;
  next meaningful experiment should run the residual-focused no-bucket
  correction at 262144/class with the same frozen base artifacts, if source
  publication/push gate is available
```

Interpretation:

```text
The residual-focused objective is a stronger match to the observed failure
mode than global aux voting or bucketed aux classification. Across seed0 and
seed1, focus 5% and focus 10% reach AUC 1.0 on this local 2048/class held-out
split, while the focus 10% label-shuffle control drops sharply. This supports
the route as a local candidate.

However, this remains a tiny saturated validation setting: the base score is
already between 0.999995 and 0.999999 AUC. The useful claim is not "formal
PRESENT r8 solved"; it is "the residual-focused correction objective is now the
best local next candidate for a medium-scale SPN/PRESENT diagnostic."
```

## Claim Scope

This is a local diagnostic plan and tooling record only. It does not report a
formal PRESENT r8 result, does not claim a breakthrough, does not replace the
262144/class trail-position postprocess gate, and does not provide formal
SPN/PRESENT evidence.
