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

## 2048/Class Train-Derived Hard Residual Slice Evaluation

Because the full held-out validation AUC is saturated, the next diagnostic
evaluated the same corrected score artifacts on the validation rows that are
hard for the frozen base score. The slice threshold is selected only from train
base residual loss:

```text
train_base = logit_mean(train trail_stats_logistic_scores, train span_raw117_matched_scores)
train_residual_loss = abs(label - sigmoid(train_base))
threshold = top residual_focus_fraction cutoff on train_residual_loss
validation_slice = validation rows where validation_base_residual_loss >= threshold
```

This avoids using validation labels to choose an after-the-fact slice. The
validation split is only scored after the train-derived threshold is frozen.

New local CLI:

```text
scripts/evaluate-residual-slice-correction
```

Artifacts:

```text
seed0 focus 5% =
  outputs/local_audits/i1_present_r8_seed0_aux_residual_focus05_slice_eval.json
seed1 focus 5% =
  outputs/local_audits/i1_present_r8_seed1_aux_residual_focus05_slice_eval.json

seed0 focus 10% =
  outputs/local_audits/i1_present_r8_seed0_aux_residual_focus10_slice_eval.json
seed1 focus 10% =
  outputs/local_audits/i1_present_r8_seed1_aux_residual_focus10_slice_eval.json

seed0 focus 20% =
  outputs/local_audits/i1_present_r8_seed0_aux_residual_focus20_slice_eval.json
seed1 focus 20% =
  outputs/local_audits/i1_present_r8_seed1_aux_residual_focus20_slice_eval.json

seed0 focus 10% label-shuffle control =
  outputs/local_audits/i1_present_r8_seed0_aux_residual_focus10_labelshuffle_slice_eval.json
seed1 focus 10% label-shuffle control =
  outputs/local_audits/i1_present_r8_seed1_aux_residual_focus10_labelshuffle_slice_eval.json
```

Train-derived validation hard-slice metrics:

| Seed | Route | Slice Rows | Base Slice AUC | Corrected Slice AUC | Delta AUC | Base Slice Residual Loss | Corrected Slice Residual Loss | Delta Loss |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | focus 5% | `140` | `0.9997938569367141` | `1.0` | `+0.00020614306328592402` | `0.07913075997560895` | `0.06574271403120033` | `-0.013388045944408622` |
| 0 | focus 10% | `229` | `0.9999226485148515` | `1.0` | `+0.00007735148514853574` | `0.05332041171749253` | `0.04405268186532514` | `-0.009267729852167388` |
| 0 | focus 20% | `432` | `0.99997854997855` | `1.0` | `+0.000021450021450042378` | `0.0308982023407884` | `0.02638943614039346` | `-0.004508766200394941` |
| 0 | focus 10% label-shuffle | `229` | `0.9999226485148515` | `0.7379331683168316` | `-0.2619894801980198` | `0.05332041171749253` | `0.3395830523144762` | `+0.2862626405969837` |
| 1 | focus 5% | `109` | `0.9983050847457627` | `1.0` | `+0.0016949152542372614` | `0.09033436224564191` | `0.07015369929492883` | `-0.02018066295071308` |
| 1 | focus 10% | `217` | `0.9995741781638563` | `1.0` | `+0.0004258218361437027` | `0.052570848049561913` | `0.04173601037835539` | `-0.01083483767120652` |
| 1 | focus 20% | `424` | `0.9998887108262108` | `0.9999777421652422` | `+0.00008903133903137572` | `0.030039898719409312` | `0.025137205725153558` | `-0.004902692994255754` |
| 1 | focus 10% label-shuffle | `217` | `0.9995741781638563` | `0.7989269289729177` | `-0.20064724919093857` | `0.052570848049561913` | `0.2924646814365919` | `+0.23989383338702996` |

Decision:

```text
decision = keep_focus05_focus10_for_medium_scale_residual_slice_gate
status = train_derived_hard_slice_diagnostic_pass_local_2048class
action =
  keep focus 5% and focus 10% as the next medium-scale candidates;
  treat focus 5% as the sharper hard-slice repair variant;
  treat focus 10% as the broader and still stable repair variant;
  keep the focus 10% label-shuffle control in future gates;
  evaluate future 262144/class runs with both global metrics and train-derived
  hard residual slice metrics
```

Interpretation:

```text
The hard-slice view is more informative than saturated global AUC. On both
seeds, focus 5% and focus 10% reduce validation residual loss on the train-
derived hard slice, and the label-shuffle control sharply worsens the same
slice. This supports the residual-focused aux-depth-word/aux-word correction as
the current strongest local candidate for medium-scale confirmation.

The claim remains local diagnostic only. The run size is 2048/class, the base
score is already extremely strong, and no formal SPN/PRESENT claim is allowed
until completed, retrieved, plan-aligned larger-scale evidence exists.
```

## 262144/Class Residual-Focus Action Planner

The medium-scale route should not reuse the older bucket-conditioned V16 gate as
the promoted candidate. A dedicated planner now emits the residual-focused
follow-up commands after the 262144/class trail-position score postprocess is
complete:

```text
scripts/plan-residual-focus-262k
```

Planner behavior:

```text
input =
  outputs/remote_results/i1_present_r8_trail_position_beamstats_262k_postprocess_status.json

pending condition =
  trail-position score artifacts are missing, incomplete, or not postprocessed

ready condition =
  status = pass
  score artifacts are complete enough to build train/validation frozen bases

source gate assessment =
  support_trail_position_score_residual_all_runs means the old trail-position
  scale gate passed;
  hold_trail_position_score_residual_mixed_runs is still allowed for this
  residual-focus diagnostic, because the purpose here is to test whether
  hard residual slices can be repaired when the old global-AUC margin gate is
  too saturated to be decisive

base =
  train/validation trail-position frozen scores
  train/validation raw117 matched compressed-span scores

candidates =
  residual-focus 5%
  residual-focus 10%

controls =
  uniform no-focus residual correction
  focus 10% label-shuffle residual correction

evaluation =
  global validation metrics from the residual-correction reports
  train-derived hard residual slice metrics from
  scripts/evaluate-residual-slice-correction

source selection =
  train-only residual bucket axis spectra from
  scripts/analyze-residual-bucket-axis-spectrum
  targets =
    residual_loss
    residual_error_at_0_5

gate =
  scripts/gate-residual-focus-262k

readiness audit =
  scripts/audit-residual-focus-262k-readiness
```

This planner does not launch remote jobs, does not SSH-poll, and does not prove
the 262144/class claim by itself. It is the command skeleton to run once the
source-publication and trail-position retrieval gates are both satisfied. If
the source gate assessment is mixed, the generated plan is only a residual
diagnostic follow-up, not promotion of the old trail-position scale gate.

The planner now emits `source_selection_commands` in addition to the candidate
and control commands. These commands use only the train compressed-span summary
features and train frozen trail+raw117 score artifacts:

```text
train_residual_loss_axis_spectrum =
  train_span_summary_features
  train_trail_position_scores
  train_raw117_scores
  target = residual_loss

train_hard_error_axis_spectrum =
  train_span_summary_features
  train_trail_position_scores
  train_raw117_scores
  target = residual_error_at_0_5
```

They are deliberately not validation-spectrum selection commands. Their role is
to identify which SPN feature families should seed the next residual expert
after the 262144/class residual-focus gate is available. They do not block or
pass the residual-focus gate, do not launch remote work, and do not make a
medium/formal claim by themselves.

The planner also emits:

```text
source_selection_summary_command =
  scripts/summarize-residual-axis-spectrum
```

This command consumes the train-only `train_residual_loss_axis_spectrum.json`
and `train_hard_error_axis_spectrum.json` reports across available seeds, then
writes:

```text
residual_axis_spectrum_summary.json
```

The summary selects recurring SPN axis groups and recommended feature prefixes
for the next residual source probe. It rejects validation feature reports by
default, so it is a train-derived source-selection aid rather than a held-out
validation structure search. `primary_*` groups are retained in `all_groups`
as secondary overlap with the current trail-position anchor, but the selected
residual-source recommendation prioritizes non-primary groups and stable
`residual_loss` evidence over sparse hard-error-only spikes. It remains
diagnostic only.

`scripts/advance-residual-focus-results` now runs this summary automatically
when every train-only source-selection report named by the action plan exists.
The generated summary is carried in the advance report via
`ran_source_selection_summary`, `source_selection_summary_status`,
`source_selection_summary_decision`, and `source_selection_summary_output`.
This automation does not alter the residual-focus gate, Pool 3 readiness, or
claim scope.

`scripts/residual-focus-status` also exposes that summary when it exists:

```text
source_selection_summary_status
source_selection_summary_decision
source_selection_recommended_feature_prefixes
source_selection_selected_groups
source_selection_report_count
source_selection_existing_report_count
source_selection_missing_report_count
source_selection_missing_reports
```

These status fields are observability only; the status branch and next action
remain controlled by retrieved residual-focus outputs, the gate, Pool 3 plan,
Pool 3 fixed-fusion evaluation, and repair plan.

For compatibility with action plans generated before
`source_selection_summary_output` existed, both
`scripts/advance-residual-focus-results` and `scripts/residual-focus-status`
fall back to:

```text
<artifact-root>/residual_axis_spectrum_summary.json
```

Local 2048/class sanity output from existing train spectrum reports:

```text
artifact =
  outputs/local_audits/i1_present_r8_residual_axis_spectrum_summary.json
recommended_feature_prefixes =
  aux_word_
  aux_depth_word_
  aux_cell_
decision = residual_axis_spectrum_stable_groups_selected
claim_scope = train-only source-selection diagnostic
```

The companion gate consumes the action plan's planned outputs after the
residual-focus commands finish. It keeps a candidate only when:

```text
1. focus05 or focus10 has a train-derived hard-slice residual-loss drop;
2. the candidate's hard-slice loss drop beats the uniform no-focus correction;
3. the focus10 label-shuffle control worsens hard-slice residual loss;
4. the result remains labeled as medium diagnostic evidence, not formal
   SPN/PRESENT evidence.
```

Current local gate status before running the residual-focus 262144/class
commands:

```text
status = pending
decision = wait_for_residual_focus_262k_outputs
missing_outputs =
  seed0 residual_uniform_slice_eval.json
  seed0 residual_focus10_labelshuffle_slice_eval.json
  seed0 residual_focus05_slice_eval.json
  seed0 residual_focus10_slice_eval.json
  seed1 residual_uniform_slice_eval.json
  seed1 residual_focus10_labelshuffle_slice_eval.json
  seed1 residual_focus05_slice_eval.json
  seed1 residual_focus10_slice_eval.json
```

Current readiness audit:

```text
status = pending
decision = residual_focus_262k_execution_not_ready
command_count = 20
control_command_count = 8
unsafe_command_count = 0
remote_checkpoint_seed_count = 0
missing_outputs_count = 8
blockers =
  gate_missing_outputs
next_action =
  finish_residual_focus_262k_outputs
```

Checkpoint resolution update:

```text
planner behavior =
  if models.json records a remote Windows checkpoint path but the retrieved
  checkpoint exists under run_root/checkpoints/<same filename>, use the local
  retrieved checkpoint for generated score-export commands and keep the remote
  path as remote_train_trail_position_checkpoint for provenance.

seed0 local checkpoint =
  outputs/remote_results/i1_present_r8_trail_position_beamstats_262k_seed0_gpu0_20260706/checkpoints/row0002_present_trail_position_stats_pairset_seed0.pt

seed1 local checkpoint =
  outputs/remote_results/i1_present_r8_trail_position_beamstats_262k_seed1_gpu1_20260706/checkpoints/row0002_present_trail_position_stats_pairset_seed1.pt

readiness impact =
  remote_checkpoint_reference_requires_remote_or_retrieved_checkpoint removed;
  the only current readiness blocker is the eight missing residual-focus
  slice-evaluation outputs listed above.
```

Current source-publication gate:

```text
script = scripts/check-launch-source
artifact = outputs/local_audits/i1_present_r8_residual_focus_262k_source_gate.json
status = fail
branch = main
upstream = origin/main
errors =
  unpushed_commits
should_push = true
```

This means the generated command plan is structurally safe and no longer blocked
by remote-only checkpoint paths. The remaining source-publication issue is that
`main` is ahead of `origin/main`; the exact ahead count changes as local
research commits accrue. The residual-focus 262144/class package should run
from pushed source for a normal remote claim, and the local readiness audit
must not be treated as permission to dirty-overlay or SSH-poll from the main
thread.

Remote package preparation:

```text
script =
  scripts/plan-residual-focus-remote-package

report =
  outputs/local_audits/i1_present_r8_residual_focus_262k_remote_package.json

launcher =
  configs/remote/generated/run_i1_present_r8_residual_focus_262k_20260707.cmd

launch_wrapper =
  configs/remote/generated/launch_i1_present_r8_residual_focus_262k_20260707.sh

monitor =
  configs/remote/generated/monitor_i1_present_r8_residual_focus_262k_20260707.sh

status = pending
decision = residual_focus_remote_package_blocked
launch_allowed = false
blockers =
  source_gate_not_pass

command_count = 20
control_command_count = 8
planned_output_count = 18
```

The package translates action-plan commands into remote Windows commands rooted
under `G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_residual_focus_262k`
and uses the existing remote 262144/class trail-position run artifacts as frozen
inputs. It is prepared only: the source-publication gate must pass before launch,
and launch still needs the normal remote handoff and local tmux monitor discipline.
The generated launch wrapper was locally dry-run while `launch_allowed=false`;
it exited before SSH and wrote:

```text
outputs/remote_results/i1_present_r8_residual_focus_262k/monitor/launch_blocked.marker
```

## 2026-07-07 Remote Launch Update

The source-publication gate was later cleared and the residual-focus remote
package became launchable:

```text
source_gate =
  outputs/local_audits/i1_present_r8_residual_focus_262k_source_gate.json
source_gate_status = pass
branch = main
upstream = origin/main
ahead = 0
behind = 0
dirty = false

default_package =
  outputs/local_audits/i1_present_r8_residual_focus_262k_remote_package.json
default_package_status = pass
default_package_decision = residual_focus_remote_package_ready
default_launch_allowed = true
```

The first default remote launch reached the remote run directory and cloned the
pushed source, but failed at command 0 before feature export:

```text
run_id = i1_present_r8_residual_focus_262k
remote_revision = 7c015c4e3628d145708e6ac3559ff14f0707b2a9
failure_stage = command_0
failure =
  ModuleNotFoundError: No module named 'blockcipher_nd'
local_retrieved_logs =
  outputs/remote_results/i1_present_r8_residual_focus_262k/logs/
status = failed_launch_not_training_result
```

This was an environment/bootstrap bug in the generated remote package, not a
negative residual-focus experiment result. The package generator was fixed to
emit:

```text
set PYTHONPATH=%SOURCE_ROOT%\src;%PYTHONPATH%
```

before running the `scripts\...` Python entrypoints. It was also extended to
support an isolated retry run id and to mark generated local `.sh` launcher and
monitor scripts executable.

Fix commits:

```text
000fae2 fix: set residual focus remote pythonpath
6ae26d5 fix: make residual focus remote scripts executable
```

Relevant verification:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/test_residual_focus_remote_package.py \
  tests/test_check_launch_source.py -q

result = 8 passed
```

The active remote run is now the isolated retry package:

```text
run_id = i1_present_r8_residual_focus_262k_retry1
retry_package =
  outputs/local_audits/i1_present_r8_residual_focus_262k_retry1_remote_package.json
retry_package_status = pass
retry_package_decision = residual_focus_remote_package_ready
retry_launch_allowed = true

remote_root =
  G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_residual_focus_262k_retry1
remote_revision =
  6ae26d50f8052c24182eb40f616c4e6d487cf815
local_launch_session =
  launch_i1_present_r8_residual_focus_262k_retry1_20260707
local_monitor_session =
  monitor_i1_present_r8_residual_focus_262k_retry1_20260707
```

One bounded read-only launch confirmation found:

```text
started_marker = present
command_0_marker = present
git_revision = 6ae26d50f8052c24182eb40f616c4e6d487cf815
pythonpath =
  G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_residual_focus_262k_retry1\source\src;
```

Current interpretation:

```text
planned = yes
launched_from_pushed_commit = yes
running_or_waiting_for_monitor = yes
completed_remotely = no
retrieved_result_outputs = no
residual_focus_gate_passed = no
formal_or_breakthrough_claim = no
```

The local monitor is responsible for retrieving the 18 planned outputs under
`outputs/local_audits/i1_present_r8_residual_focus_262k/`. Until those outputs
exist and `scripts/gate-residual-focus-262k` is run, the experiment status
remains running/pending, not complete.

## Claim Scope

This is a local diagnostic plan and tooling record only. It does not report a
formal PRESENT r8 result, does not claim a breakthrough, does not replace the
262144/class trail-position postprocess gate, and does not provide formal
SPN/PRESENT evidence.
