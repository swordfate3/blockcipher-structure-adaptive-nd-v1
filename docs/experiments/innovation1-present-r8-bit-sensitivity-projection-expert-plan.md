# Innovation 1 PRESENT r8 Bit-Sensitivity Projection Expert Plan

**Date:** 2026-07-07

**Status:** v0/v1 local seed0 screens held; full 262144/class activation still
blocked on active trail-position score artifacts; no remote launch asset

## Question

Can a bit-sensitivity-guided projection expert become a real non-neighbor
PRESENT r8 expert under the current matched-negative trail-position protocol?

The target is not a higher-capacity model. The target is a different, compact
representation selected from stable trail-position residual sensitivity axes.

## Dependency

The medium-scale route remains inactive until:

```text
active_summary_branch = wait_for_trail_position_262k_results
required_postprocess = scripts/postprocess-trail-position-result
required_report = scripts/render-trail-position-report
required_runs =
  i1_present_r8_trail_position_beamstats_262k_seed0_gpu0_20260706
  i1_present_r8_trail_position_beamstats_262k_seed1_gpu1_20260706
required_score_rows = 262144
```

If those runs are still running or score artifacts are missing, do not start
the medium-scale version of this experiment.

## 2048/Class V0 Local Screen

A seed0 local diagnostic was run against the existing 2048/class frozen
trail-position artifacts to test the mechanics before the active 262144/class
artifacts exist:

```text
run_id = i1_present_r8_bit_sensitivity_projection_2048_seed0_local_20260707
selection_split = train
validation_split = validation
top_k = 64 raw feature axes
score_split_export = train support added to scripts/export-checkpoint-scores
gate = outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_gate.json
decision = hold_projection_duplicate_or_weak
action = do_not_promote_projection_to_diverse_pool_or_remote_scale
```

Metrics:

| Artifact | AUC |
|---|---:|
| Same-input global control | `0.8542919158935547` |
| Trail-position anchor | `0.9985876083374023` |
| Bit-sensitivity raw-axis projection | `0.49335479736328125` |

Gate deltas:

```text
projection_margin_vs_global_auc = -0.36093711853027344
projection_margin_vs_anchor_auc = -0.5052328109741211
hold_reasons = projection_auc_below_gate, does_not_clear_global_control
anchor_projection_error_jaccard = 0.02153558052434457
anchor_projection_probability_correlation = -0.012520744853315167
```

Interpretation:

```text
The v0 train-only raw-axis projection is not a useful non-neighbor expert.
Its low overlap with trail-position is not useful diversity; it comes with
near-random validation AUC. Do not run seed1 or remote scale for this exact
raw-axis projection. Keep the tooling because it is needed for aligned
train/validation feature export and future structure-aware projection variants.
```

Next route refinement:

```text
do_not_expand_v0_raw_axis_projection
wait_for_262144_class_trail_position_score_artifacts
if projection is revisited, use grouped/structured axes or residual summaries,
not individual raw feature columns alone
```

## V1 Grouped-Axis Local Screen

The 2026-07-07 follow-up implemented grouped-axis projection support without
starting another remote branch:

```text
selector = scripts/select-bit-sensitivity-projection
new selector args = --group-size, --top-groups
scorer = scripts/apply-bit-sensitivity-projection
projection_unit = contiguous_axis_group
selection_split = train only
status = local seed0 screen held
```

Rationale:

```text
The v0 single-axis mask was near random. That failure is consistent with the
external evidence: SPN neural distinguishers benefit from structured
multi-pair/derived representations and convolutional or grouped views, not
isolated scalar feature columns.
```

The grouped selector ranks contiguous feature blocks using the same train-only
residual/class statistics, expands the selected block axes for auditability,
and writes `selected_groups` into the frozen mask. The scorer reads those
groups and scores the mean group response as one projection unit. This keeps
the validation split untouched and preserves compatibility with the existing
frozen-score postprocess gate.

Local 2048/class seed0 results:

| Variant | Projection units | Validation AUC | Decision |
|---|---:|---:|---|
| Group size 4, top 16 groups | 16 | `0.5030102729797363` | hold |
| Group size 8, top 8 groups | 8 | `0.5734086036682129` | hold |
| Group size 16, top 4 groups | 4 | `0.5104804039001465` | hold |
| Group size 32, top 2 groups | 2 | `0.5292434692382812` | hold |

Comparison anchors:

| Artifact | AUC |
|---|---:|
| Same-input global control | `0.8542919158935547` |
| Trail-position anchor | `0.9985876083374023` |

Gate outputs:

```text
group4 decision = hold_projection_duplicate_or_weak
group8 decision = hold_projection_duplicate_or_weak
group16 decision = hold_projection_duplicate_or_weak
group32 decision = hold_projection_duplicate_or_weak
shared hold reason = does_not_clear_global_control
```

Interpretation:

```text
Grouped-axis projection is less dead than raw single-axis projection, but it
still loses badly to the same-input global-stat control. The best grouped
variant, group8, reaches only 0.5734 AUC versus 0.8543 for the global control.
Do not run seed1 or remote scale for this exact contiguous-group mean variant.
Keep the tooling because it is useful for future structure-aware residual
summaries, but do not promote the current grouped projection as a diverse
expert.
```

Allowed next use:

```text
only after 262144/class trail-position score artifacts are retrieved and
verified, use grouped export/scoring as a diagnostic support tool; do not
repeat this exact group-mean screen unless a new structural residual summary
changes the hypothesis
```

Not allowed:

```text
remote launch from grouped-axis tooling alone
claiming that grouped-axis projection is a trained model result
claiming diverse-expert readiness without same-protocol AUC and overlap gates
```

## V2 Trail-Position Structural-Stats Local Screen

The next local-only follow-up changes the representation rather than merely
changing the number of contiguous raw axes. `export-bit-sensitivity-features`
now supports:

```text
--feature-view trail_position_stats
```

This deterministic view reuses the PRESENT trail-position model's internal
position statistics, compressing the raw validation matrix from `39936` bits to
`3708` structure-aware features. The selector still uses only train-split
scores/features, and the frozen scorer is still applied only on the held-out
validation split.

2048/class local screen artifacts:

```text
train_features =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed{seed}_train_trail_stats_features
validation_features =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed{seed}_validation_trail_stats_features
mask =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed{seed}_trail_stats_mask.json
score_artifact =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed{seed}_trail_stats_scores
gate =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed{seed}_trail_stats_gate.json
seed0 decision = projection_expert_ready_for_local_screen
seed1 decision = projection_expert_ready_for_local_screen
```

Metrics:

| Seed | Global control AUC | Structural-stats projection AUC | Trail-position anchor AUC | Projection margin vs global |
|---:|---:|---:|---:|---:|
| 0 | `0.8542919158935547` | `0.9096775054931641` | `0.9985876083374023` | `+0.055385589599609375` |
| 1 | `0.8728437423706055` | `0.9378452301025391` | `0.9982948303222656` | `+0.0650014877319336` |

Gate overlap and ensemble diagnostics:

```text
seed0 anchor_projection_error_jaccard = 0.057441253263707574
seed1 anchor_projection_error_jaccard = 0.08157099697885196
seed0 best_ensemble_delta_vs_best_single_auc = -0.001667022705078125
seed1 best_ensemble_delta_vs_best_single_auc = -0.001583099365234375
```

Train-fitted stacking diagnostic:

```text
cli = scripts/evaluate-stacked-ensemble
fit_split = train score artifacts
eval_split = held-out validation score artifacts
feature_space = logits
claim_scope = train-fitted validation-evaluated frozen-score stacking diagnostic only

seed0 stacking_artifact =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_trail_stats_stacking.json
seed0 stacked_validation_auc = 0.998295783996582
seed0 best_single_validation_auc = 0.9985876083374023
seed0 best_fixed_ensemble_validation_auc = 0.9969205856323242
seed0 delta_stacked_vs_best_single_auc = -0.0002918243408203125
seed0 delta_stacked_vs_fixed_ensemble_auc = +0.0013751983642578125

seed1 stacking_artifact =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed1_trail_stats_stacking.json
seed1 stacked_validation_auc = 0.9975728988647461
seed1 best_single_validation_auc = 0.9982948303222656
seed1 best_fixed_ensemble_validation_auc = 0.9967117309570312
seed1 delta_stacked_vs_best_single_auc = -0.0007219314575195312
seed1 delta_stacked_vs_fixed_ensemble_auc = +0.0008611679077148438

both decisions = stacked_ensemble_diagnostic_no_best_single_gain
```

Train-holdout selected stacking diagnostic:

```text
cli = scripts/evaluate-stacked-ensemble
selection_split = deterministic train holdout only
train_holdout_fraction = 0.25
candidate_feature_spaces = logits, probabilities
candidate_l2 = 0.0, 0.0001, 0.001, 0.01
candidate_standardize = both

seed0 stacking_artifact =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_trail_stats_stacking_train_holdout.json
seed0 selected = logits, l2=0.0, standardize=false
seed0 stacked_validation_auc = 0.9985494613647461
seed0 best_single_validation_auc = 0.9985876083374023
seed0 delta_stacked_vs_best_single_auc = -0.00003814697265625
seed0 delta_stacked_vs_fixed_ensemble_auc = +0.001628875732421875

seed1 stacking_artifact =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed1_trail_stats_stacking_train_holdout.json
seed1 selected = probabilities, l2=0.0, standardize=false
seed1 stacked_validation_auc = 0.9984188079833984
seed1 best_single_validation_auc = 0.9982948303222656
seed1 delta_stacked_vs_best_single_auc = +0.0001239776611328125
seed1 delta_stacked_vs_fixed_ensemble_auc = +0.0017070770263671875

decision = mixed_train_holdout_stacking_diagnostic
```

Train-holdout selection-seed stability:

```text
stability_cli = scripts/summarize-stacked-selection
selection_seeds = 0, 1, 2, 3, 4

seed0 delta_stacked_vs_best_single_auc:
  min = -0.0000400543212890625
  max = -0.00003814697265625
  mean = -0.0000385284423828125
  positive_selection_seeds = 0 / 5

seed1 delta_stacked_vs_best_single_auc:
  min = +0.0001239776611328125
  max = +0.0001239776611328125
  mean = +0.0001239776611328125
  positive_selection_seeds = 5 / 5

decision = stable_but_mixed_train_holdout_stacking_diagnostic
```

Route-level stacking stability:

```text
route_stability_cli = scripts/summarize-stacked-route
strict_route_report =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_trail_stats_stacking_route_stability_strict.json
relaxed_route_report =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_trail_stats_stacking_route_stability_relaxed.json

strict passed_seed_count = 0 / 2
relaxed passed_seed_count = 1 / 2
relaxed positive_seed_fraction = 0.5
relaxed delta_mean_vs_best_single_auc:
  min = -0.0000385284423828125
  max = +0.0001239776611328125
  mean = +0.000042724609375

decision = stable_but_mixed_cross_seed_stacking_diagnostic
```

Compressed structural-stat logistic expert:

```text
cli = scripts/fit-compressed-feature-expert
feature_view = trail_position_stats
fit_split = train feature artifacts
score_split = held-out validation feature artifacts
feature_count = 3708
model = logistic
steps = 800
learning_rate = 0.05
l2 = 0.001
standardize = true, with train statistics only
decision = compressed_feature_expert_local_screen_positive_needs_controls

seed0 report =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_trail_stats_logistic_report.json
seed0 validation_auc = 1.0
seed0 validation_accuracy = 0.99951171875
seed0 calibrated_validation_accuracy = 1.0

seed1 report =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed1_trail_stats_logistic_report.json
seed1 validation_auc = 1.0
seed1 validation_accuracy = 0.99951171875
seed1 calibrated_validation_accuracy = 1.0
```

Shuffle-label control:

```text
control_cli = scripts/fit-compressed-feature-expert --shuffle-train-labels --shuffle-seed 0
control_decision = compressed_feature_expert_shuffle_train_labels_control
control_scope = train labels are shuffled only for fitting; score artifacts and metrics still use true labels

seed0 control_report =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_trail_stats_logistic_shuffle_train_labels_report.json
seed0 control_validation_auc = 0.4979085922241211
seed0 control_validation_accuracy = 0.5029296875
seed0 control_calibrated_validation_accuracy = 0.51904296875

seed1 control_report =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed1_trail_stats_logistic_shuffle_train_labels_report.json
seed1 control_validation_auc = 0.5246953964233398
seed1 control_validation_accuracy = 0.51318359375
seed1 control_calibrated_validation_accuracy = 0.5322265625
```

Compressed expert ensemble check:

```text
seed0 ensemble_report =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_trail_stats_logistic_ensemble.json
seed0 best_single = compressed_feature_logistic_expert
seed0 best_single_auc = 1.0
seed0 best_ensemble_auc = 1.0
seed0 delta_best_ensemble_vs_single_auc = 0.0

seed1 ensemble_report =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed1_trail_stats_logistic_ensemble.json
seed1 best_single = compressed_feature_logistic_expert
seed1 best_single_auc = 1.0
seed1 best_ensemble_auc = 0.999995231628418
seed1 delta_best_ensemble_vs_single_auc = -0.00000476837158203125
```

Compressed expert route gate:

```text
route_gate_cli = scripts/summarize-compressed-feature-expert
route_gate =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_trail_stats_compressed_expert_route_gate.json

normal_passed_seed_count = 2 / 2
shuffle_control_passed_seed_count = 2 / 2
ensemble_gain_passed_seed_count = 0 / 2
validation_auc_mean = 1.0
shuffle_control_auc_mean = 0.5113019943237305
ensemble_delta_vs_best_single_auc_mean = -0.000002384185791015625

decision = compressed_feature_local_positive_controls_pass_not_ensemble_gain
```

Compressed feature sparsity audit:

```text
sparsity_cli = scripts/audit-compressed-feature-sparsity
ranking = train-only abs(class_mean_difference) / train_std
top_k = 1, 2, 4, 8, 16, 32, 64, 128, 256

seed0 sparsity_report =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_trail_stats_sparse_audit.json
seed0 decision = sparse_compressed_feature_local_screen_positive
seed0 validation_auc_by_top_k:
  k=1   0.6603231430053711
  k=2   0.6603231430053711
  k=4   0.7419681549072266
  k=8   0.7881331443786621
  k=16  0.8615026473999023
  k=32  0.9242277145385742
  k=64  0.966217041015625
  k=128 0.9928836822509766
  k=256 0.9996299743652344

seed1 sparsity_report =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed1_trail_stats_sparse_audit.json
seed1 decision = sparse_compressed_feature_local_screen_positive
seed1 validation_auc_by_top_k:
  k=1   0.639317512512207
  k=2   0.639317512512207
  k=4   0.6965384483337402
  k=8   0.7919750213623047
  k=16  0.8639602661132812
  k=32  0.9326858520507812
  k=64  0.9727020263671875
  k=128 0.995387077331543
  k=256 0.9996557235717773
```

Interpretation:

```text
This is the first bit-sensitivity projection variant that clears the
same-input global control on both local seeds. It supports the representation
hypothesis: structure-aware residual summaries are materially better than
raw-axis masks or contiguous raw-axis means. However, simple score aggregation
with the trail-position anchor still does not beat the best single
trail-position expert. Train-fitted logistic stacking improves over the fixed
ensemble on both seeds, but it also remains below the best single
trail-position anchor. Train-holdout-selected stacking improves the calibration
picture: seed0 is nearly tied with the best single anchor and seed1 slightly
beats it. A five-selection-seed stability sweep shows this is not a one-off
holdout accident, and the route-level summary makes the cross-seed gate
explicit: strict selection stability passes 0/2 seeds, while relaxed
per-seed stability passes only seed1. Because the improvement is not two-seed
positive, treat V2 as a two-seed local non-neighbor diagnostic candidate and a
useful frozen-score toolbox item, not as a completed ensemble improvement or
remote-launch result.

The compressed structural-stat logistic expert is a stronger local diagnostic:
it fits a tiny logistic model on the exported train-split structural-stat
features and reaches perfect held-out validation AUC on both local seeds. The
new shuffle-label control drops to near-random AUC on both seeds, so the signal
does not survive when the fit labels are randomized. This makes the route worth
keeping, but it is still a 2048/class local screen with 1024/class validation
score artifacts, not remote evidence, not formal SPN/PRESENT evidence, and not
a multi-network improvement. The best ensemble ties the compressed expert on
seed0 and loses very slightly on seed1, so the immediate result is a stronger
single compressed SPN-structural expert, not successful diverse aggregation.
The route-level gate now encodes that distinction directly, so future 262144/class
retrieved artifacts must pass the same normal/shuffle/ensemble split before this
route can be promoted.

The sparsity audit adds an important architecture hint. One or two train-ranked
structural dimensions are only weak positive, 64 dimensions are still below the
0.99 local gate, and 128 dimensions clear 0.99 AUC on both seeds. This argues
against a trivial single-feature leakage explanation and favors a medium-sparse
SPN structural-stat expert: future model work should preserve grouped
trail-position/statistic structure rather than collapsing to a tiny scalar rule.

Sparse-feature decoding uses:

decode_cli = scripts/decode-compressed-feature-sparsity
seed0 decode_report =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_trail_stats_sparse_decode.json
seed1 decode_report =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed1_trail_stats_sparse_decode.json

seed0 top-256 family_counts:
  depth_word_cell_span = 108
  depth_cell_span = 64
  word_span = 34
  depth_word_span = 33
  cell_span = 15
  global_pair_density = 1
  global_trail_density = 1
seed0 top-256 depth_counts:
  depth0 = 44, depth1 = 51, depth2 = 54, depth3 = 56

seed1 top-256 family_counts:
  depth_word_cell_span = 119
  depth_cell_span = 61
  word_span = 30
  depth_word_span = 29
  cell_span = 15
  global_pair_density = 1
  global_trail_density = 1
seed1 top-256 depth_counts:
  depth0 = 54, depth1 = 48, depth2 = 54, depth3 = 53

The top sparse dimensions are dominated by span-type structural statistics:
depth_word_cell_span, depth_cell_span, word_span, depth_word_span, and
cell_span. The useful features are spread across all four trail depths rather
than concentrated in one depth or one cell. This points away from a single
leakage-like scalar and toward a grouped span/statistic-aware SPN expert.

Span-family restricted expert:

span_family_cli = scripts/fit-compressed-feature-expert
span_family_filter =
  --include-feature-family depth_word_cell_span
  --include-feature-family depth_cell_span
  --include-feature-family word_span
  --include-feature-family depth_word_span
  --include-feature-family cell_span
selected_feature_count = 731 / 3708

seed0 span_family_report =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_trail_stats_span_family_report.json
seed0 validation_auc = 0.9999723434448242
seed0 validation_accuracy = 0.99560546875
seed0 calibrated_validation_accuracy = 0.998046875
seed0 shuffle_train_labels_auc = 0.4802093505859375

seed1 span_family_report =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed1_trail_stats_span_family_report.json
seed1 validation_auc = 0.9999513626098633
seed1 validation_accuracy = 0.99658203125
seed1 calibrated_validation_accuracy = 0.99853515625
seed1 shuffle_train_labels_auc = 0.47839784622192383

This strengthens the span/statistic-aware architecture direction: the five
span families alone retain almost all of the compressed structural-stat signal
while the shuffle-label control stays near random. It is still a 2048/class
local diagnostic only, not remote evidence, not formal SPN/PRESENT evidence,
and not a multi-network improvement.

Span-family attribution:

span_family_attribution_cli = scripts/audit-compressed-feature-families

seed0 attribution_report =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_trail_stats_span_family_attribution.json
seed0 single_family_validation_auc:
  depth_word_cell_span = 0.9999923706054688
  depth_cell_span = 0.9865589141845703
  word_span = 0.9422855377197266
  depth_word_span = 0.9392290115356445
  cell_span = 0.8775739669799805
seed0 leave_out_depth_word_cell_span_auc = 0.9918107986450195

seed1 attribution_report =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed1_trail_stats_span_family_attribution.json
seed1 single_family_validation_auc:
  depth_word_cell_span = 0.999969482421875
  depth_cell_span = 0.9866390228271484
  word_span = 0.9360342025756836
  depth_word_span = 0.9317569732666016
  cell_span = 0.8646869659423828
seed1 leave_out_depth_word_cell_span_auc = 0.9919548034667969

Interpretation: depth_word_cell_span is the dominant span-family backbone. It
is nearly saturated by itself on both local seeds. Removing it produces the
largest leave-one-out drop, though the remaining four span families still score
about 0.992 AUC, so the other families carry a real but secondary structural
signal. The next learned architecture should use depth_word_cell_span as the
primary grouped channel and add lower-rank depth_cell/word/cell span channels
as auxiliary context rather than treating all 731 span dimensions uniformly.

Structured span-block exporter:
  cli = scripts/export-compressed-span-blocks
  kind = compressed_spn_span_blocks
  input = existing trail_position_stats feature artifact
  outputs =
    depth_word_cell_span.npy [rows, depth, trailword, cell]
    depth_cell_span.npy [rows, depth, cell]
    word_span.npy [rows, word]
    depth_word_span.npy [rows, depth, trailword]
    cell_span.npy [rows, cell]
  role =
    depth_word_cell_span -> primary_backbone
    all other span blocks -> auxiliary_context
  claim_scope =
    structure-preserving export only; no training, scoring, label change,
    negative-sample change, remote evidence, or formal SPN/PRESENT evidence

Structured span-summary diagnostic:
  route_summary_cli = scripts/summarize-compressed-span-route
  route_summary =
    outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_span_summary_route.json
  route_summary_decision = compressed_span_summary_retains_flat_signal_controls_pass
  summary_feature_view = compressed_span_summary
  summary_feature_count = 273
  flat_span_feature_count = 731
  feature_reduction_ratio = 0.3734610123119015
  summary_source = pooled depth/word/cell views over compressed_spn_span_blocks
  model = compressed_span_summary_logistic_expert
  fit_split = train
  score_split = validation
  seed0 validation_auc = 0.9999141693115234
  seed1 validation_auc = 0.9998435974121094
  seed0 shuffle_train_labels_validation_auc = 0.5048818588256836
  seed1 shuffle_train_labels_validation_auc = 0.4824662208557129
  max_auc_drop_vs_flat_span = 0.00010776519775390625

Interpretation: the compact 273-dimensional structured summary retains nearly
all local span-family signal while the shuffled-label controls stay near
random. This supports the grouped span/statistic-aware architecture direction
more strongly than a flat near-neighbor ensemble. It is still a 2048/class
local diagnostic only, not remote evidence and not formal SPN/PRESENT evidence.

Primary/auxiliary prefix attribution:
  cli = scripts/fit-compressed-feature-expert
  new_filter = --include-feature-prefix
  primary_prefix = primary_
  auxiliary_prefix = aux_
  primary_feature_count = 133
  auxiliary_feature_count = 140
  seed0 primary_validation_auc = 0.9997234344482422
  seed1 primary_validation_auc = 0.9992923736572266
  seed0 auxiliary_validation_auc = 0.9964427947998047
  seed1 auxiliary_validation_auc = 0.9976606369018555
  seed0 primary_shuffle_validation_auc = 0.45572566986083984
  seed1 primary_shuffle_validation_auc = 0.5348129272460938
  seed0 auxiliary_shuffle_validation_auc = 0.5421538352966309
  seed1 auxiliary_shuffle_validation_auc = 0.5033082962036133

Interpretation: primary depth_word_cell-derived summary channels are the
strong backbone. Auxiliary depth/cell/word summaries are weaker but still
strongly positive on both local seeds, and their shuffled-label controls remain
near random. This supports a compact grouped expert with a primary backbone
and lower-rank auxiliary context, not a flat all-feature logistic model as the
final architecture.

Two-branch grouped expert diagnostic:
  cli = scripts/fit-compressed-span-grouped-expert
  model = compressed_span_grouped_logistic_expert
  decision = compressed_span_grouped_expert_local_screen_positive_needs_controls
  feature_model = two_branch_logistic
  branch_features = primary_logit + auxiliary_logit
  branch_fit_split = train
  combiner_fit_split = train
  score_split = validation
  seed0 grouped_validation_auc = 0.9997968673706055
  seed1 grouped_validation_auc = 0.9996414184570312
  seed0 primary_branch_validation_auc = 0.9997234344482422
  seed1 primary_branch_validation_auc = 0.9992923736572266
  seed0 auxiliary_branch_validation_auc = 0.9964427947998047
  seed1 auxiliary_branch_validation_auc = 0.9976606369018555
  seed0 full_summary_validation_auc = 0.9999141693115234
  seed1 full_summary_validation_auc = 0.9998435974121094

Interpretation: the two-branch grouped expert improves over primary-only and
auxiliary-only on both local seeds, so the auxiliary context adds usable
residual information when combined with the primary backbone. It is still
slightly below the full 273-dimensional flat summary logistic model, so this is
not yet an accuracy improvement over the best compact summary anchor. The value
is architectural: it turns the span-summary route into an explicit
primary-backbone plus lower-rank-auxiliary-context design that can later be
upgraded to a real grouped SPN model. This is 2048/class local diagnostic
evidence only; it is not remote evidence, not formal SPN/PRESENT evidence, and
not a multi-network improvement.

Semantic/hybrid grouped branch-logit diagnostic:
  cli = scripts/fit-compressed-span-grouped-expert --group-mode semantic|hybrid
  semantic_feature_model = semantic_group_logistic
  hybrid_feature_model = hybrid_group_logistic
  semantic_branch_count = 12
  hybrid_branch_count = 14
  semantic_groups =
    primary_depth, primary_trailword, primary_cell, primary_depth_cell,
    primary_depth_trailword, primary_global, aux_depth_cell, aux_word,
    aux_depth_word, aux_cell, aux_word_global, aux_cell_global
  hybrid_groups = primary, auxiliary + semantic_groups
  seed0 semantic_group_validation_auc = 0.998713493347168
  seed1 semantic_group_validation_auc = 0.9978828430175781
  seed0 semantic_l2zero_validation_auc = 0.9994039535522461
  seed1 semantic_l2zero_validation_auc = 0.9988164901733398
  seed0 hybrid_group_validation_auc = 0.9992799758911133
  seed1 hybrid_group_validation_auc = 0.9986753463745117
  decision = semantic_or_hybrid_branch_logit_decomposition_hold

Interpretation: finer semantic branch-logit decomposition is not the next
accuracy route. Even after a no-L2/longer-fit sanity pass, the 12-way semantic
combiner remains below the coarse two-branch result, and the hybrid
primary/auxiliary plus semantic combiner also remains below the coarse
two-branch result. The likely issue is information loss from precompressing
each semantic group into one fitted logit. The next architectural follow-up
should preserve within-group raw features or block tensors while applying
structured regularization, rather than adding more prefit branch logits.

Raw interaction summary diagnostic:
  cli = scripts/fit-compressed-span-interaction-expert
  feature_model = raw_plus_primary_auxiliary_interactions_logistic
  raw_feature_count = 273
  interaction_selection_split = train
  top_primary = 8
  top_auxiliary = 8
  interaction_count = 64
  total_feature_count = 337
  model = compressed_span_interaction_logistic_expert
  seed0 interaction_validation_auc = 0.9999170303344727
  seed1 interaction_validation_auc = 0.9998636245727539
  seed0 delta_vs_full_summary_auc = 0.000002860 ... positive
  seed1 delta_vs_full_summary_auc = 0.000020027 ... positive
  seed0 interaction_shuffle_validation_auc = 0.5185546875
  seed1 interaction_shuffle_validation_auc = 0.48958492279052734
  decision = raw_interaction_summary_tiny_positive_controls_pass_local

Interpretation: preserving all raw 273 summary features and adding a small
train-selected primary/auxiliary interaction bank is the first local
architecture diagnostic in this span-summary line that beats the full-summary
linear anchor on both seeds. The margin is tiny, so this is not a strong
result, not remote evidence, and not formal SPN/PRESENT evidence. The shuffled
train-label controls are near random, which supports that the positive signal
is tied to train-label-selected SPN coordinate interactions rather than the
mere presence of extra dimensions. The next step should be a stricter
interaction gate or a block-preserving model that keeps raw group information,
not another prefit branch-logit decomposition.

Strict train-internal holdout follow-up:
  cli = scripts/fit-compressed-span-interaction-expert
  added_gate = --selection-holdout-fraction 0.5 --selection-seed
  selection_fit_split_mode = train_internal_holdout
  interaction_selection_rows = 2048
  fit_rows = 2048
  seed0 holdout_interaction_validation_auc = 0.9998722076416016
  seed1 holdout_interaction_validation_auc = 0.9998645782470703
  seed0 delta_vs_full_summary_auc = -0.000041961669921875
  seed1 delta_vs_full_summary_auc = 0.0000209808349609375
  seed0 delta_vs_original_interaction_auc = -0.00004482269287109375
  seed1 delta_vs_original_interaction_auc = 0.00000095367431640625
  seed0 holdout_shuffle_validation_auc = 0.49273252487182617
  seed1 holdout_shuffle_validation_auc = 0.5002899169921875
  decision = raw_interaction_holdout_mixed_local_diagnostic

Interpretation: once interaction selection and logistic fitting are separated
inside the train split, the tiny raw-interaction gain is no longer uniformly
positive. Seed1 remains slightly above the full 273D summary anchor, but seed0
falls below it by about `4.2e-5` AUC. The shuffle controls stay near random, so
the route is not a label-control failure; however, this stricter gate says the
current flat cross-product expansion is not strong enough for a remote scale-up
by itself. Keep the implementation as a diagnostic and use the result to guide
the next architecture toward block-preserving/raw group tensor interaction
modeling with explicit regularization, rather than promoting this exact
hand-selected interaction bank as the next scaled candidate.

Semantic block interaction diagnostic:
  cli = scripts/fit-compressed-span-block-interaction-expert
  feature_model = raw_plus_semantic_block_interactions_logistic
  raw_feature_count = 273
  primary_group_count = 6
  auxiliary_group_count = 6
  block_pair_count = 36
  block_interaction_stat_count = 4
  block_interaction_feature_count = 144
  total_feature_count = 417
  l2 = 0.001
  seed0 block_interaction_validation_auc = 0.9999065399169922
  seed1 block_interaction_validation_auc = 0.999908447265625
  seed0 delta_vs_full_summary_auc = -0.00000762939453125
  seed1 delta_vs_full_summary_auc = 0.000064849853515625
  seed0 delta_vs_strict_flat_holdout_auc = 0.000034332275390625
  seed1 delta_vs_strict_flat_holdout_auc = 0.0000438690185546875
  seed0 block_interaction_shuffle_validation_auc = 0.5331153869628906
  seed1 block_interaction_shuffle_validation_auc = 0.5094890594482422
  l2_0.003_seed0_auc = 0.9999065399169922
  l2_0.003_seed1_auc = 0.9999074935913086
  l2_0.01_seed0_auc = 0.9998979568481445
  l2_0.01_seed1_auc = 0.9999008178710938
  decision = semantic_block_interaction_mixed_local_diagnostic

Interpretation: semantic block interactions are a cleaner SPN-inductive-bias
diagnostic than train-selected single-feature cross-products because the block
features are label-independent: every primary semantic block is paired with
every auxiliary semantic block using four standardized aggregate product
statistics. This improves over the strict flat-interaction holdout on both
seeds and is clearly positive on seed1 versus the full 273D summary anchor, but
seed0 remains slightly below the full summary anchor. The shuffle controls are
near random but seed0 is mildly high at `0.5331`, so this should be treated as a
useful mixed diagnostic and a basis for the next block/tensor model, not as a
remote-scale candidate. The quick L2 sweep did not improve the result.

Semantic low-rank block interaction diagnostic:
  cli = scripts/fit-compressed-span-low-rank-interaction-expert
  feature_model = raw_plus_semantic_low_rank_block_interactions_logistic
  raw_feature_count = 273
  primary_group_count = 6
  auxiliary_group_count = 6
  block_pair_count = 36
  rank1_low_rank_interaction_feature_count = 36
  rank1_total_feature_count = 309
  rank2_low_rank_interaction_feature_count = 144
  rank2_total_feature_count = 417
  seed0 rank1_low_rank_validation_auc = 0.9999256134033203
  seed1 rank1_low_rank_validation_auc = 0.9999008178710938
  seed0 rank1_delta_vs_full_summary_auc = 0.000011444091796875
  seed1 rank1_delta_vs_full_summary_auc = 0.000057220458984375
  seed0 rank1_delta_vs_block_stat_auc = 0.000019073486328125
  seed1 rank1_delta_vs_block_stat_auc = -0.00000762939453125
  seed0 rank1_low_rank_shuffle_validation_auc = 0.4889411926269531
  seed1 rank1_low_rank_shuffle_validation_auc = 0.4727621078491211
  seed0 rank2_low_rank_validation_auc = 0.999913215637207
  seed1 rank2_low_rank_validation_auc = 0.9998798370361328
  seed0 rank2_low_rank_shuffle_validation_auc = 0.4826202392578125
  seed1 rank2_low_rank_shuffle_validation_auc = 0.44483137130737305
  decision = semantic_low_rank_rank1_positive_vs_full_mixed_vs_blockstat_local

Interpretation: rank1 low-rank block interactions are the best current
block/tensor diagnostic in this local line. The projection is unsupervised on
train features only, so it avoids train-label feature picking; shuffle-label
controls are cleanly near random or inverted. Rank1 beats the full 273D summary
anchor on both seeds and beats the semantic block-stat aggregate on seed0, but
still trails block-stat by `7.6e-6` AUC on seed1. Rank2 adds more interaction
dimensions but is weaker than rank1 on both seeds. Keep rank1 low-rank as the
next candidate baseline for a true learned low-rank/block-tensor module; do
not promote it to remote scale until it clears a broader local gate.

Learned low-rank block interaction diagnostic:
  cli = scripts/fit-compressed-span-learned-low-rank-interaction-expert
  feature_model = raw_plus_learned_semantic_low_rank_block_interactions
  raw_feature_count = 273
  primary_group_count = 6
  auxiliary_group_count = 6
  block_pair_count = 36
  rank = 1
  learned_low_rank_interaction_count = 36
  total_feature_count = 309
  steps = 300
  learning_rate = 0.01
  weight_decay = 0.001
  seed0 learned_low_rank_validation_auc = 0.9998531341552734
  seed1 learned_low_rank_validation_auc = 0.9995737075805664
  seed0 delta_vs_full_summary_auc = -0.00006103515625
  seed1 delta_vs_full_summary_auc = -0.00026988983154296875
  seed0 delta_vs_unsupervised_rank1_auc = -0.000072479248046875
  seed1 delta_vs_unsupervised_rank1_auc = -0.00032711029052734375
  seed0 learned_low_rank_shuffle_validation_auc = 0.5293331146240234
  seed1 learned_low_rank_shuffle_validation_auc = 0.47176551818847656
  decision = learned_low_rank_rank1_hold_local_diagnostic

Interpretation: the learned rank1 block-tensor module is now implemented as a
reusable diagnostic CLI, but this first fixed-budget local gate does not
improve the route. Shuffle-label controls stay near random, so the failure mode
is not obvious label leakage; the learned supervised projections simply trail
the stronger unsupervised rank1 SVD projection and even the full 273D summary
anchor on both seeds at this budget. Do not remote-scale this learned rank1
variant. Keep the unsupervised rank1 low-rank expert as the current strongest
block/tensor candidate, and use the learned module only as a scaffold for a
future stricter ablation such as frozen unsupervised initialization, projection
regularization, or a raw-branch-disabled interaction-only test.

SVD-frozen learned low-rank diagnostic:
  cli = scripts/fit-compressed-span-learned-low-rank-interaction-expert
  projection_init = svd
  freeze_projections = true
  trainable_projection_parameter_count = 0
  rank = 1
  total_feature_count = 309
  steps = 2000
  learning_rate = 0.01
  weight_decay = 0.001
  seed0 svd_frozen_validation_auc = 0.9999265670776367
  seed1 svd_frozen_validation_auc = 0.9999094009399414
  seed0 delta_vs_unsupervised_rank1_auc = 0.00000095367431640625
  seed1 delta_vs_unsupervised_rank1_auc = 0.00000858306884765625
  seed0 svd_frozen_shuffle_validation_auc = 0.46834707260131836
  seed1 svd_frozen_shuffle_validation_auc = 0.6085009574890137
  seed1 weight_decay_0.01_validation_auc = 0.9998893737792969
  seed1 weight_decay_0.01_shuffle_validation_auc = 0.6176691055297852
  decision = svd_frozen_learned_rank1_recovers_auc_but_fails_seed1_shuffle_control

Interpretation: initializing and freezing the learned projections from
train-feature SVD recovers the useful rank1 low-rank signal and even gives a
tiny main-AUC gain over the existing unsupervised rank1 logistic expert on both
local seeds. However, the same PyTorch classifier path produces a high seed1
shuffle-label control (`0.6085` AUC), and stronger weight decay does not fix
that control while preserving the main gain. Treat this as an attribution
result: the previous learned-random failure was mostly projection drift/random
initialization, but the SVD-frozen learned classifier is not control-clean
enough to replace the existing unsupervised rank1 logistic expert or to justify
remote scale-up.

Interaction-only low-rank diagnostic:
  cli = scripts/fit-compressed-span-low-rank-interaction-expert
  option = --interaction-only
  feature_model = semantic_low_rank_block_interactions_only_logistic
  raw_features_included = false
  raw_feature_count = 273
  low_rank_interaction_feature_count = 36
  total_feature_count = 36
  rank = 1
  steps = 2000
  learning_rate = 0.05
  l2 = 0.001
  seed0 interaction_only_validation_auc = 0.5190114974975586
  seed1 interaction_only_validation_auc = 0.5553302764892578
  seed0 interaction_only_delta_vs_full_summary_auc = -0.48090267181396484
  seed1 interaction_only_delta_vs_full_summary_auc = -0.44451332092285156
  seed0 interaction_only_delta_vs_rank1_low_rank_auc = -0.4809141159057617
  seed1 interaction_only_delta_vs_rank1_low_rank_auc = -0.44457054138183594
  seed0 interaction_only_shuffle_validation_auc = 0.5091586112976074
  seed1 interaction_only_shuffle_validation_auc = 0.4804563522338867
  decision = interaction_only_low_rank_weak_not_primary_signal_source

Interpretation: the 36 rank1 primary-by-auxiliary interaction features alone
carry only a weak above-random signal at this local scale, while shuffle-label
controls are cleanly near random. Therefore the near-perfect rank1 low-rank
expert is not powered by the interaction bank alone; its useful gain comes
from adding a small interaction correction on top of the raw 273D compressed
span summary. Do not spend remote scale on interaction-only low-rank. The next
better-targeted ablation should compress or partition the raw span-summary
families to locate which raw semantic blocks carry the dominant PRESENT r8
signal.

Raw span-family localization diagnostic:
  cli = scripts/fit-compressed-feature-expert
  selector = --include-feature-prefix
  feature_source = compressed_span_summary raw 273D
  seeds = 0,1
  samples_per_class = 2048
  strongest_single_family = primary_depth_trailword
  seed0 primary_depth_trailword_auc = 0.9997749328613281
  seed1 primary_depth_trailword_auc = 0.999262809753418
  seed0 primary_depth_cell_auc = 0.9990720748901367
  seed1 primary_depth_cell_auc = 0.9978551864624023
  seed0 aux_depth_cell_auc = 0.9923276901245117
  seed1 aux_depth_cell_auc = 0.994171142578125
  compact_combo = primary_depth_trailword + aux_depth_cell
  compact_combo_feature_count = 60
  seed0 compact_combo_auc = 0.9999017715454102
  seed1 compact_combo_auc = 0.9998178482055664
  seed0 compact_combo_delta_vs_full_summary_auc = -0.00001239776611328125
  seed1 compact_combo_delta_vs_full_summary_auc = -0.00002574920654296875
  seed0 compact_combo_shuffle_auc = 0.4941110610961914
  seed1 compact_combo_shuffle_auc = 0.4871025085449219
  seed0 compact_combo_l2_0_auc = 0.9999017715454102
  seed1 compact_combo_l2_0_auc = 0.999821662902832
  decision = compact_raw_primary_depth_trailword_aux_depth_cell_anchor_local

Interpretation: the raw-family screen localizes the dominant PRESENT r8
compressed-span signal to primary trajectory/depth structure, especially
`primary_depth_trailword`, with `aux_depth_cell` providing a strong compact
complement. A 60D raw combo nearly matches the 273D full-summary expert on both
seeds and has clean shuffle-label controls. This is a better next anchor than
interaction-only low-rank: it preserves almost all of the raw signal with much
lower dimensionality, but it still trails full 273D and rank1 raw+interaction,
so keep it as a compact local diagnostic rather than a remote-scale candidate.
```

## Fixed Protocol

The first screen must keep the active PRESENT r8 protocol fixed:

```text
cipher = PRESENT-80
rounds = 8
samples_per_class = 2048
pairs_per_sample = 16
feature_source = present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits
sample_structure = plaintext_integral_nibble_difference_matched_negative
integral_active_nibble = 0
difference_profile = present_zhang_wang2022_mcnd
negative_mode = encrypted_random_plaintexts
train_key = 0x00000000000000000000
validation_key = 0x11111111111111111111
seeds = 0, 1
```

The screen must not change labels, negative samples, validation data, metric
computation, or plan-alignment logic.

## Candidate And Controls

Minimum matrix:

| Row | Model / artifact | Role |
|---:|---|---|
| 0 | `present_pairset_global_stats` | same-input global control |
| 1 | `present_trail_position_stats_pairset` | strongest current trail-position anchor |
| 2 | `present_r8_bit_sensitivity_projection_expert` | candidate non-neighbor projection expert |
| 3 | shuffled or mismatch projection control | leakage / axis-selection control |

The candidate is only eligible after a train-only selector writes a frozen mask
artifact:

```text
train_feature_dir = outputs/local_audits/i1_present_r8_bit_sensitivity_projection_train_features_seed{seed}
validation_feature_dir = outputs/local_audits/i1_present_r8_bit_sensitivity_projection_validation_features_seed{seed}
mask_artifact = outputs/local_audits/i1_present_r8_bit_sensitivity_projection_mask_seed{seed}.json
sensitivity_report = outputs/local_audits/i1_present_r8_bit_sensitivity_projection_seed{seed}.json
selection_split = train
```

Prepared feature exports:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/export-bit-sensitivity-features \
  --eval-plan configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_262k_seed{seed_number}.csv \
  --eval-row-index 1 \
  --split train \
  --output-dir outputs/local_audits/i1_present_r8_bit_sensitivity_projection_train_features_seed{seed}

UV_CACHE_DIR=/tmp/uv-cache uv run scripts/export-bit-sensitivity-features \
  --eval-plan configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_262k_seed{seed_number}.csv \
  --eval-row-index 1 \
  --split validation \
  --reference-artifact outputs/remote_results/<run_id>/score_artifacts/trail_position \
  --output-dir outputs/local_audits/i1_present_r8_bit_sensitivity_projection_validation_features_seed{seed}
```

The validation export must pass the reference-artifact label/sample-id alignment
check before the frozen projection scorer is allowed to run.

Prepared selector:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/select-bit-sensitivity-projection \
  --features outputs/local_audits/i1_present_r8_bit_sensitivity_projection_train_features_seed{seed}/features.npy \
  --control-artifact outputs/remote_results/<run_id>/score_artifacts/global_stats_control \
  --anchor-artifact outputs/remote_results/<run_id>/score_artifacts/trail_position \
  --output-mask outputs/local_audits/i1_present_r8_bit_sensitivity_projection_mask_seed{seed}.json \
  --output-report outputs/local_audits/i1_present_r8_bit_sensitivity_projection_seed{seed}.json \
  --top-k 64
```

Prepared grouped selector variant:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/select-bit-sensitivity-projection \
  --features outputs/local_audits/i1_present_r8_bit_sensitivity_projection_train_features_seed{seed}/features.npy \
  --control-artifact outputs/remote_results/<run_id>/score_artifacts/global_stats_control \
  --anchor-artifact outputs/remote_results/<run_id>/score_artifacts/trail_position \
  --output-mask outputs/local_audits/i1_present_r8_bit_sensitivity_projection_grouped_mask_seed{seed}.json \
  --output-report outputs/local_audits/i1_present_r8_bit_sensitivity_projection_grouped_seed{seed}.json \
  --group-size 8 \
  --top-groups 8
```

The selector writes only a train-only mask/report. It is not a model result and
must not be interpreted as candidate AUC.

Prepared frozen scorer:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/apply-bit-sensitivity-projection \
  --features outputs/local_audits/i1_present_r8_bit_sensitivity_projection_validation_features_seed{seed}/features.npy \
  --mask outputs/local_audits/i1_present_r8_bit_sensitivity_projection_mask_seed{seed}.json \
  --reference-artifact outputs/remote_results/<run_id>/score_artifacts/trail_position \
  --output-dir outputs/local_audits/i1_present_r8_bit_sensitivity_projection_scores_seed{seed} \
  --output-report outputs/local_audits/i1_present_r8_bit_sensitivity_projection_scores_seed{seed}.json \
  --run-id i1_present_r8_bit_sensitivity_projection_seed{seed}
```

The scorer writes a standard frozen-score artifact so the candidate can be
checked by the existing ensemble/diversity tooling. It is still not a trained
neural model and cannot be promoted without the local gate below.

Prepared postprocess gate:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run scripts/postprocess-bit-sensitivity-projection \
  --global-artifact outputs/remote_results/<run_id>/score_artifacts/global_stats_control \
  --anchor-artifact outputs/remote_results/<run_id>/score_artifacts/trail_position \
  --projection-artifact outputs/local_audits/i1_present_r8_bit_sensitivity_projection_scores_seed{seed} \
  --output outputs/local_audits/i1_present_r8_bit_sensitivity_projection_gate_seed{seed}.json
```

This gate is a frozen-score diagnostic only. It checks protocol/label/sample-id
alignment, strict negative mode, projection expert-family metadata, the
candidate AUC margin over the same-input global control, and error overlap with
the trail-position anchor. It does not launch remote training and does not make
formal SPN/PRESENT claims.

## Gate

Promote to a compatible local-screen expert only if all of these hold:

```text
postprocess decision = projection_expert_ready_for_local_screen
projection_auc >= configured min_projection_auc
projection_auc >= same_input_global_control_auc + configured min_margin_vs_global
projection_error_jaccard_with_trail_position <= configured max_error_jaccard_with_anchor
score_artifacts_exported = true and aligned
negative_mode = encrypted_random_plaintexts
expert_family = bit_sensitivity_projection
```

If the candidate passes the gate, the next step is still only a local
frozen-score diversity or ensemble screen against the global and trail-position
artifacts. Do not jump directly to a remote run or `1000000/class`.

## Hold Conditions

Hold the route if:

```text
trail-position 262144/class artifacts are not verified
candidate is a single-seed spike
candidate loses to same-input global control on either seed
candidate duplicates the trail-position error set
mask selection uses validation evidence
mismatch controls separate the classes
postprocess decision = hold_projection_duplicate_or_weak
postprocess decision = fail_protocol_alignment
```

## Claim Scope

Allowed:

```text
local non-neighbor expert screen
bit-sensitivity-guided representation diagnostic
possible future diverse-pool input if score artifacts pass
```

Not allowed:

```text
formal PRESENT evidence
breakthrough claim
SOTA claim
remote-launch basis before local gate
diverse ensemble claim without low-overlap score evidence
```

## V4 Compact Raw Plus Low-Rank Interaction Diagnostic

The 2026-07-07 follow-up added selected raw-prefix support to the existing
low-rank interaction CLI so the compact raw anchor can be tested with a small
controlled interaction correction:

```text
cli = scripts/fit-compressed-span-low-rank-interaction-expert
new option = --include-raw-feature-prefix
feature_model = selected_raw_plus_semantic_low_rank_block_interactions_logistic
selected_raw_prefixes = primary_depth_trailword_, aux_depth_cell_
selected_raw_feature_count = 60
low_rank_interaction_feature_count = 36
feature_count = 96
rank = 1
fit_split = train
score_split = held-out validation
```

Artifacts:

```text
seed0 report =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_span_compact_raw_plus_low_rank_rank1_report.json
seed1 report =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed1_span_compact_raw_plus_low_rank_rank1_report.json
seed0 shuffle =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_span_compact_raw_plus_low_rank_rank1_shuffle_report.json
seed1 shuffle =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed1_span_compact_raw_plus_low_rank_rank1_shuffle_report.json
```

Metrics:

| Seed | Compact raw + rank1 AUC | Delta vs 60D compact raw | Delta vs full 273D | Delta vs 309D allraw+rank1 | Shuffle AUC |
|---:|---:|---:|---:|---:|---:|
| 0 | `0.9998960494995117` | `-0.0000057220458984` | `-0.0000181198120117` | `-0.0000295639038086` | `0.4553308486938477` |
| 1 | `0.9998569488525391` | `+0.0000391006469727` | `+0.0000133514404297` | `-0.0000438690185547` | `0.4464559555053711` |

Decision:

```text
decision = compact_raw_plus_low_rank_rank1_mixed_local_diagnostic
action = do_not_promote_to_remote_or_formal_claim
```

Interpretation:

```text
The selected 96D feature set has clean shuffle controls and remains very strong
locally, but it is mixed against the 60D compact raw anchor and loses to the
309D allraw+rank1 expert on both seeds. The low-rank interaction term is a
small correction, not a stable new primary signal source. Keep the selected
raw-prefix CLI support because it makes compact ablations reproducible, but do
not spend a medium remote slot on this exact 96D variant.
```

## V5 Raw-Family Add-One Diagnostic

The next local diagnostic used the existing raw-prefix filter to add one raw
family at a time around the 60D compact raw anchor:

```text
base_anchor = primary_depth_trailword_ + aux_depth_cell_
screen = base_anchor + one extra raw family
seeds = 0, 1
samples_per_class = 2048 local diagnostic only
```

The strongest stable addition was `aux_depth_word_`:

| Variant | Feature Count | Seed0 AUC | Seed1 AUC |
|---|---:|---:|---:|
| 60D compact raw anchor | 60 | `0.9999017715454102` | `0.9998178482055664` |
| Anchor + `aux_depth_word_` | 113 | `0.9999208450317383` | `0.9998998641967773` |

One follow-up compact raw combination added the small `aux_word_global_` family:

```text
selected_raw_prefixes =
  primary_depth_trailword_
  aux_depth_cell_
  aux_depth_word_
  aux_word_global_
feature_count = 117
model = raw-only logistic
```

Artifacts:

```text
seed0 report =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_span_raw_combo_anchor_plus_aux-depth-word_aux-word-global_report.json
seed1 report =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed1_span_raw_combo_anchor_plus_aux-depth-word_aux-word-global_report.json
seed0 shuffle =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_span_raw_combo_anchor_plus_aux-depth-word_aux-word-global_shuffle_report.json
seed1 shuffle =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed1_span_raw_combo_anchor_plus_aux-depth-word_aux-word-global_shuffle_report.json
```

Metrics:

| Seed | 117D Raw AUC | Delta vs 60D compact raw | Delta vs full 273D | Delta vs 309D allraw+rank1 | Shuffle AUC |
|---:|---:|---:|---:|---:|---:|
| 0 | `0.9999246597290039` | `+0.0000228881835938` | `+0.0000104904174805` | `-0.0000009536743164` | `0.4848761558532715` |
| 1 | `0.9999103546142578` | `+0.0000925064086914` | `+0.0000667572021484` | `+0.0000095367431641` | `0.5067415237426758` |

Decision:

```text
decision = compact_raw_117d_family_anchor_positive_local_diagnostic
action = keep_as_best_compact_raw_local_candidate
```

Interpretation:

```text
The 117D raw-family anchor is a better local direction than the 96D
compact-raw-plus-low-rank variant. It beats the full 273D raw summary on both
seeds, matches or nearly matches the 309D allraw+rank1 route, and has clean
shuffle-label controls. This is still only a 2048/class local diagnostic; do
not call it remote evidence or formal SPN/PRESENT evidence. The next
plan-aligned scale step should prioritize this raw-family representation over
the exact 96D low-rank correction.
```

## V6 Raw117 Plus Low-Rank Interaction Control

The next controlled test re-added the rank1 low-rank interaction bank to the
best compact raw-family anchor:

```text
cli = scripts/fit-compressed-span-low-rank-interaction-expert
feature_model = selected_raw_plus_semantic_low_rank_block_interactions_logistic
selected_raw_prefixes =
  primary_depth_trailword_
  aux_depth_cell_
  aux_depth_word_
  aux_word_global_
selected_raw_feature_count = 117
low_rank_interaction_feature_count = 36
feature_count = 153
rank = 1
fit_split = train
score_split = held-out validation
samples_per_class = 2048 local diagnostic only
```

Artifacts:

```text
seed0 report =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_span_raw117_plus_low_rank_rank1_report.json
seed1 report =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed1_span_raw117_plus_low_rank_rank1_report.json
seed0 shuffle reports =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_span_raw117_plus_low_rank_rank1_shuffle_report.json
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_span_raw117_plus_low_rank_rank1_shuffle7400_report.json
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_span_raw117_plus_low_rank_rank1_shuffle7500_report.json
seed1 shuffle reports =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed1_span_raw117_plus_low_rank_rank1_shuffle_report.json
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed1_span_raw117_plus_low_rank_rank1_shuffle7401_report.json
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed1_span_raw117_plus_low_rank_rank1_shuffle7501_report.json
```

Metrics:

| Seed | 153D Raw117 + Rank1 AUC | Delta vs 117D raw | Delta vs full 273D | Delta vs 309D allraw+rank1 | Shuffle AUC min/mean/max |
|---:|---:|---:|---:|---:|---:|
| 0 | `0.9999237060546875` | `-0.0000009536743164` | `+0.0000095367431641` | `-0.0000019073486328` | `0.5209646225 / 0.5373158455 / 0.5526294708` |
| 1 | `0.9999132156372070` | `+0.0000028610229492` | `+0.0000696182250977` | `+0.0000123977661133` | `0.5321292877 / 0.5471987724 / 0.5624542236` |

Decision:

```text
decision = raw117_plus_low_rank_rank1_hold_due_mixed_gain_and_shuffle_control
action = keep_117d_raw_family_anchor_as_cleaner_candidate
```

Interpretation:

```text
The 153D raw117+rank1 variant is not a better promotion target than the 117D
raw-only anchor. Its main AUC is mixed relative to 117D raw-only, and repeated
shuffle-label controls stay above the raw-only control distribution. The result
does not prove a correctness failure, but the tiny main-AUC gain is not worth
the weaker control behavior. Keep this as an attribution boundary and
prioritize the 117D raw-family representation for the next plan-aligned scale
step after source-publication and remote-monitor gates are clean.
```

## V7 Raw117 Logistic Setting Sensitivity

The next local diagnostic checked whether the 117D raw-family anchor was
limited by the default logistic fitting settings:

```text
cli = scripts/fit-compressed-feature-expert
selected_raw_prefixes =
  primary_depth_trailword_
  aux_depth_cell_
  aux_depth_word_
  aux_word_global_
feature_count = 117
steps = 2000
learning_rate = 0.05
setting_sweep =
  standardize = true, false
  l2 = 0, 0.00001, 0.0001, 0.001, 0.01
samples_per_class = 2048 local diagnostic only
```

Representative metrics:

| Setting | Seed0 AUC | Seed1 AUC | Interpretation |
|---|---:|---:|---|
| `standardize=true, l2=0` | `0.9999246597290039` | `0.9999132156372070` | tied best |
| `standardize=true, l2=0.0001` | `0.9999246597290039` | `0.9999132156372070` | tied best with light regularization |
| `standardize=true, l2=0.001` | `0.9999246597290039` | `0.9999103546142578` | current default, effectively tied |
| `standardize=true, l2=0.01` | `0.9999122619628906` | `0.9998855590820312` | too much regularization |
| `standardize=false, l2=0.001` | `0.9753389358520508` | `0.9899339675903320` | standardization is required |

Matched shuffle-label controls for the tied-best light-regularized setting and
the default setting were nearly identical:

| Setting | Seed0 shuffle AUC min/mean/max | Seed1 shuffle AUC min/mean/max |
|---|---:|---:|
| `standardize=true, l2=0.0001` | `0.5214757919 / 0.5347898801 / 0.5439753532` | `0.4511165619 / 0.4956715902 / 0.5712594986` |
| `standardize=true, l2=0.001` | `0.5215826035 / 0.5350863139 / 0.5443248749` | `0.4506673813 / 0.4956979752 / 0.5719518661` |

Decision:

```text
decision = raw117_logistic_setting_sensitivity_standardization_required_l2_not_material
action = keep_117d_raw_family_anchor; do_not_claim_hyperparameter_breakthrough
```

Interpretation:

```text
The 117D raw-family anchor depends on train-split standardization. Removing
standardization destroys much of the local signal. The exact l2 value is not a
meaningful bottleneck within the low-regularization range: l2=0 and l2=0.0001
tie for best AUC, while the existing l2=0.001 default is effectively tied and
has matched shuffle behavior. This supports the representation hypothesis more
than a training-tuning explanation. Keep the candidate defined by its SPN raw
families rather than by a fragile l2 setting.
```

## V8 Raw117 Family Dropout Attribution

The next local diagnostic tested which of the four selected raw families are
structurally important by dropping one family at a time from the 117D anchor:

```text
cli = scripts/fit-compressed-feature-expert
baseline_prefixes =
  primary_depth_trailword_
  aux_depth_cell_
  aux_depth_word_
  aux_word_global_
dropout = leave one prefix family out
steps = 2000
learning_rate = 0.05
l2 = 0.001
standardize = true
samples_per_class = 2048 local diagnostic only
```

Metrics:

| Dropped family | Feature count | Seed0 AUC | Seed0 delta vs 117D | Seed1 AUC | Seed1 delta vs 117D |
|---|---:|---:|---:|---:|---:|
| none, 117D anchor | 117 | `0.9999246597290039` | `0.0` | `0.9999103546142578` | `0.0` |
| `primary_depth_trailword_` | 81 | `0.9966125488281250` | `-0.0033121109008789` | `0.9976892471313477` | `-0.0022211074829102` |
| `aux_depth_cell_` | 93 | `0.9998550415039062` | `-0.0000696182250977` | `0.9996814727783203` | `-0.0002288818359375` |
| `aux_depth_word_` | 64 | `0.9999246597290039` | `+0.0000000000000000` | `0.9998874664306641` | `-0.0000228881835938` |
| `aux_word_global_` | 113 | `0.9999208450317383` | `-0.0000038146972656` | `0.9998998641967773` | `-0.0000104904174805` |

Decision:

```text
decision = raw117_family_dropout_primary_depth_trailword_main_aux_depth_cell_required
action = keep_117d_anchor; prioritize primary_depth_trailword + aux_depth_cell as the core,
         treat aux_depth_word and aux_word_global as small complementary families
```

Interpretation:

```text
The 117D anchor is not an arbitrary feature bag. The primary depth/trailword
family is the main signal axis: removing it costs about 0.002-0.003 AUC even
when all auxiliary families remain. The aux depth/cell family is the important
second axis and costs about 7e-5 to 2.3e-4 AUC when removed. The aux depth/word
and aux word/global families are smaller complements: they help seed1 and
slightly polish seed0, but they do not replace the two-family core. Future
compact SPN architectures should preserve the primary-depth/trailword and
aux-depth/cell views explicitly before spending capacity on interaction terms.
```

## V9 Grouped Late-Fusion Branch Diagnostic

The next architecture diagnostic tested whether the family-dropout insight can
be converted directly into a branch-logit late-fusion expert:

```text
cli = scripts/fit-compressed-span-grouped-expert
samples_per_class = 2048 local diagnostic only
core_dual_branch =
  group_mode = coarse
  primary_prefix = primary_depth_trailword_
  auxiliary_prefix = aux_depth_cell_
semantic_grouped =
  group_mode = semantic
hybrid_grouped =
  group_mode = hybrid
branch_steps = 2000
combiner_steps = 2000
learning_rate = 0.05
l2 = 0.001
standardize = true
```

Metrics:

| Model | Feature / branch count | Seed0 AUC | Seed1 AUC | Decision |
|---|---:|---:|---:|---|
| Flat core raw anchor | 60 raw features | `0.9999017715454102` | `0.9998178482055664` | baseline |
| Flat 117D raw-family anchor | 117 raw features | `0.9999246597290039` | `0.9999103546142578` | keep |
| Core dual-branch late fusion | 2 branch logits | `0.9998054504394531` | `0.9995899200439453` | hold |
| Semantic grouped late fusion | 12 branch logits | `0.9993896484375000` | `0.9987850189208984` | hold |
| Semantic grouped late fusion, l2=0 | 12 branch logits | `0.9994039535522461` | `0.9988164901733398` | hold |
| Hybrid grouped late fusion | 14 branch logits | `0.9992799758911133` | `0.9986753463745117` | hold |

Decision:

```text
decision = grouped_late_fusion_holds_due_to_raw_family_information_loss
action = do_not_promote_branch_logit_fusion; preserve within-family raw detail in next architecture
```

Interpretation:

```text
The family-dropout result should not be implemented as one logit per family.
Compressing each branch to a scalar logit before fusion loses useful
within-family structure. Even the targeted two-branch core model loses to the
flat 60D raw anchor, and broader semantic/hybrid grouped models lose more. The
next SPN-aware model should still expose primary-depth/trailword and
aux-depth/cell as first-class views, but it should preserve their internal
coordinates or use a shallow structured layer over those coordinates instead
of late-fusing scalar branch scores.
```

## V10 Raw117 Prefix Sparsity Diagnostic

The follow-up checked whether the 117D raw-family anchor can be reduced to a
smaller train-ranked coordinate set without losing the local signal. This uses
the same `2048/class` span-summary feature artifacts and ranks features with
train labels only, then scores held-out validation:

```text
cli = scripts/audit-compressed-feature-sparsity
samples_per_class = 2048 local diagnostic only
feature_scope =
  primary_depth_trailword_
  aux_depth_cell_
  aux_depth_word_
  aux_word_global_
selected_scope_feature_count = 117
top_k = 8, 12, 16, 24, 32, 48, 60, 80, 117
steps = 2000
learning_rate = 0.05
l2 = 0.001
standardize = true
```

Artifacts:

```text
seed0 =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_raw117_prefix_sparsity_audit.json
seed1 =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed1_raw117_prefix_sparsity_audit.json
```

Metrics:

| Top-k inside raw117 | Seed0 AUC | Seed1 AUC |
|---:|---:|---:|
| 8 | `0.9906158447265625` | `0.9952144622802734` |
| 12 | `0.9979648590087891` | `0.9970254898071289` |
| 16 | `0.9990558624267578` | `0.9987335205078125` |
| 24 | `0.9997768402099609` | `0.9989719390869141` |
| 32 | `0.9998035430908203` | `0.9994058609008789` |
| 48 | `0.9998989105224609` | `0.9995336532592773` |
| 60 | `0.9998931884765625` | `0.9996490478515625` |
| 80 | `0.9999227523803711` | `0.9996881484985352` |
| 117 | `0.9999246597290039` | `0.9999103546142578` |

Decision:

```text
decision = raw117_prefix_sparsity_full117_remains_best_local_diagnostic
action = keep_117d_raw_family_anchor; do_not_promote_sparse_topk_as_replacement
```

Interpretation:

```text
The sparse screen is useful, but it does not justify replacing the 117D anchor.
Very small coordinate sets already recover most of the signal: top8 reaches
about 0.991/0.995 AUC, and top24 already reaches about 0.9998/0.9990 AUC.
However, the full 117D scope remains best on both seeds, with the seed1 gap
from top80 to top117 still about 2.22e-4 AUC. The top-ranked coordinates are
mostly aux depth/cell global or depth means plus a few primary depth/trailword
coordinates, so the representation is not arbitrary. The next architecture
should keep the 117D raw-family anchor as the clean compact candidate and use
the top-k ranking as an attribution map, not as a promoted replacement.
```

## V11 Raw117 Pairwise Interaction Diagnostic

The next coordinate-preserving check tested whether simple train-selected
pairwise products can improve the 117D raw-family anchor without reintroducing
the full 273D raw span summary. This required adding raw-prefix selection to
`scripts/fit-compressed-span-interaction-expert`, so the raw branch is exactly
the 117D family scope while the interaction branch uses selected
primary/auxiliary coordinates:

```text
cli = scripts/fit-compressed-span-interaction-expert
samples_per_class = 2048 local diagnostic only
raw_feature_scope =
  primary_depth_trailword_
  aux_depth_cell_
  aux_depth_word_
  aux_word_global_
raw_feature_count = 117
steps = 2000
learning_rate = 0.05
l2 = 0.001
standardize = true
```

Artifacts:

```text
core top4x4 =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_raw117_core_pair_interaction_top4_report.json
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed1_raw117_core_pair_interaction_top4_report.json
core top8x8 =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_raw117_core_pair_interaction_report.json
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed1_raw117_core_pair_interaction_report.json
core top12x8 =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_raw117_core_pair_interaction_top12x8_report.json
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed1_raw117_core_pair_interaction_top12x8_report.json
primary x aux_depth_word top8x8 =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_raw117_primary_auxword_pair_interaction_report.json
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed1_raw117_primary_auxword_pair_interaction_report.json
```

Metrics:

| Variant | Feature count | Interaction count | Seed0 AUC | Delta vs raw117 | Seed1 AUC | Delta vs raw117 |
|---|---:|---:|---:|---:|---:|---:|
| Raw117 anchor | 117 | 0 | `0.9999246597290039` | `0.0` | `0.9999103546142578` | `0.0` |
| Primary x aux-depth-cell top4x4 | 133 | 16 | `0.9999227523803711` | `-0.0000019073486328` | `0.9999027252197266` | `-0.0000076293945312` |
| Primary x aux-depth-cell top8x8 | 181 | 64 | `0.9999217987060547` | `-0.0000028610229492` | `0.9998884201049805` | `-0.0000219345092773` |
| Primary x aux-depth-cell top12x8 | 213 | 96 | `0.9999265670776367` | `+0.0000019073486328` | `0.9998846054077148` | `-0.0000257492065430` |
| Primary x aux-depth-word top8x8 | 181 | 64 | `0.9999132156372070` | `-0.0000114440917969` | `0.9998893737792969` | `-0.0000209808349609` |

Decision:

```text
decision = raw117_pairwise_interactions_hold_due_no_stable_gain
action = keep_117d_raw_family_anchor; do_not_promote_simple_pairwise_product_layer
```

Interpretation:

```text
The result rejects the simplest coordinate-preserving interaction layer. The
only positive delta is seed0 top12x8 at about +1.9e-6 AUC, but the same variant
loses about 2.6e-5 AUC on seed1. Smaller core interactions and aux-depth-word
interactions lose on both seeds. Therefore the current 117D standardized raw
family anchor is still cleaner than adding explicit pairwise product features.
Future architecture work should not spend the next slot on hand-selected
pairwise products; it should either preserve raw117 for scale confirmation or
test a genuinely different non-neighbor expert family.
```

## V12 Trail-Position And Raw117 Frozen-Score Alignment Diagnostic

The raw117 anchor now has an explicit same-validation frozen-score check
against the recovered 2048-row trail-position score artifacts. This answers a
narrow ensemble question without training another model:

```text
question =
  Are the trail-position neural expert and the 117D compressed SPN structural
  expert aligned on the same held-out validation rows, and does fixed
  score-level aggregation improve over the best single expert?

scale = 2048 total validation rows / 1024 per class local diagnostic only
negative_mode = encrypted_random_plaintexts
pairs_per_sample = 16
rounds = PRESENT-80 r8
```

Alignment artifacts:

```text
seed0 alignment =
  outputs/local_audits/i1_present_r8_seed0_trail_global_raw117_alignment_check.json
seed1 alignment =
  outputs/local_audits/i1_present_r8_seed1_trail_global_raw117_alignment_check.json
```

Both alignment checks pass for labels, sample IDs, validation key, feature
encoding, strict negative mode, sample structure, and pairs-per-sample:

```text
seed0 status = pass, rows = 2048
seed1 status = pass, rows = 2048
```

The first three-artifact diagnostic included the same-input global-statistics
control, the trail-position expert, and the raw117 compressed structural
expert:

```text
seed0 report =
  outputs/local_audits/i1_present_r8_seed0_trail_global_raw117_fixed_ensemble.json
seed1 report =
  outputs/local_audits/i1_present_r8_seed1_trail_global_raw117_fixed_ensemble.json

seed0 best_single = compressed_feature_logistic_expert
seed0 best_single_auc = 0.9999246597290039
seed0 best_ensemble = auc_positive_weighted_logit_mean
seed0 best_ensemble_auc = 0.9999408721923828
seed0 delta_best_ensemble_vs_single_auc = +0.00001621246337890625

seed1 best_single = compressed_feature_logistic_expert
seed1 best_single_auc = 0.9999103546142578
seed1 best_ensemble = auc_positive_weighted_logit_mean
seed1 best_ensemble_auc = 0.9999113082885742
seed1 delta_best_ensemble_vs_single_auc = +0.00000095367431640625
```

This result should not be reported as a diverse expert-pool success because the
global-statistics row is a control, not a candidate expert. The candidate-only
diagnostic is therefore the relevant one:

```text
seed0 candidate report =
  outputs/local_audits/i1_present_r8_seed0_trail_raw117_candidate_fixed_ensemble.json
seed1 candidate report =
  outputs/local_audits/i1_present_r8_seed1_trail_raw117_candidate_fixed_ensemble.json

seed0 trail_position_auc = 0.9985876083374023
seed0 raw117_auc = 0.9999246597290039
seed0 best_candidate_ensemble = logit_mean
seed0 best_candidate_ensemble_auc = 0.9999418258666992
seed0 delta_best_candidate_ensemble_vs_single_auc = +0.0000171661376953125
seed0 trail_raw117_probability_correlation = 0.9459612966412756
seed0 trail_raw117_error_jaccard_at_0_5 = 0.057692307692307696
seed0 diverse_pool_decision = diverse_expert_pool_not_ready
seed0 diverse_pool_errors = too_few_eligible_families

seed1 trail_position_auc = 0.9982948303222656
seed1 raw117_auc = 0.9999103546142578
seed1 best_candidate_ensemble = auc_positive_weighted_logit_mean
seed1 best_candidate_ensemble_auc = 0.9999189376831055
seed1 delta_best_candidate_ensemble_vs_single_auc = +0.00000858306884765625
seed1 trail_raw117_probability_correlation = 0.9417126952090996
seed1 trail_raw117_error_jaccard_at_0_5 = 0.06153846153846154
seed1 diverse_pool_decision = diverse_expert_pool_not_ready
seed1 diverse_pool_errors = too_few_eligible_families
```

Decision:

```text
decision = aligned_candidate_fusion_tiny_positive_but_diverse_pool_not_ready
action =
  keep raw117 as a compact structural anchor and as a scale-confirmation
  candidate; do not claim that multi-network aggregation is solved
```

Interpretation:

```text
The good news is that raw117 is not merely a detached local audit artifact: it
is aligned to the same held-out validation rows as the trail-position neural
expert, and fixed score-level aggregation gives a tiny positive AUC delta on
both local seeds. The caution is equally important. The deltas are only about
8.6e-6 to 1.7e-5 AUC, the two candidate experts remain highly correlated
around 0.94-0.95, and the candidate pool still has only two eligible families.
This is evidence for a useful aligned compact representation, not evidence
that a broad multi-network SPN ensemble is ready.

The next meaningful use is medium-scale reuse after the active 262144/class
trail-position artifacts are retrieved: export the same raw117/compact
structural score artifact on those exact validation rows, then rerun the
candidate-only fixed and train-split-calibrated fusion gates. Until then, the
right architecture hypothesis remains structured SPN-coordinate modeling over
the raw117 families, not adding more near-neighbor networks.
```

## V13 Raw117 Candidate Stacking Stability Diagnostic

The V12 candidate-only fixed fusion was extended with a train-holdout stacking
stability diagnostic across five selection seeds. This checks whether a fitted
two-score calibration layer is more reliable than the simple frozen fixed
fusion rule.

Artifacts:

```text
seed0 stability =
  outputs/local_audits/i1_present_r8_seed0_trail_raw117_candidate_stacked_selection_stability.json
seed1 stability =
  outputs/local_audits/i1_present_r8_seed1_trail_raw117_candidate_stacked_selection_stability.json

selection_seeds = 0, 1, 2, 3, 4
train_holdout_fraction = 0.25
candidate_feature_spaces = logits, probabilities
candidate_l2 = 0.0, 0.0001, 0.001, 0.01
candidate_standardize = true, false
```

Important scope limitation:

```text
validation raw117 artifact =
  compressed_span_summary, feature_count = 117
train structural artifact used by stacking =
  trail_position_stats logistic scores, feature_count = 3708
strict train raw117 score artifact found = false
```

Because stacking consumes only frozen score columns, the diagnostic is still a
useful two-expert calibration check. It should not be described as strict
raw117 train-fitted calibration until a matching train raw117 score artifact is
generated and substituted.

Metrics:

```text
seed0 positive_selection_seeds = 0 / 5
seed0 delta_stacked_vs_best_single_auc:
  min = -0.0000133514404296875
  max = -0.0000133514404296875
  mean = -0.0000133514404296875
seed0 same_selection = true
seed0 dominant_selection = logits, l2=0.0, standardize=true

seed1 positive_selection_seeds = 3 / 5
seed1 delta_stacked_vs_best_single_auc:
  min = -0.0000362396240234375
  max = +0.00000667572021484375
  mean = -0.00001049041748046875
seed1 same_selection = false
seed1 dominant_selection = logits, l2=0.0, standardize=false

both decisions = mixed_or_unstable_stacked_selection_diagnostic
```

Comparison to V12 fixed fusion:

```text
seed0 fixed_fusion_delta_vs_best_single_auc = +0.0000171661376953125
seed1 fixed_fusion_delta_vs_best_single_auc = +0.00000858306884765625

seed0 stacking_mean_delta_vs_best_single_auc = -0.0000133514404296875
seed1 stacking_mean_delta_vs_best_single_auc = -0.00001049041748046875
```

Decision:

```text
decision = fixed_fusion_tiny_positive_stacking_not_promoted
action =
  keep candidate-only fixed fusion as the clean local aligned-fusion diagnostic;
  do not promote train-fitted stacking as an ensemble improvement
```

Interpretation:

```text
The multi-network question has now been tested in a narrow but useful form.
Trail-position plus raw117 gives a tiny positive fixed-fusion signal on both
local seeds, but train-fitted stacking does not make that signal stronger or
more stable. Seed0 loses to the best single expert under all five selection
seeds. Seed1 sometimes beats the best single expert by about 6.7e-6 AUC, but
other selection seeds lose by about 3.6e-5 AUC, so the average effect is
negative and selection is not stable.

This reinforces the current architecture direction. The useful progress is the
SPN-aware compact structural representation itself, not a generic stacking
layer over two highly correlated experts. The next medium-scale step should be
to wait for retrieved 262144/class trail-position score artifacts, export
raw117 on exactly those rows, and rerun the same fixed-fusion/stability gates.
Only after a third structurally different candidate clears its own controls
should the route be called a diverse multi-network pool.
```

## V14 Matched Train Raw117 Stacking Follow-Up

The V13 limitation was resolved locally by exporting matched train and
validation raw117 score artifacts from the same compressed-span-summary feature
scope:

```text
seed0 train raw117 scores =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_train_span_raw117_matched_scores
seed0 validation raw117 scores =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_span_raw117_matched_scores
seed0 report =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_span_raw117_matched_report.json

seed1 train raw117 scores =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed1_train_span_raw117_matched_scores
seed1 validation raw117 scores =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed1_span_raw117_matched_scores
seed1 report =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed1_span_raw117_matched_report.json

feature_view = compressed_span_summary
feature_count = 117
include_feature_prefixes =
  aux_depth_cell_
  aux_depth_word_
  aux_word_global_
  primary_depth_trailword_
```

The regenerated validation raw117 metrics match the V12 raw117 anchor:

```text
seed0 matched raw117 validation_auc = 0.9999246597290039
seed1 matched raw117 validation_auc = 0.9999103546142578
```

Matched train-holdout stacking artifacts:

```text
seed0 matched stability =
  outputs/local_audits/i1_present_r8_seed0_trail_raw117_matched_candidate_stacked_selection_stability.json
seed1 matched stability =
  outputs/local_audits/i1_present_r8_seed1_trail_raw117_matched_candidate_stacked_selection_stability.json
```

Metrics:

```text
seed0 positive_selection_seeds = 5 / 5
seed0 delta_stacked_vs_best_single_auc:
  min = +0.000003814697265625
  max = +0.000003814697265625
  mean = +0.000003814697265625
seed0 same_selection = true
seed0 dominant_selection = logits, l2=0.0, standardize=false
seed0 delta_stacked_vs_fixed_ensemble_auc = -0.0000133514404296875

seed1 positive_selection_seeds = 4 / 5
seed1 delta_stacked_vs_best_single_auc:
  min = -0.00001811981201171875
  max = +0.00001049041748046875
  mean = +0.00000438690185546875
seed1 same_selection = false
seed1 dominant_selection = logits, l2=0.0, standardize=false
seed1 best delta_stacked_vs_fixed_ensemble_auc = +0.0000019073486328125
seed1 worst delta_stacked_vs_fixed_ensemble_auc = -0.000026702880859375
```

Decision:

```text
decision = matched_raw117_stacking_tiny_positive_but_not_promoted_over_fixed_fusion
action =
  keep fixed fusion as the cleaner two-expert local gate;
  keep matched raw117 train scores for future scale/postprocess reuse;
  do not call this a diverse multi-network success
```

Interpretation:

```text
Generating matched train raw117 score artifacts improves the stacking evidence:
the previous feature-scope caveat is removed, seed0 becomes stably positive,
and seed1 is positive on four out of five selection seeds. But the gain is only
around 4e-6 AUC on average. Seed0 still loses to the simpler fixed-fusion rule,
and seed1 remains selection-sensitive with one negative selection seed.

So the ranking is now clearer:
1. raw117 remains a strong compact SPN structural expert.
2. fixed fusion is the cleanest current two-expert aggregation diagnostic.
3. matched stacking is useful as a calibration audit, but not a stronger route.
4. broader multi-network aggregation still needs at least one more genuinely
   different, control-clearing expert family before promotion.

The next scale action should reuse the matched raw117 export path on the exact
retrieved 262144/class trail-position validation rows.
```

## V15 Trail + Raw117 Reliability/Residual Bucket Diagnostic

A local frozen-score residual diagnostic was added to avoid treating
"multi-network" as merely stacking correlated probabilities. The diagnostic
uses train-derived bucket edges from two aligned score artifacts and then
applies those same edges to held-out validation scores:

```text
cli =
  scripts/analyze-reliability-residual-buckets

seed0 report =
  outputs/local_audits/i1_present_r8_seed0_trail_raw117_reliability_residual_buckets.json
seed1 report =
  outputs/local_audits/i1_present_r8_seed1_trail_raw117_reliability_residual_buckets.json

model_order =
  present_trail_position_stats_pairset
  compressed_feature_logistic_expert
bucket_count = 5
claim_scope =
  local frozen-score reliability/residual bucket diagnostic only;
  not a trained third expert, not remote evidence, and not formal SPN/PRESENT evidence
```

Validation summary:

```text
seed0:
  decision = reliability_residual_bucket_route_candidate_local
  candidate_buckets = 9
  trail_auc = 0.9985876083374023
  raw117_auc = 0.9999246597290039
  best_fixed_ensemble_auc = 0.9999418258666992
  disagreement_rate_at_0_5 = 0.02392578125
  error_jaccard_at_0_5 = 0.057692307692307696
  trail_wrong_raw117_correct_rate_at_0_5 = 0.021484375
  raw117_wrong_trail_correct_rate_at_0_5 = 0.00244140625
  both_wrong_count_at_0_5 = 3

seed1:
  decision = reliability_residual_bucket_route_candidate_local
  candidate_buckets = 9
  trail_auc = 0.9982948303222656
  raw117_auc = 0.9999103546142578
  best_fixed_ensemble_auc = 0.9999189376831055
  disagreement_rate_at_0_5 = 0.02978515625
  error_jaccard_at_0_5 = 0.06153846153846154
  trail_wrong_raw117_correct_rate_at_0_5 = 0.0263671875
  raw117_wrong_trail_correct_rate_at_0_5 = 0.00341796875
  both_wrong_count_at_0_5 = 4
```

The strongest repeated pattern is not a generic ensemble effect. The lowest
`min_confidence` bucket and lowest `logit_gap_abs` bucket concentrate cases
where raw117 corrects trail-position:

```text
seed0 min_confidence bucket0:
  rows = 417
  disagreement_rate = 0.11510791366906475
  correction_gap = 0.091127
  both_wrong_lift = 0.005729

seed0 logit_gap_abs bucket0:
  rows = 415
  disagreement_rate = 0.07710843373493977
  correction_gap = 0.053012
  both_wrong_lift = 0.005764

seed1 min_confidence bucket0:
  rows = 383
  disagreement_rate = 0.1566579634464752
  correction_gap = 0.120104
  both_wrong_lift = 0.008491

seed1 logit_gap_abs bucket0:
  rows = 405
  disagreement_rate = 0.09382716049382717
  correction_gap = 0.059259
  both_wrong_lift = 0.007923
```

Decision:

```text
decision = reliability_residual_bucket_route_candidate_local
action =
  design a frozen residual expert or control gate before training scale-up;
  do not call current two-score buckets a trained third expert;
  use these buckets to target interpretable SPN reliability features
```

Interpretation:

```text
This is the first useful local evidence that a third route should focus on
residual reliability rather than another near-neighbor score average. The
diagnostic identifies train-derived validation buckets where the two strong
experts genuinely disagree and where raw117 repeatedly fixes trail-position
errors. That makes a reliability/residual expert worth designing, but it is
still only a local frozen-score diagnosis. The next implementation step should
turn the bucket signal into an interpretable frozen candidate or control gate
and then test whether it improves over raw117/fixed fusion under the same
held-out scoring protocol.
```

A follow-up local probe tested a deliberately simple train-derived bucket
router over existing frozen scores:

```text
probe =
  outputs/local_audits/i1_present_r8_trail_raw117_residual_bucket_router_probe.json
per-bucket candidate modes =
  trail-position score
  raw117 score
  probability mean
  logit mean
selection =
  choose the best mode per train-derived bucket on train labels,
  then apply the fixed bucket choices to validation
```

Probe result:

```text
seed0 best_single_auc = 0.9999246597290039
seed0 best_fixed_ensemble_auc = 0.9999418258666992
seed0 best_bucket_router_auc = 0.9999246597290039
seed0 best_bucket_router_feature = signed_logit_delta_model1_minus_model0
seed0 best_bucket_router_modes = raw117 in all buckets
seed0 delta_vs_best_fixed_auc = -0.0000171661376953125

seed1 best_single_auc = 0.9999103546142578
seed1 best_fixed_ensemble_auc = 0.9999189376831055
seed1 best_bucket_router_auc = 0.9999103546142578
seed1 best_bucket_router_feature = signed_logit_delta_model1_minus_model0
seed1 best_bucket_router_modes = raw117 in all buckets
seed1 delta_vs_best_fixed_auc = -0.00000858306884765625
```

This is a useful brake. The residual buckets explain where raw117 corrects
trail-position, but a cheap score router does not improve over raw117 and loses
to fixed fusion. Therefore the next route should not be a per-bucket score
switch. It should create an interpretable residual feature or control-clearing
expert that uses the bucket structure to model cases not already absorbed by
raw117.
