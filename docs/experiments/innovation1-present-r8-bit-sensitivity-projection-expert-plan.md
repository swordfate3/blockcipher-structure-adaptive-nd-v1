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
