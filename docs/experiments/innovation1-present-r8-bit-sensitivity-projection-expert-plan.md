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

Interpretation:

```text
This is the first bit-sensitivity projection variant that clears the
same-input global control on both local seeds. It supports the representation
hypothesis: structure-aware residual summaries are materially better than
raw-axis masks or contiguous raw-axis means. However, simple score aggregation
with the trail-position anchor still does not beat the best single
trail-position expert. Train-fitted logistic stacking improves over the fixed
ensemble on both seeds, but it also remains below the best single
trail-position anchor. Treat V2 as a two-seed local non-neighbor diagnostic
candidate, not as a completed ensemble improvement or remote-launch result.
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
