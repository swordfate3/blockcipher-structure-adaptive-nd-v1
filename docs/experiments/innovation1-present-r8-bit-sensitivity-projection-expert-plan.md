# Innovation 1 PRESENT r8 Bit-Sensitivity Projection Expert Plan

**Date:** 2026-07-07

**Status:** conditional local-screen plan; blocked on active trail-position
262144/class score artifacts; no remote launch asset

## Question

Can a bit-sensitivity-guided projection expert become a real non-neighbor
PRESENT r8 expert under the current matched-negative trail-position protocol?

The target is not a higher-capacity model. The target is a different, compact
representation selected from stable trail-position residual sensitivity axes.

## Dependency

This plan is inactive until:

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
this experiment.

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
