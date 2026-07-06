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
mask_artifact = outputs/local_audits/i1_present_r8_bit_sensitivity_projection_mask_seed{seed}.json
sensitivity_csv = outputs/local_audits/i1_present_r8_bit_sensitivity_projection_seed{seed}.csv
selection_split = train
```

## Gate

Promote to a compatible frozen-score expert only if all of these hold:

```text
candidate_auc > same_input_global_control_auc on both seeds
candidate_auc >= 0.55 on both seeds
mean_candidate_auc >= mean_global_control_auc + 0.01
candidate_error_overlap_with_trail_position < global_control_error_overlap_with_trail_position
active_nibble_mismatch_auc <= 0.55
input_difference_mismatch_auc <= 0.55
score_artifacts_exported = true
```

If the candidate passes the local gate, the next step is still only a
`65536/class` diagnostic plan. Do not jump directly to `1000000/class`.

## Hold Conditions

Hold the route if:

```text
trail-position 262144/class artifacts are not verified
candidate is a single-seed spike
candidate loses to same-input global control on either seed
candidate duplicates the trail-position error set
mask selection uses validation evidence
mismatch controls separate the classes
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

