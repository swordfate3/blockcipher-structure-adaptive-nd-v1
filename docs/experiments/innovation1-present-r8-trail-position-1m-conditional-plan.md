# Innovation 1 PRESENT r8 Trail-Position 1M Conditional Plan

**Date:** 2026-07-07

**Status:** conditional plan only / no launch / waiting for 262144/class
seed0+seed1 retrieval

**Scope:** PRESENT-80 r8 trail-position beamstats route with strict
`encrypted_random_plaintexts` negatives. This document defines the next scale
gate only if the active `262144/class` watcher-managed run completes,
retrieves, verifies, and passes the residual/error-overlap gate.

## Why This Plan Exists

The active route is:

```text
run_ids =
  i1_present_r8_trail_position_beamstats_262k_seed0_gpu0_20260706
  i1_present_r8_trail_position_beamstats_262k_seed1_gpu1_20260706

candidate = present_trail_position_stats_pairset
same-input neural control = present_pairset_global_stats
feature = present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits
sample_structure = plaintext_integral_nibble_difference_matched_negative
negative_mode = encrypted_random_plaintexts
scale = 262144/class
```

Latest local-only state at plan creation:

```text
status = running
train_matrix rows = 0 for both seeds
score_artifacts = not ready
postprocess_allowed = false
cache_total_progress = 62.5%
cache_negative_class_progress = 25.0%
```

This is progress, not a result. The purpose of this document is to make the
next decision explicit before results arrive, so a positive 262k diagnostic
does not get over-promoted into a formal claim or an immediate remote launch.

## Activation Trigger

This 1M plan is activated only if all of the following are true:

```text
1. both 262144/class seed runs have train_matrix rows >= 2
2. both runs have complete global_stats_control and trail_position score artifacts
3. score artifact verification passes with expected rows = 262144
4. postprocess decision = support_trail_position_score_residual_all_runs
5. each seed candidate clears the same-input global control by the declared AUC margin
6. error-overlap analysis does not flag high-overlap or threshold-side risk
7. deterministic/mismatch controls still support active-nibble/input-difference specificity
```

If any trigger item fails, do not prepare a remote 1M launch. Instead update the
262k experiment record and return to SPN-aware representation search or
non-neighbor expert discovery.

## Research Question

If the medium diagnostic passes:

```text
Does the PRESENT r8 trail-position representation still beat the same-input
global-statistics neural control at 1000000/class across multiple seeds?
```

This asks whether the position-aware residual survives scale. It does not ask
whether this is a diverse ensemble result, and it does not replace Zhang/Wang
r7 Case2 reproduction.

## Fixed Protocol

Keep these fields unchanged from the 262k protocol:

| Field | Value |
|---|---|
| Cipher | `PRESENT-80` |
| Rounds | `8` |
| Pairs per sample | `16` |
| Feature | `present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits` |
| Sample structure | `plaintext_integral_nibble_difference_matched_negative` |
| Difference profile | `present_zhang_wang2022_mcnd` |
| Difference member | `0` |
| Integral active nibble | `0` |
| Negative mode | `encrypted_random_plaintexts` |
| Train key | `0x00000000000000000000` |
| Validation key | `0x11111111111111111111` |
| Loss | `mse` |
| Optimizer | `adam` |
| Learning rate | `0.0001` |
| Weight decay | `0.00001` |
| Checkpoint metric | `val_auc` |
| Restore best checkpoint | `true` |

Do not change labels, validation data, negative construction, metric
computation, sample IDs, or plan-alignment logic between the 262k gate and the
1M confirmation.

## Lean Training Matrix

Use two rows per seed:

| Row | Model | Role |
|---:|---|---|
| 0 | `present_pairset_global_stats` | same-input neural/global-stat control |
| 1 | `present_trail_position_stats_pairset` | position-aware candidate |

Recommended seed ladder:

```text
phase A = seed0 and seed1 at 1000000/class, only after 262k both-seed pass
phase B = optional seed2/seed3 only if phase A is positive but variance is large
```

Do not add near-neighbor ensemble rows to this training matrix. A future
diverse-pool plan may consume the frozen score artifacts only after this route
and at least one non-neighbor expert have compatible aligned scores.

## Required Companion Controls

The 1M training rows are not enough by themselves. Before any stronger claim,
the evidence package must include:

```text
same-input global-stat neural control at 1000000/class
frozen score artifacts for both rows and both seeds
score-artifact verification summaries
trail-position residual/error-overlap reports
deterministic split baseline comparison
active-nibble mismatch control
input-difference mismatch control
pair-order reverse assessment
```

If deterministic/mismatch controls are too expensive at 1000000/class, the 1M
result can only be called scale confirmation of the neural residual, not formal
attribution evidence. The report must then explicitly cite the completed 262k
or smaller same-protocol control gate and mark the attribution bridge as
limited.

## Gate

Primary continue gate:

```text
candidate_auc >= global_control_auc + 0.001 on every launched 1M seed
min_candidate_auc_margin_vs_global >= 0.001
candidate_calibrated_accuracy >= global_control_calibrated_accuracy on every seed
postprocess decision remains support_trail_position_score_residual_all_runs
```

Hold gate:

```text
candidate clears global control by AUC but has high error overlap,
or candidate only improves one seed,
or threshold-side corrections are worse than the control.
```

Stop gate:

```text
candidate fails to clear same-input global control on any 1M seed,
or score artifacts fail verification,
or mismatch controls show the signal is explained by active-nibble/input-difference leakage.
```

## Claim Scope

Allowed after a passing phase A:

```text
PRESENT r8 trail-position has two-seed 1000000/class scale-confirmation
evidence over a same-input global-stat neural control under the fixed
matched-negative integral protocol.
```

Not allowed:

```text
breakthrough
SOTA
formal SPN/PRESENT claim
Zhang/Wang r7 Case2 reproduction claim
diverse multi-network ensemble claim
proof that generic trail features solve r8/r9
```

Formal-style route evidence would still require a complete result archive,
documented controls, multi-seed variance interpretation, and conservative
claim wording.

## Remote Readiness Requirements

Before any launch asset is generated:

```text
262k decision report is updated and committed
docs/experiments record contains per-seed metrics and artifact paths
1M plan CSV exists and is checked for fixed protocol fields
dataset/feature cache is disk-backed under G:\lxy
progress JSONL emits during cache generation
score export writes labels, probabilities, logits, sample_ids, and models.json
verification summary is required before done markers
launcher uses cmd.exe /c, not cmd.exe /k
run-owned clean clone or clean source gate is used
local watcher/retrieval tmux is prepared before launch handoff
```

This plan does not authorize an immediate remote run. A future launch should be
created only from a pushed commit and only after the 262k gate activates this
document.

## Next Action

Current action:

```text
wait_for_trail_position_262k_results
```

When both 262k runs are ready:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/postprocess-trail-position-result \
  --run-root outputs/remote_results/i1_present_r8_trail_position_beamstats_262k_seed0_gpu0_20260706 \
  --run-root outputs/remote_results/i1_present_r8_trail_position_beamstats_262k_seed1_gpu1_20260706 \
  --expected-score-rows 262144 \
  --output outputs/remote_results/i1_present_r8_trail_position_beamstats_262k_postprocess_status.json

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/render-trail-position-report \
  --postprocess outputs/remote_results/i1_present_r8_trail_position_beamstats_262k_postprocess_status.json \
  --output outputs/remote_results/i1_present_r8_trail_position_beamstats_262k_decision_report.md
```

Only if that decision report passes the activation trigger should a concrete
1M CSV and remote readiness config be prepared.
