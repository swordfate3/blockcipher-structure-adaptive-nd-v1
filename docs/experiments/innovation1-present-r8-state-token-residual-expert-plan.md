# Innovation 1 PRESENT r8 State-Token Residual Expert Plan

**Date:** 2026-07-07

**Status:** planned / local-only while `i1_present_r8_residual_focus_262k_retry1`
is running

## Question

The current active branch is the 262144/class residual-focus diagnostic. While
that branch is running, the next architecture work should not launch another
remote job or add more near-neighbor score averaging. The next useful question
is:

```text
Can a state-token SPN residual expert preserve the depth/word/cell coordinates
that the compressed span diagnostics found useful, then learn a small correction
over hard residual slices of the frozen trail-position + raw117 base?
```

This is a different hypothesis from a plain multi-network ensemble:

```text
not: average more similar trail/global/raw117 scores
yes: learn a compact residual correction over structured SPN state tokens
```

## Current Gate

Local planner:

```text
scripts/plan-state-token-residual-expert
```

Current generated plan:

```text
artifact = outputs/local_audits/i1_present_r8_state_token_residual_expert_plan.json
status = pending
decision = wait_for_residual_focus_outputs_before_state_token_expert
should_launch_remote = false
missing_output_count = 18
```

The planner reads only local status artifacts. It does not SSH, sync, launch
remote jobs, train a model, or claim a result.

## Evidence Behind The Candidate

Local diagnostics already narrowed the useful structural source:

```text
feature_view = trail_position_stats
strong families =
  depth_word_cell_span
  depth_cell_span
  word_span
  depth_word_span
  cell_span
```

The strongest single family was `depth_word_cell_span`, and the useful signal
appeared at medium sparse size rather than in one or two magic scalar features.
That points to a grouped coordinate-aware expert:

```text
token = stat_family x trail_depth x trail_word x cell_index
input = span/statistic features aligned to frozen score artifacts
objective = residual correction of frozen trail_position + raw117 base
```

The external literature thread also favors representation-first work:

- A. Gohr, "Improving Attacks on Round-Reduced Speck32/64 Using Deep
  Learning," CRYPTO 2019 / IACR ePrint 2019/037,
  <https://eprint.iacr.org/2019/037>.
- L. Zhang and Z. Wang, "Improving Differential-Neural Distinguisher Model For
  DES, Chaskey, and PRESENT," arXiv:2204.06341,
  <https://arxiv.org/abs/2204.06341>.
- L. Zhang et al., "Neural-Inspired Advances in Integral Cryptanalysis,"
  arXiv:2505.10790, <https://arxiv.org/abs/2505.10790>.

The takeaway is narrow: input organization and cipher-structure evidence matter
more than blindly widening the pool of similar neural networks. This plan does
not rely on the previously mis-cited arXiv:2505.10792 entry.

## Candidate Design

Route:

```text
route = present_r8_state_token_residual_expert
model_family = state_token_residual_graph
base_scores =
  trail_position_anchor
  matched_raw117_compressed_spn_structural_expert
objective = frozen_base_residual_correction
```

Model idea:

```text
1. Build tokens from span/statistic families instead of flattening all 3708
   structural dimensions equally.
2. Keep explicit coordinates: stat_family, trail_depth, trail_word, cell_index.
3. Feed the token encoder frozen base reliability views such as base logit,
   confidence, residual-focus rank, and model disagreement.
4. Train only a small additive correction on top of the frozen base score.
5. Score held-out validation after all token grouping and residual-focus choices
   are frozen from the train split.
```

This should be a residual expert, not another global classifier. A global
classifier may rediscover the already saturated trail-position/raw117 signal,
which is not the missing piece.

## Implementation Status

Initial local model skeleton:

```text
model_key = present_state_token_residual
class = PresentStateTokenResidualDistinguisher
input_feature_view = trail_position_stats
input_feature_count = 3708
selected_span_tokens = 731
token_families =
  word_span
  cell_span
  depth_word_cell_span
  depth_cell_span
  depth_word_span
```

The model does not consume the raw 39936-bit beamstats training matrix directly.
It consumes the deterministic `trail_position_stats` feature artifact exported
by `scripts/export-bit-sensitivity-features --feature-view trail_position_stats`.
That distinction matters: until a route-specific feature-artifact training CLI
or plan-aligned runner exists, this is a local feature-artifact expert, not a
remote training-matrix row.

Current validated smoke:

```text
tests/test_state_token_residual_model.py
  forward shape = [batch, 1]
  selected_span_feature_bits = 731
  attention_weights shape = [batch, 731]
  registry key = present_state_token_residual
```

Initial local feature-artifact fitter:

```text
script = scripts/fit-state-token-residual-expert
fit_split = train feature artifact
score_split = held-out validation feature artifact
output =
  EnsembleScoreArtifact train/validation score directories
  JSON report with train/validation metrics and guardrails
claim_scope = local feature-artifact diagnostic only
```

This makes the next local check executable without changing the remote training
protocol. The fitter is intentionally separate from `scripts/train` because its
input is the exported 3708D `trail_position_stats` artifact, not the raw
beamstats pair matrix used by the current residual-focus remote run.

First 2048/class local feature-artifact smoke:

```text
feature_artifacts =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed{0,1}_*_trail_stats_features
settings =
  steps = 20
  token_dim = 8
  hidden_bits = 16
  batch_size = 256
```

| Seed | State-Token Validation AUC | State-Token Validation Accuracy | Label-Shuffle Validation AUC | Label-Shuffle Calibrated Accuracy |
|---:|---:|---:|---:|---:|
| 0 | `0.9979028701782227` | `0.97607421875` | `0.29055213928222656` | `0.5` |
| 1 | `0.995269775390625` | `0.96826171875` | `0.28820180892944336` | `0.5` |

Interpretation:

```text
decision = state_token_residual_feature_artifact_smoke_pass_controls
status = local_2048class_feature_artifact_diagnostic_only
```

The state-token model learns a strong train-fitted signal from the structured
span tokens, and the label-shuffle control collapses to calibrated-random
accuracy. This validates the route as an executable local candidate. It does
not yet beat the previous flat `trail_position_stats` logistic diagnostic, which
reached saturated local AUC on the same feature view. Therefore the next useful
work is not to promote this as a global classifier; it is to use the tokenized
model for the intended residual-correction or token-coordinate control setting
after the residual-focus 262144/class artifacts are available.

Token-coordinate shuffle control was then added to the fitter:

```text
flag = --shuffle-token-coordinates
control = permute family/depth/word/cell coordinate ids across the same 731
  span feature values
```

| Seed | Normal State-Token AUC | Token-Coordinate-Shuffle AUC | Delta |
|---:|---:|---:|---:|
| 0 | `0.9979028701782227` | `0.9968814849853516` | `-0.0010213851928710938` |
| 1 | `0.995269775390625` | `0.99542236328125` | `+0.000152587890625` |

Interpretation:

```text
decision = hold_coordinate_structure_claim_for_current_state_token_smoke
status = local_2048class_coordinate_control_diagnostic
```

The coordinate-shuffle control does not collapse. That is useful negative
evidence: the current small state-token expert mostly benefits from the selected
span scalar values, not from proven use of the true depth/word/cell coordinate
layout. The model and fitter remain useful local infrastructure, but the route
must not claim that token coordinates are already validated. The next
architecture iteration should either optimize the intended residual-correction
objective directly or add a stronger coordinate-dependent mechanism whose
coordinate-shuffle control is expected to matter.

The next local implementation moved from global classification to the intended
frozen-base correction objective:

```text
script = scripts/fit-state-token-residual-correction-expert
base_artifacts =
  train/validation trail_position_stats logistic score artifact
  train/validation matched raw117 compressed SPN structural score artifact
objective =
  base_logit_mean + state_token_residual_correction
feature_view = trail_position_stats
strict_negative_mode = encrypted_random_plaintexts
```

The correction fitter validates train/validation feature dirs, aligned labels
and sample ids, strict negative mode, and the required `trail_position_stats`
view. It freezes the two base score artifacts and trains only the additive
state-token correction term.

First 2048/class frozen-base correction smoke:

| Seed | Mode | Base Validation AUC | Corrected Validation AUC | Delta AUC | Base Accuracy | Corrected Accuracy |
|---:|---|---:|---:|---:|---:|---:|
| 0 | uniform | `0.9999990463256836` | `0.9999990463256836` | `0.0` | `0.9990234375` | `0.9990234375` |
| 1 | uniform | `0.999995231628418` | `0.999995231628418` | `0.0` | `0.9990234375` | `0.9990234375` |
| 0 | focus10 | `0.9999990463256836` | `0.9999990463256836` | `0.0` | `0.9990234375` | `0.9990234375` |
| 1 | focus10 | `0.999995231628418` | `0.999995231628418` | `0.0` | `0.9990234375` | `0.9990234375` |

Artifacts:

```text
outputs/local_audits/i1_present_r8_state_token_residual_correction_seed0_report.json
outputs/local_audits/i1_present_r8_state_token_residual_correction_seed1_report.json
outputs/local_audits/i1_present_r8_state_token_residual_correction_focus10_seed0_report.json
outputs/local_audits/i1_present_r8_state_token_residual_correction_focus10_seed1_report.json
```

Interpretation:

```text
decision = state_token_residual_correction_diagnostic_no_base_gain
status = local_2048class_frozen_base_correction_diagnostic_only
```

The correction objective is now executable and guarded, but the available
2048/class validation artifacts are already near-saturated under the frozen
trail+raw117 base. Uniform and focus10 correction variants did not improve AUC
or accuracy. This is not evidence that state-token correction has failed at
meaningful scale; it says this tiny local validation screen has too little
remaining headroom for a global AUC gain. The route should wait for the active
262144/class residual-focus artifacts before any promotion or rejection.

The additive correction head is now zero-initialized by default:

```text
default = base_logit_mean + 0.0 correction at step 0
ablation flag = --no-zero-init-correction-head
```

This is a semantic/stability fix rather than a performance claim. The residual
expert should begin as the frozen base and then learn only deviations justified
by the train split. A short local 20-step focus10 diagnostic with zero init kept
global AUC and accuracy unchanged and lowered hard-slice residual loss less than
the earlier random-head smoke:

| Seed | Zero-Init Focus Residual-Loss Delta | Prior Random-Head Focus Residual-Loss Delta |
|---:|---:|---:|
| 0 | `-0.00026168495156135563` | `-0.0006984658851611619` |
| 1 | `-0.00012316551115307273` | `-0.0004869211776279414` |

Interpretation:

```text
decision = keep_zero_init_as_default_semantic_guard
status = local_2048class_training_semantics_fix_not_performance_win
```

The true-label focus10 correction lowered residual loss on the hard slice,
while label-shuffle controls made residual loss much worse. Token-coordinate
shuffle controls still partially lowered residual loss, so these diagnostics
show label-dependent residual signal but do not prove true depth/word/cell
coordinate-layout dependence.

The next local attribution guard adds a stricter value-only token control:

```text
flag = --drop-token-coordinates
control = zero and freeze family/depth/word/cell embeddings while keeping the
  same selected span feature values
```

This differs from coordinate shuffle. Shuffle asks whether wrong coordinates
hurt. Drop asks whether coordinates add any value beyond the selected scalar
span features at all. A state-token route should not be promoted as
coordinate-aware unless it beats both the shuffled-coordinate and dropped-
coordinate controls under the same train/validation protocol.

The first 2048/class drop-coordinate diagnostic used the same 20-step local
feature-artifact setup as the earlier state-token smoke:

```text
script = scripts/fit-state-token-residual-expert
flag = --drop-token-coordinates
steps = 20
token_dim = 8
hidden_bits = 16
batch_size = 256
```

| Seed | Normal AUC | Coordinate-Shuffle AUC | Drop-Coordinate AUC |
|---:|---:|---:|---:|
| 0 | `0.9979028701782227` | `0.9968814849853516` | `0.997344970703125` |
| 1 | `0.995269775390625` | `0.99542236328125` | `0.9963893890380859` |

The drop-coordinate control does not collapse. It remains a strong value-only
token classifier, so the current global state-token implementation is better
described as a learned selected-span-value expert than as proven coordinate-
layout evidence.

A fair zero-init residual-correction coordinate attribution check was also run
with focus10 and the frozen trail+raw117 base:

| Seed | True Focus Residual-Loss Delta | Coordinate-Shuffle Delta | Drop-Coordinate Delta |
|---:|---:|---:|---:|
| 0 | `-0.00026168495156135563` | `-0.00025688090503773325` | `-0.00035369516459912015` |
| 1 | `-0.00012316551115307273` | `-0.00011917128751281342` | `-0.00013718681714702807` |

Global AUC and accuracy remained unchanged in all three zero-init focus10
correction variants. The hard-slice residual-loss drops are real local
diagnostics, but they survive both coordinate shuffle and coordinate drop.

Interpretation:

```text
decision = hold_state_token_coordinate_layout_claim
status = local_2048class_value_only_control_matches_coordinate_model
action =
  keep state-token/drop-coordinate controls in the gate;
  do not promote current state-token as coordinate-aware;
  next architecture work should either add a stronger coordinate-dependent
  mechanism or prioritize residual family/source selection after the active
  262144/class residual-focus artifacts are retrieved
```

## Required Controls

Do not promote the route unless these controls are present:

```text
same_input_global_control
uniform_residual_control
label_shuffle_control
token_coordinate_shuffle_control
token_coordinate_drop_control
train_only_selection_control
```

Control meanings:

```text
same_input_global_control =
  proves the token expert beats a collapsed same-input view

uniform_residual_control =
  proves residual focusing matters more than a generic correction

label_shuffle_control =
  catches leakage or invalid attribution

token_coordinate_shuffle_control =
  proves depth/word/cell coordinates matter, not just feature count

token_coordinate_drop_control =
  proves coordinate embeddings add value beyond the selected span scalar values

train_only_selection_control =
  proves validation labels did not choose token families, masks, buckets, or
  thresholds
```

## Activation Policy

While the active residual-focus 262144/class branch is pending:

```text
allowed =
  finalize_state_token_experiment_plan
  prepare_local_smoke_tests_only

forbidden =
  launch_state_token_remote
  scale_state_token_to_1m
  claim_state_token_candidate
  change_labels_or_negative_mode
```

If residual-focus gate fails or holds:

```text
decision = repair_residual_focus_before_state_token_expert
allowed = repair_residual_focus_source_or_objective
```

If residual-focus gate passes:

```text
decision = state_token_residual_expert_local_plan_ready
allowed =
  write_state_token_model_smoke_test
  implement_local_state_token_smoke_only
  compare_against_same_input_controls
```

## First Implementable Checks After Gate Pass

The first implementation must be local and boring:

```text
1. state_token_forward_smoke
   Prove the model accepts the planned tokenized feature shape.

2. local_2048class_residual_slice_screen
   Use `scripts/fit-state-token-residual-expert` on existing train/validation
   `trail_position_stats` feature artifacts, then compare the frozen score
   artifact against the compressed logistic/span baselines and controls.

3. same_protocol_control_gate
   Require the route to beat same-input/global, uniform residual, label-shuffle,
   token-coordinate-shuffle, and token-coordinate-drop controls before any
   remote promotion.
```

## Claim Scope

This plan is an architecture route and local action guard. It does not report a
new PRESENT r8 result, does not upgrade 2048/class local diagnostics to remote
evidence, and does not make a formal SPN/PRESENT claim. The current remote
residual-focus run must finish, be retrieved, and pass its own gate before this
route can become more than local preparation.
