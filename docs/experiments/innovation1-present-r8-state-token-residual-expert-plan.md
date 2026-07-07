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

## Required Controls

Do not promote the route unless these controls are present:

```text
same_input_global_control
uniform_residual_control
label_shuffle_control
token_coordinate_shuffle_control
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
   Compare state-token correction against frozen trail+raw117 base and controls.

3. same_protocol_control_gate
   Require the route to beat same-input/global, uniform residual, label-shuffle,
   and token-coordinate-shuffle controls before any remote promotion.
```

## Claim Scope

This plan is an architecture route and local action guard. It does not report a
new PRESENT r8 result, does not upgrade 2048/class local diagnostics to remote
evidence, and does not make a formal SPN/PRESENT claim. The current remote
residual-focus run must finish, be retrieved, and pass its own gate before this
route can become more than local preparation.
