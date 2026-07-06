# Innovation 1 SPN Bit-Sensitivity Non-Neighbor Route

**Date:** 2026-07-07

**Status:** conditional research route; no remote launch while the active
trail-position 262144/class watchers are running

## Current State

The active PRESENT-80 r8 trail-position route is still waiting for retrieved
medium-scale evidence:

```text
active_branch = wait_for_trail_position_262k_results
active_runs = seed0, seed1
samples_per_class = 262144
status = running
train_matrix_rows = 0
postprocess_allowed = false
```

That means the next useful work is not another remote SPN launch. The safe
parallel work is route design that can consume the trail-position score
artifacts after they exist.

## External Evidence Check

The direction is based on current local evidence plus the external literature
thread already used by this project:

- Zhang and Wang 2022, "Improving Differential-Neural Distinguisher Model For
  DES, Chaskey, and PRESENT", arXiv:2204.06341,
  <https://arxiv.org/abs/2204.06341>.
- Zhang et al. 2025, "Neural-Inspired Advances in Integral Cryptanalysis",
  arXiv:2505.10790, <https://arxiv.org/abs/2505.10790>.
- Tashkilkhanova et al. 2025, "Enhancing RX Neural Cryptanalysis: Advanced
  Data Formatting and Weighted Differential Techniques for Block Ciphers",
  arXiv:2511.06336, <https://arxiv.org/abs/2511.06336>.
- Zhang et al. 2025, "More Efficient Deep Learning-Based Distinguishing Attacks
  with Multiple Ciphertext Pairs", arXiv:2505.10792,
  <https://arxiv.org/abs/2505.10792>.
- The local paper index also flags bit selection, GPD, PRESENT entropy, and
  multi-ciphertext-pair work as representation/feature-search evidence rather
  than broad model-ensemble evidence.

The 2026-07-07 re-check adds one concrete implementation lesson. Recent
RX-neural work uses bit sensitivity tests to reduce the input data format, then
uses the freed input budget to include more ciphertext-pair evidence. That
pattern fits this route better than a wider generic ensemble: first identify
stable, train-only informative axes; then test whether the compact evidence can
support a non-neighbor expert or higher pair count without changing labels,
negative construction, or validation keys.

The takeaway is not "add more weak networks." The stronger route is:

```text
SPN-aware data/feature representation
train-only bit/axis sensitivity compression
multi-pair evidence only under the same protocol
then structure-aware architecture over that representation
then diverse aggregation only after a real non-neighbor expert exists
```

## Why This Route

The current trail-position route is strong locally, but it is not enough to
start averaging nearby variants:

```text
near-neighbor frozen-score aggregation = no gain over best trail-position expert
cell-value histogram = weak-positive but loses to same-input global control
raw matrix Inception = fails the both-seed global-control gate
handwritten InvP aggregate statistics = held
```

The missing ingredient for the user's multi-network idea is not count. It is a
third expert family that is both:

```text
weak-positive under the same protocol
low-overlap / non-neighbor relative to trail-position
```

Bit-sensitivity-guided projection is a better candidate than another generic
network because it can use the trail-position artifacts to choose a compact,
frozen representation and then test whether that representation carries a
different error pattern.

## Proposed Non-Neighbor Expert

Name:

```text
present_r8_bit_sensitivity_projection_expert
```

Hypothesis:

```text
The trail-position route may contain stable depth/cell/word sensitivity axes.
A compact projection expert built from those axes can become a non-neighbor
weak-positive expert, or can explain that the trail-position signal is already
too concentrated for useful aggregation.
```

The expert must be selected on training-split evidence only, then frozen before
validation:

```text
selection_input = trail-position candidate/control frozen scores plus features
selection_split = train only
validation_split = held-out validation key
allowed_projection_sources = depth, word, cell, prefix/trail blocks, activity
disallowed_shortcuts = label leakage, validation-selected masks, changed negatives
```

The feature-export tooling added for this route now makes that discipline
executable:

```text
train export -> features.npy / labels.npy / sample_ids.npy / metadata.json
validation export -> same files, plus reference-artifact alignment check
selector -> train-only frozen mask
scorer -> validation-only frozen projection scores
gate -> same-protocol margin and error-overlap checks
```

This differs from the held SGP route because SGP searched unstable raw axes
directly. This route starts from the current strongest controlled
trail-position residual and asks which axes explain its residual errors.

## 2026-07-07 V0 Raw-Axis Screen

A seed0 local 2048/class screen tested the simplest executable version of this
idea: select 64 individual raw feature axes on the train split, freeze the
mask, score the held-out validation split, and compare against the existing
global-control and trail-position frozen score artifacts.

```text
gate = outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_gate.json
decision = hold_projection_duplicate_or_weak
global_control_auc = 0.8542919158935547
trail_position_anchor_auc = 0.9985876083374023
projection_auc = 0.49335479736328125
projection_margin_vs_global_auc = -0.36093711853027344
projection_margin_vs_anchor_auc = -0.5052328109741211
```

The low error overlap with the trail-position anchor is not a diversity win:
the projection is near random. This rejects the raw individual-axis projection
as a candidate expert. It does not reject the broader bit-sensitivity idea,
because the external evidence points to feature-format compression and
multi-pair evidence, not necessarily single raw-axis masks.

Updated implication:

```text
do_not_expand_v0_raw_axis_projection
do_not_run_remote_or_seed1_for_this_exact_v0_screen
keep train/validation-aligned export and gate tooling
if revisiting projection, use grouped structural axes, residual summaries, or
multi-pair compressed formats rather than isolated raw feature columns
```

## 2026-07-07 V1 Grouped-Axis Screen

The next concrete step was implemented and screened locally, not launched as a
new remote experiment:

```text
scripts/select-bit-sensitivity-projection --group-size <n> --top-groups <k>
scripts/apply-bit-sensitivity-projection
projection_unit = contiguous_axis_group
mask fields = selected_groups, selected_axes, selected_group_count
scorer behavior = average each selected group as one frozen projection unit
```

This is a deliberately small correction to the v0 failure. It keeps the same
train-only mask discipline, but changes the projection unit from an isolated
raw scalar to a contiguous structure block. That is closer to the external
evidence thread:

```text
Zhang/Wang-style PRESENT gains are representation-driven.
Multi-pair neural distinguisher work treats input formatting as part of the
attack, not as a neutral preprocessing detail.
Bit-selection/sensitivity work is useful when it compresses structured input
so the saved budget can be spent on better evidence, not when it creates a
standalone single-bit oracle.
```

2048/class seed0 validation results:

| Variant | Projection units | AUC |
|---|---:|---:|
| group4/top16 | 16 | `0.5030102729797363` |
| group8/top8 | 8 | `0.5734086036682129` |
| group16/top4 | 4 | `0.5104804039001465` |
| group32/top2 | 2 | `0.5292434692382812` |

Anchors:

```text
same_input_global_control_auc = 0.8542919158935547
trail_position_anchor_auc = 0.9985876083374023
best_grouped_projection_auc = 0.5734086036682129
decision = hold_projection_duplicate_or_weak
hold_reason = does_not_clear_global_control
```

Important limitation:

```text
grouped-axis projection is a weak diagnostic signal, not an expert
no grouped-axis variant clears the same-input global control
no remote launch is justified from this exact contiguous-group mean variant
```

Updated implication:

```text
do_not_expand_v1_contiguous_group_mean_projection
do_not_run_remote_or_seed1_for_this_exact_v1_screen
preserve grouped tooling for future structure-aware residual summaries
next local-only route should change the representation, not merely the number
of contiguous axes averaged together
```

The first valid use is after watcher retrieval:

```text
1. postprocess 262144/class trail-position score artifacts
2. export train/validation bit-sensitivity features aligned to those artifacts
3. select grouped masks on train only
4. apply grouped frozen scorer on validation only
5. run postprocess-bit-sensitivity-projection against global and trail anchors
```

For the next variant, the hypothesis should move from simple contiguous group
means to structural residual summaries: for example, per-depth/per-cell
residual response maps, compressed multi-pair block statistics, or explicit
trail-family residual features. Those are more aligned with the literature
than adding more weak projection scorers to an ensemble.

## 2026-07-07 V2 Structural-Stats Projection Screen

The next variant implements that representation shift directly:

```text
feature_view = trail_position_stats
raw_input_bits = 39936
structural_stats_bits = 3708
selection = train split only
scoring = held-out validation split only
```

Instead of selecting isolated raw bits or contiguous raw columns, the selector
works on deterministic per-depth/per-word/per-cell statistics already used by
the strongest trail-position model family. This is still a frozen projection,
not a trained network.

2048/class local result:

```text
seed0 global_control_auc = 0.8542919158935547
seed0 structural_stats_projection_auc = 0.9096775054931641
seed0 trail_position_anchor_auc = 0.9985876083374023
seed0 margin_vs_global_auc = +0.055385589599609375
seed0 error_jaccard_with_anchor = 0.057441253263707574

seed1 global_control_auc = 0.8728437423706055
seed1 structural_stats_projection_auc = 0.9378452301025391
seed1 trail_position_anchor_auc = 0.9982948303222656
seed1 margin_vs_global_auc = +0.0650014877319336
seed1 error_jaccard_with_anchor = 0.08157099697885196

both decisions = projection_expert_ready_for_local_screen
```

Updated interpretation:

```text
V0 raw-axis projection failed.
V1 contiguous raw-axis group means remained weak.
V2 structural-stat projection clears the same-input global control on seed0
and seed1.
```

This is the strongest evidence so far for the non-neighbor projection idea,
but the claim is still narrow: two-seed local diagnostic only. The low
error-overlap pattern survives seed1, but naive frozen-score aggregation still
does not beat the best single trail-position expert:

```text
seed0 best_ensemble_delta_vs_best_single_auc = -0.001667022705078125
seed1 best_ensemble_delta_vs_best_single_auc = -0.001583099365234375
```

So the route should not be sold as "many networks already improve the best
model." The real result is better: a structurally different projection expert
now exists locally and clears strong controls on two seeds. The next work is to
test whether it survives larger retrieved trail-position artifacts and whether
more careful stacking/calibration can exploit its low-overlap errors.

That calibration question has now been tested once with a train-fitted
logistic stacking diagnostic:

```text
cli = scripts/evaluate-stacked-ensemble
fit_split = train frozen-score artifacts
eval_split = held-out validation frozen-score artifacts
feature_space = logits

seed0 stacked_validation_auc = 0.998295783996582
seed0 best_single_validation_auc = 0.9985876083374023
seed0 delta_stacked_vs_best_single_auc = -0.0002918243408203125
seed0 delta_stacked_vs_fixed_ensemble_auc = +0.0013751983642578125

seed1 stacked_validation_auc = 0.9975728988647461
seed1 best_single_validation_auc = 0.9982948303222656
seed1 delta_stacked_vs_best_single_auc = -0.0007219314575195312
seed1 delta_stacked_vs_fixed_ensemble_auc = +0.0008611679077148438

both decisions = stacked_ensemble_diagnostic_no_best_single_gain
```

This is the right kind of ensemble test because the weights are fitted on the
train split and scored on held-out validation artifacts. It improves over the
naive fixed ensemble, so calibration matters. It still does not beat the best
single trail-position expert, so the current result remains a diagnostic for a
real non-neighbor representation, not evidence that multi-network aggregation
has already improved the strongest local candidate.

Promotion remains hard:

```text
both seeds must clear same-input global control
both seeds must be weak-positive
error overlap with trail-position must be low for the right reason
mismatch controls must stay near random
```

## Activation Gate

Do not implement or launch this as a meaningful experiment until the active
trail-position 262144/class postprocess reaches one of these states:

```text
support_trail_position_score_residual_all_runs
mixed_trail_position_score_residual
hold_trail_position_score_residual_due_high_overlap
```

If the 262144/class artifacts are still missing, stale, or unverified, the only
allowed actions are documentation, local tooling, or tests that do not touch the
active remote runs.

## Local Screen Gate

The first executable step must be a local diagnostic screen, not a remote
scale-up:

```text
rounds = 8
samples_per_class <= 2048
pairs_per_sample = 16
sample_structure = plaintext_integral_nibble_difference_matched_negative
negative_mode = encrypted_random_plaintexts
train_key = 0x00000000000000000000
validation_key = 0x11111111111111111111
same_budget_baseline = present_pairset_global_stats
anchor = present_trail_position_stats_pairset
```

Advance only if the projection expert:

```text
beats same-input global control on both seeds
is weak-positive on both seeds
has lower error overlap with trail-position than the global control does
does not pass active-nibble or input-difference mismatch controls
exports compatible frozen scores
```

Hold immediately if it is a single-seed spike, loses to the same-input global
control, or simply duplicates the trail-position error set.

## Claim Scope

Allowed:

```text
conditional non-neighbor expert route
local diagnostic screen plan
representation-search hypothesis
```

Not allowed:

```text
do_not_launch_remote_now
do_not_claim_formal_spn_present_evidence
do_not_claim_breakthrough
do_not_claim_diverse_ensemble_ready
do_not_treat_262144_class_as_formal_evidence
```

## Next Action

Wait for the active 262144/class trail-position artifacts. When they are ready,
run score postprocess first, then decide:

```text
pass with low overlap -> prepare 1M trail-position confirmation first
pass but high overlap -> prioritize this projection expert as an explanation/diversity screen
mixed/weak -> use sensitivity analysis diagnostically, not as remote launch basis
```

## 2026-07-07 Literature-Integrated Next Route

The latest local evidence and the external literature point to the same next
research shape:

```text
not = wider pool of near-neighbor neural models
yes = better SPN evidence presentation, then calibrated aggregation
```

Relevant external pressure:

```text
Zhang/Wang 2022 PRESENT/DES/Chaskey work:
  derived multi-ciphertext-pair features and PRESENT-specific kernel choices
  matter for PRESENT distinguishers.

Neural-inspired integral cryptanalysis:
  neural models are useful as feature explorers for integral/SPN properties,
  so the right abstraction is often the state/data view, not just model size.

LLM-neural-distinguisher negative signal:
  a very different model family does not automatically improve a strong
  neural distinguisher; prompt/data representation carries much of the value.
```

Current local pressure:

```text
V0 raw-axis bit projection -> rejected
V1 contiguous raw-axis group projection -> rejected
V2 trail-position structural-stat projection -> clears same-input global
control on seed0 and seed1
train-fitted stacking -> improves over fixed ensemble but still loses to
best single trail-position anchor on both seeds
```

Therefore the next method-level route is:

```text
SPN compressed evidence pooling
```

This means converting the trail-position/bit-sensitivity signal into a compact
state-cell evidence table that can support more samples or more pair evidence
without simply concatenating more raw bits. The candidate should reuse
structural summaries already shown to matter:

```text
per depth / active cell
per word / P-layer position
S-box DDT-consistent beam statistics
train-only sensitivity-selected structural axes
optional pair-count or cell-count pooling summaries
```

The first implementation must be a local tooling/diagnostic step only while the
active 262144/class remote run is incomplete. Do not launch a new remote branch
or spend a GPU slot until the existing watcher retrieves both seed score
artifact sets.

Concrete next gate after 262144/class artifacts are ready:

```text
1. postprocess trail-position seed0/seed1 score artifacts
2. classify the anchor as pass / mixed / high-overlap hold
3. if pass or high-overlap hold, reuse the retrieved train/validation score
   artifacts to export the V2 structural-stat projection at the same scale
4. fit stacking only on train artifacts
5. score only on held-out validation artifacts
6. require improvement over best single anchor before calling aggregation useful
```

Do not run a validation hyperparameter sweep and call the best variant a
research gain. Any calibration choice beyond the current default should be
chosen by a train-side rule or a nested split before reporting validation AUC.

Promotion rule:

```text
promote_to_medium_scale_followup only if:
  same-budget global control is cleared
  deterministic/mismatch controls remain non-explanatory
  frozen scores are compatible
  error overlap with trail-position is low or explains a real residual
  train-fitted aggregation beats the best single validation AUC

otherwise:
  keep as representation diagnostic or discard
```
