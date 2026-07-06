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
