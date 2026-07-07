# Innovation 1 SPN Route Arbitration Memo

**Date:** 2026-07-07

**Status:** research-route decision memo; no new training result

## Question

The project needs a route that can improve SPN/PRESENT distinguishers without
blindly following either of these weak defaults:

```text
default_bad_route_1 = keep stacking similar weak neural networks
default_bad_route_2 = keep tuning tiny 2048/class local variants until one looks good
```

This memo re-ranks the next Innovation 1 SPN actions using current local
evidence plus a fresh source check.

## Source Check

Verified sources used for this memo:

- A. Gohr, "Improving Attacks on Round-Reduced Speck32/64 Using Deep Learning,"
  CRYPTO 2019, <https://eprint.iacr.org/2019/037>.
- L. Zhang and Z. Wang, "Improving Differential-Neural Distinguisher Model For
  DES, Chaskey, and PRESENT," arXiv:2204.06341,
  <https://arxiv.org/abs/2204.06341>.
- L. Zhang et al., "Neural-Inspired Advances in Integral Cryptanalysis,"
  arXiv:2505.10790, <https://arxiv.org/abs/2505.10790>.
- A. Tashkilkhanova et al., "Enhancing RX Neural Cryptanalysis: Advanced Data
  Formatting and Weighted Differential Techniques for Block Ciphers,"
  arXiv:2511.06336, <https://arxiv.org/abs/2511.06336>.

Source hygiene correction:

```text
arXiv:2505.10792 is not a multiple-ciphertext-pair block-cipher neural
distinguisher paper. It is an unrelated retrieval-augmented generation / LLM
paper. Do not cite it as cryptanalysis evidence.
```

## Current Local Evidence

The strongest current r8 branch is still the trail-position residual route.
It has local diagnostic evidence and a pending 262144/class remote retrieval
path, but no completed 262144/class score artifact bundle yet:

```text
status = pending
decision = wait_for_trail_position_262k_score_artifacts
should_run = false
source_status = pending
source_decision = wait_for_trail_position_score_artifacts
```

The best current local third-family candidate is the V16/V17 bucket residual
pool:

```text
trail-position neural expert
+ matched raw117 compressed SPN structural expert
+ bucket-conditioned residual feature expert
```

Machine-readable local gate:

```text
artifact = outputs/local_audits/i1_present_r8_bucket_residual_controls_gate.json
status = pass
decision = bucket_conditioned_residual_controls_pass_local_diagnostic
min_bucket_vs_nobucket_auc_delta = +0.0000286102294921875
min_three_vs_two_auc_delta = +0.0000057220458984375
max_shuffle_label_validation_auc = 0.5435142517089844
max_trainbucket_shuffle_three_vs_two_delta = -0.00002002716064453125
max_valbucket_shuffle_three_vs_two_delta = -0.00006961822509765625
claim_scope = local 2048/class frozen-score control gate only
```

Interpretation:

```text
keep_as_262k_migration_candidate
do_not_claim_breakthrough
do_not_launch_a_new_remote_branch_from_this_local_gate
```

## Literature-Aligned Takeaway

The source check supports a representation-first route:

```text
1. choose SPN/integral-aware evidence format
2. compress or bucket that evidence using train-only rules
3. test whether a structure-aware expert beats same-input controls
4. aggregate only after the expert is weak-positive and non-neighbor
```

It does not support a count-first route:

```text
1. train many nearby neural networks
2. average scores
3. call the resulting tiny gain an SPN architecture improvement
```

Zhang/Wang-style PRESENT work is representation- and derived-feature-heavy.
Neural-inspired integral work treats neural methods as a way to discover or
prioritize cryptanalytic structure. RX-neural data-format work points in the
same direction: data formatting and evidence allocation can matter as much as,
or more than, generic network capacity.

## Route Ranking

| Rank | Route | Decision | Reason |
|---:|---|---|---|
| 1 | Wait for 262144/class trail-position score artifacts, then run V16/V17 bucket residual planner + gate | Keep as active main branch | This is the strongest controlled local branch and now has an automatic migration gate. |
| 2 | Design a structural residual source over train-only selected SPN evidence buckets | Plan only, no remote launch yet | This extends the literature-aligned representation-first idea without changing labels or negatives. |
| 3 | Linear-combination integral residual expert | Track as a machine-readable backup route, blocked until residual-focus/source-selection are ready | Neural-inspired integral work supports searching for structure-preserving linear/integral combinations, but this must be train-only and residual-conditioned. |
| 4 | Diverse expert pool | Validator only | Useful after a real non-neighbor expert exists; not the next main experiment by itself. |
| 5 | Raw matrix Inception / cell histogram / contiguous bit projection variants already screened | Hold | Current evidence is weak, near-random, or below same-input controls. |

## Next Implementable Idea After 262k

If 262k trail-position score artifacts pass, run the already planned V16/V17
sequence. If that gate passes or is borderline-positive, the next new research
idea should not be another average of nearby experts. It should be:

```text
candidate_route = residual_bucket_axis_spectrum
selection_split = train only
validation_split = held-out validation key
inputs =
  trail-position train/validation score artifacts
  raw117 train/validation score artifacts
  compressed SPN span features
feature_family =
  per-bucket residual sensitivity spectrum over depth/word/cell groups
controls =
  shuffle labels
  shuffle train bucket values
  shuffle validation bucket values
  no-bucket same-117D expert
  same-input global/trail controls
```

The key difference from earlier weak projection attempts is that the selection
unit is not a raw feature axis or contiguous column block. It is a residual
bucket-conditioned structural spectrum:

```text
for each reliability bucket:
  summarize which SPN depth/word/cell groups explain the remaining trail+raw117
  errors
freeze summaries from train split
score held-out validation split without changing labels, negatives, or keys
```

This is a better "multiple networks" story: not many similar networks, but
specialized structural experts conditioned on a real residual regime.

## 2026-07-07 V2 Route Addition

The route summary now tracks an explicit backup route:

```text
candidate_route = linear_combo_integral_residual
status_now = blocked_by_residual_focus
selection_split = train_only
requires_source_selection = true
decision_now =
  wait_for_residual_focus_outputs_before_linear_combo_integral_expert
```

This is not a new remote launch and not a benchmark change. It is a route-level
commitment that, if residual-focus passes but Pool 3 is too near-neighbor or too
fragile, the next SPN-adaptive expert should be built from train-selected
linear/integral structural combinations rather than from more nearby trail
variants.

Design constraint:

```text
input family =
  residual-axis train summaries
  stable SPN depth/word/cell groups
  integral-style linear-combination candidates over compressed structural bits
forbidden =
  validation-selected feature search
  random-ciphertext negatives
  immediate remote launch while residual-focus pending
  weighted/fitted ensemble claims before fixed-score controls pass
```

Rationale from source-checked literature:

- Zhang/Wang PRESENT work reinforces that multiple-ciphertext/derived feature
  formatting matters for SPN neural distinguishers.
- Neural-inspired integral work supports using neural methods to identify
  useful integral or linear-combination structure, not just to add capacity.
- RX/data-format work reinforces the same practical bias: change the evidence
  format and allocation before increasing the count of near-identical models.

## Non-Actions

Do not do these next:

```text
do_not_start_new_remote_training_until_262k_status_is_ready_or_current_run_fails_cleanly
do_not_expand_2048_class_variant_search
do_not_average_more_near_neighbor_trail_global_raw117_scores
do_not_launch_linear_combo_integral_residual_before_residual_focus_gate_and_train_source_selection
do_not_cite_arxiv_2505_10792_as_neural_cryptanalysis
do_not_make_formal_or_breakthrough_claims_from_2048_or_262144_class_evidence
```

## Claim Scope

This memo is research triage only. It does not report a new model result, does
not upgrade local 2048/class evidence to remote evidence, and does not make a
formal SPN/PRESENT claim. It is meant to keep the next implementation aligned
with source-checked, representation-first neural cryptanalysis.
