# Innovation 1 PRESENT Diverse Expert Pool Plan

**Date:** 2026-07-05

**Status:** family-readiness gate implemented / secondary validator / no remote launch

**Scope:** PRESENT/SPN neural distinguishers under strict
`encrypted_random_plaintexts` negatives. This plan extends the frozen-score
neural ensemble route from a near-neighbor diagnostic into a diversity-aware
expert selection route. It is application-level evidence only unless later
plans add paper-scale multi-seed confirmation.

## Motivation

The first recovered neural ensemble screen trained and evaluated:

```text
run_id = i1_present_neural_ensemble_r7_65k_seed0_gpu0_20260705
rounds = 7
samples_per_class = 65536
pairs_per_sample = 16
sample_structure = zhang_wang_case2_official_mcnd
negative_mode = encrypted_random_plaintexts
```

It produced:

| Field | Value |
|---|---:|
| Best single | `present_nibble_ddt_graph` |
| Best single AUC | `0.789112608414` |
| Best ensemble | `probability_mean` |
| Best ensemble AUC | `0.790061685257` |
| Delta vs best single AUC | `0.000949076843` |
| Gate margin | `0.001000000000` |
| Decision | `weak_neural_ensemble_positive_below_gate` |

That result is useful but narrow. It answers whether a small pool of related
SPN-aware views has mild score-level complementarity. It does **not** answer
whether genuinely different neural representation families can combine into a
stronger distinguisher.

The next question is therefore:

```text
Can a diversity-constrained pool of different PRESENT/SPN expert families
improve over the strongest single expert and over a near-neighbor ensemble?
```

## Non-Goals

This plan does not:

```text
change validation labels
change negative sample definition
change metric computation
fit ensemble weights on final validation labels
claim raw single-sample SOTA
claim PRESENT r8/r9 breakthrough
claim formal evidence from 65536/class or 262144/class single-seed screens
```

It also does not mechanically widen the previous r7 three-model pool. More
models are useful only if they add representation-level diversity and pass
quality gates.

## Expert Families

Each candidate score artifact should be assigned an `expert_family`. The family
is a research label used for diversity selection, not a benchmark change.

| Family | Representative models or routes | Intended difference |
|---|---|---|
| `raw_mcnd` | `present_zhang_wang_keras_mcnd` | raw ciphertext pair bit convolution / Zhang-Wang style |
| `invp_cell` | `present_nibble_invp_only_spn_only` | InvP(DeltaC) organized as PRESENT 4-bit cells |
| `ddt_graph` | `present_nibble_ddt_graph` | InvP cells with S-box DDT/P-layer graph priors |
| `p_layer_graph` | `present_nibble_invp_p_layer_graph_spn_only` | fixed public P-layer message passing |
| `pair_evidence` | `present_nibble_invp_pair_consistency_spn_only`, pair mixer variants | multi-pair weak evidence pooling |
| `inverse_round_matrix` | `present_matrix_trail_hybrid_pairset_invp_sinv` | inverse-round / structural inverse-S matrix representation |
| `projection_feature` | r8 projection/truncated feature rows | low-dimensional weak projection evidence |

Near-neighbor SPN graph variants may still be included as controls, but a
claimed diverse pool must contain at least one non-neighbor family beyond
`invp_cell` / `ddt_graph` / `p_layer_graph`.

## Approaches Considered

### Approach A: Keep Adding Similar SPN Graph Models

```text
raw MCND + InvP-only + DDT graph + P-layer graph + S-box prior gate
```

Trade-off:

```text
Pros: easy to run with existing code.
Cons: likely high error correlation; repeats the near-neighbor ensemble pattern.
```

Decision:

```text
Use only as a control, not as the main diverse-expert route.
```

### Approach B: Train One Large Multi-Branch Super-Model

Train raw, InvP, DDT, pair-evidence, and inverse-round branches jointly.

Trade-off:

```text
Pros: may learn useful cross-family fusion.
Cons: mixes architecture, representation, training dynamics, and aggregation;
hard to attribute gains; higher implementation and GPU risk.
```

Decision:

```text
Defer until frozen-score diversity evidence says which families matter.
```

### Approach C: Frozen Diverse Expert Pool

Train or reuse separately defined experts, export aligned frozen logits, then
select and aggregate experts under explicit diversity constraints.

Trade-off:

```text
Pros: controlled, explainable, reuses existing score artifact path, separates
quality from diversity, and avoids final-validation fitted weights.
Cons: needs score artifacts/checkpoints for each family; cross-round pools need
careful scope labels and should not be mixed into raw r7 claims.
```

Decision:

```text
Use this approach first.
```

## Design

The diverse expert pool adds a selection layer on top of existing frozen score
artifacts:

```text
score artifacts -> family/diversity manifest -> fixed diversity report
                -> predeclared subset selection -> fixed-rule ensemble
                -> gate and plan update
```

The score artifact schema already includes model metadata. The diverse route
adds a sidecar manifest or metadata fields:

```text
expert_family
expert_scope
protocol_group
source_run_id
candidate_status
```

Recommended meanings:

| Field | Meaning |
|---|---|
| `expert_family` | one of the families listed above |
| `expert_scope` | `same_protocol_r7`, `r8_high_round_screen`, `r9_high_round_screen`, etc. |
| `protocol_group` | artifacts that may be directly ensembled because labels/sample IDs/protocol match |
| `candidate_status` | `strong_anchor`, `weak_positive`, `near_neighbor_control`, `rejected`, `pending` |

Only artifacts with identical labels, sample IDs, and protocol identity may be
directly combined. Cross-round and cross-sample-structure outputs are not raw
score ensembles; they can only be compared as route evidence or evaluated in a
separate application-level setting with explicitly matched data.

## Candidate Pools

### Pool 0: Recovered r7 Near-Neighbor Control

Purpose:

```text
Document the current limit of similar-model aggregation.
```

Rows:

| Family | Artifact |
|---|---|
| `raw_mcnd` | recovered `zhang_wang` score artifact |
| `invp_cell` | recovered `invp_only` score artifact |
| `ddt_graph` | recovered `ddt_graph` score artifact |

Expected status:

```text
weak_neural_ensemble_positive_below_gate
```

This pool is not the diverse route; it is the reference control.

### Pool 1: r7 Diverse Retrospective Pool

Launch only if compatible checkpoints/artifacts exist or can be recovered
without retraining. Candidate families:

```text
raw_mcnd
invp_cell
ddt_graph
p_layer_graph or sbox_prior_gate
pair_evidence if same r7 protocol artifact exists
```

This can test whether a fourth family reduces error overlap enough to pass the
diverse gate.

### Pool 2: High-Round Diverse Screen Pool

Use after relevant r8/r9 screens complete and only within matching protocol
groups. Candidate families:

```text
raw/high-round anchor
pair_evidence
inverse_round_matrix
projection_feature if weak-positive and low-correlation
```

This pool is mainly for route selection at r8/r9. It must not be reported as a
PRESENT r7 same-protocol improvement.

### Pool 3: r8 Residual-Guided Diverse Expert Pool

Use only after the active PRESENT r8 residual-focus 262144/class gate completes
and passes. This pool is the first concrete path from the residual-focused
structure work back into the user's multi-network idea.

Candidate families:

```text
trail_position_anchor =
  PRESENT r8 trail-position neural frozen score

compressed_span_structural =
  matched raw117 compressed SPN structural expert

residual_focus_aux_word =
  residual-focused aux_depth_word_ + aux_word_ correction expert,
  promoted only if scripts/gate-residual-focus-262k keeps focus05 or focus10

near_neighbor_controls =
  global trail-position control,
  uniform no-focus residual correction,
  focus10 label-shuffle residual correction
```

Promotion requirements:

```text
residual_focus_gate_status = pass
decision = keep_residual_focus_262k_hard_slice_candidate
passing_candidates contains focus05 or focus10
label_shuffle_control worsens hard-slice residual loss
uniform_no_focus does not match the kept focus candidate
all artifacts share the same PRESENT r8 protocol group and sample ids
```

If those requirements pass, the next ensemble test should still be a fixed
frozen-score diagnostic, not a learned final-validation stack:

```text
best_single
trail_position + raw117 fixed logit/probability fusion
trail_position + raw117 + residual_focus fixed fusion
trail_position + raw117 + uniform residual control
trail_position + raw117 + label-shuffle residual control
```

The expected claim scope is:

```text
application-level medium diagnostic evidence for residual-guided diverse expert
combination
```

It is not:

```text
raw single-sample SOTA
PRESENT r8 solved
formal SPN/PRESENT evidence
```

If the residual-focus 262144/class gate fails, do not run this pool. The next
action should return to residual-source repair or a different SPN-aware feature
family, not broader aggregation.

## Selection Rule

Start from all valid artifacts in one `protocol_group`. Compute per-model
quality and pairwise diversity:

```text
auc
calibrated_accuracy
probability_correlation
logit_correlation
disagreement_rate_at_0_5
double_fault_rate_at_0_5
error_jaccard_at_0_5
oracle_accuracy_at_0_5
```

A candidate can enter the diverse subset only if:

```text
auc > 0.5
and candidate_status != rejected
and protocol identity aligns with the pool
```

For a pool to be called diverse, require:

```text
family_count >= 4
and at least one expert_family outside {raw_mcnd, invp_cell, ddt_graph, p_layer_graph}
and at least one non-neighbor expert has error_jaccard_at_0_5 <= 0.65
  against the best single or current InvP/DDT anchor
```

If fewer than four families are available, report it as:

```text
near-neighbor or partial-diversity diagnostic
```

not as a diverse expert pool.

## Ensemble Rules

First pass remains fixed and unfitted:

```text
probability_mean
logit_mean
sum_logodds
auc_positive_weighted_logit_mean
rank_average
```

No logistic stacking or learned weights are allowed until a separate calibration
split exists. Validation-label-fitted weights are invalid for route claims.

## Gates

### Near-Neighbor Control Gate

Use the old neural ensemble threshold:

```text
best ensemble AUC >= best single AUC + 0.001
and max error_jaccard_at_0_5 <= 0.85
```

This gate can keep a candidate pool, but it does not prove diverse expert
benefit.

### Diverse Expert Gate

For a pool with `family_count >= 4`:

```text
best ensemble AUC >= best single AUC + 0.0015
and best ensemble AUC >= near-neighbor control best ensemble AUC + 0.0005
and max error_jaccard_at_0_5 <= 0.80
and at least one non-neighbor expert has error_jaccard_at_0_5 <= 0.65
```

Decision table:

| Result | Decision |
|---|---|
| Passes diverse gate | `support_diverse_expert_pool_route` |
| Positive AUC but fails diversity requirement | `near_neighbor_or_correlated_positive_only` |
| Diversity exists but ensemble does not beat best single | `diversity_not_useful_for_current_pool` |
| Only 2-3 nearby families available | `partial_diversity_diagnostic_only` |
| Protocol/sample alignment fails | `invalid_diverse_pool_alignment` |

## Implementation Plan Sketch

Keep implementation small and inspectable:

1. Add optional expert-family metadata support to the score-artifact export path
   or a sidecar manifest reader.
2. Add a diversity-aware postprocess module that consumes existing
   `neural_ensemble_summary.json` plus artifact metadata.
3. Add a CLI such as `scripts/postprocess-diverse-expert-pool`.
4. Add tests with synthetic score artifacts covering:
   - near-neighbor pool rejected as not diverse,
   - diverse pool passing the family and error-overlap constraints,
   - protocol mismatch rejection.
5. Run the recovered r7 Pool 0 as a baseline control.
6. Do not launch remote training until a specific missing family is chosen and
   a lean 2-4 row plan is written.

## 2026-07-06 Family-Readiness Gate Implementation

Implemented:

```text
score artifact metadata:
  scripts/export-checkpoint-scores --expert-family <family> --candidate-status <status>

gate:
  blockcipher_nd.evaluation.neural_ensemble.assess_diverse_expert_pool

summary field:
  neural_ensemble_summary.json["diverse_expert_pool"]

postprocess behavior:
  if ensemble AUC improves but diverse_expert_pool.status == fail,
  decision = keep_near_neighbor_ensemble_control_not_diverse_pool
```

The readiness gate requires:

```text
eligible family count >= 3
at least one non-neighbor family outside invp_cell/ddt_graph/p_layer_graph
non-neighbor pairwise error_jaccard_at_0_5 <= 0.75
each eligible expert AUC >= 0.52
candidate_status != rejected
```

This turns the user's "multiple different neural networks" idea into a
checkable condition. The current recovered r7 pool can remain useful as Pool 0,
but it should be labeled as a near-neighbor control unless its artifacts include
a genuine non-neighbor family with acceptable error overlap.

Suggested family labels for future score exports:

| Route | `expert_family` |
|---|---|
| Zhang/Wang raw MCND | `raw_mcnd` |
| InvP/P-layer aligned anchor | `invp_cell` |
| DDT graph | `ddt_graph` |
| P-layer graph | `p_layer_graph` |
| pair evidence pooling | `pair_evidence` |
| inverse-round/integral matrix | `inverse_round_matrix` |
| r8 trail-position residual | `trail_position` |
| projection/truncated weak feature | `projection_feature` |

The r8 trail-position route is therefore a possible future non-neighbor expert
only after a same-protocol score artifact exists and the residual/deterministic
controls still pass at that evidence scale.

## Literature-Informed Priority Update

A 2026-07-05 literature refresh changed the priority order. Current external
and local evidence points to SPN-aware feature/input representation as the
better first-order improvement route:

```text
SPN feature/input search > new SPN architecture variant > diverse ensemble
```

See:

```text
docs/research/innovation1-spn-adaptation-literature-refresh-20260705.md
```

Therefore this diverse expert pool should be treated as a **secondary
validator**. It should not consume the next remote slot unless a non-neighbor
expert family, such as inverse-round/integral matrix features or pair-evidence
pooling, has already produced compatible weak-positive score artifacts.

## Immediate Next Action

Do not start a new remote run from this plan yet.

As of 2026-07-07, the active mainline action is the remote
`i1_present_r8_residual_focus_262k_retry1` run documented in:

```text
docs/experiments/innovation1-present-r8-residual-bucket-axis-spectrum-plan.md
```

Next concrete step for this plan, only when the residual-focus 262144/class
gate finishes:

```text
If scripts/gate-residual-focus-262k keeps focus05 or focus10, instantiate Pool
3 as a same-protocol fixed-fusion diagnostic. If the gate fails, do not broaden
the ensemble; repair the residual source first.
```

The local Pool 3 fixed-fusion diagnostic is now wired as:

```text
scripts/evaluate-residual-guided-diverse-pool
```

It is deliberately narrow. It reads the Pool 3 plan and aligned frozen
validation score artifacts, then compares:

```text
trail_position + raw117
trail_position + raw117 + selected residual_focus
trail_position + raw117 + uniform residual control
trail_position + raw117 + label-shuffle residual control
```

The tool blocks when the Pool 3 plan is pending or held, and it reports missing
score artifacts instead of trying to generate or retrieve them. It does not fit
ensemble weights, contact the remote host, launch training, or upgrade the
claim beyond application-level medium diagnostic evidence.
If the residual-focus fusion fails to strictly beat the base/control comparisons
for a seed, the evaluator returns `hold`, and the upper-level residual-focus
postprocess reports `repair_residual_guided_pool3_before_scaleup` rather than
promoting the pool.

`scripts/advance-residual-focus-results` and
`scripts/watch-residual-focus-results` now call this fixed-fusion evaluator
automatically after the residual-focus gate passes, the Pool 3 plan is ready,
and all per-seed validation score artifacts are present. If the artifacts are
not present yet, the route reports `wait_for_pool3_score_artifacts` and remains
non-terminal so the local watcher can keep waiting for later retrieval/sync
cycles rather than pretending an ensemble result exists.

The recovered r7 Pool 0 remains the near-neighbor control for historical
context, but it is no longer the next mainline action. The next aggregation work
should be residual-guided and same-protocol, not a wider pile of similar SPN
graph variants.

## Claim Scope

Allowed after local Pool 0 postprocess:

```text
The recovered r7 neural ensemble is a near-neighbor control with weak positive
aggregation below gate.
```

Allowed only after a future diverse gate passes:

```text
Under a fixed protocol group, frozen score aggregation of multiple
representation-level expert families improves over both the best single expert
and the near-neighbor control at the tested diagnostic scale.
```

Not allowed:

```text
raw neural distinguisher SOTA
PRESENT higher-round breakthrough
formal evidence without 1000000/class multi-seed confirmation
cross-protocol ensemble claims without matched labels/sample IDs
```
