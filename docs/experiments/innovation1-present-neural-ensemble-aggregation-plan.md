# Innovation 1 PRESENT Neural Ensemble Aggregation Plan

**Date:** 2026-07-05

**Status:** design prepared / implementation not started / no remote launch

**Scope:** PRESENT-80, strict `encrypted_random_plaintexts` negatives, same
Zhang/Wang Case2 validation protocol unless a later plan explicitly narrows a
round-specific diagnostic. This is an application-level score aggregation
route, not a raw single-sample SOTA route.

## Research Question

The current strongest Innovation 1 SPN route is:

```text
route = present_nibble_invp_only_spn_only
evidence = two_seed_1000000/class positive with attribution control
seed0 AUC = 0.797470988906
seed1 AUC = 0.797347588554
max control AUC = 0.793621524954
allowed claim = InvP/P-layer aligned SPN view has stable positive evidence
```

At the same time, several later r8/r9 screens either stopped or became
diagnostic-only:

```text
r8 pair-set 1M = stop/rethink pair-set scale
r9 weak-probe 262144/class = near random; stop from-scratch r9/r10 branch
r9 curriculum 262144/class = near random; stop r9 curriculum branch
r9 difference screen 65536/class = all candidates near random; stop difference screen
r8 projection screen 65536/class = weak projection hold; no ensemble-confirmed result
```

The open question is therefore not "can many weak rows be averaged until a
number improves?" The useful question is:

```text
Do independently trained SPN-aware neural views make different validation
errors under the same strict PRESENT protocol, so that fixed score-level
aggregation improves over the best single model?
```

If yes, this becomes an application-level weak-signal amplification route. If
no, it argues that current positive evidence is concentrated in the InvP/P-layer
aligned view and should be improved architecturally rather than by ensembling.

## Non-Goals

This plan does not claim:

```text
raw single-sample SOTA
PRESENT r9 breakthrough
formal multi-seed publication evidence
new SPN architecture evidence by itself
```

It also does not change:

```text
validation labels
negative sample definition
sample structure
metric computation
difference profile
validation key
plan/result alignment rules
```

## Existing Code To Reuse

The repository already has two related but narrower aggregation tools:

```text
src/blockcipher_nd/evaluation/pairset_aggregation.py
scripts/evaluate-pairset-aggregation
```

These aggregate many pair logits from one frozen single-pair scorer and test
whether learned pair-set models beat independent score aggregation.

```text
src/blockcipher_nd/cli/evaluate_projection_ensemble.py
scripts/evaluate-projection-ensemble
```

This trains a small projection-feature matrix and evaluates same-validation
projection ensembles with:

```text
probability_mean
logit_mean
auc_weighted_logit_mean
probability/logit correlation
disagreement rate
double-fault rate
error Jaccard
```

The new route should reuse those metric ideas, but it needs a more general
artifact path for already trained neural models and checkpoints.

## Candidate Approaches

### Approach A: Train-Time Multi-Head Ensemble

Train several heads in one model and aggregate inside the model.

Trade-off:

```text
Pros: one run, easy artifact management.
Cons: mixes architecture change with aggregation; hard to know whether gains
come from shared representation, head diversity, or aggregation.
```

Decision:

```text
Do not use as the first step.
```

### Approach B: Same-Plan Multi-Model Matrix Ensemble

Train a lean matrix of 3-4 models in one protocol, export same-validation
logits, and ensemble them after training.

Trade-off:

```text
Pros: controlled, reproducible, easy to compare to best single model.
Cons: may spend GPU time retraining models for scores that already exist in
summaries but not as per-sample artifacts.
```

Decision:

```text
Use for the first implementable smoke/medium route after the score artifact
format exists.
```

### Approach C: Retrospective Checkpoint Score Ensemble

Load selected checkpoints from already completed strong runs, regenerate the
same deterministic validation split, export logits/probabilities, and ensemble.

Trade-off:

```text
Pros: directly answers whether existing strong routes have complementary
errors; minimal new training.
Cons: requires checkpoints to be present/retrieved and exact validation split
identity to be reproducible. Some older runs may only have summary JSONL.
```

Decision:

```text
Use as the preferred research direction after an inventory confirms which
checkpoints are available locally or retrievable from run artifacts.
```

## Recommended Design

Use a two-stage route:

```text
Stage 1 = score artifact infrastructure and local smoke
Stage 2 = medium or paper-scale ensemble evaluation using available strong
          checkpoints or a tightly controlled same-plan multi-model run
```

Stage 1 should not launch remote training. It should define and test a stable
artifact schema:

```text
labels.npy
probabilities.npy
logits.npy
sample_ids.npy or deterministic split metadata
models.json
summary.json
```

The artifact identity must include:

```text
cipher
rounds
seed
samples_per_class
validation_samples_per_class
pairs_per_sample
feature_encoding
negative_mode
sample_structure
difference_profile
difference_member
train_key
validation_key
checkpoint_metric
restore_best_checkpoint
model_key
model_options
run_id
checkpoint_path
git_commit
```

Stage 2 should evaluate only fixed, predeclared score rules first:

```text
best_single
probability_mean
logit_mean
sum_logodds
auc_positive_weighted_logit_mean
rank_average
```

Logistic stacking is allowed only after a separate calibration split exists.
It must not fit weights on the same validation labels used for final reporting.

## Initial Candidate Pool

Do not start from the near-random r9 difference screen. First candidate pool:

| Role | Candidate | Reason |
|---|---|---|
| Strong anchor | `present_nibble_invp_only_spn_only` | strongest route with two-seed 1M/class evidence and attribution control |
| Same-protocol baseline | `present_zhang_wang_keras_mcnd` | local Zhang/Wang-style anchor and reference comparator |
| Pair evidence candidate | `present_nibble_invp_pair_consistency_spn_only` or pair mixer | tests whether pair evidence errors complement InvP-only |
| Structural weak view | DDT graph, topology-aware, active auxiliary, or transition-spectrum row | only include if same protocol and non-near-random under the chosen round/scale |

Keep the first non-smoke ensemble matrix to 3-4 models. Larger pools are
reserved for a later attribution study.

## Scale Ladder

```text
local smoke = tiny samples; proves artifact and metric path only
65536/class = screen/diagnostic; useful for weak complementarity triage
262144/class = medium diagnostic; minimum for route decision
1000000/class multi-seed = formal-style support, still application-level
```

Do not call a 65536/class ensemble a formal result. Do not call a 262144/class
single-seed ensemble a breakthrough.

## Gate

Primary metric:

```text
val_auc
```

Supporting metrics:

```text
accuracy
calibrated_accuracy
val_loss or log_loss if available
probability/logit correlation
disagreement rate
double-fault rate
error Jaccard
oracle accuracy at threshold 0.5
```

Decision table:

| Result | Decision |
|---|---|
| best ensemble AUC >= best single AUC + 0.001 and double-fault/error overlap is not high | keep neural ensemble aggregation route; prepare medium confirmation |
| best ensemble AUC improves < 0.001 but calibrated accuracy improves | treat as calibration/threshold effect; no route claim |
| best ensemble AUC <= best single AUC | stop ensemble as a main improvement route for this pool |
| ensemble improves only when fitted on the final validation labels | invalid; split leakage |
| labels/sample order differ across score artifacts | invalid; fix artifact identity before metrics |

## Claim Scope

Allowed if gate passes:

```text
Under the fixed PRESENT protocol, predeclared score-level aggregation of
complementary SPN-aware neural views improves over the best single model at the
tested diagnostic scale.
```

Not allowed:

```text
The ensemble is a new raw neural distinguisher SOTA.
The ensemble proves a single architecture is better.
The ensemble establishes PRESENT r9 success.
The ensemble result is formal without 1000000/class multi-seed confirmation.
```

## Required Implementation Plan Before Launch

Before any non-smoke remote launch, implement and verify:

```text
1. Generic probability/logit artifact writer for matrix rows or checkpoints.
2. Generic ensemble evaluator that consumes frozen artifacts only.
3. Artifact alignment validation for labels, sample ids, and protocol fields.
4. Diversity/error-overlap report.
5. Postprocess gate and docs update hook.
6. Local smoke using tiny PRESENT/SPECK rows.
7. Remote readiness checks for G:\lxy artifact paths if medium+ training is needed.
```

## Current Next Action

```text
Do not launch remote.
Do not add a large model matrix yet.
Next implementation step: build the generic frozen-score ensemble artifact
path and smoke it locally, then decide whether existing checkpoints are
available for retrospective evaluation or whether a lean same-plan 262144/class
matrix is needed.
```

