# Innovation 1 PRESENT Neural Ensemble Aggregation Plan

**Date:** 2026-07-05

**Status:** training completed 3/3 rows; score export failed before ensemble evaluation

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

## Implemented Local Tooling

Status as of 2026-07-05:

```text
score artifact core = implemented
checkpoint score export CLI = implemented
frozen artifact ensemble CLI = implemented
local tiny smoke = implemented as tests only; not research evidence
remote launch = completed training; post-training score export failed on tooling dependency
```

Export per-checkpoint scores:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/export-checkpoint-scores \
  --checkpoint <checkpoint.pt> \
  --eval-plan <same_protocol_eval_plan.csv> \
  --eval-row-index 0 \
  --model-key <model_key> \
  --hidden-bits <hidden_bits> \
  --batch-size 256 \
  --device auto \
  --output-dir <score_artifact_dir>
```

Evaluate a fixed-rule frozen ensemble:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate-neural-ensemble \
  --artifacts <score_artifact_dir_a> <score_artifact_dir_b> [<score_artifact_dir_c>] \
  --output <ensemble_summary.json>
```

The evaluator validates labels, sample ids, and protocol identity before
reporting:

```text
probability_mean
logit_mean
sum_logodds
auc_positive_weighted_logit_mean
rank_average
diversity/error-overlap metrics
```

Current claim scope remains application-level score aggregation only. The tiny
smoke tests verify code paths, not SPN/PRESENT model quality. A meaningful
PRESENT ensemble result still requires aligned strong checkpoints or a lean
same-protocol model matrix, followed by the scale ladder in this document.

## Checkpoint Inventory And First Screen

Inventory result as of 2026-07-05:

```text
local retrieved PyTorch checkpoints for strong PRESENT runs = not found
remote G:\lxy PyTorch checkpoints under project/runs roots = not found
available local .pt files = smoke-only checkpoints, not PRESENT evidence
retrospective ensemble from existing strong checkpoints = blocked by missing checkpoints
```

The route therefore moves to a checkpoint-producing same-protocol screen rather
than a retrospective score export. The first planned screen is:

```text
plan = configs/experiment/innovation1/innovation1_spn_present_neural_ensemble_r7_65k_seed0.csv
rounds = 7
samples_per_class = 65536
pairs_per_sample = 16
negative_mode = encrypted_random_plaintexts
sample_structure = zhang_wang_case2_official_mcnd
difference_profile = present_zhang_wang2022_mcnd
validation_key = 0x11111111111111111111
claim_scope = checkpoint-producing ensemble diagnostic only
```

Rows:

| Row | Model | Role |
|---|---|---|
| 0 | `present_zhang_wang_keras_mcnd` | same-protocol Zhang/Wang-style baseline |
| 1 | `present_nibble_invp_only_spn_only` | strongest InvP/P-layer aligned anchor |
| 2 | `present_nibble_ddt_graph` | structural view for error-complementarity triage |

Train the checkpoint matrix:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_neural_ensemble_r7_65k_seed0.csv \
  --epochs 18 \
  --batch-size 1024 \
  --hidden-bits 32 \
  --device cuda:0 \
  --learning-rate 0.0001 \
  --optimizer adam \
  --weight-decay 0.00001 \
  --loss mse \
  --lr-scheduler official_cyclic \
  --max-learning-rate 0.002 \
  --checkpoint-metric val_auc \
  --restore-best-checkpoint \
  --checkpoint-output-dir <run_root>/checkpoints \
  --early-stopping-patience 8 \
  --early-stopping-min-delta 0.0001 \
  --train-eval-interval 0 \
  --sample-structure zhang_wang_case2_official_mcnd \
  --negative-mode encrypted_random_plaintexts \
  --key-rotation-interval 0 \
  --dataset-cache-root <run_root>/dataset_cache \
  --dataset-cache-chunk-size 8192 \
  --dataset-cache-workers 4 \
  --output <run_root>/results/train_matrix.jsonl \
  --progress-output <run_root>/logs/train_matrix_progress.jsonl
```

The `--checkpoint-output-dir` path writes one selected checkpoint per plan row,
for example:

```text
row0001_present_zhang_wang_keras_mcnd_seed0.pt
row0002_present_nibble_invp_only_spn_only_seed0.pt
row0003_present_nibble_ddt_graph_seed0.pt
```

After training, export score artifacts from each checkpoint against the same
plan row and evaluate:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/export-checkpoint-scores \
  --checkpoint <run_root>/checkpoints/row0001_present_zhang_wang_keras_mcnd_seed0.pt \
  --eval-plan configs/experiment/innovation1/innovation1_spn_present_neural_ensemble_r7_65k_seed0.csv \
  --eval-row-index 0 \
  --model-key present_zhang_wang_keras_mcnd \
  --hidden-bits 32 \
  --batch-size 1024 \
  --device cuda:0 \
  --output-dir <run_root>/score_artifacts/zhang_wang

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/export-checkpoint-scores \
  --checkpoint <run_root>/checkpoints/row0002_present_nibble_invp_only_spn_only_seed0.pt \
  --eval-plan configs/experiment/innovation1/innovation1_spn_present_neural_ensemble_r7_65k_seed0.csv \
  --eval-row-index 1 \
  --model-key present_nibble_invp_only_spn_only \
  --hidden-bits 32 \
  --batch-size 1024 \
  --device cuda:0 \
  --output-dir <run_root>/score_artifacts/invp_only

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/export-checkpoint-scores \
  --checkpoint <run_root>/checkpoints/row0003_present_nibble_ddt_graph_seed0.pt \
  --eval-plan configs/experiment/innovation1/innovation1_spn_present_neural_ensemble_r7_65k_seed0.csv \
  --eval-row-index 2 \
  --model-key present_nibble_ddt_graph \
  --hidden-bits 32 \
  --batch-size 1024 \
  --device cuda:0 \
  --output-dir <run_root>/score_artifacts/ddt_graph

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate-neural-ensemble \
  --artifacts \
    <run_root>/score_artifacts/zhang_wang \
    <run_root>/score_artifacts/invp_only \
    <run_root>/score_artifacts/ddt_graph \
  --output <run_root>/results/neural_ensemble_summary.json
```

Gate remains:

```text
best ensemble AUC >= best single AUC + 0.001
and double-fault/error overlap is not high
=> keep ensemble route and prepare 262144/class confirmation
```

If this 65536/class screen does not improve over the best single model, stop
this candidate pool and return to architecture/data-representation changes
rather than widening the ensemble mechanically.

## Prepared Remote Launch Package

Status as of 2026-07-05:

```text
remote config = configs/remote/innovation1_spn_present_neural_ensemble_r7_65k_seed0_gpu0_20260705.json
launcher = configs/remote/generated/run_i1_present_neural_ensemble_r7_65k_seed0_gpu0_20260705.cmd
monitor = configs/remote/generated/monitor_i1_present_neural_ensemble_r7_65k_seed0_gpu0_20260705.sh
run_id = i1_present_neural_ensemble_r7_65k_seed0_gpu0_20260705
remote launch = started from GitHub-pushed commit
source commit recorded remotely = 0537802e6f45fbf5a254c0a1049ba3fa9e5c928a
local monitor = ne_ens_r7_65k_g0_20260705
launcher session = ne_ens_launcher_r7_65k_g0_20260705
```

Before launch, verify:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness \
  --config configs/remote/innovation1_spn_present_neural_ensemble_r7_65k_seed0_gpu0_20260705.json
```

The generated launcher is designed to:

```text
1. create logs/results/checkpoints/score_artifacts under G:\lxy
2. run scripts\check-remote-readiness on the remote clone before training
3. train the 3-row checkpoint-producing matrix with --checkpoint-output-dir
4. verify all three selected checkpoints exist
5. export aligned frozen score artifacts for Zhang/Wang, InvP-only, and DDT graph
6. evaluate scripts\evaluate-neural-ensemble into results\neural_ensemble_summary.json
7. write done/failed markers for the local monitor
```

The generated monitor retrieves:

```text
logs/
results/
checkpoints/
score_artifacts/
```

When `train_matrix.jsonl`, all three score artifacts, and
`neural_ensemble_summary.json` are present locally, the monitor runs:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/postprocess-neural-ensemble
```

and appends a guarded `Retrieved Neural Ensemble Result` block to this plan.
The gate keeps the route only when best ensemble AUC improves over best single
AUC by at least `0.001` and pairwise error overlap is not above the configured
threshold.

## Running Status

Status as of the local artifact check on 2026-07-05 after monitor retrieval:

```text
remote readiness = pass
retrieved train rows = 3 / 3
retrieved checkpoints = 3 / 3
training status = completed remotely
score export status = failed on missing matplotlib import dependency
neural_ensemble_summary.json = not retrieved
postprocess gate = not run
claim status = no ensemble claim yet
```

Retrieved training rows:

| Row | Model | AUC | Calibrated accuracy | Best epoch | Epochs ran | Status |
|---|---|---:|---:|---:|---:|---|
| 1 | `present_zhang_wang_keras_mcnd` | `0.761419387999922` | `0.693603515625` | `6` | `14` | training complete; checkpoint retrieved |
| 2 | `present_nibble_invp_only_spn_only` | `0.7837966117076576` | `0.710235595703125` | `16` | `18` | training complete; checkpoint retrieved |
| 3 | `present_nibble_ddt_graph` | `0.7869770601391792` | `0.7142486572265625` | `14` | `18` | training complete; checkpoint retrieved |

Local artifacts:

```text
run_root = outputs/remote_results/i1_present_neural_ensemble_r7_65k_seed0_gpu0_20260705
train_matrix = results/train_matrix.jsonl
checkpoints = checkpoints/row0001_*.pt, row0002_*.pt, row0003_*.pt
score_artifacts = missing
ensemble_summary = missing
failure log = logs/i1_present_neural_ensemble_r7_65k_seed0_gpu0_20260705_export_zhang_wang_stderr.txt
```

Failure diagnosis:

```text
The first score export imported blockcipher_nd.cli.evaluate_pairset_aggregation
for two small helper functions. That import pulled blockcipher_nd.evaluation,
which eagerly imported blockcipher_nd.evaluation.plots and required matplotlib.
The remote training environment did not have matplotlib, so the post-training
export failed before any neural ensemble scores were produced.
```

Next action:

```text
Repair the lightweight import boundary for scripts/export-checkpoint-scores,
push the fix, then rerun only score export + scripts/evaluate-neural-ensemble
from the retrieved/remote checkpoints. Do not retrain for this recovery unless
checkpoint reuse fails.
```

This status is not evidence that the PRESENT neural ensemble screen has passed
or failed. It is evidence that the three candidate neural networks trained at
the planned 65536/class diagnostic scale. Final ensemble interpretation still
requires all three score artifacts, `neural_ensemble_summary.json`, and the
guarded `scripts/postprocess-neural-ensemble` gate.
