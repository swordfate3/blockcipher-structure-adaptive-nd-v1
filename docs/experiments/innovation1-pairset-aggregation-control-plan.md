# Innovation 1 Pair-Set Aggregation Control Plan

**Date:** 2026-06-30

**Status:** deferred attribution control / assets prepared / do not launch
while topology-aware network seed0 is active

**Scope:** PRESENT-80 r7, Zhang/Wang 2022 Case2 `m=16`, strict encrypted-random-plaintext negatives.

## Research Question

Does a learned SPN pair-set model extract useful cross-pair structure, or is the
observed multi-pair signal mostly explained by independent single-pair scores
combined with a simple aggregation rule?

This is a required control before claiming that pair-set evidence pooling is a
structure-adaptive data/model contribution.

## Why This Exists

The current strongest supported Innovation 1 route is:

```text
present_nibble_invp_only_spn_only
evidence = two-seed 1000000/class positive confirmation
attribution = paper-scale controls support true InvP/P-layer alignment
```

The project also contains a joint pair-set candidate:

```text
present_nibble_invp_pair_consistency_spn_only
```

That model pools 16 pair embeddings with mean/max/std and learned evidence
pooling. Medium-scale results previously showed only a small advantage over
InvP-only. Before expanding this direction, we need to separate two effects:

```text
Effect A: more independent pair evidence improves the distinguisher.
Effect B: the neural model learns non-trivial consistency across the 16 pairs.
```

Only Effect B supports a pair-set structure innovation claim.

## Trigger

Do not launch this run while:

```text
i1_spn_topology_aware_network_r7_262k_seed0_gpu0_20260701
```

is running. The DDT graph branch that originally blocked this plan has already
completed both seed0 and seed1 as weak diagnostic evidence, so it no longer
blocks pair-set attribution. This plan becomes actionable after one of these
conditions:

```text
1. The topology-aware network branch is retrieved, validated, plan-aligned,
   and its gate or route decision asks for pair-set attribution rather than
   candidate-trail / transition consistency.

2. A future positive pair-set route needs this frozen single-pair aggregation
   control before any cross-pair structure claim.

3. User explicitly chooses pair-set data representation as the next route.
```

## Single Hypothesis

Holding cipher, rounds, sample structure, negative mode, train/validation keys,
optimizer, scheduler, checkpoint metric, and scale fixed:

```text
Learned pair-set consistency should beat frozen single-pair score aggregation
by at least +0.001 AUC if it captures real cross-pair structure.
```

If it does not beat frozen aggregation, pair-set pooling should be treated as
statistical aggregation or diagnostic-only, not as a primary structural
innovation.

## Required Implementation

This plan needs one small evaluation route before launch:

```text
frozen_single_pair_invp_logodds_aggregate
```

Expected behavior:

```text
1. Train or load a single-pair InvP-only scorer with pairs_per_sample = 1.
2. For a Zhang/Wang Case2 m=16 sample, slice the 2048-bit input into 16
   independent 128-bit ciphertext-pair views.
3. Apply the frozen single-pair scorer to each pair.
4. Aggregate logits or calibrated log-odds with fixed rules:
   - mean logit
   - sum log-odds
   - top-k mean or top-k logsumexp
5. Report the best predeclared aggregation rule on validation only.
6. Evaluate the chosen rule on the same held-out validation/test protocol as
   the joint pair-set model.
```

The frozen aggregator must not update weights on 16-pair samples. Otherwise it
becomes another learned pair-set model and no longer controls for independent
score aggregation.

## Implementation Status

Completed local foundation:

```text
module = src/blockcipher_nd/evaluation/pairset_aggregation.py
tests = tests/test_pairset_aggregation.py
capability = split a 16-pair sample into pair views, apply a frozen scorer to
             each pair, aggregate logits/log-odds, and compute binary metrics

module = src/blockcipher_nd/training/trainer.py
tests = tests/test_training_metrics.py
capability = opt-in --checkpoint-output support for saving the selected
             PyTorch checkpoint payload after training

script = scripts/evaluate-pairset-aggregation
tests = tests/test_pairset_aggregation_cli.py
capability = load a saved frozen single-pair scorer, generate pair-set
             evaluation data from a plan row, and write aggregation metrics

script = scripts/gate-pairset-aggregation
tests = tests/test_pairset_aggregation_gate.py
capability = compare learned pair-set AUC against frozen single-pair
             aggregation and the InvP anchor, then emit continue/weak/stop gate

script = scripts/postprocess-pairset-aggregation
tests = tests/test_pairset_aggregation_postprocess.py
capability = validate learned result alignment, plot learned curves, run the
             pair-set aggregation gate, write summaries, and update plan docs

config = configs/experiment/innovation1/innovation1_spn_present_pairset_aggregation_control_single_pair_smoke.csv
capability = train a tiny single-pair InvP-only scorer checkpoint for frozen
             aggregation smoke; pairs_per_sample = 1; not accuracy evidence

config = configs/experiment/innovation1/innovation1_spn_present_pairset_aggregation_control_smoke.csv
capability = train tiny 16-pair InvP anchor and learned pair-consistency rows
             for postprocess/gate smoke; not accuracy evidence

config = configs/experiment/innovation1/innovation1_spn_present_pairset_aggregation_control_single_pair_r7_262k.csv
capability = MEDIUM 262144/class stage-A single-pair scorer checkpoint row;
             not formal reproduction or breakthrough evidence

config = configs/experiment/innovation1/innovation1_spn_present_pairset_aggregation_control_r7_262k.csv
capability = MEDIUM 262144/class stage-B InvP anchor plus learned pair-set
             consistency rows; not formal reproduction or breakthrough evidence

remote_config = configs/remote/innovation1_spn_present_pairset_aggregation_control_single_pair_r7_262k_gpu1_20260630.json
capability = readiness-checked stage-A metadata for single-pair checkpoint run

remote_config = configs/remote/innovation1_spn_present_pairset_aggregation_control_r7_262k_gpu1_20260630.json
capability = readiness-checked stage-B metadata for learned pair-set plus
             frozen aggregation gate run

launcher = configs/remote/generated/run_i1_pairset_aggregation_control_r7_262k_seed0_gpu1_20260630.cmd
capability = stage-aware Windows launcher: stage A checkpoint training, stage B
             learned/anchor training, frozen aggregation summary generation

monitor = configs/remote/generated/monitor_i1_pairset_aggregation_control_r7_262k_seed0_gpu1_20260630.sh
capability = local tmux watcher script: pull logs/results/checkpoints and run
             postprocess-pairset-aggregation when all artifacts are ready
```

Therefore this plan is local-smoke-verified and remote-launch-asset-prepared,
but still not launched. The current code and configs make the core aggregation
math, scorer artifact persistence, frozen aggregation CLI, learned-pairset
smoke matrix, 262144/class staged plan rows, stage-aware launcher/watcher, and
gate/postprocess decision path testable and reusable. It must still wait for
the current topology-aware network run to be retrieved and gated, or for the
user to explicitly choose pair-set attribution over the candidate-trail data
representation route.

## Local Smoke Readiness

Purpose:

```text
Prove the pair-set attribution-control plumbing works end to end:
single-pair checkpoint -> frozen aggregation summary -> learned pair-set
results -> postprocess/gate artifacts.
```

Smoke commands:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_pairset_aggregation_control_single_pair_smoke.csv \
  --epochs 1 \
  --batch-size 4 \
  --hidden-bits 8 \
  --device cpu \
  --checkpoint-output outputs/smoke/pairset_aggregation_control/single_pair_invp.pt \
  --output outputs/smoke/pairset_aggregation_control/single_pair_results.jsonl \
  --progress-output outputs/smoke/pairset_aggregation_control/single_pair_progress.jsonl

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_pairset_aggregation_control_smoke.csv \
  --epochs 1 \
  --batch-size 4 \
  --hidden-bits 8 \
  --device cpu \
  --output outputs/smoke/pairset_aggregation_control/learned_pairset_results.jsonl \
  --progress-output outputs/smoke/pairset_aggregation_control/learned_pairset_progress.jsonl

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate-pairset-aggregation \
  --checkpoint outputs/smoke/pairset_aggregation_control/single_pair_invp.pt \
  --eval-plan configs/experiment/innovation1/innovation1_spn_present_pairset_aggregation_control_smoke.csv \
  --eval-row-index 0 \
  --samples-per-class 8 \
  --pairs-per-sample 16 \
  --scorer-model-key present_nibble_invp_only_spn_only \
  --scorer-hidden-bits 8 \
  --scorer-model-options '{"spn_mixer_depth":1,"activation":"relu","norm":"layernorm"}' \
  --scorer-pairs-per-sample 1 \
  --aggregation-mode sum_logodds \
  --batch-size 4 \
  --device cpu \
  --output outputs/smoke/pairset_aggregation_control/frozen_aggregation_summary.json

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/postprocess-pairset-aggregation \
  --plan configs/experiment/innovation1/innovation1_spn_present_pairset_aggregation_control_smoke.csv \
  --learned-results outputs/smoke/pairset_aggregation_control/learned_pairset_results.jsonl \
  --frozen-summary outputs/smoke/pairset_aggregation_control/frozen_aggregation_summary.json \
  --output-dir outputs/smoke/pairset_aggregation_control/postprocess \
  --run-id pairset_aggregation_control_smoke \
  --expected-rows 2
```

Expected status:

```text
local smoke may pass or fail the research gate depending on random tiny metrics;
only CLI execution, protocol alignment, artifact creation, and gate plumbing are
meaningful at this scale.
```

Completed local smoke:

```text
date = 2026-06-30
status = pass
single_pair_checkpoint = outputs/smoke/pairset_aggregation_control/single_pair_invp.pt
learned_rows = 2/2
frozen_summary_status = pass
postprocess_status = pass
validation_status = pass
pairset_aggregation_gate_status = pass
generated_artifacts = curves.svg, history.csv, gate.json, summary.json, summary.md
claim_scope = plumbing/readiness only; tiny metric decision is ignored
```

## Candidate Matrix

First non-smoke scale:

```text
262144/class
seed = 0
expected_rows = 1 stage-A scorer row + 2 stage-B learned/anchor rows
claim_scope = medium diagnostic only
```

Lean matrix:

| Row | Route | Role |
|---:|---|---|
| 0 | `present_nibble_invp_only_spn_only` | current supported InvP 16-pair anchor |
| 1 | `present_nibble_invp_pair_consistency_spn_only` | learned joint pair-set candidate |
| 2 | `frozen_single_pair_invp_logodds_aggregate` | independent score aggregation control |
| 3 | optional `frozen_single_pair_delta_only_logodds_aggregate` | checks whether aggregation alone works without InvP |

Do not include unrelated graph/DDT models in this matrix. This plan asks only
whether pair-set learning beats independent score aggregation.

Prepared 262144/class configs:

```text
stage_a_plan = configs/experiment/innovation1/innovation1_spn_present_pairset_aggregation_control_single_pair_r7_262k.csv
stage_a_remote_config = configs/remote/innovation1_spn_present_pairset_aggregation_control_single_pair_r7_262k_gpu1_20260630.json
stage_a_expected_rows = 1
stage_a_output = single-pair InvP checkpoint for frozen aggregation

stage_b_plan = configs/experiment/innovation1/innovation1_spn_present_pairset_aggregation_control_r7_262k.csv
stage_b_remote_config = configs/remote/innovation1_spn_present_pairset_aggregation_control_r7_262k_gpu1_20260630.json
stage_b_expected_rows = 2
stage_b_output = InvP 16-pair anchor plus learned pair-consistency result JSONL
```

Launch status:

```text
not launched
blocked_by_policy = current topology-aware network seed0 still active
stage_aware_launcher = configs/remote/generated/run_i1_pairset_aggregation_control_r7_262k_seed0_gpu1_20260630.cmd
stage_aware_monitor = configs/remote/generated/monitor_i1_pairset_aggregation_control_r7_262k_seed0_gpu1_20260630.sh
remaining_requirement = wait for topology-aware result gate, or explicit user
                        choice to prioritize pair-set attribution
windows_launcher_note = scorer model options intentionally use default spn_mixer_depth=2, activation=relu, norm=layernorm to avoid fragile cmd.exe JSON quoting
```

Readiness stage lock:

```text
scripts/check-remote-readiness enforces pairset_aggregation_stage_lock:

stage A:
  pairset_stage = single_pair_scorer_checkpoint
  checkpoint_output under G:\lxy\blockcipher-structure-adaptive-nd-runs

stage B:
  pairset_stage = learned_pairset_plus_frozen_aggregation_gate
  requires_checkpoint under G:\lxy\blockcipher-structure-adaptive-nd-runs
  frozen_aggregation_output under G:\lxy\blockcipher-structure-adaptive-nd-runs
```

Bounded local health check after launch:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/monitor-health \
  --run-id i1_pairset_aggregation_control_r7_262k_seed0_gpu1_20260630 \
  --plan configs/experiment/innovation1/innovation1_spn_present_pairset_aggregation_control_r7_262k.csv \
  --plan-doc docs/experiments/innovation1-pairset-aggregation-control-plan.md \
  --expected-rows 2 \
  --postprocess-kind pairset_aggregation \
  --output /tmp/i1_pairset_monitor_health_latest.json
```

This command is for bounded local status checks only. It must not replace the
tmux watcher loop and must not be used to SSH-poll the remote workstation from
the main thread. For this route, `result_ready` requires the learned pair-set
JSONL, the stage-A single-pair result JSONL, `checkpoints/single_pair_invp.pt`,
and `results/frozen_aggregation_summary.json`; a learned JSONL without those
auxiliary artifacts remains `waiting_for_auxiliary_artifacts` while the remote
run is still active.

## Protocol

| Field | Value |
|---|---|
| Cipher | `PRESENT-80` |
| Rounds | `7` |
| Difference profile | `present_zhang_wang2022_mcnd` |
| Sample structure | `zhang_wang_case2_official_mcnd` |
| Pairs per sample | `16` for joint/evaluation samples |
| Single-pair scorer | `pairs_per_sample = 1` checkpoint |
| Negative mode | `encrypted_random_plaintexts` |
| Train key | `0x00000000000000000000` |
| Validation key | `0x11111111111111111111` |
| Scheduler | `official_cyclic` |
| Checkpoint metric | `val_auc` |
| Primary metric | `val_auc` |

## Decision Gates

Continue:

```text
pair_consistency_auc >= frozen_aggregate_auc + 0.001
and pair_consistency_auc >= invp_only_anchor_auc + 0.001
```

Weak continue:

```text
pair_consistency_auc is best, but margin < 0.001
```

Action:

```text
repeat 262144/class seed1 before any 1M run
```

Stop:

```text
pair_consistency_auc <= frozen_aggregate_auc
or pair_consistency_auc <= invp_only_anchor_auc
```

Action:

```text
Do not scale pair-consistency to 1M as a main route. Treat it as aggregation
or diagnostic context, and return to topology/DDT, active-pattern, or new SPN
feature construction.
```

## Evidence Language

Allowed if the gate passes:

```text
At 262144/class diagnostic scale, learned pair-set consistency outperformed
the frozen single-pair aggregation control under the same PRESENT r7 protocol.
```

Not allowed:

```text
formal route evidence
proof of multi-pair structural learning
breakthrough
SOTA
```

## Remote Launch Requirements

Before any meaningful remote launch:

```text
1. Implement the frozen aggregation route and tests.
2. Add smoke CSV with tiny samples.
3. Run local CPU smoke.
4. Add 262144/class plan CSV and remote config.
5. Verify disk-backed cache/progress paths under G:\lxy.
6. Run scripts/check-remote-readiness.
7. Generate stage-aware launcher/watcher:
   - stage A trains single-pair scorer with --checkpoint-output
   - stage B trains learned/anchor rows
   - frozen aggregation summary evaluates the saved stage-A checkpoint
   - postprocess-pairset-aggregation runs after both artifacts exist
8. Commit and push.
9. Launch from the pushed commit only after the topology-aware result gate or
   an explicit user route choice makes pair-set attribution the next action.
10. Hand off retrieval/postprocess to local tmux watcher.
```

## Relationship To Current Topology-Aware Run

Current running run:

```text
run_id = i1_spn_topology_aware_network_r7_262k_seed0_gpu0_20260701
status = running / watcher-managed
```

Completed DDT graph context:

```text
i1_spn_ddt_graph_r7_262k_seed0_gpu0_20260630 -> weak_ddt_graph_signal
i1_spn_ddt_graph_r7_262k_seed1_gpu1_20260630 -> weak_ddt_graph_signal
decision = do not promote DDT graph to 1M yet
```

This pair-set aggregation plan should not preempt the active topology-aware
network result. If topology-aware stops, the current default next data/feature
branch is candidate-trail / transition consistency; pair-set aggregation remains
a prepared attribution control to use when a pair-set claim becomes relevant or
when the user explicitly chooses this route.
