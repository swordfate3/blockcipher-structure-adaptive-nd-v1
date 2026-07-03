# Innovation 1 Trail-Family Consistency Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `skills/blockcipher-auto-research/SKILL.md` for evidence gates, and use
> Karpathy-style coding discipline for implementation. This is a conditional
> experiment plan under `docs/experiments/`, not a historical agent plan.

**Goal:** Prepare the next SPN/PRESENT structure-adaptive data route after the
active candidate-trail and prepared bit-transition-spectrum diagnostics resolve.
This route tests whether real samples are better characterized by consistency
with a small family of plausible SPN differential trails than by per-cell or
bit-transition summaries alone.

**Architecture:** Keep PRESENT-80 r7 Zhang/Wang 2022 Case2 protocol fixed. Build
deterministic, cacheable trail-family features from `DeltaC`, `InvP(DeltaC)`,
active cell patterns, DDT legality/probability, and pair-to-pair agreement
against a compact candidate trail family. Compare only against the strongest
same-scale InvP-only anchor and false-family / shuffled-alignment controls.

**Status:** conditional next-hypothesis plan only.

```text
do_not_launch_until =
  1. i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702 is retrieved,
     validated, postprocessed, and gated; and
  2. candidate-trail does not select seed1 confirmation as the active branch; and
  3. bit-transition-spectrum either stops or is explicitly deprioritized by its
     gate/plan after documented evidence.

claim_scope = planned route only; no evidence yet
implementation_status = local smoke runner/gate/postprocess implemented; medium seed0 readiness assets prepared
remote_config_status = prepared but gated; do not launch until trigger
```

This plan exists so the project does not stall if candidate-trail and
bit-transition-spectrum are tied or negative. It is not permission to bypass the
current candidate-trail gate.

## Trigger

Start this route only if one of these is true:

```text
1. candidate-trail seed0 decision = stop_candidate_trail_route and
   bit-transition-spectrum seed0 decision = stop_transition_spectrum_route.

2. candidate-trail seed0 is weak/tied, bit-transition-spectrum is weak/tied,
   and the documented branch decision prefers trail-family evidence over another
   variance seed.

3. transition-spectrum is blocked by shuffled-P control matching the true route,
   leaving higher-level trail-family consistency as the next SPN hypothesis.
```

Do not start this route if:

```text
candidate-trail seed0 decision = support_candidate_trail_route
or transition-spectrum seed0 decision = support_transition_spectrum_route
```

In those cases, run the corresponding 262144/class seed1 confirmation first.

## Research Question

Holding cipher, rounds, sample structure, negative mode, train/validation keys,
scale, checkpoint metric, and primary metric fixed:

```text
Do real PRESENT r7 Case2 samples show stronger agreement with a compact family
of plausible SPN differential trails than encrypted-random-plaintext negatives,
beyond what InvP-only, candidate-trail cell summaries, or bit-transition
spectrum features capture?
```

## Single Hypothesis

The previous routes test local or low-order structure:

```text
InvP-only                 -> true P-layer aligned output-difference view
candidate-trail           -> per-cell transition consistency
bit-transition-spectrum   -> bit-level P-layer movement statistics
```

The remaining gap is a higher-level trail-family signal:

```text
real samples should concentrate around a small set of compatible active-cell
and S-box transition families across the 16 pairs, while encrypted-random
negatives should have weaker family agreement or larger trail-family entropy.
```

This route changes only the feature representation:

```text
feature route = trail-family consistency
```

It must not change negative samples, validation key, sample structure, metric,
checkpoint selection, train/validation split logic, or sample scale.

## Fixed Protocol

| Field | Value |
|---|---|
| Cipher | `PRESENT-80` |
| Rounds | `7` |
| Difference profile | `present_zhang_wang2022_mcnd` |
| Sample structure | `zhang_wang_case2_official_mcnd` |
| Pairs/sample | `16` |
| Negative mode | `encrypted_random_plaintexts` |
| Train key | `0x00000000000000000000` |
| Validation key | `0x11111111111111111111` |
| Key rotation | `0` |
| Primary metric | `val_auc` / `auc` |
| Checkpoint metric | `val_auc` |
| First scale | `262144/class` |
| Evidence level | medium diagnostic only |
| Cache | disk-backed feature cache with metadata, progress, and parameter-matched reuse |

## Feature Sketch

Use deterministic features derived from each 16-pair sample:

```text
per pair:
  DeltaC
  InvP(DeltaC)
  active cell mask over 16 nibbles
  DDT-supported input-difference candidates per active output cell
  top-k local transition scores or normalized DDT counts

per sample:
  family best score
  family top-k score margin
  family entropy
  count of pairs explained by best family
  mean/std/min/max family compatibility over 16 pairs
  impossible-transition count under best family
  active-cell overlap with best family
  disagreement between top two trail families
```

Controls:

```text
false_family_control:
  use deterministic shuffled or random trail-family templates with the same
  number of active cells and same feature dimensions.

invp_anchor:
  inject the same-scale InvP-only anchor row.

simple_statistics_control:
  use active counts / DDT aggregate counts without family identity, if the first
  implementation can support it without broadening the matrix too much.
```

The first implementation should be a deterministic feature route plus linear/MLP
diagnostics, not a new large neural architecture.

## Minimal Matrix

First non-smoke scale:

```text
262144/class
seed = 0
```

Rows:

| Row | Model/route | Role |
|---:|---|---|
| 0 | `present_nibble_invp_only_spn_only` | strongest same-scale anchor |
| 1 | `trail_family_consistency_linear` | linear feature sufficiency diagnostic |
| 2 | `trail_family_consistency_mlp` | small nonlinear feature diagnostic |
| 3 | `trail_family_consistency_false_family` | false-family/control alignment |

Do not include Zhang/Wang baseline unless a protocol audit requires it. The
question is whether the trail-family route improves beyond the current best
internal InvP-only anchor under the same scale.

## Gates

Primary metric:

```text
auc
```

Support:

```text
best true trail-family route >= InvP-only same-scale anchor + 0.001 AUC
and calibrated_accuracy is not worse than InvP-only
and true trail-family route >= false-family control + 0.001 AUC
```

Action:

```text
launch 262144/class seed1 confirmation before any 1M run
```

Weak:

```text
best true trail-family route is at or above InvP-only but margin < 0.001 AUC
or true-vs-false-family margin is positive but < 0.001 AUC
```

Action:

```text
run seed1 only if candidate-trail and transition-spectrum are also weak/negative
and this route is the best available next SPN feature hypothesis
```

Stop:

```text
true trail-family route <= InvP-only
or calibrated_accuracy regresses
or false-family control matches/exceeds true trail-family route
```

Action:

```text
do not scale this feature route
switch to active-pattern auxiliary-head attribution, cross-cipher GIFT/SKINNY
transfer planning, or formalize InvP-only as the cleaner structure-adaptive
route with broader multi-seed attribution
```

## Implementation Plan

### Task 1: Define Trail-Family Templates

Status: local feature foundation implemented.

Required behavior:

```text
derive a compact deterministic family from PRESENT active-cell patterns and DDT
support; do not use labels or validation statistics to choose templates
```

Suggested files:

```text
created src/blockcipher_nd/features/spn_trail_family.py
modified tests/test_project_structure.py
```

Implemented behavior:

```text
present_pair_trail_family_template:
  builds deterministic, label-free per-pair active-mask / confidence / margin /
  disagreement / score views from existing PRESENT candidate-evidence layers.

present_pair_trail_family_features:
  emits fixed per-pair summary features for smoke diagnostics.

present_pairset_trail_family_features:
  emits pair-set agreement, consensus, entropy, margin, aggregate pair-feature,
  and global mask statistics for one multi-pair sample.

false_family:
  applies a deterministic cell-shift control with matched dimensions, intended
  for future true-family vs false-family attribution checks.
  As of commit 209fcde follow-up implementation, the shifted active masks are
  also used when building pair-level active-mask summary features, so the
  control no longer leaves the per-pair active-mask block fully true-family
  aligned.
  Trail-family cache_version is 2 after this control-semantics change; later
  runs should not reuse version-1 false-family caches for route evidence.
```

Verification:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_project_structure.py -k "trail_family or bit_transition_spectrum"
```

This implementation is not result evidence and does not authorize launching the
trail-family medium run before the trigger conditions above are met.

### Task 2: Add Dataset/CLI Matrix Route

Status: local smoke runner implemented.

Required behavior:

```text
read JSON matrix config
generate/reuse disk-backed trail-family features
emit one JSONL row per matrix row
include metrics.auc, metrics.calibrated_accuracy, auc, accuracy, calibrated_accuracy
record feature_cache_workers and cache metadata/progress
```

Suggested files:

```text
created src/blockcipher_nd/tasks/innovation1/spn_trail_family.py
created src/blockcipher_nd/cli/spn_trail_family_matrix.py
created scripts/spn-trail-family-matrix
```

Implemented behavior:

```text
runner_script = scripts/spn-trail-family-matrix
feature_route = trail_family_consistency
rows = external InvP anchor, linear, MLP, false-family control
cache = disk-backed features.npy / labels.npy with metadata and progress JSONL
workers = feature_cache_workers, using the same map-style chunk interface as
          the bit-transition-spectrum route
```

This runner is implemented for smoke/readiness and future gated use. The
262144/class remote matrix is still blocked by the trigger conditions at the top
of this document.

### Task 3: Add Smoke Config And Gate

Status: local smoke config, gate, and postprocess implemented.

Required behavior:

```text
tiny smoke first; no accuracy claims
gate compares true route vs InvP anchor and false-family control
postprocess writes next_action_readiness.json
```

Suggested files:

```text
created configs/experiment/innovation1/innovation1_spn_present_trail_family_smoke.json
created src/blockcipher_nd/planning/trail_family_gate.py
created src/blockcipher_nd/planning/trail_family_postprocess.py
created scripts/gate-trail-family
created scripts/postprocess-trail-family
```

Implemented behavior:

```text
gate compares:
  best(trail_family_consistency_linear, trail_family_consistency_mlp)
  vs present_nibble_invp_only_spn_only anchor
  vs trail_family_consistency_false_family control

postprocess writes:
  *_trail_family_gate.json
  *_postprocess_summary.json
  *_postprocess_summary.md
  *_next_action_readiness.json
```

Smoke command:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/spn-trail-family-matrix \
  --config configs/experiment/innovation1/innovation1_spn_present_trail_family_smoke.json \
  --output /tmp/i1_trail_family_smoke.jsonl

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/gate-trail-family \
  --results /tmp/i1_trail_family_smoke.jsonl \
  --expected-rows 4 \
  --require-false-family-control

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/postprocess-trail-family \
  --results /tmp/i1_trail_family_smoke.jsonl \
  --output-dir /tmp/i1_trail_family_postprocess \
  --run-id i1_trail_family_smoke_verify \
  --plan configs/experiment/innovation1/innovation1_spn_present_trail_family_smoke.json \
  --expected-rows 4
```

Smoke verification status:

```text
matrix = pass, emitted 4 JSONL rows
gate = pass, weak_trail_family_signal on tiny smoke only
postprocess = pass, validation_status = pass, next_action_readiness emitted
claim_scope = smoke/readiness only, not model evidence
```

### Task 4: Add Medium Plan/Remote Config

Status: seed0 and seed1 readiness assets prepared; launches remain blocked by gate.

Created files:

```text
configs/experiment/innovation1/innovation1_spn_present_trail_family_r7_262k_seed0.json
configs/experiment/innovation1/innovation1_spn_present_trail_family_r7_262k_seed1.json
configs/remote/innovation1_spn_present_trail_family_r7_262k_seed0_gpu1_20260702.json
configs/remote/innovation1_spn_present_trail_family_r7_262k_seed1_gpu1_20260702.json
configs/remote/generated/run_i1_trail_family_r7_262k_seed0_gpu1_20260702.cmd
configs/remote/generated/run_i1_trail_family_r7_262k_seed1_gpu1_20260702.cmd
configs/remote/generated/monitor_i1_trail_family_r7_262k_seed0_gpu1_20260702.sh
configs/remote/generated/monitor_i1_trail_family_r7_262k_seed1_gpu1_20260702.sh
```

Medium seed0 matrix:

```text
run_id = i1_trail_family_r7_262k_seed0_gpu1_20260702
scale = 262144/class
seed = 0
device = cuda:1
feature_cache_root = G:\lxy\blockcipher-structure-adaptive-nd-runs\trail_family_cache
feature_cache_workers = 4
rows = external InvP anchor, linear, mlp, false_family
expected_rows = 4
runner_script = scripts/spn-trail-family-matrix
monitor = local tmux watcher / sub-agent, with postprocess-trail-family
claim_scope = medium diagnostic only
```

Medium seed1 confirmation / variance-check matrix:

```text
run_id = i1_trail_family_r7_262k_seed1_gpu1_20260702
scale = 262144/class
seed = 1
device = cuda:1
feature_cache_root = G:\lxy\blockcipher-structure-adaptive-nd-runs\trail_family_cache
feature_cache_workers = 4
rows = external InvP anchor, linear, mlp, false_family
expected_rows = 4
runner_script = scripts/spn-trail-family-matrix
monitor = local tmux watcher / sub-agent, with postprocess-trail-family
launch_gate = support_trail_family_route or weak_trail_family_signal from seed0
claim_scope = medium diagnostic confirmation/variance only
```

Launch rule:

```text
do not launch the medium remote config until candidate-trail and
transition-spectrum gates select or explicitly document this branch

do not launch seed1 until seed0 is retrieved, validated, plan-aligned, and
postprocessed as support_trail_family_route or weak_trail_family_signal
```

## Readiness Requirements

Before any meaningful remote launch:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_project_structure.py -k "trail_family or remote_readiness"

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness \
  --config configs/remote/innovation1_spn_present_trail_family_r7_262k_seed0_gpu1_20260702.json

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness \
  --config configs/remote/innovation1_spn_present_trail_family_r7_262k_seed1_gpu1_20260702.json
```

The readiness gate must enforce:

```text
runner_script = scripts/spn-trail-family-matrix
sample_structure = zhang_wang_case2_official_mcnd
negative_mode = encrypted_random_plaintexts
validation_key = 0x11111111111111111111
key_rotation_interval = 0
feature_cache_root under G:\lxy\blockcipher-structure-adaptive-nd-runs
cmd.exe /c only
expected_rows = 4
```

## Current Action

```text
local smoke runner/gate/postprocess implemented
medium seed0 and seed1 plan/remote/launcher/monitor prepared
candidate-trail seed0 gate = stop_candidate_trail_route
bit-transition-spectrum seed0 gate = stop_transition_spectrum_route
trail-family seed0 is now the active next branch
```

## Launch Record

### i1_trail_family_r7_262k_seed0_gpu1_20260702

```text
status = launched / watcher_handoff
date = 2026-07-03
trigger_source_1 = i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702
trigger_decision_1 = stop_candidate_trail_route
trigger_source_2 = i1_bit_transition_spectrum_r7_262k_seed0_gpu1_20260702
trigger_decision_2 = stop_transition_spectrum_route
claim_scope = medium diagnostic only; not formal evidence
```

Candidate-trail and bit-transition-spectrum were both retrieved, validated,
postprocessed, plan-aligned, and gated to stop. This satisfies the trigger for
trail-family seed0. The route remains a `262144/class` medium diagnostic and
must not be described as paper-scale evidence.

Launch target:

```text
run_id = i1_trail_family_r7_262k_seed0_gpu1_20260702
plan = configs/experiment/innovation1/innovation1_spn_present_trail_family_r7_262k_seed0.json
remote_config = configs/remote/innovation1_spn_present_trail_family_r7_262k_seed0_gpu1_20260702.json
launcher = configs/remote/generated/run_i1_trail_family_r7_262k_seed0_gpu1_20260702.cmd
monitor = configs/remote/generated/monitor_i1_trail_family_r7_262k_seed0_gpu1_20260702.sh
expected_rows = 4
scale = 262144/class
seed = 0
device = cuda:1
feature_cache_workers = 4
claim_scope = medium diagnostic only
```

Readiness refresh before launch:

```text
command = UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness --config configs/remote/innovation1_spn_present_trail_family_r7_262k_seed0_gpu1_20260702.json
status = pass
checked_invariants = plan_exists, expected_rows_matches_plan, run_id_task_archive_alignment, github_ssh_repo, cmd_exe_c_only_policy, g_lxy_artifact_policy, training_protocol_matches_plan, medium_scale_dataset_cache, trail_family_protocol_lock
warnings = remote config relies on runner/plan defaults for optimizer/loss/scheduler/checkpoint fields
```

Next execution step:

```text
launch from pushed commit
hand off monitoring/retrieval/postprocess to local tmux watcher
do not SSH-poll from the main thread after launch
```

Launch execution:

```text
source_commit = d28fc9c
remote_launcher_uploaded_to = G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_trail_family_r7_262k_seed0_gpu1_20260702\run_i1_trail_family_r7_262k_seed0_gpu1_20260702.cmd
windows_task = i1_trail_family_r7_262k_seed0_gpu1_20260702
windows_task_command = cmd.exe /c G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_trail_family_r7_262k_seed0_gpu1_20260702\run_i1_trail_family_r7_262k_seed0_gpu1_20260702.cmd
local_monitor_session = monitor_i1_trail_family_seed0_20260702
local_result_root = outputs/remote_results/i1_trail_family_r7_262k_seed0_gpu1_20260702
handoff_status = local watcher started; first heartbeat fresh
```

Initial bounded monitor-health check:

```text
status = running
results_jsonl_exists = false
results_jsonl_line_count = 0 / 4
done_markers = none
failed_markers = none
needs_main_thread_intervention = false
heartbeat = fresh
launch_state = launch_progress_observed
```

## Claim Scope

Until at least `1000000/class` multi-seed evidence exists, this route can only
support one of these statements:

```text
trail-family consistency smoke passed
trail-family consistency medium diagnostic positive
trail-family consistency tied with InvP-only
trail-family consistency negative under official Case2 protocol
```

It must not be described as:

```text
formal route evidence
breakthrough
SOTA
proof that trail families solve PRESENT r7
```

## Relation To Other Routes

```text
InvP-only                 -> structure-aligned data representation
candidate-trail route     -> local transition-consistency feature attribution
bit-transition-spectrum   -> bit-level P-layer movement statistics
trail-family consistency  -> higher-level active-pattern / transition-family agreement
```

The useful paper direction is whichever route survives same-protocol scale,
seed, and attribution gates most cleanly.
