# Innovation 1 Active-Pattern Auxiliary Head Plan

**Status:** launched / watcher-managed medium diagnostic seed0

**Scope:** PRESENT-80 r7, Zhang/Wang 2022 Case2 `m=16`, strict encrypted-random-plaintext negatives.

This plan is a current experiment plan, not the archived 2026-06-22 standalone
active-pattern screen. The archived screen remains non-launchable and should not
be reused as evidence.

## Research Question

Holding cipher, rounds, sample structure, negative mode, train/validation keys,
sample scale, scheduler, checkpoint metric, and primary metric fixed:

```text
Can an InvP/P-layer aligned SPN distinguisher learn a better real-vs-random
decision if it is also trained to predict SPN active-pattern or trail-consistency
auxiliary targets derived from the same public ciphertext-pair evidence?
```

## Single Hypothesis

The current strongest completed Innovation 1 evidence says that
`InvP(DeltaC)` / P-layer aligned SPN views are useful. Candidate-trail,
bit-transition-spectrum, DDT graph, topology-aware graph, and trail-family
routes ask whether hand-built SPN structure signals can beat that anchor as
standalone representations or architecture changes.

This route tests a different hypothesis:

```text
active-pattern and local trail signals may be too compressed or noisy as a
standalone real-vs-random input, but still useful as auxiliary supervision that
regularizes the InvP-only encoder toward SPN-relevant internal features.
```

Only one factor should change in the first experiment:

```text
model/training objective = InvP-only main classifier + active-pattern auxiliary head
```

Do not change the benchmark, negative samples, validation key, sample structure,
metric computation, checkpoint metric, or scale to make this look better.

## Trigger

Historical pre-launch rule:

```text
i1_trail_family_r7_262k_seed0_gpu1_20260702
```

had to be retrieved, validated, postprocessed, and gated before this route could
start.

That trigger has now fired:

```text
trail-family seed0 decision = stop_trail_family_route
active-auxiliary seed0 status = launched / watcher_handoff
```

This plan became actionable because one of these became true:

```text
1. trail-family seed0 gates to stop/tied/negative and pair-set aggregation is
   not the only desired fallback.
2. pair-set aggregation control shows learned cross-pair pooling is mostly
   independent-score aggregation, so the next useful model route is auxiliary
   supervision rather than another pair-set pooling variant.
3. User explicitly chooses active-pattern auxiliary supervision as the next
   SPN structure-adaptive neural network route.
```

If trail-family seed0 gates positive or weak-positive, run the configured
trail-family seed1 confirmation/variance check first.

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
| Scheduler | `official_cyclic` |
| Checkpoint metric | `val_auc` |
| Primary metric | `val_auc`, then calibrated accuracy |
| First meaningful scale | `262144/class` |
| Evidence level | medium diagnostic only |
| Cache | disk-backed dataset or feature cache with progress and parameter-matched reuse |

## Candidate Design

Base model:

```text
present_nibble_invp_only_spn_only
```

Main head:

```text
binary real-vs-random classification
loss_main = existing binary loss under the same training protocol
```

Auxiliary head options, first implementation chooses exactly one:

```text
A. active-cell mask over 16 PRESENT nibbles derived from InvP(DeltaC)
B. active-cell count bucket / density over each of the 16 pairs
C. trail-family consensus mask from public candidate evidence
```

First route should prefer `A` because it is deterministic, cheap, and directly
tests whether SPN cell activity supervision helps the already-supported InvP
encoder. Do not combine A/B/C in the first run.

Objective:

```text
loss = main_loss + lambda_aux * aux_loss
lambda_aux first value = 0.1
```

Expected model output during training may be a structured payload internally,
but result JSONL must still report the standard main-task metrics:

```text
auc
accuracy
calibrated_accuracy
loss
```

Auxiliary accuracy/F1 is diagnostic only. A high auxiliary metric without main
task improvement is not a successful Innovation 1 result.

## Minimal Matrix

First non-smoke scale:

```text
262144/class
seed = 0
```

Rows:

| Row | Route | Role |
|---:|---|---|
| 0 | `present_nibble_invp_only_spn_only` | strongest same-scale anchor |
| 1 | `present_nibble_invp_active_aux_spn_only` | candidate auxiliary-head route |
| 2 | `present_nibble_invp_active_aux_shuffled_targets` | target-control route |

Keep the matrix lean. Do not add Zhang/Wang, DDT graph, topology graph, and
trail-family rows unless the gate needs them for a documented protocol audit.

## Controls

Required control:

```text
shuffled_targets:
  preserve active target marginal frequencies but break sample-to-target
  alignment. If true auxiliary supervision and shuffled-target supervision tie,
  the auxiliary route is not evidence that active-pattern structure helped.
```

Optional later controls:

```text
lambda_aux = 0.0       # should match InvP-only training objective
lambda_aux = 0.03/0.3 # only after the first route shows a signal
mask-only aux          # no trail-family target until active mask is understood
```

## Gates

Support:

```text
active_aux_auc >= InvP-only anchor AUC + 0.001
and calibrated_accuracy is not worse than InvP-only
and active_aux_auc >= shuffled-target control AUC + 0.001
```

Action:

```text
launch 262144/class seed1 confirmation before any 1M run
```

Weak:

```text
active_aux_auc is at or above InvP-only but margin < 0.001
or true-vs-shuffled margin is positive but < 0.001
```

Action:

```text
run seed1 only if this is the best available next SPN structure hypothesis
after trail-family and pair-set gates
```

Stop:

```text
active_aux_auc <= InvP-only
or calibrated_accuracy regresses
or shuffled-target control matches/exceeds true auxiliary route
```

Action:

```text
do not scale active-pattern auxiliary supervision as a main route;
switch to the prepared S-box transition prior gate seed0 route, because it tests
local S-box DDT reliability gating over the same supported InvP cell view.
If S-box prior also stops, return to InvP route consolidation, cross-cipher
transfer planning, pair-set evidence pooling, or a new SPN graph/data hypothesis.
```

## Implementation Tasks

Task 1: model interface.

```text
Add an InvP-only SPN model variant with an auxiliary active-mask head, or add an
opt-in auxiliary-head wrapper around the existing InvP-only encoder.
```

Implementation constraint:

```text
Do not disturb existing binary model outputs for ordinary runners.
The auxiliary route should be opt-in and route-specific.
```

Task 2: target builder.

```text
Build deterministic active-cell targets from the same public ciphertext-pair
features used by the model. No labels, validation statistics, or secret-key
information may be used to create auxiliary targets.
```

Task 3: trainer path.

```text
Add a route-specific training loop or adapter that computes main binary loss
plus auxiliary loss while still emitting standard result rows and history.
```

Task 4: smoke and readiness.

```text
Create a tiny smoke config first.
Run CPU smoke to prove shape, loss, metrics, and result JSONL.
Only then prepare 262144/class remote config with disk-backed cache/progress.
```

## Readiness Requirements

Before any meaningful remote launch:

```text
1. docs/experiments plan updated with run_id, rows, gate, and claim scope.
2. CPU smoke passes.
3. tests cover deterministic active target construction and shuffled-target control.
4. remote config passes scripts/check-remote-readiness.
5. generated launcher uses cmd.exe /c and keeps all artifacts under G:\lxy.
6. local tmux watcher or sub-agent is ready to retrieve and postprocess.
7. code/config/docs are committed and pushed.
```

## Local Smoke Implementation

Status as of 2026-07-03:

```text
implementation_status = local smoke runner and medium seed0 readiness config implemented
evidence_level = smoke only; no model-quality claim
script = scripts/spn-active-auxiliary-matrix
smoke_config = configs/experiment/innovation1/innovation1_spn_present_active_auxiliary_smoke.json
medium_seed0_plan = configs/experiment/innovation1/innovation1_spn_present_active_auxiliary_r7_262k_seed0.json
medium_seed0_remote_config = configs/remote/innovation1_spn_present_active_auxiliary_r7_262k_seed0_gpu1_20260703.json
medium_seed1_plan = configs/experiment/innovation1/innovation1_spn_present_active_auxiliary_r7_262k_seed1.json
medium_seed1_remote_config = configs/remote/innovation1_spn_present_active_auxiliary_r7_262k_seed1_gpu1_20260703.json
```

Implemented rows:

| Row | Route | Role |
|---:|---|---|
| 0 | `present_nibble_invp_only_spn_only` | external anchor placeholder for matrix shape |
| 1 | `present_nibble_invp_active_aux_spn_only` | true active-mask auxiliary target |
| 2 | `present_nibble_invp_active_aux_shuffled_targets` | shuffled-target control |

Verified local smoke command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/spn-active-auxiliary-matrix \
  --config configs/experiment/innovation1/innovation1_spn_present_active_auxiliary_smoke.json \
  --output /tmp/i1_active_auxiliary_smoke.jsonl
```

Smoke output:

```text
3 JSONL rows emitted; candidate/control rows include auc, calibrated_accuracy,
auxiliary_loss, and auxiliary_target fields.
```

Medium seed0 readiness status:

```text
prepared_not_launched = true
samples_per_class = 262144
dataset_cache_root = G:\lxy\blockcipher-structure-adaptive-nd-runs\active_auxiliary_cache
dataset_cache_workers = 4
runner_script = scripts/spn-active-auxiliary-matrix
required_control = present_nibble_invp_active_aux_shuffled_targets
```

Gate/postprocess implementation status:

```text
gate_script = scripts/gate-active-auxiliary
postprocess_script = scripts/postprocess-active-auxiliary
monitor_health_postprocess_kind = active_auxiliary
gate_inputs = InvP-only anchor, true active-auxiliary candidate, shuffled-target control
positive_branch = seed1 confirmation assets prepared; launch only after seed0 gate selects support/weak
stop_branch = do not scale active-auxiliary; switch to prepared S-box transition prior gate seed0
```

The postprocess path writes local result-alignment, active-auxiliary gate,
summary, Markdown summary, next-action-readiness, and an idempotent plan-doc
result block. Seed1 confirmation assets now exist so a positive/weak seed0 gate
can proceed through the normal pushed-commit remote launch path without an extra
asset-generation step. If seed0 stops, postprocess points to the prepared
S-box transition prior gate seed0 remote config rather than requiring a new
route-design step.

Current next action after launch:

```text
1. Wait for the local watcher/sub-agent to retrieve active-auxiliary seed0.
2. Do not launch active-auxiliary seed1 until seed0 is retrieved, validated,
   plan-aligned, postprocessed, and gated as support/weak.
3. If seed0 stops or ties, switch to the prepared S-box transition prior gate
   seed0 route after readiness refresh from the pushed commit.
4. Do not treat the 262144/class seed0 result as formal evidence.
```

## Claim Scope

Allowed after a passing 262144/class seed0 gate:

```text
active-pattern auxiliary supervision shows medium diagnostic signal under the
official PRESENT r7 Case2 protocol.
```

Not allowed:

```text
formal route evidence
proof that active patterns solve PRESENT r7
breakthrough
SOTA
```

This route can only become a main claim after at least `1000000/class`
multi-seed, same-protocol, strict-negative, plan-aligned evidence.

## Launch Record

### i1_active_auxiliary_r7_262k_seed0_gpu1_20260703

```text
status = launched / watcher_handoff
date = 2026-07-04
trigger_source = i1_trail_family_r7_262k_seed0_gpu1_20260702
trigger_decision = stop_trail_family_route
trigger_claim_scope = trail-family diagnostic gate; not paper-scale, formal, or breakthrough evidence
claim_scope = medium diagnostic only; not formal evidence
```

Trail-family seed0 was retrieved, validated, postprocessed, and gated to
`stop_trail_family_route`. That satisfies this route's fallback trigger. This
active-auxiliary run remains a `262144/class` medium diagnostic and must not be
described as formal route evidence.

Launch target:

```text
run_id = i1_active_auxiliary_r7_262k_seed0_gpu1_20260703
plan = configs/experiment/innovation1/innovation1_spn_present_active_auxiliary_r7_262k_seed0.json
remote_config = configs/remote/innovation1_spn_present_active_auxiliary_r7_262k_seed0_gpu1_20260703.json
launcher = configs/remote/generated/run_i1_active_auxiliary_r7_262k_seed0_gpu1_20260703.cmd
monitor = configs/remote/generated/monitor_i1_active_auxiliary_r7_262k_seed0_gpu1_20260703.sh
expected_rows = 3
scale = 262144/class
seed = 0
device = cuda:1
dataset_cache_workers = 4
claim_scope = medium diagnostic only
```

Readiness refresh before launch:

```text
command = UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness --config configs/remote/innovation1_spn_present_active_auxiliary_r7_262k_seed0_gpu1_20260703.json
status = pass
checked_invariants = plan_exists, expected_rows_matches_plan, run_id_task_archive_alignment, github_ssh_repo, cmd_exe_c_only_policy, g_lxy_artifact_policy, training_protocol_matches_plan, medium_scale_dataset_cache, active_auxiliary_protocol_lock
warnings = remote config relies on runner/plan defaults for optimizer/loss/scheduler/checkpoint fields; dataset_cache_workers=4 is conservative and plan-aligned
```

Launch execution:

```text
source_commit = 4b593d3
remote_launcher_uploaded_to = G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_active_auxiliary_r7_262k_seed0_gpu1_20260703\run_i1_active_auxiliary_r7_262k_seed0_gpu1_20260703.cmd
windows_task_command = cmd.exe /c G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_active_auxiliary_r7_262k_seed0_gpu1_20260703\run_i1_active_auxiliary_r7_262k_seed0_gpu1_20260703.cmd
local_monitor_session = monitor_i1_active_auxiliary_seed0_20260703
local_result_root = outputs/remote_results/i1_active_auxiliary_r7_262k_seed0_gpu1_20260703
handoff_status = local watcher started; first heartbeat fresh
```

Initial bounded monitor-health check:

```text
status = running
results_jsonl_exists = false
results_jsonl_line_count = 0 / 3
done_markers = none
failed_markers = none
needs_main_thread_intervention = false
early_scp_warning = remote logs/results not created yet; normal before launcher emits first artifacts
```

### Launch Repair: 2026-07-04 10:06 +0800

```text
status_before_repair = remote_artifacts_missing
needs_main_thread_intervention = true
symptom = repeated watcher sync found no remote logs/ and no remote results/
remote_run_dir = G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_active_auxiliary_r7_262k_seed0_gpu1_20260703
run_dir_status = exists
launcher_status = exists
logs_status_before_repair = missing
results_status_before_repair = missing
python_process_before_repair = none
```

Diagnosis:

```text
The original Windows start-based launch path did not actually execute the
launcher far enough to create logs/, results/, torch info, git artifacts, or the
started marker. This was a launch/quoting problem, not a model-quality result
and not evidence for or against active-auxiliary.
```

Repair action:

```text
1. Verified the launcher existed under G:\lxy.
2. Ran the launcher once in a foreground SSH diagnostic to confirm it creates
   logs/, source/, readiness artifacts, and starts the Python runner.
3. Stopped that foreground diagnostic process.
4. Relaunched the same existing launcher with Windows native background process
   creation:
   wmic process call create "cmd.exe /c call G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_active_auxiliary_r7_262k_seed0_gpu1_20260703\run_i1_active_auxiliary_r7_262k_seed0_gpu1_20260703.cmd"
5. Verified a background python.exe process is running the planned
   scripts\spn-active-auxiliary-matrix command.
```

Post-repair bounded status:

```text
status = running
needs_main_thread_intervention = false
remote logs/ = exists
remote source/ = exists
remote python process = running
local logs pulled = yes
progress_file = logs/active_auxiliary_progress.jsonl
latest_progress = active_auxiliary_cache_start / train / 524288 total rows
observed_cache_chunks = active_auxiliary_positive_chunk reached 16384 class rows before relaunch; background run restarted cache generation against the same disk cache path
results_jsonl_exists = false
claim_scope = still medium diagnostic only; no result yet
main_thread_policy = watcher/sub-agent should resume retrieval; do not launch active-auxiliary seed1 or S-box prior until seed0 is retrieved, validated, postprocessed, and gated
```

### Sidecar Failure Check: 2026-07-04 10:11 +0800

```text
status = failed
needs_main_thread_intervention = true
postprocess_allowed = false
results_jsonl_exists = false
results_jsonl_line_count = 0 / 3
done_markers = none
failed_marker = logs/i1_active_auxiliary_r7_262k_seed0_gpu1_20260703_failed.marker
remote_git_revision = d1579f9953541d241cdc24963c63b8c0bdf038d0
readiness_status = pass
```

Latest pulled progress before the failure marker:

```text
progress_file = outputs/remote_results/i1_active_auxiliary_r7_262k_seed0_gpu1_20260703/logs/active_auxiliary_progress.jsonl
latest_event = active_auxiliary_positive_chunk
split = train
samples_per_class = 262144
total_rows = 524288
class_rows_done = 40960 / 262144
cache_total_progress_percent = 7.812
cache_rows_per_second = 79.186
cache_eta_seconds = 6104
```

Local artifacts did not contain a Python traceback:

```text
stdout_log = empty
stderr_log = empty
torch_info = torch 2.5.1+cu118 / cuda 11.8 / 2 CUDA devices
monitor_health_exit_code = 4
```

Decision:

```text
claim_scope = failed launch/run artifact state; no model-quality metric
gate_decision = no gate, no postprocess, no seed1, no S-box prior launch from this result
required_next_action = main-thread bounded diagnosis or repair plan, because monitor-health explicitly requested intervention
```

### Stale Failed-Marker Correction: 2026-07-04 10:22 +0800

```text
corrected_status = running
claim_scope = still no model-quality metric
root_cause = stale failed marker from the foreground diagnostic/repair phase
```

Bounded remote diagnosis showed that the `failed.marker` was not the final state
of the repaired background run:

```text
failed_marker_mtime = 2026-07-04 10:05 +0800
remote_progress_mtime = 2026-07-04 10:18 +0800
remote_python_process = python.exe PID 47268 still running planned spn-active-auxiliary-matrix command
stdout_log = empty
stderr_log = empty
```

The local watcher had stopped at 10:11 because it treated any pulled
`failed.marker` as terminal, even though the repaired background process kept
updating `active_auxiliary_progress.jsonl`. This was a monitor/tooling failure,
not active-auxiliary model evidence.

Repair:

```text
1. Updated monitor-health to report stale_failed_markers when progress/heartbeat
   proves a later run is still active.
2. Updated the active-auxiliary seed0/seed1 watcher scripts to ignore a failed
   marker that is older than the active progress file.
3. Restarted only the local watcher; the remote training process was not
   restarted or modified.
```

Post-repair bounded status:

```text
status = running
needs_main_thread_intervention = false
stale_failed_markers = logs/i1_active_auxiliary_r7_262k_seed0_gpu1_20260703_failed.marker
latest_event = active_auxiliary_positive_chunk
split = train
class_rows_done = 139264 / 262144
cache_class_progress_percent = 53.125
results_jsonl_exists = false
postprocess_allowed = false
main_thread_policy = watcher/sub-agent continues retrieval; no seed1 or S-box prior until seed0 result is retrieved and gated
```

### Final Failure Diagnosis: 2026-07-04 19:33 +0800

Bounded local `monitor-health` later reported:

```text
status = stale_monitor
needs_main_thread_intervention = true
postprocess_allowed = false
results_jsonl_exists = false
done_marker = none
failed_marker = logs/i1_active_auxiliary_r7_262k_seed0_gpu1_20260703_failed.marker
```

A single bounded remote diagnosis was then allowed by the monitor-health gate.
Remote state:

```text
remote_time = 2026-07-04T19:33:55+08:00
progress_mtime = 2026-07-04T11:25:39+08:00
latest_progress_event = active_auxiliary_cache_done
latest_progress_split = validation
validation_samples_per_class = 65536
results_jsonl_exists = false
done_marker = false
failed_marker = true
failed_marker_mtime = 2026-07-04T11:47:33+08:00
matching_training_process = none
```

Remote stderr root cause:

```text
torch.OutOfMemoryError: CUDA out of memory.
Tried to allocate 16.00 GiB.
GPU 1 total capacity = 47.99 GiB
GPU 1 free at failure = 10.25 GiB
failure site = _evaluate_active_aux_model -> model(x) over the full validation tensor
```

Interpretation:

```text
This is an implementation/runtime failure during validation evaluation, not
active-auxiliary route evidence. The run produced no result rows, no gate, and
no model-quality conclusion.
```

Repair:

```text
1. Changed _evaluate_active_aux_model to evaluate validation features in bounded
   batches using the configured batch_size.
2. Aggregates logits across batches and computes auxiliary loss as an element
   weighted mean.
3. Avoids converting read-only memmap validation slices directly into torch
   tensors by copying each evaluation batch.
4. Added a regression test proving active-auxiliary evaluation calls the model
   in bounded batches instead of one full validation forward pass.
```

Verification:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_project_structure.py -k "active_auxiliary"
```

Result:

```text
21 passed, 217 deselected
```

Retry rule:

```text
Relaunch active-auxiliary seed0 from the pushed repair commit before any
active-auxiliary seed1 or S-box transition prior branch. Reuse the disk-backed
cache when metadata matches, but write retry run artifacts to a fresh run_id so
the failed OOM evidence remains auditable.
```

Prepared retry:

```text
run_id = i1_active_auxiliary_r7_262k_seed0_gpu1_retry1_20260704
plan = configs/experiment/innovation1/innovation1_spn_present_active_auxiliary_r7_262k_seed0_retry1.json
remote_config = configs/remote/innovation1_spn_present_active_auxiliary_r7_262k_seed0_gpu1_retry1_20260704.json
launcher = configs/remote/generated/run_i1_active_auxiliary_r7_262k_seed0_gpu1_retry1_20260704.cmd
monitor = configs/remote/generated/monitor_i1_active_auxiliary_r7_262k_seed0_gpu1_retry1_20260704.sh
expected_rows = 3
scale = 262144/class
claim_scope = medium diagnostic retry after validation OOM repair; no prior failed metric
```

Readiness:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness \
  --config configs/remote/innovation1_spn_present_active_auxiliary_r7_262k_seed0_gpu1_retry1_20260704.json
```

Result:

```text
status = pass
checked_invariants = plan_exists, expected_rows_matches_plan,
  run_id_task_archive_alignment, github_ssh_repo, cmd_exe_c_only_policy,
  g_lxy_artifact_policy, training_protocol_matches_plan,
  medium_scale_dataset_cache, active_auxiliary_protocol_lock
```
