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
