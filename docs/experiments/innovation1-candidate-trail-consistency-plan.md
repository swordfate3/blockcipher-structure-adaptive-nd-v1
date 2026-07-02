# Innovation 1 Candidate-Trail Consistency Plan

**Date:** 2026-07-01

**Status:** active next data-representation branch / medium seed0 launch
prepared after topology-aware network route stopped

**Scope:** PRESENT-80 r7, Zhang/Wang 2022 Case2 `m=16`, strict
encrypted-random-plaintext negatives.

## Why This Exists

The old active-pattern screen is archived and not launchable. It used a small
standalone active-statistics route and did not match the current remote workflow
or current official Case2 protocol. However, active S-box patterns and candidate
trail consistency remain useful SPN-structure signals if they are used as
attribution features or auxiliary evidence for a real-vs-random distinguisher.

This plan defines the re-entry route for that idea. It should not revive the old
active-only screen. It should test whether candidate trail / transition
consistency features add real distinguishing signal beyond the current
InvP/DDT/pair-set evidence routes.

## Current Evidence Boundary

Current strongest completed Innovation 1 route:

```text
present_nibble_invp_only_spn_only
evidence = two-seed 1000000/class positive confirmation
attribution = paper-scale controls support true InvP/P-layer alignment
```

Recently completed route:

```text
i1_spn_ddt_graph_r7_262k_seed0_gpu0_20260630
i1_spn_ddt_graph_r7_262k_seed1_gpu1_20260630
status = retrieved / validated / postprocessed / plan-aligned
decision = weak_ddt_graph_signal in both seeds
manual branch decision = do not promote DDT graph to 1M yet
```

Resolved predecessor route:

```text
i1_spn_topology_aware_network_r7_262k_seed1_gpu1_20260701
status = retrieved / validated / postprocessed / plan-aligned
decision = stop_topology_aware_network_route
reason = true-P graph did not beat InvP-only or shuffled-P controls
```

Candidate-trail consistency is therefore a prepared next data/feature
representation branch selected after the topology-aware network route stopped.

## Trigger

This plan becomes actionable only after one of these conditions:

```text
1. The active topology-aware network run is retrieved, validated,
   postprocessed, plan-aligned, and its gate is tied/negative.

2. The active topology-aware network run is weak, and the next branch decision
   prefers a data/feature representation hypothesis over another topology seed.

3. The topology-aware network route is positive, but attribution needs a richer
   trail-consistency diagnostic before any formal method claim.

4. User explicitly chooses candidate-trail / active-pattern feature
   representation as the next route after the current active run is resolved.
```

This route became launchable because the blocking topology-aware route is
resolved:

```text
i1_spn_topology_aware_network_r7_262k_seed1_gpu1_20260701
decision = stop_topology_aware_network_route
```

Current active candidate-trail run:

```text
run_id = i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702
status = launched from pushed commit / local watcher-managed / running
local_root = outputs/remote_results/i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702
latest_bounded_check = 2026-07-02 18:51:27+08:00 heartbeat, no failed markers
progress = candidate feature-cache generation, 229376/262144 class rows, 229376/524288 total rows
progress_percent = 87.5% of current class, 43.75% of total cache
latest_event = candidate_cache_positive_chunk
results_jsonl = not yet present
postprocess_allowed = false
needs_main_thread_intervention = false
claim_scope = running only; no candidate-trail evidence yet
```

Prepared conditional seed1 follow-up:

```text
run_id = i1_candidate_trail_consistency_r7_262k_seed1_gpu1_20260702
experiment_config = configs/experiment/innovation1/innovation1_spn_present_candidate_trail_consistency_r7_262k_seed1.json
remote_config = configs/remote/innovation1_spn_present_candidate_trail_consistency_r7_262k_seed1_gpu1_20260702.json
launcher = configs/remote/generated/run_i1_candidate_trail_consistency_r7_262k_seed1_gpu1_20260702.cmd
monitor = configs/remote/generated/monitor_i1_candidate_trail_consistency_r7_262k_seed1_gpu1_20260702.sh
status = readiness asset only / do not launch while seed0 is running
trigger = seed0 candidate-trail gate returns support_candidate_trail_route or weak_candidate_trail_signal
scale = 262144/class
seed = 1
feature_cache_workers = 4
claim_scope = conditional medium diagnostic confirmation or variance check only
```

This seed1 asset exists so the result-ready branch can run
`scripts/check-remote-readiness` against a concrete config instead of stopping
on manual config creation. It must not be launched until seed0 is retrieved,
validated, plan-aligned, postprocessed, and the candidate-trail gate selects a
support or weak-signal branch.

Watcher/postprocess handoff:

```text
monitor = configs/remote/generated/monitor_i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702.sh
behavior = sync logs/results, wait for 4 non-empty JSONL rows, then run scripts/postprocess-candidate-trail
plan_doc_update = docs/experiments/innovation1-candidate-trail-consistency-plan.md
postprocess_artifacts =
  outputs/remote_results/i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702/
    i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702_candidate_trail_gate.json
    i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702_postprocess_summary.json
    i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702_postprocess_summary.md
    i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702_next_action_readiness.json
```

After the watcher records `postprocess_done`, the next bounded local action is
to inspect the postprocess summary and `next_action_readiness` artifact, run
`scripts/plan-next-action` if an explicit readiness re-check is needed, commit
the updated experiment document, and only then follow the recorded gate branch.
If the watcher records `postprocess_failed`, inspect
`monitor/postprocess_stderr.log` before making any route decision.

## Research Question

Holding cipher, rounds, sample structure, negative mode, train/validation keys,
optimizer, scheduler, checkpoint metric, and scale fixed:

```text
Do candidate trail / transition consistency features expose SPN differential
propagation information that is not already captured by InvP-only, DDT graph,
or learned pair-set consistency?
```

## Single Hypothesis

For PRESENT r7 Case2 `m=16`, real differential samples should show more
coherent candidate-transition evidence across S-box cells and across the 16
pairs than encrypted-random-plaintext negatives.

The route should test exactly one factor:

```text
feature route = candidate trail / transition consistency
```

Do not mix this first test with new topology layers, new pair aggregation
training, changed negative samples, changed validation key, changed metric, or
new sample structure.

## Required Protocol

Use the current official protocol:

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
| Scheduler | `official_cyclic` |
| Checkpoint metric | `val_auc` |
| Dataset storage | disk-backed cache with progress/reuse metadata |

Implementation update, 2026-07-01:

```text
scripts/spn-candidate-evidence now defaults to zhang_wang_case2_official_mcnd,
validation_key = 0x11111111111111111111,
key_rotation_interval = 0,
and disk-backed candidate feature cache can generate/reuse official Case2
candidate-trail features in local smoke.
```

The remote readiness gate must reject any candidate-trail remote config that
uses:

```text
sample_structure = zhang_wang_case2_mcnd
validation_key = 0xffffffffffffffffffff
key_rotation_interval > 0
missing feature_mode or feature_mode not in {cell_structured, cell_structured_shuffled}
```

unless it is clearly labeled as a historical diagnostic that is not comparable
with the current official Case2 evidence chain.

Readiness tooling update, 2026-07-01:

```text
scripts/check-remote-readiness enforces candidate-trail/candidate-evidence
protocol lock:
  sample_structure = zhang_wang_case2_official_mcnd
  negative_mode = encrypted_random_plaintexts
  validation_key = 0x11111111111111111111
  key_rotation_interval = 0
  feature_mode in {cell_structured, cell_structured_shuffled}
  feature_cache_root or dataset_cache_root under G:\lxy\blockcipher-structure-adaptive-nd-runs
```

Feature-cache implementation update, 2026-07-02:

```text
scripts/spn-candidate-evidence supports feature_cache_workers for future
candidate-trail runs. The option parallelizes deterministic chunk feature
generation for the route-specific feature cache, records the worker count in
results/progress, and is gated by scripts/check-remote-readiness.
The worker count is not part of the cache identity because it changes execution
strategy, not the deterministic dataset contents; this allows a later 4-worker
request to reuse a compatible cache produced by a 1-worker run.
The already launched seed0 run keeps its original plan/config; future seed1 or
next-branch candidate-trail runs may explicitly set feature_cache_workers > 1
after local smoke/readiness validation.
```

Progress observability update, 2026-07-02:

```text
Future candidate-trail feature-cache progress records include a `time` field.
scripts/monitor-health reports cache_rows_per_second and cache_eta_seconds when
at least two timestamped cache-progress records are available. The active seed0
run was launched before this observability update, so its ETA fields remain null
even though percentage progress is available.
```

The historical `configs/remote/innovation1_spn_candidate_evidence_r7_65536_gpu0_20260623.json`
is intentionally not launch-ready under the current protocol lock.

## Feature Sketch

Build deterministic per-sample features from ciphertext pair data and
PRESENT/SPN structure:

```text
per cell:
  DeltaC nibble
  InvP(DeltaC) nibble
  active DeltaC / active InvP flags
  DDT legality / probability summaries
  candidate input-difference support
  top candidate transition margin

across cells:
  active count
  active-position histogram
  high-probability transition count
  impossible/low-probability transition count
  transition entropy or margin

across 16 pairs:
  mean/std/max of candidate consistency scores
  agreement of active positions
  disagreement between likely local transitions
```

These features should be deterministic and cacheable. The first implementation
should remain a diagnostic feature route, not a new large network.

## Minimal Matrix

First non-smoke scale:

```text
262144/class
seed = 0
```

Keep the matrix lean:

| Row | Model/route | Role |
|---:|---|---|
| 0 | `present_nibble_invp_only_spn_only` | current supported InvP anchor |
| 1 | `candidate_trail_consistency_linear` | feature-only linear diagnostic |
| 2 | `candidate_trail_consistency_mlp` | small nonlinear feature diagnostic |

Optional control if implementation cost is small:

| Row | Model/route | Role |
|---:|---|---|
| 3 | `candidate_trail_consistency_shuffled_cells` | false active-position/control alignment |

Do not include Zhang/Wang baseline in the first matrix unless a protocol audit
requires it. Compare against the current same-scale InvP anchor and previously
retrieved Zhang/Wang context.

## Gates

Primary metric:

```text
val_auc
```

Continue only if:

```text
best candidate-trail route >= InvP-only same-scale anchor + 0.001 AUC
and calibrated_accuracy is not worse than InvP-only
```

Weak continue:

```text
best candidate-trail route is best but margin < 0.001 AUC
```

Action:

```text
run a second 262144/class seed before any 1M scale
```

Stop:

```text
candidate-trail route <= InvP-only
or shuffled/false-alignment control matches the true route
```

Action:

```text
record as diagnostic/negative evidence and do not scale this branch
```

## Implementation Readiness Checklist

Before any remote launch:

```text
1. Add a tiny local smoke plan under configs/experiment/innovation1/.
2. Ensure disk-backed feature cache writes features.npy / labels.npy or
   equivalent chunks plus metadata and progress JSONL.
3. Ensure parameter-matched cache reuse/resume behavior.
4. Add unit tests for deterministic features, cache reuse, and protocol lock.
5. Add postprocess/gate script before remote launch.
6. Add remote config and generated cmd/monitor scripts only after smoke passes.
7. Run scripts/check-remote-readiness before launch.
8. Commit and push before launching from the pushed commit.
```

Completed local foundation:

```text
config = configs/experiment/innovation1/innovation1_spn_present_candidate_trail_consistency_smoke.json

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/spn-candidate-evidence \
  --config configs/experiment/innovation1/innovation1_spn_present_candidate_trail_consistency_smoke.json

result:
  sample_structure = zhang_wang_case2_official_mcnd
  negative_mode = encrypted_random_plaintexts
  key_rotation_interval = 0
  route/model = candidate_trail_consistency_linear or candidate_trail_consistency_mlp
  training_model = linear or mlp
```

The smoke config is a plan-driven readiness check only. It uses tiny
`samples_per_class=2`, `pairs_per_sample=1`, and `epochs=1`; it is not accuracy
evidence and must not be compared against InvP/DDT results.

Prepared conditional remote smoke config:

```text
remote_config = configs/remote/innovation1_spn_present_candidate_trail_consistency_smoke_gpu1_20260701.json
status = readiness asset only / do not launch while topology-aware route is active
purpose = verify future candidate-trail remote launch plumbing can reference the
          JSON smoke plan and G:\lxy feature cache under the current protocol
```

This config is not a medium diagnostic. It remains a conditional smoke asset
until the topology-aware branch gate selects candidate-trail as the next route.

Cell-structured control update, 2026-07-01:

```text
Local cell-structured candidate-trail features are implemented for smoke and
diagnostic preparation. The default candidate-trail smoke now uses
feature_mode = cell_structured, and the shuffled-cell control uses
feature_mode = cell_structured_shuffled with a fixed deterministic cell
permutation.
```

Completed local foundation:

```text
1. `feature_mode = cell_structured` exposes a stable PRESENT layer/cell feature
   axis before pair-set aggregation.
2. `feature_mode = cell_structured_shuffled` applies a fixed shuffled cell
   permutation for `candidate_trail_consistency_shuffled_cells`.
3. Local tests prove true and shuffled feature tensors have the same shape but
   distinct values while preserving the official protocol fields.
4. CLI smoke supports `--model shuffled_cells --feature-mode
   cell_structured_shuffled` and emits the gate-aligned result key
   `candidate_trail_consistency_shuffled_cells`.
```

First medium diagnostic launch assets, 2026-07-02:

```text
run_id = i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702
plan = configs/experiment/innovation1/innovation1_spn_present_candidate_trail_consistency_r7_262k_seed0.json
remote_config = configs/remote/innovation1_spn_present_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702.json
launcher = configs/remote/generated/run_i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702.cmd
monitor = configs/remote/generated/monitor_i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702.sh
expected_rows = 4
scale = 262144/class
claim_scope = medium diagnostic only
```

The matrix compares the completed same-scale InvP-only anchor from the
topology seed1 run against candidate-trail linear/MLP rows and the shuffled-cell
control. This is acceptable because the research question is whether
candidate-trail / transition consistency adds signal beyond the current
same-scale InvP representation; it is not a fresh Zhang/Wang reproduction.

Local smoke update, 2026-07-02:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/spn-candidate-evidence \
  --config configs/experiment/innovation1/innovation1_spn_present_candidate_trail_consistency_smoke.json \
  --output /tmp/i1_candidate_trail_smoke.jsonl

result = pass
output contract now includes metrics.auc, metrics.calibrated_accuracy,
top-level auc, accuracy, calibrated_accuracy, and selected_model for
postprocess/gate compatibility.
```

Gate tooling update, 2026-07-01:

```text
script = scripts/gate-candidate-trail
postprocess = scripts/postprocess-candidate-trail
module = src/blockcipher_nd/planning/candidate_trail_gate.py
postprocess_module = src/blockcipher_nd/planning/candidate_trail_postprocess.py
purpose = compare candidate_trail_consistency_linear/mlp against
          present_nibble_invp_only_spn_only and optional shuffled-cell control
decisions =
  support_candidate_trail_route
  weak_candidate_trail_signal
  stop_candidate_trail_route
```

The gate and postprocess wrappers are local tooling only. The smoke plan and
conditional remote-smoke readiness asset exist and pass local checks, but they
do not make the route medium-scale launchable. The 262144/class plan, remote
config, generated launcher, and monitor should be created only after the
topology-aware branch gate selects candidate-trail as the next route.

Output-contract update, 2026-07-01:

```text
scripts/spn-candidate-evidence now writes gate-aligned result keys:
  model = candidate_trail_consistency_linear for --model linear
  model = candidate_trail_consistency_mlp    for --model mlp

The original learner type remains visible as:
  training_model = linear or mlp
```

This prevents a retrieved candidate-trail JSONL from passing training but later
failing local gate/postprocess with `missing_candidate_models`.

Matrix runner update, 2026-07-01:

```text
script = scripts/spn-candidate-evidence-matrix
remote_config runner_script = scripts/spn-candidate-evidence-matrix
plan shape = JSON object with common + rows
purpose = produce one candidate-trail JSONL row per planned row
supported rows =
  external_anchor -> inject completed InvP-only same-scale metrics
  candidate/model=linear -> candidate_trail_consistency_linear
  candidate/model=mlp -> candidate_trail_consistency_mlp
  candidate/model=shuffled_cells -> candidate_trail_consistency_shuffled_cells
```

This closes the 262144/class preparation gap: the future medium diagnostic can
emit the 4 rows expected by `scripts/postprocess-candidate-trail` without
manually concatenating per-model outputs. It still does not authorize launching
the medium remote run while the topology-aware branch is active.

Readiness rule:

```text
candidate-trail JSON matrix remote configs must explicitly set
runner_script = scripts/spn-candidate-evidence-matrix
```

This prevents a future medium launcher from accidentally calling the generic
`scripts/train` entrypoint against a route-specific JSON matrix.

## Claim Scope

Until at least `1000000/class` multi-seed evidence exists, this route can only
support one of these statements:

```text
candidate-trail consistency smoke passed
candidate-trail consistency medium diagnostic positive
candidate-trail consistency tied with InvP-only
candidate-trail consistency negative under official Case2 protocol
```

It must not be described as:

```text
formal route evidence
breakthrough
SOTA
proof that trail features solve PRESENT r7
```

## Relation To Other Routes

This route is a data/feature representation branch. It should be interpreted
beside, not instead of, the structure-network branches:

```text
InvP-only                -> structure-aligned data representation
DDT graph                -> explicit S-box prior + topology network
topology-aware network   -> true P-layer message passing over InvP cells
pair-set aggregation     -> cross-pair evidence attribution
candidate-trail route    -> transition-consistency feature attribution
```

The useful paper direction is whichever route survives same-protocol scale,
seed, and attribution gates most cleanly.
