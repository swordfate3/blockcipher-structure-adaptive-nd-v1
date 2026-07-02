# Innovation 1 SPN Topology-Aware Network Conditional Plan

**Date:** 2026-07-01

**Status:** active route / seed0 weak signal retrieved / seed1 relaunch prepared

**Scope:** PRESENT-80 r7, Zhang/Wang 2022 Case2 `m=16`, strict
encrypted-random-plaintext negatives. This plan is prepared as the next
network-architecture route activated after the DDT graph seed1 variance check
returned a stable weak diagnostic signal rather than stable support.

## Why This Plan Exists

The current strongest completed Innovation 1 route is:

```text
model = present_nibble_invp_only_spn_only
scale = 1000000/class
seed0 AUC = 0.797470988906
seed1 AUC = 0.797347588554
claim_scope = two-seed paper-scale positive confirmation, not formal evidence
```

The completed paper-scale attribution controls support the explanation that
true `InvP(DeltaC)` / P-layer alignment carries useful SPN structure:

```text
decision = support_invp_structural_attribution
attribution_margin = 0.003726063600
```

The DDT graph branch tested whether adding S-box DDT priors and a topology
graph improves beyond this InvP-only anchor. Seed0 and seed1 were both weak
diagnostic signals:

```text
run_id = i1_spn_ddt_graph_r7_262k_seed0_gpu0_20260630
decision = weak_ddt_graph_signal
margin_vs_same_graph_no_ddt_auc = 0.000472726912
required_margin = 0.001

run_id = i1_spn_ddt_graph_r7_262k_seed1_gpu1_20260630
decision = weak_ddt_graph_signal
margin_vs_same_graph_no_ddt_auc = 0.000626281631
required_margin = 0.001

manual branch decision = do not promote DDT graph to 1M yet; activate
topology-aware network route
```

Because seed1 did not turn DDT graph into stable support, the next useful
question is not "add more hand-crafted DDT features". It is:

```text
Can the network architecture itself make better use of the already-supported
InvP/P-layer structure?
```

## Research Question

Can an explicit PRESENT topology-aware network, using the same InvP-only
structural input, improve over the current InvP-only token-mixer anchor?

More concretely:

```text
Does fixed PRESENT P-layer message passing over 16 S-box/nibble cells add a
real structure prior beyond the same input processed by a generic token mixer?
```

## Single Hypothesis

Change exactly one factor:

```text
network topology = generic token mixing -> fixed PRESENT P-layer message passing
```

Keep unchanged:

```text
cipher = PRESENT-80
rounds = 7
sample_structure = zhang_wang_case2_official_mcnd
negative_mode = encrypted_random_plaintexts
pairs_per_sample = 16
feature family = InvP(DeltaC)-only nibble/cell view
loss = mse
optimizer = adam
lr_scheduler = official_cyclic
checkpoint_metric = val_auc
validation key = 0x11111111111111111111
```

Do not add in this first topology-aware network route:

```text
raw Zhang/Wang MCND branch
DDT priors
candidate trail features
new negative samples
new sample structure
new pair-set aggregation objective
multi-task auxiliary loss
```

## Candidate Model Sketch

Proposed model key:

```text
present_nibble_invp_p_layer_graph_spn_only
```

Input:

```text
raw ciphertext pairs -> DeltaC -> InvP(DeltaC)
```

Per pair:

```text
nodes = 16 PRESENT S-box cells
node feature = 4-bit InvP(DeltaC) nibble
edge/message source = fixed PRESENT P-layer nibble adjacency
```

Network:

```text
shared nibble encoder
2-3 fixed P-layer message-passing blocks
pair-level embeddings
same evidence pooling family as current InvP route
binary classifier
```

Required topology control:

```text
present_nibble_invp_shuffled_p_layer_graph_spn_only
```

This control must keep the same input and model budget but replace true P-layer
adjacency with deterministic shuffled adjacency. No topology claim is allowed
without this control.

Implementation note:

```text
src/blockcipher_nd/models/structure/spn/present_p_layer_mixer.py already has
PresentPLayerMixerBlock and PRESENT nibble adjacency logic. Reuse this instead
of creating a second graph convention.
```

## First Matrix

First non-smoke scale:

```text
run_id = i1_spn_topology_aware_network_r7_262k_seed0_gpu0_20260701
262144/class
seed = 0
```

Keep the matrix lean:

| Row | Model | Role |
|---:|---|---|
| 0 | `present_nibble_invp_only_spn_only` | current strongest same-input anchor |
| 1 | `present_nibble_invp_p_layer_graph_spn_only` | true-P topology candidate |
| 2 | `present_nibble_invp_shuffled_p_layer_graph_spn_only` | shuffled topology control |

Do not include Zhang/Wang in this matrix. The question is not whether InvP is
better than raw MCND; that is already supported. The question is whether
topology-aware message passing improves the already-supported InvP route.

## Gates

Primary metric:

```text
val_auc
```

Support:

```text
true-P graph AUC >= InvP-only anchor AUC + 0.001
and true-P graph AUC >= shuffled-P graph AUC + 0.001
and calibrated_accuracy is not worse than InvP-only
```

Weak signal:

```text
true-P graph is best but one required margin is < 0.001
```

Action:

```text
run 262144/class seed1 variance check before any 1M run
```

Stop:

```text
true-P graph <= InvP-only anchor
or true-P graph <= shuffled-P graph
```

Action:

```text
record as tied/negative topology evidence; do not scale this architecture.
Switch to candidate-trail / transition-consistency representation or a new
data-structure hypothesis.
```

## Branch Rules

This plan became actionable after the DDT seed1 result was retrieved,
validated, postprocessed, and manually decided as weak DDT graph signal.

If DDT seed1 gives stable support:

```text
Do not launch this route immediately.
First decide whether to run a lean 1M DDT confirmation matrix.
```

If DDT seed1 is weak, tied, negative, or unstable:

```text
[x] Activate this topology-aware network route.
[x] Implement the model and shuffled control.
[x] Run smoke/readiness.
[x] Commit and push.
[x] Launch 262144/class seed0 from the pushed commit.
[x] Hand off retrieval/postprocess to tmux watcher.
```

Seed0 result and seed1 relaunch update:

```text
[x] seed0 retry was retrieved, validated, and postprocessed.
[x] seed0 decision = weak_topology_aware_network_signal.
[x] action = run_262k_seed1_variance_check_before_scaling.
[x] previous seed1 accelerator attempt failed because AMP/BF16 hit an EvidencePooling scatter dtype bug.
[x] bug fixed in commit cb83eff; this relaunch uses the normal FP32 training path, not amp-bf16, to keep the variance check comparable to seed0.
[x] dataset_cache_workers is updated from 4 to 8 for generation speed only, based on the completed cache-only Map/Reduce benchmark.
```

If DDT seed1 fails operationally:

```text
Repair or rerun DDT seed1 first unless the failure proves the DDT route is not
actionable. Do not use an infrastructure failure as evidence against DDT.
```

## Local Readiness Checklist

Before any meaningful remote launch:

```text
1. Add model classes and registry keys.
2. Add forward-shape tests for candidate and shuffled control.
3. Add smoke CSV with tiny samples_per_class.
4. Run CPU smoke through scripts/train.
5. Add 262144/class CSV only after smoke passes.
6. Add remote config with disk dataset cache under G:\lxy.
7. Run scripts/check-remote-readiness.
8. Commit and push before remote launch.
```

Current implementation artifacts:

```text
model keys:
- present_nibble_invp_p_layer_graph_spn_only
- present_nibble_invp_shuffled_p_layer_graph_spn_only

smoke plan:
configs/experiment/innovation1/innovation1_spn_present_topology_aware_network_smoke.csv

medium plan:
configs/experiment/innovation1/innovation1_spn_present_topology_aware_network_r7_262k.csv
```

Current launch record:

```text
run_id = i1_spn_topology_aware_network_r7_262k_seed0_gpu0_20260701
remote_config = configs/remote/innovation1_spn_present_topology_aware_network_r7_262k_gpu0_20260701.json
status = launch_stalled / operational failure / no model evidence
local_monitor = outputs/remote_results/i1_spn_topology_aware_network_r7_262k_seed0_gpu0_20260701/monitor/monitor.log
claim_scope = medium diagnostic only
```

Operational note, 2026-07-01:

```text
The first seed0 launch stalled before training. Local monitor-health now reports
status = launch_stalled with launch_state.reason =
torch_info_empty_before_git_or_training. A read-only remote check found only:
  logs/*gpu_info.txt
  logs/*launch_env.txt
  logs/*torch_info.txt        size 0
  logs/*torch_info_stderr.txt size 0
  empty results/
No started marker, git artifact, progress JSONL, stdout/stderr, done marker,
failed marker, or matching python.exe process was present.

Interpretation:
  This is an infrastructure/launcher failure, not training evidence and not a
  result for or against the topology-aware network route.
```

Retry launch record:

```text
run_id = i1_spn_topology_aware_network_r7_262k_seed0_gpu0_retry1_20260701
remote_config = configs/remote/innovation1_spn_present_topology_aware_network_r7_262k_gpu0_retry1_20260701.json
status = prepared for relaunch from pushed commit after launch_stalled diagnosis
local_monitor = outputs/remote_results/i1_spn_topology_aware_network_r7_262k_seed0_gpu0_retry1_20260701/monitor/monitor.log
claim_scope = medium diagnostic retry only; same protocol and matrix as the original seed0 plan
```

Prepared conditional seed1 assets:

```text
plan = configs/experiment/innovation1/innovation1_spn_present_topology_aware_network_r7_262k_seed1.csv
remote_config = configs/remote/innovation1_spn_present_topology_aware_network_r7_262k_seed1_gpu1_20260701.json
launcher = configs/remote/generated/run_i1_spn_topology_aware_network_r7_262k_seed1_gpu1_20260701.cmd
monitor = configs/remote/generated/monitor_i1_spn_topology_aware_network_r7_262k_seed1_gpu1_20260701.sh
launch_condition = seed0 gate is support_topology_aware_network_route or weak_topology_aware_network_signal
current_status = prepared only / not launched
```

Postprocess automation:

```text
script = scripts/postprocess-topology-aware-result
module = src/blockcipher_nd/planning/topology_aware_postprocess.py
summary_artifact = outputs/remote_results/<run_id>/<run_id>_postprocess_summary.json
next_action_artifact = outputs/remote_results/<run_id>/<run_id>_next_action_readiness.json
```

The postprocess summary now emits a structured `next_action` in addition to
human-readable next steps:

| Gate decision | Next action branch | Remote launch? | Meaning |
|---|---|---:|---|
| `support_topology_aware_network_route` | `topology_aware_seed1_confirmation` | yes | launch prepared seed1 as 262144/class confirmation |
| `weak_topology_aware_network_signal` | `topology_aware_seed1_variance_check` | yes | launch prepared seed1 as 262144/class variance check |
| `stop_topology_aware_network_route` | `candidate_trail_consistency` | no | stop this architecture and switch to the data/feature representation plan |

The `*_next_action_readiness.json` artifact validates the selected remote
config when a seed1 launch is requested. If the branch is
`candidate_trail_consistency`, it intentionally does not launch a remote job;
the next step is to create the candidate-trail medium plan/config from
`docs/experiments/innovation1-candidate-trail-consistency-plan.md` only after
recording the topology-aware result.

## Claim Scope

Allowed if 262144/class seed0 passes support gate:

```text
medium diagnostic evidence that fixed PRESENT P-layer message passing may add
value over the current InvP-only token-mixer route.
```

Not allowed:

```text
breakthrough
SOTA
formal route evidence
paper-ready proof
claim that SPN topology-aware networks are generally superior
```

Formal route evidence would still require at least:

```text
1000000/class
multiple seeds
same-protocol baseline and InvP anchor
true-P vs shuffled-P attribution
complete artifacts and documented gates
```

## Retrieved Topology-Aware Network Result

<!-- topology-aware-postprocess:i1_spn_topology_aware_network_r7_262k_seed0_gpu0_retry1_20260701:start -->
### i1_spn_topology_aware_network_r7_262k_seed0_gpu0_retry1_20260701 Topology-Aware Network Result

| Field | Value |
|---|---|
| Run ID | `i1_spn_topology_aware_network_r7_262k_seed0_gpu0_retry1_20260701` |
| Postprocess status | `pass` |
| Validation status | `pass` |
| Topology-aware gate status | `pass` |
| Decision | `weak_topology_aware_network_signal` |
| Action | `run_262k_seed1_variance_check_before_scaling` |
| Interpretation | `true-P graph is best but below at least one required margin; treat as weak diagnostic signal` |
| Margin vs InvP AUC | `0.001328941929` |
| Margin vs shuffled AUC | `0.000486004283` |
| Calibrated delta vs InvP | `0.000713348389` |
| Required margin | `0.001000000000` |
| Next action branch | `topology_aware_seed1_variance_check` |
| Next action should launch remote | `True` |
| Next action launch config | `configs/remote/innovation1_spn_present_topology_aware_network_r7_262k_seed1_gpu1_20260701.json` |
| Next action readiness command | `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness --config configs/remote/innovation1_spn_present_topology_aware_network_r7_262k_seed1_gpu1_20260701.json` |
| Next action run id | `i1_spn_topology_aware_network_r7_262k_seed1_gpu1_20260701` |
| Next steps | `Record this as weak topology-aware network signal.; Run the remote readiness gate: UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness --config configs/remote/innovation1_spn_present_topology_aware_network_r7_262k_seed1_gpu1_20260701.json; Launch configs/remote/innovation1_spn_present_topology_aware_network_r7_262k_seed1_gpu1_20260701.json as a 262144/class seed1 variance check from the pushed commit.; Hand off seed1 monitoring and retrieval to a local tmux watcher or sub-agent.; Keep claim scope diagnostic only.` |
| Claim scope | `262144/class medium diagnostic topology-aware network gate; not paper-scale, formal, or breakthrough evidence` |
| Results JSONL | `outputs/remote_results/i1_spn_topology_aware_network_r7_262k_seed0_gpu0_retry1_20260701/results/i1_spn_topology_aware_network_r7_262k_seed0_gpu0_retry1_20260701.jsonl` |
| Validation report | `outputs/remote_results/i1_spn_topology_aware_network_r7_262k_seed0_gpu0_retry1_20260701/i1_spn_topology_aware_network_r7_262k_seed0_gpu0_retry1_20260701_local_result_gate.json` |
| Topology-aware gate | `outputs/remote_results/i1_spn_topology_aware_network_r7_262k_seed0_gpu0_retry1_20260701/i1_spn_topology_aware_network_r7_262k_seed0_gpu0_retry1_20260701_topology_aware_gate.json` |
| Curves | `outputs/remote_results/i1_spn_topology_aware_network_r7_262k_seed0_gpu0_retry1_20260701/i1_spn_topology_aware_network_r7_262k_seed0_gpu0_retry1_20260701_curves.svg` |
| History CSV | `outputs/remote_results/i1_spn_topology_aware_network_r7_262k_seed0_gpu0_retry1_20260701/i1_spn_topology_aware_network_r7_262k_seed0_gpu0_retry1_20260701_history.csv` |
| Summary JSON | `outputs/remote_results/i1_spn_topology_aware_network_r7_262k_seed0_gpu0_retry1_20260701/i1_spn_topology_aware_network_r7_262k_seed0_gpu0_retry1_20260701_postprocess_summary.json` |
| Summary Markdown | `outputs/remote_results/i1_spn_topology_aware_network_r7_262k_seed0_gpu0_retry1_20260701/i1_spn_topology_aware_network_r7_262k_seed0_gpu0_retry1_20260701_postprocess_summary.md` |
| Next action readiness | `outputs/remote_results/i1_spn_topology_aware_network_r7_262k_seed0_gpu0_retry1_20260701/i1_spn_topology_aware_network_r7_262k_seed0_gpu0_retry1_20260701_next_action_readiness.json` |

Model rows:

| Model | AUC | Calibrated Accuracy |
|---|---:|---:|
| `present_nibble_invp_only_spn_only` | `0.792120689293` | `0.717407226562` |
| `present_nibble_invp_p_layer_graph_spn_only` | `0.793449631223` | `0.718120574951` |
| `present_nibble_invp_shuffled_p_layer_graph_spn_only` | `0.792963626940` | `0.717285156250` |
<!-- topology-aware-postprocess:i1_spn_topology_aware_network_r7_262k_seed0_gpu0_retry1_20260701:end -->
