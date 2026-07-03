# Innovation 1 S-box Transition Prior Gate Plan

**Status:** planned next architecture/data-representation route / do not launch
while trail-family seed0 is running.

**Scope:** PRESENT-80 r7, Zhang/Wang 2022 Case2 `m=16`, strict
encrypted-random-plaintext negatives.

This is a current experiment plan under `docs/experiments/`, not a historical
agent plan. It exists so that if trail-family and pair-set controls do not
produce a clean next route, the project has a sharper SPN-specific hypothesis
than "try another generic graph network."

## Research Question

Holding cipher, rounds, sample structure, negative mode, train/validation keys,
scale, checkpoint metric, and primary metric fixed:

```text
Can a PRESENT InvP-only distinguisher improve if S-box DDT transition priors are
used as an explicit gate over cell evidence, rather than only appended as raw
features or encoded by a generic P-layer graph?
```

## Why This Route Exists

Completed evidence says:

```text
InvP/P-layer aligned view:
  stable positive 1M/class two-seed evidence with attribution controls.

DDT graph:
  two 262144/class seeds are weak positive only; DDT priors are consistently
  helpful but below the +0.001 AUC gate against same-graph no-DDT controls.

Topology-aware P-layer graph:
  seed1 stopped the route because true-P topology did not beat InvP-only or the
  shuffled-P control.

Candidate-trail and bit-transition spectrum:
  stopped as standalone compressed feature routes.

Trail-family:
  currently running as the active data-representation branch.
```

Interpretation:

```text
The useful signal is likely not "DDT as another feature column" or "P-layer
message passing by itself." The stronger hypothesis is that local S-box
transition likelihood should modulate which InvP cells and pair evidence the
classifier trusts.
```

This route therefore tests a different mechanism:

```text
S-box transition prior = gate / attention / reliability weight
```

not:

```text
S-box transition prior = extra flat feature only
```

## Trigger

Do not launch this route while:

```text
i1_trail_family_r7_262k_seed0_gpu1_20260702
```

is running or not yet postprocessed.

This plan becomes actionable only if one of these is true:

```text
1. trail-family seed0 gates stop/tied/negative and pair-set aggregation does
   not become the selected next attribution control.

2. pair-set aggregation shows the multi-pair signal is mostly frozen
   single-pair score aggregation, so the next useful innovation must change
   local cell evidence weighting rather than pair-set pooling.

3. User explicitly selects the S-box transition prior gate route.
```

If trail-family gates support/weak-positive, run its seed1 confirmation first.
If active-auxiliary becomes the selected route, run its seed0 before this plan.

## Single Hypothesis

Only one factor changes in the first experiment:

```text
model mechanism = InvP-only SPN encoder + DDT-derived per-cell evidence gate
```

Everything else must stay fixed:

```text
cipher, rounds, sample_structure, negative_mode, train key, validation key,
sample scale, scheduler, checkpoint metric, metric computation, and validation
split.
```

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
| Learning rate | `0.0001` |
| Max learning rate | `0.002` |
| Checkpoint metric | `val_auc` |
| Early stopping | patience `8`, min delta `0.0001` |
| Primary metric | `val_auc`, then calibrated accuracy |
| First meaningful scale | `262144/class` |
| Evidence level | medium diagnostic only |
| Cache | disk-backed feature or dataset cache with progress and parameter-matched reuse |

## Candidate Design

Base view:

```text
InvP(DeltaC) nibble/cell tokens, same as present_nibble_invp_only_spn_only.
```

Prior source:

```text
For each output-difference cell, compute DDT-derived statistics:
  max DDT count / probability over possible input differences
  number of legal input differences
  normalized entropy over legal input differences
  high-probability transition flag
  active/inactive flag
```

Gate:

```text
gate_i = sigmoid(MLP([cell_embedding_i, ddt_prior_i]))
cell_embedding_i' = cell_embedding_i * (1 + alpha * gate_i)
```

or an equivalent residual reliability gate. The gate should modulate cell
evidence before sample-level pooling. It must not use labels, validation
statistics, real key material, or guessed-key information.

Controls:

```text
no_ddt_gate:
  same InvP cell encoder and gating capacity, but DDT prior inputs zeroed or
  replaced by active/count-only features.

shuffled_ddt_prior:
  DDT prior vectors deterministically permuted across cells or DDT rows while
  preserving marginal ranges.

invp_anchor:
  same-scale present_nibble_invp_only_spn_only anchor row.
```

The control question is not "can a bigger model do better." It is:

```text
Does a real S-box transition prior gate beat the same-capacity gate without the
true transition prior?
```

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
| 1 | `present_nibble_invp_sbox_prior_gate` | true DDT-prior gate candidate |
| 2 | `present_nibble_invp_no_ddt_gate` | same-capacity no-DDT gate control |
| 3 | `present_nibble_invp_shuffled_sbox_prior_gate` | shuffled-prior control |

Do not include Zhang/Wang, active-aux, trail-family, pair-set, and topology
rows in the first matrix. This matrix asks only whether transition-prior gating
adds signal over InvP-only and same-capacity controls.

## Gates

Support:

```text
true_prior_gate_auc >= InvP-only anchor AUC + 0.001
and true_prior_gate_auc >= max(no_ddt_gate_auc, shuffled_prior_gate_auc) + 0.001
and calibrated_accuracy is not worse than InvP-only
```

Action:

```text
prepare and launch 262144/class seed1 confirmation before any 1M run.
```

Weak:

```text
true_prior_gate is best but one required margin is positive and < 0.001.
```

Action:

```text
run seed1 only if this is the best remaining SPN structure hypothesis after
trail-family, pair-set aggregation, and active-auxiliary gates.
```

Stop:

```text
true_prior_gate <= InvP-only anchor
or true_prior_gate <= no_ddt_gate
or true_prior_gate <= shuffled_prior_gate
or calibrated_accuracy regresses materially.
```

Action:

```text
Do not scale this route. Treat S-box transition priors as weak diagnostic
context and return to InvP-only route consolidation, cross-cipher transfer, or
a new SPN data representation hypothesis.
```

## Implementation Tasks

Task 1: prior builder.

```text
Add deterministic PRESENT DDT prior features derived from public DeltaC /
InvP(DeltaC), with tests for shape, determinism, no label use, and shuffled
control behavior.
```

Task 2: model variant.

```text
Add an opt-in InvP-only SPN model variant with a residual per-cell prior gate.
Keep existing InvP-only model outputs untouched.
```

Task 3: smoke config.

```text
Create a tiny smoke matrix with InvP anchor, true prior gate, no-DDT gate, and
shuffled-prior gate.
```

Task 4: medium readiness assets.

```text
After smoke passes, prepare 262144/class seed0 remote config with disk-backed
cache/progress under G:\lxy and watcher-owned retrieval/postprocess.
```

Task 5: result gate and postprocess.

```text
Add route-specific gate/postprocess artifacts so watcher retrieval can
automatically validate result rows, compare true prior against InvP/no-DDT/
shuffled controls, update this plan, and emit a next-action readiness report.
```

## Implementation Status

Status as of 2026-07-03:

```text
implementation_status = local model aliases, smoke config, gate/postprocess,
  and 262144/class seed0 remote assets implemented
evidence_level = smoke only; no model-quality claim
smoke_config = configs/experiment/innovation1/innovation1_spn_present_sbox_transition_prior_gate_smoke.csv
medium_seed0_plan = configs/experiment/innovation1/innovation1_spn_present_sbox_transition_prior_gate_r7_262k_seed0.csv
medium_seed0_remote_config = configs/remote/innovation1_spn_present_sbox_transition_prior_gate_r7_262k_seed0_gpu1_20260703.json
medium_seed0_launcher = configs/remote/generated/run_i1_sbox_prior_gate_r7_262k_seed0_gpu1_20260703.cmd
medium_seed0_monitor = configs/remote/generated/monitor_i1_sbox_prior_gate_r7_262k_seed0_gpu1_20260703.sh
medium_seed1_plan = configs/experiment/innovation1/innovation1_spn_present_sbox_transition_prior_gate_r7_262k_seed1.csv
medium_seed1_remote_config = configs/remote/innovation1_spn_present_sbox_transition_prior_gate_r7_262k_seed1_gpu1_20260703.json
medium_seed1_launcher = configs/remote/generated/run_i1_sbox_prior_gate_r7_262k_seed1_gpu1_20260703.cmd
medium_seed1_monitor = configs/remote/generated/monitor_i1_sbox_prior_gate_r7_262k_seed1_gpu1_20260703.sh
gate_script = scripts/gate-sbox-prior
postprocess_script = scripts/postprocess-sbox-prior
implemented_model_aliases =
  present_nibble_invp_sbox_prior_gate
  present_nibble_invp_no_ddt_gate
  present_nibble_invp_shuffled_sbox_prior_gate
remote_config_status = prepared but deferred; do not launch while trail-family is running
```

Implemented local controls:

| Route | Role |
|---|---|
| `present_nibble_invp_only_spn_only` | strongest same-scale anchor placeholder for smoke wiring |
| `present_nibble_invp_sbox_prior_gate` | true S-box DDT transition prior gate |
| `present_nibble_invp_no_ddt_gate` | same-capacity gate with DDT prior channels zeroed |
| `present_nibble_invp_shuffled_sbox_prior_gate` | deterministic shuffled-prior control |

The smoke path only checks model construction, forward/training wiring, and
official-protocol task parsing. It is not evidence that the prior gate improves
PRESENT r7.

Prepared medium seed0 matrix:

| Row | Model | Role |
|---:|---|---|
| 0 | `present_nibble_invp_only_spn_only` | same-run InvP anchor |
| 1 | `present_nibble_invp_sbox_prior_gate` | true S-box DDT transition prior gate |
| 2 | `present_nibble_invp_no_ddt_gate` | same-capacity no-DDT gate control |
| 3 | `present_nibble_invp_shuffled_sbox_prior_gate` | deterministic shuffled-prior control |

Prepared launch rule:

```text
Do not launch while i1_trail_family_r7_262k_seed0_gpu1_20260702 is still
running. If trail-family gates stop/tied/negative, or the user explicitly
selects this route, run readiness from the pushed commit and launch via the
generated cmd + local tmux watcher handoff.
```

Prepared seed1 confirmation rule:

```text
Do not launch seed1 until seed0 is retrieved, validated, plan-aligned, and its
S-box prior gate returns support_sbox_prior_route or weak_sbox_prior_signal.
Seed1 remains medium diagnostic confirmation/variance evidence, not formal
route evidence.
```

## Readiness Requirements

Before meaningful remote launch:

```text
1. This plan has the selected run_id and exact rows.
2. CPU smoke passes.
3. Unit tests cover DDT prior construction and shuffled/no-DDT controls.
4. scripts/check-remote-readiness passes.
5. Generated launcher uses cmd.exe /c and no project artifacts outside G:\lxy.
6. Local tmux watcher or sub-agent is ready to retrieve and postprocess.
7. Code/config/docs are committed and pushed.
```

Prepared readiness command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness \
  --config configs/remote/innovation1_spn_present_sbox_transition_prior_gate_r7_262k_seed0_gpu1_20260703.json
```

Prepared monitor-health command after launch:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/monitor-health \
  --root outputs/remote_results \
  --run-id i1_sbox_prior_gate_r7_262k_seed0_gpu1_20260703 \
  --tmux-session monitor_i1_sbox_prior_gate_r7_262k_seed0_gpu1_20260703 \
  --plan configs/experiment/innovation1/innovation1_spn_present_sbox_transition_prior_gate_r7_262k_seed0.csv \
  --plan-doc docs/experiments/innovation1-sbox-transition-prior-gate-plan.md \
  --expected-rows 4 \
  --postprocess-kind sbox_prior
```

Prepared seed1 readiness command after seed0 support/weak:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness \
  --config configs/remote/innovation1_spn_present_sbox_transition_prior_gate_r7_262k_seed1_gpu1_20260703.json
```

## Claim Scope

Allowed after a passing 262144/class seed0 gate:

```text
S-box transition prior gating shows medium diagnostic signal under the official
PRESENT r7 Case2 protocol.
```

Not allowed:

```text
formal route evidence
proof that DDT priors solve PRESENT r7
breakthrough
SOTA
```

This route can only become route-level evidence after at least a positive
262144/class seed1 confirmation, and can only approach formal evidence after
`1000000/class` multi-seed, strict-negative, plan-aligned results.
