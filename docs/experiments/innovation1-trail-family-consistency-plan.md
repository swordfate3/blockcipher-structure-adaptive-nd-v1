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
implementation_status = local feature foundation implemented; CLI/gate/postprocess/remote not implemented
remote_config_status = do not create until trigger
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
```

Verification:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_project_structure.py -k "trail_family or bit_transition_spectrum"
```

This implementation is not result evidence and does not authorize launching the
trail-family medium run before the trigger conditions above are met.

### Task 2: Add Dataset/CLI Matrix Route

Status: pending by design.

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
create src/blockcipher_nd/tasks/innovation1/spn_trail_family.py
create src/blockcipher_nd/cli/spn_trail_family_matrix.py
create scripts/spn-trail-family-matrix
```

### Task 3: Add Smoke Config And Gate

Status: pending by design.

Required behavior:

```text
tiny smoke first; no accuracy claims
gate compares true route vs InvP anchor and false-family control
postprocess writes next_action_readiness.json
```

Suggested files:

```text
create configs/experiment/innovation1/innovation1_spn_present_trail_family_smoke.json
create src/blockcipher_nd/planning/trail_family_gate.py
create src/blockcipher_nd/planning/trail_family_postprocess.py
create scripts/gate-trail-family
create scripts/postprocess-trail-family
```

### Task 4: Add Medium Plan/Remote Config Only After Trigger

Status: pending by design.

Suggested files:

```text
create configs/experiment/innovation1/innovation1_spn_present_trail_family_r7_262k_seed0.json
create configs/remote/innovation1_spn_present_trail_family_r7_262k_seed0_gpu*_YYYYMMDD.json
modify docs/experiments/innovation1-trail-family-consistency-plan.md
```

Launch rule:

```text
do not create or launch the medium remote config until candidate-trail and
transition-spectrum gates select this branch
```

## Readiness Requirements

Before any meaningful remote launch:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_project_structure.py -k "trail_family or remote_readiness"

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness \
  --config configs/remote/<trail-family-medium-config>.json
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
local feature foundation implemented
no smoke config yet
no remote config yet
waiting for candidate-trail seed0 gate, then transition-spectrum gate if selected
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
