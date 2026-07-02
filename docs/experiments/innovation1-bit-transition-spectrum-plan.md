# Innovation 1 Bit-Level Transition Spectrum Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `skills/blockcipher-auto-research/SKILL.md` for evidence gates, and use Karpathy-style coding discipline for implementation. This is a current experiment plan under `docs/experiments/`, not a historical agent plan.

**Goal:** Prepare the next SPN/PRESENT structure-adaptive data route after the active candidate-trail diagnostic resolves, testing whether bit-level transition-spectrum features expose signal that cell-level candidate-trail summaries miss.

**Architecture:** Keep the official PRESENT-80 r7 Zhang/Wang Case2 protocol fixed. Build deterministic, cacheable transition-spectrum features from `DeltaC`, `InvP(DeltaC)`, active bit/cell masks, and P-layer bit-to-cell movement statistics; compare only against the strongest same-scale InvP-only anchor and a false-alignment control.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, project `scripts/` CLI wrappers, disk-backed feature cache, JSONL/SVG/CSV result artifacts, remote Windows A6000 under `G:\lxy`.

---

## Status

```text
status = conditional next-branch plan
do_not_launch_until = i1_candidate_trail_consistency_r7_262k_seed0_gpu1_20260702 is retrieved, validated, postprocessed, and gated
claim_scope = planned route only; no evidence yet
implementation_status = local route implemented; medium remote config intentionally not created until trigger
```

This plan is not a replacement for the active candidate-trail run. It is a prepared branch so the project can move immediately if candidate-trail is weak or negative.

## Trigger

Start this route only if one of these is true:

```text
1. candidate-trail seed0 decision = stop_candidate_trail_route
2. candidate-trail seed0 decision = weak_candidate_trail_signal and manual branch choice prefers a new feature route over seed1 variance
3. candidate-trail seed0 support is blocked by shuffled-cell control matching or exceeding the true route
```

Do not start this route if:

```text
candidate-trail seed0 decision = support_candidate_trail_route
```

In that case, launch candidate-trail 262144/class seed1 first.

## Research Question

Holding cipher, rounds, sample structure, negative mode, train/validation keys, scale, checkpoint metric, and primary metric fixed:

```text
Do bit-level P-layer transition-spectrum features reveal SPN differential
propagation structure beyond InvP-only and cell-level candidate-trail summaries?
```

## Single Hypothesis

The current strongest positive route shows that `InvP(DeltaC)` is useful. The topology-aware network route showed that a coarse fixed P-layer graph does not automatically improve results. The next plausible gap is therefore bit-level movement information:

```text
real differential samples should have a different distribution of bit movement,
active-bit concentration, active-cell fan-in/fan-out, and cross-pair transition
stability than encrypted-random-plaintext negatives.
```

This route changes only the feature representation:

```text
feature route = bit-level transition spectrum
```

It must not change negative samples, validation key, sample structure, metric, checkpoint selection, or train/validation split logic.

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

For each ciphertext pair:

```text
DeltaC = C xor C'
DeltaV = InvP(DeltaC)
active_bit_c = bit mask of DeltaC
active_bit_v = bit mask of DeltaV
active_cell_c = nibble active mask of DeltaC
active_cell_v = nibble active mask of DeltaV
```

P-layer transition spectrum features:

```text
bit active count before/after InvP
nibble active count before/after InvP
per-source-cell outgoing active-bit count
per-target-cell incoming active-bit count
source-target cell transition count matrix, flattened 16x16 or compressed
fan-in/fan-out entropy
top-k active transition concentration
transition overlap between DeltaC and InvP(DeltaC)
pair-to-pair mean/std/span over 16 pairs
```

False-alignment control:

```text
replace true PRESENT InvP mapping with fixed deterministic shuffled bit mapping
same feature dimensions
same model family
same training protocol
```

The first implementation should be a deterministic feature route, not a new large neural architecture.

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
| 1 | `bit_transition_spectrum_linear` | linear feature sufficiency diagnostic |
| 2 | `bit_transition_spectrum_mlp` | small nonlinear feature diagnostic |
| 3 | `bit_transition_spectrum_shuffled_p` | false P-layer alignment control |

Do not include Zhang/Wang baseline unless a protocol audit requires it. The relevant question is whether the new route beats the current best internal InvP-only anchor under the same scale.

## Gates

Primary metric:

```text
auc
```

Support:

```text
best true transition-spectrum route >= InvP-only same-scale anchor + 0.001 AUC
and calibrated_accuracy is not worse than InvP-only
and true transition-spectrum route >= shuffled-P control + 0.001 AUC
```

Action:

```text
launch 262144/class seed1 confirmation before any 1M run
```

Weak:

```text
best true transition-spectrum route is at or above InvP-only
but margin < 0.001 AUC
or true-vs-shuffled margin is positive but < 0.001 AUC
```

Action:

```text
run seed1 only if candidate-trail is also weak/negative and this route is the best available next feature hypothesis
```

Stop:

```text
true transition-spectrum route <= InvP-only
or calibrated_accuracy regresses
or shuffled-P control matches/exceeds true transition-spectrum route
```

Action:

```text
do not scale this feature route
switch to trail-family consistency, active-pattern auxiliary head, or cross-cipher GIFT/SKINNY transfer planning
```

## Implementation Plan

### Task 1: Add Feature Extractor

Status: completed.

**Files:**

```text
create src/blockcipher_nd/features/spn_transition_spectrum.py
modify tests/test_project_structure.py
```

Required API:

```python
def present_bit_transition_spectrum_features(
    pairs: list[tuple[int, int]],
    *,
    width: int,
    shuffled: bool = False,
    shuffle_seed: int = 20260702,
) -> np.ndarray:
    ...
```

Test requirements:

```text
shape is stable for PRESENT width 64
true and shuffled features have identical shape
true and shuffled features differ on a nontrivial pair set
features are finite float32
empty pair list raises ValueError
```

### Task 2: Add Dataset/CLI Route

Status: completed.

**Files:**

```text
create src/blockcipher_nd/tasks/innovation1/spn_transition_spectrum.py
create src/blockcipher_nd/cli/spn_transition_spectrum_matrix.py
create scripts/spn-transition-spectrum-matrix
modify tests/test_project_structure.py
```

Required behavior:

```text
read JSON matrix config
generate/reuse disk-backed transition-spectrum features
emit one JSONL row per matrix row
include metrics.auc, metrics.calibrated_accuracy, auc, accuracy, calibrated_accuracy
record feature_cache_workers, feature_cache_root, feature_cache_chunk_size
stream feature-cache rows directly into features.npy / labels.npy memmap before flush
```

### Task 3: Add Smoke Config And Gate

Status: completed.

**Files:**

```text
create configs/experiment/innovation1/innovation1_spn_present_bit_transition_spectrum_smoke.json
create src/blockcipher_nd/planning/transition_spectrum_gate.py
create scripts/gate-transition-spectrum
modify tests/test_project_structure.py
```

Smoke config:

```text
samples_per_class = 2
pairs_per_sample = 1
epochs = 1
negative_mode = encrypted_random_plaintexts
sample_structure = zhang_wang_case2_official_mcnd
validation_key = 0x11111111111111111111
key_rotation_interval = 0
```

### Task 4: Add Medium Plan/Remote Config Only After Trigger

Status: pending by design.

**Files:**

```text
create configs/experiment/innovation1/innovation1_spn_present_bit_transition_spectrum_r7_262k_seed0.json
create configs/remote/innovation1_spn_present_bit_transition_spectrum_r7_262k_seed0_gpu*_YYYYMMDD.json
modify docs/experiments/innovation1-bit-transition-spectrum-plan.md
```

Launch rule:

```text
do not create or launch the medium remote config until the active candidate-trail result is retrieved and its gate selects this branch
```

## Readiness Requirements

Before any meaningful remote launch:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_project_structure.py -k "transition_spectrum or candidate_trail or remote_readiness"

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness \
  --config configs/remote/<transition-spectrum-medium-config>.json
```

The readiness gate must enforce:

```text
runner_script = scripts/spn-transition-spectrum-matrix
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
local implementation complete for feature extraction, matrix runner, smoke config,
gate, postprocess, monitor-health integration, remote readiness gate, and
streamed disk-backed feature cache
postprocess writes <run_id>_next_action_readiness.json for the next branch handoff
no remote launch yet
waiting for candidate-trail seed0 gate
```

Implemented assets:

```text
src/blockcipher_nd/features/spn_transition_spectrum.py
src/blockcipher_nd/tasks/innovation1/spn_transition_spectrum.py
src/blockcipher_nd/cli/spn_transition_spectrum_matrix.py
src/blockcipher_nd/planning/transition_spectrum_gate.py
src/blockcipher_nd/planning/transition_spectrum_postprocess.py
scripts/spn-transition-spectrum-matrix
scripts/gate-transition-spectrum
scripts/postprocess-transition-spectrum
configs/experiment/innovation1/innovation1_spn_present_bit_transition_spectrum_smoke.json
```

Verification already exercised:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_project_structure.py -k "bit_transition_spectrum or transition_spectrum"
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/spn-transition-spectrum-matrix \
  --config configs/experiment/innovation1/innovation1_spn_present_bit_transition_spectrum_smoke.json \
  --output /tmp/transition_spectrum_stream_smoke.jsonl
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/gate-transition-spectrum \
  --results /tmp/transition_spectrum_stream_smoke.jsonl \
  --expected-rows 4
```

Do not create the `262144/class` medium plan or remote config until the
candidate-trail seed0 result is retrieved and its gate selects this branch.
