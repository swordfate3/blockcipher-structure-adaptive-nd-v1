# PRESENT Case 3 Three-Channel Topology-Residual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and locally adjudicate the frozen H2 hypothesis that the Liu Case 3 tensor `(C0, C1, InvP(C0 xor C1))` adds complementary signal to the existing PRESENT-80 r7 InvP Token-Mixer.

**Architecture:** Preserve the H1 Token-Mixer, pair aggregation, classifier, and scalar residual fusion. Add a parallel three-channel state-matrix adapter whose only role-specific difference is true inverse P, deterministic shuffled P, or identity mapping on the difference channel; reuse the existing immutable four-role gate engine through an H2-specific specification.

**Tech Stack:** Python 3.10, PyTorch, NumPy disk-backed caches, CSV matrices, JSONL results/progress, SVG/CSV evaluation artifacts, pytest, Ruff, Git.

---

## Frozen Contract

The authoritative design is
`docs/experiments/innovation1-present-case3-three-channel-topology-residual-design.md`.
The exact roles are:

```python
CASE3_MODEL_ROLES = {
    "anchor": "present_nibble_invp_only_spn_only",
    "candidate": "present_nibble_case3_invp_topology_residual_spn_only",
    "shuffled_p": "present_nibble_case3_shuffled_p_topology_residual_spn_only",
    "raw_triple": "present_nibble_case3_raw_topology_residual_spn_only",
}
```

All three hybrids use:

```python
HYBRID_OPTIONS = {
    "spn_mixer_depth": 2,
    "token_mlp_ratio": 2,
    "local_channels": 16,
    "local_depth": 1,
    "local_kernel_size": 3,
    "local_residual_scale_init": 0.1,
    "activation": "relu",
    "norm": "layernorm",
    "local_norm": "batchnorm2d",
    "dropout": 0.0,
}
```

The gate uses the existing `PRESENT_CASE2_ATTRIBUTION_PROTOCOL`, but its semantic
contract names Liu Case 3 and its representation role is `raw_triple`. The
three seed0 margins are all `0.003`.

## File Map

- Create `src/blockcipher_nd/models/structure/spn/present_case3_topology_residual.py`: exact three-channel tensor construction and shared hybrid.
- Modify SPN export and registry files: expose and construct three fixed variants.
- Create `src/blockcipher_nd/planning/present_case3_topology_residual_gate.py`: H2 role/spec/decision wrapper over the shared gate.
- Create `src/blockcipher_nd/cli/gate_present_case3_topology_residual.py` and `scripts/gate-present-case3-topology-residual`: thin gate CLI.
- Create `tests/test_present_case3_topology_residual.py`: tensor semantics, architecture, capacity, initialization, registry, and gradient tests.
- Create `tests/test_present_case3_topology_residual_gate.py`: strict H2 protocol, matrix, CLI, and decisions.
- Create two four-row CSV matrices under `configs/experiment/innovation1/`: R0 `64/class` and R1 `8192/class`.
- Modify `src/blockcipher_nd/evaluation/plots.py`: four explicit non-colliding plot labels.
- Update the H2 design and this plan only after actual artifacts pass the gate.

### Task 1: Three-Channel Tensor And Hybrid Model

**Files:**
- Create: `tests/test_present_case3_topology_residual.py`
- Create: `src/blockcipher_nd/models/structure/spn/present_case3_topology_residual.py`

- [ ] **Step 1: Write failing tensor-semantics tests**

Independently construct the expected tensor and prove channels 0 and 1 remain
raw while only channel 2 is mapped:

```python
@pytest.mark.parametrize("mapping_mode", ["true", "shuffled", "raw"])
def test_case3_local_view_has_exact_channels(mapping_mode: str) -> None:
    model = PresentNibbleCase3TopologyResidualSpnOnlyDistinguisher(
        input_bits=16 * 128, mapping_mode=mapping_mode
    )
    features = _raw_features()
    pairs = features.reshape(2, 16, 2, 64)
    delta = (pairs[:, :, 0] - pairs[:, :, 1]).abs()
    indices = (
        torch.arange(64)
        if mapping_mode == "raw"
        else present_inverse_p_indices(mapping_mode)
    )
    channels = torch.stack(
        [pairs[:, :, 0], pairs[:, :, 1], delta.index_select(2, indices)], dim=2
    )
    expected = channels.reshape(2, 16, 3, 16, 4).permute(0, 1, 2, 4, 3)
    actual = model.local_case3_view(features)
    assert actual.shape == (2, 16, 3, 4, 16)
    torch.testing.assert_close(actual, expected)
```

Add assertions that flipping the mapping changes channel 2 only, invalid input
widths/modes fail, and the forward/backward outputs and gradients are finite.

- [ ] **Step 2: Run the focused test and verify RED**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_present_case3_topology_residual.py -q
```

Expected: collection failure because the new module does not exist.

- [ ] **Step 3: Implement the minimal shared model**

Subclass no H1 role class; use the same components explicitly so the H1 module
is unchanged. The core view and stem must be:

```python
def local_case3_view(self, features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2 or features.shape[1] != self.input_bits:
        raise ValueError(
            f"expected {self.input_bits} input bits, got {tuple(features.shape)}"
        )
    pairs = features.float().reshape(
        features.shape[0], self.pairs_per_sample, 2, 64
    )
    difference = (pairs[:, :, 0] - pairs[:, :, 1]).abs()
    mapped_difference = difference.index_select(2, self.mapping_indices)
    channels = torch.stack(
        [pairs[:, :, 0], pairs[:, :, 1], mapped_difference], dim=2
    )
    return channels.reshape(
        features.shape[0], self.pairs_per_sample, 3, 16, 4
    ).permute(0, 1, 2, 4, 3)
```

```python
self.local_stem = nn.Sequential(
    nn.Conv2d(3, local_channels, kernel_size=1),
    conv2d_norm(local_norm, local_channels),
    build_activation(activation),
)
```

Flatten batch and pair only before Conv2D, retain the channel axis, pool spatial
mean/max, project to the Token-Mixer embedding, and fuse with scalar `alpha`.
Add fixed subclasses for `true`, `shuffled`, and `raw`.

- [ ] **Step 4: Run the model tests and verify GREEN**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_present_case3_topology_residual.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the model core**

```bash
git add tests/test_present_case3_topology_residual.py src/blockcipher_nd/models/structure/spn/present_case3_topology_residual.py
git commit -m "feat: add PRESENT Case3 residual adapter"
```

### Task 2: Exports And Registry

**Files:**
- Modify: `src/blockcipher_nd/models/structure/spn/__init__.py`
- Modify: `src/blockcipher_nd/models/structure/__init__.py`
- Modify: `src/blockcipher_nd/registry/model_families/spn.py`
- Modify: `tests/test_present_case3_topology_residual.py`

- [ ] **Step 1: Add failing registry and initialization tests**

For all three model keys, assert exact class, forwarded options, default
`pair_bits=128`, invalid explicit options preserved, equal total/trainable/local
parameter counts, identical hybrid state dictionaries under the same seed, and
backbone/classifier equality with a separately seeded anchor.

- [ ] **Step 2: Verify the registry tests fail**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_present_case3_topology_residual.py -q
```

Expected: unknown model keys or missing exports.

- [ ] **Step 3: Register the exact model family**

Add an explicit dictionary beside `topology_residual_models`:

```python
case3_topology_residual_models = {
    "present_nibble_case3_invp_topology_residual_spn_only":
        PresentNibbleCase3InvPTopologyResidualSpnOnlyDistinguisher,
    "present_nibble_case3_shuffled_p_topology_residual_spn_only":
        PresentNibbleCase3ShuffledPTopologyResidualSpnOnlyDistinguisher,
    "present_nibble_case3_raw_topology_residual_spn_only":
        PresentNibbleCase3RawTopologyResidualSpnOnlyDistinguisher,
}
```

Forward the same option names and defaults as H1. Preserve an explicitly invalid
zero instead of replacing it with a truthy fallback.

- [ ] **Step 4: Run focused and project-structure tests**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_present_case3_topology_residual.py tests/test_project_structure.py -q
```

Expected: H2 registry tests pass; no existing model/export regression.

- [ ] **Step 5: Commit registry integration**

```bash
git add src/blockcipher_nd/models/structure/spn/__init__.py src/blockcipher_nd/models/structure/__init__.py src/blockcipher_nd/registry/model_families/spn.py tests/test_present_case3_topology_residual.py
git commit -m "feat: register PRESENT Case3 residual models"
```

### Task 3: H2 Strict Gate And CLI

**Files:**
- Create: `src/blockcipher_nd/planning/present_case3_topology_residual_gate.py`
- Create: `src/blockcipher_nd/cli/gate_present_case3_topology_residual.py`
- Create: `scripts/gate-present-case3-topology-residual`
- Create: `tests/test_present_case3_topology_residual_gate.py`

- [ ] **Step 1: Write failing decision and protocol tests**

Reuse H1 fixture helpers but emit H2 model keys. Cover: readiness, promotion
only when all three margins are at least `0.003`, weak positive margins,
candidate losing to each comparator, missing/duplicate rows, NaN, wrong sample
count, wrong negative mode, wrong effective key schedule, cache non-reuse,
history/checkpoint mismatch, wrong model options, and CLI JSON exit status.

- [ ] **Step 2: Verify RED**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_present_case3_topology_residual_gate.py -q
```

Expected: the H2 gate module and CLI are missing.

- [ ] **Step 3: Implement the immutable H2 gate spec**

Call `evaluate_four_role_attribution()` with an H2 `FourRoleGateSpec`. Map the
shared gate's representation slot to `raw_triple`, freeze the hybrid options,
and implement these seed0 decisions:

```python
if architecture_margin <= 0 or topology_margin <= 0 or representation_margin <= 0:
    return "reject_h2", "stop_h2_and_keep_token_mixer_anchor"
if min(architecture_margin, topology_margin, representation_margin) >= 0.003:
    return "promote_seed1", "run_identical_h2_seed1_local_gate"
return "weak_or_fragile_no_scale", "inspect_histories_once_and_stop_h2_scaling"
```

The H2 spec must not authorize medium or remote scale even after a two-seed
call; any two-seed path returns either a bounded follow-up recommendation or a
no-scale decision.

- [ ] **Step 4: Verify gate and legacy gates**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_present_case3_topology_residual_gate.py tests/test_invp_topology_residual_gate.py tests/test_invp_state_matrix_conv2d_gate.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit the strict gate**

```bash
git add src/blockcipher_nd/planning/present_case3_topology_residual_gate.py src/blockcipher_nd/cli/gate_present_case3_topology_residual.py scripts/gate-present-case3-topology-residual tests/test_present_case3_topology_residual_gate.py
git commit -m "feat: add PRESENT Case3 strict gate"
```

### Task 4: Frozen Experiment Matrices And Plot Labels

**Files:**
- Create: `configs/experiment/innovation1/innovation1_spn_present_case3_topology_residual_smoke_seed0.csv`
- Create: `configs/experiment/innovation1/innovation1_spn_present_case3_topology_residual_8192_seed0.csv`
- Modify: `src/blockcipher_nd/evaluation/plots.py`
- Modify: `tests/test_present_case3_topology_residual_gate.py`
- Modify: `tests/test_topology_residual_plot_labels.py`

- [ ] **Step 1: Write failing exact-matrix and label tests**

Assert four rows, exact role order, one seed, same protocol fields/options,
`64/class` for R0, `8192/class` for R1, no seed1 matrix, and these visible
labels: `InvP token mixer`, `Case3 true InvP`, `Case3 shuffled P`,
`Case3 raw triple`.

- [ ] **Step 2: Verify RED**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_present_case3_topology_residual_gate.py tests/test_topology_residual_plot_labels.py -q
```

- [ ] **Step 3: Add the two matrices and four plot aliases**

Copy H1 protocol fields exactly. Change only family/network/model identifiers,
H2 evidence text, and the three hybrid model keys. Do not add validation data,
labels, negative modes, keys, or optimizer changes.

- [ ] **Step 4: Validate matrix parsing and labels**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_present_case3_topology_residual_gate.py tests/test_topology_residual_plot_labels.py tests/test_project_structure.py -q
```

- [ ] **Step 5: Commit experiment readiness files**

```bash
git add configs/experiment/innovation1/innovation1_spn_present_case3_topology_residual_smoke_seed0.csv configs/experiment/innovation1/innovation1_spn_present_case3_topology_residual_8192_seed0.csv src/blockcipher_nd/evaluation/plots.py tests/test_present_case3_topology_residual_gate.py tests/test_topology_residual_plot_labels.py
git commit -m "experiment: freeze PRESENT Case3 local gate"
```

### Task 5: Implementation Verification And Publication

**Files:** all Task 1-4 files

- [ ] **Step 1: Run focused verification**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_present_case3_topology_residual.py tests/test_present_case3_topology_residual_gate.py tests/test_invp_topology_residual.py tests/test_invp_topology_residual_gate.py tests/test_invp_state_matrix_conv2d_gate.py tests/test_topology_residual_plot_labels.py tests/test_project_structure.py -q
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/blockcipher_nd/models/structure/spn/present_case3_topology_residual.py src/blockcipher_nd/planning/present_case3_topology_residual_gate.py src/blockcipher_nd/cli/gate_present_case3_topology_residual.py tests/test_present_case3_topology_residual.py tests/test_present_case3_topology_residual_gate.py
git diff --check
```

Expected: all focused tests and static checks pass.

- [ ] **Step 2: Push the complete implementation head**

```bash
git push origin main
```

Do not start R0 from unpublished code.

### Task 6: R0 Readiness

**Files:** ignored artifacts under `outputs/local_smoke/` and `outputs/local_cache/`; update the H2 design after passage

- [ ] **Step 1: Run the dedicated R0 matrix from a clean absent cache**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_case3_topology_residual_smoke_seed0.csv \
  --results outputs/local_smoke/i1_present_case3_topology_residual_smoke_seed0/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_case3_topology_residual_smoke_seed0/progress.jsonl \
  --dataset-cache-root outputs/local_cache/i1_present_case3_topology_residual_smoke_seed0 \
  --epochs 1 --batch-size 32 --hidden-bits 32 \
  --dataset-cache-chunk-size 64 --dataset-cache-workers 1 --device cpu
```

- [ ] **Step 2: Validate rows, plots, histories, cache, progress, and neutral gate**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_case3_topology_residual_smoke_seed0.csv \
  --results outputs/local_smoke/i1_present_case3_topology_residual_smoke_seed0/results.jsonl \
  --expected-rows 4 \
  --output outputs/local_smoke/i1_present_case3_topology_residual_smoke_seed0/validation.json
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-results \
  --results outputs/local_smoke/i1_present_case3_topology_residual_smoke_seed0/results.jsonl \
  --output outputs/local_smoke/i1_present_case3_topology_residual_smoke_seed0/curves.svg \
  --history-csv outputs/local_smoke/i1_present_case3_topology_residual_smoke_seed0/history.csv \
  --title i1_present_case3_topology_residual_smoke_seed0
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/gate-present-case3-topology-residual \
  --results outputs/local_smoke/i1_present_case3_topology_residual_smoke_seed0/results.jsonl \
  --progress outputs/local_smoke/i1_present_case3_topology_residual_smoke_seed0/progress.jsonl \
  --expected-seeds 0 --samples-per-class 64 --epochs 1 --readiness-only \
  --output outputs/local_smoke/i1_present_case3_topology_residual_smoke_seed0/readiness_gate.json
```

Expected: validation and gate report `status=pass`; the gate reports
`decision=implementation_ready` and `research_decision_applied=false`. Parse the
SVG as XML and assert all four visible labels.

- [ ] **Step 3: Record and publish R0 only if every check passes**

Update the H2 design with run ID, artifact paths, readiness status, effective
`per_pair_random` metadata, and the authorized R1 command. Commit and push.

### Task 7: R1 `8192/class` Seed0 Adjudication

**Files:** ignored R1 artifacts; update both H2 docs after passage

- [ ] **Step 1: Run R1 automatically after R0 passes**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_case3_topology_residual_8192_seed0.csv \
  --epochs 10 --batch-size 256 --hidden-bits 32 --device cpu \
  --dataset-cache-root outputs/local_cache/i1_present_case3_topology_residual_8192_seed0 \
  --dataset-cache-chunk-size 512 --dataset-cache-workers 4 \
  --progress-output outputs/local_smoke/i1_present_case3_topology_residual_8192_seed0/progress.jsonl \
  --output outputs/local_smoke/i1_present_case3_topology_residual_8192_seed0/results.jsonl
```

Expected: `8192/class` training and the runner's frozen `4096/class` validation,
four complete local CPU rows, 10 epochs unless the frozen early-stop policy
fires, durable progress, and exact disk-cache reuse. Do not reuse the R0 cache.

- [ ] **Step 2: Generate and validate all artifacts**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_case3_topology_residual_8192_seed0.csv \
  --results outputs/local_smoke/i1_present_case3_topology_residual_8192_seed0/results.jsonl \
  --expected-rows 4 \
  --output outputs/local_smoke/i1_present_case3_topology_residual_8192_seed0/validation.json
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-results \
  --results outputs/local_smoke/i1_present_case3_topology_residual_8192_seed0/results.jsonl \
  --output outputs/local_smoke/i1_present_case3_topology_residual_8192_seed0/curves.svg \
  --history-csv outputs/local_smoke/i1_present_case3_topology_residual_8192_seed0/history.csv \
  --title i1_present_case3_topology_residual_8192_seed0
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/gate-present-case3-topology-residual \
  --results outputs/local_smoke/i1_present_case3_topology_residual_8192_seed0/results.jsonl \
  --progress outputs/local_smoke/i1_present_case3_topology_residual_8192_seed0/progress.jsonl \
  --expected-seeds 0 --samples-per-class 8192 --epochs 10 \
  --output outputs/local_smoke/i1_present_case3_topology_residual_8192_seed0/case3_gate.json
```

Require exactly four result rows and four complete histories. Validate plan
alignment, cache identity/reuse, progress terminals, effective
`per_pair_random`, restored best checkpoints, finite metrics, plot XML, visible
labels, and the H2 strict gate.

- [ ] **Step 3: Follow the gate without discretion**

Run seed1 only for `promote_seed1`. For `weak_or_fragile_no_scale` or
`reject_h2`, inspect histories once and stop. Never launch remote or larger H2
scale under this plan.

- [ ] **Step 4: Document, verify, commit, and push the verdict**

Record exact AUCs, three margins, gate decision, claim scope, artifact paths,
and recommended next action. Run focused tests and `git diff --check`, then make
a scoped result-documentation commit and push it.
