# InvP Topology-Residual Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and adjudicate the frozen H1 hypothesis that a small InvP-aligned local residual adapter can add complementary topology signal to the strongest same-protocol PRESENT-80 r7 InvP token-mixer anchor.

**Architecture:** Keep the existing true-InvP token encoder, pair aggregation, and classifier as the common backbone. Add one parallel state-matrix adapter whose only role-specific difference is true InvP, deterministic shuffled-P, or raw DeltaC mapping, then fuse each pair as `token_embedding + alpha * local_embedding` with learned scalar `alpha=0.1`. Reuse the existing strict four-role attribution validator through a small immutable gate specification instead of copying the 1,187-line Conv2D gate.

**Tech Stack:** Python 3.10, PyTorch, NumPy disk-backed dataset cache, CSV experiment matrices, JSONL progress/results, pytest, Ruff, Git worktrees.

---

## Frozen Evidence Contract

This plan implements the authoritative H1 section in
`docs/experiments/innovation1-present-invp-state-matrix-conv2d-design.md`.
It does not reopen pure Conv2D, DDT/beam-stat inputs, or E1 graph scaling.

The four roles are exactly:

```python
TOPOLOGY_RESIDUAL_MODEL_ROLES = {
    "anchor": "present_nibble_invp_only_spn_only",
    "candidate": "present_nibble_invp_topology_residual_spn_only",
    "shuffled_p": "present_nibble_shuffled_p_topology_residual_spn_only",
    "delta_only": "present_nibble_delta_topology_residual_spn_only",
}
```

The anchor options remain:

```json
{"spn_mixer_depth":2,"activation":"relu","norm":"layernorm"}
```

All hybrid options are exactly:

```json
{"spn_mixer_depth":2,"token_mlp_ratio":2,"local_channels":16,"local_depth":1,"local_kernel_size":3,"local_residual_scale_init":0.1,"activation":"relu","norm":"layernorm","local_norm":"batchnorm2d","dropout":0.0}
```

The frozen protocol is PRESENT-80 r7, `8192/class` training and `4096/class`
validation for H1 seed0, 16 pairs/sample, encrypted-random-plaintext negatives,
Zhang/Wang official MCND construction, effective `per_pair_random` keys for
both splits, MSE, Adam, official cyclic schedule, best `val_auc` checkpoint,
and local CPU execution. This is diagnostic evidence, not formal training or a
breakthrough claim.

## File Map

- Create `src/blockcipher_nd/models/structure/spn/present_invp_topology_residual.py`: shared local adapter, shared hybrid model, and the three mapping-specific classes.
- Modify `src/blockcipher_nd/models/structure/spn/__init__.py`, `src/blockcipher_nd/models/structure/__init__.py`, `src/blockcipher_nd/registry/model_families/spn.py`, and `tests/test_project_structure.py`: export, build, and recognize the three models.
- Modify `src/blockcipher_nd/planning/invp_state_matrix_conv2d_gate.py`: parameterize its existing strict four-role validation core while retaining the current public Conv2D API and byte-for-byte decision semantics.
- Create `src/blockcipher_nd/planning/invp_topology_residual_gate.py`: H1 specification and public H1 decision wrapper.
- Create `src/blockcipher_nd/cli/gate_invp_topology_residual.py` and `scripts/gate-invp-topology-residual`: thin gate CLI.
- Create `tests/test_invp_topology_residual.py` and `tests/test_invp_topology_residual_gate.py`: tensor, capacity, initialization, registry, protocol, cache, history, CLI, and decision tests.
- Create `configs/experiment/innovation1/innovation1_spn_present_invp_topology_residual_smoke_seed0.csv` and `configs/experiment/innovation1/innovation1_spn_present_invp_topology_residual_8192_seed0.csv`: frozen R0 and seed0 matrices.
- Modify `docs/experiments/innovation1-present-invp-state-matrix-conv2d-design.md`: record R0 readiness and H1 seed0 result only after artifacts exist and pass the strict gate.

### Task 1: Shared Topology-Residual Model

**Files:**
- Create: `tests/test_invp_topology_residual.py`
- Create: `src/blockcipher_nd/models/structure/spn/present_invp_topology_residual.py`

- [ ] **Step 1: Write failing tensor-semantics and architecture tests**

Create tests that instantiate the base hybrid with `input_bits=2048`, assert
`local_state_matrix_view()` returns `[batch, 16, 4, 16]`, and independently
reconstruct the expected absolute ciphertext difference. For true and shuffled
views, apply `present_inverse_p_indices("true")` and
`present_inverse_p_indices("shuffled")`; for Delta, use `torch.arange(64)`.
Assert the adapter has one `1x1` stem, exactly one 3x3 residual block, mean/max
pooling projected to 128, scalar `alpha == 0.1`, output `[batch, 1]`, and finite
gradients after `output.mean().backward()`.

```python
@pytest.mark.parametrize("mapping_mode", ["true", "shuffled", "delta"])
def test_topology_residual_state_matrix_has_exact_mapping(mapping_mode: str):
    model = PresentNibbleTopologyResidualSpnOnlyDistinguisher(
        input_bits=2048, mapping_mode=mapping_mode
    )
    features = torch.randint(0, 2, (2, 2048), dtype=torch.float32)
    raw = features.reshape(2, 16, 2, 64)
    delta = (raw[:, :, 0] - raw[:, :, 1]).abs()
    indices = (
        torch.arange(64)
        if mapping_mode == "delta"
        else present_inverse_p_indices(mapping_mode)
    )
    expected = delta.index_select(2, indices).reshape(2, 16, 16, 4).permute(0, 1, 3, 2)
    torch.testing.assert_close(model.local_state_matrix_view(features), expected)
```

- [ ] **Step 2: Run the model test and verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_invp_topology_residual.py -q
```

Expected: collection fails because
`blockcipher_nd.models.structure.spn.present_invp_topology_residual` does not
exist.

- [ ] **Step 3: Implement the minimal shared model**

Use the existing `_PresentNibblePAlignedSpnEncoder` as the common true-InvP
token backbone and `PresentStateMatrixResidualBlock` for the local block. In
constructor order, instantiate `spn_encoder` and the unchanged anchor
`classifier` first, then register mapping indices and instantiate the adapter.
Expose `local_state_matrix_view`, `encode_local_pairs`, `encode_fused_pairs`,
and `forward` so semantics are directly testable.

```python
class PresentNibbleTopologyResidualSpnOnlyDistinguisher(nn.Module):
    def __init__(
        self,
        input_bits: int,
        pair_bits: int = 128,
        base_channels: int = 32,
        spn_token_dim: int | None = None,
        spn_mixer_depth: int = 2,
        token_mlp_ratio: int = 2,
        local_channels: int = 16,
        local_depth: int = 1,
        local_kernel_size: int = 3,
        local_residual_scale_init: float = 0.1,
        activation: str = "relu",
        norm: str = "layernorm",
        local_norm: str = "batchnorm2d",
        dropout: float = 0.0,
        mapping_mode: str = "true",
    ) -> None:
        super().__init__()
        # Validate pair_bits, divisibility, local_depth == 1, odd kernel,
        # positive channels, and mapping_mode before creating modules.
        self.spn_encoder = _PresentNibblePAlignedSpnEncoder(
            input_bits=input_bits,
            pair_bits=pair_bits,
            base_channels=base_channels,
            spn_token_dim=spn_token_dim,
            spn_mixer_depth=spn_mixer_depth,
            token_mlp_ratio=token_mlp_ratio,
            activation=activation,
            norm=norm,
            dropout=dropout,
            view_mode="inv_p",
            p_alignment="true",
        )
        embedding_bits = self.spn_encoder.embedding_bits
        self.classifier = nn.Sequential(
            build_norm(norm, embedding_bits * 2),
            nn.Linear(embedding_bits * 2, max(64, base_channels * 8)),
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(max(64, base_channels * 8), 1),
        )
        self.register_buffer("mapping_indices", mapping_indices, persistent=False)
        self.local_stem = nn.Sequential(
            nn.Conv2d(1, local_channels, 1),
            conv2d_norm(local_norm, local_channels),
            build_activation(activation),
        )
        self.local_blocks = nn.ModuleList([
            PresentStateMatrixResidualBlock(
                local_channels, local_kernel_size, activation, local_norm, dropout
            )
        ])
        self.local_projection = nn.Linear(local_channels * 2, embedding_bits)
        self.alpha = nn.Parameter(torch.tensor(float(local_residual_scale_init)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        fused = self.encode_fused_pairs(features.float())
        summary = torch.cat([fused.mean(1), fused.max(1).values], dim=1)
        return self.classifier(summary)
```

Add mapping subclasses that force `mapping_mode` to `true`, `shuffled`, and
`delta`; reject attempts to override it rather than silently accepting two
conflicting sources of truth.

- [ ] **Step 4: Run the model tests and verify GREEN**

Run the Task 1 command. Expected: all tests pass with finite forward/backward.

- [ ] **Step 5: Commit Task 1**

```bash
git add tests/test_invp_topology_residual.py src/blockcipher_nd/models/structure/spn/present_invp_topology_residual.py
git commit -m "feat: add InvP topology residual adapter"
```

### Task 2: Registration, Capacity, and Common Initialization

**Files:**
- Modify: `src/blockcipher_nd/models/structure/spn/__init__.py`
- Modify: `src/blockcipher_nd/models/structure/__init__.py`
- Modify: `src/blockcipher_nd/registry/model_families/spn.py`
- Modify: `tests/test_invp_topology_residual.py`
- Modify: `tests/test_project_structure.py`

- [ ] **Step 1: Add failing registry/capacity/initialization tests**

Assert all three exact model keys build through `build_spn_model`; all use the
frozen option dictionary; all three hybrids have identical total/trainable
counts; their `spn_encoder` and `classifier` state dictionaries are identical
when rebuilt with the same seed; adapter shapes are equal; and only
`mapping_indices` differ. Assert an anchor and candidate built from the same
seed have identical common `spn_encoder` and `classifier` tensors.

```python
def test_hybrid_roles_have_equal_capacity_and_common_initialization():
    models = []
    for cls in HYBRID_CLASSES:
        torch.manual_seed(7)
        models.append(cls(input_bits=2048, **FROZEN_OPTIONS))
    assert len({sum(p.numel() for p in m.parameters()) for m in models}) == 1
    for name, tensor in models[0].spn_encoder.state_dict().items():
        torch.testing.assert_close(tensor, models[1].spn_encoder.state_dict()[name])
        torch.testing.assert_close(tensor, models[2].spn_encoder.state_dict()[name])
```

- [ ] **Step 2: Run focused tests and verify RED**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_invp_topology_residual.py tests/test_project_structure.py -q
```

Expected: registry/export assertions fail for the three new keys.

- [ ] **Step 3: Export and register exactly three hybrid keys**

Add the classes to both structure `__init__.py` files. In
`registry/model_families/spn.py`, add one mapping of model keys to classes and
forward exactly the frozen supported values:

```python
topology_residual_models = {
    "present_nibble_invp_topology_residual_spn_only": PresentNibbleInvPTopologyResidualSpnOnlyDistinguisher,
    "present_nibble_shuffled_p_topology_residual_spn_only": PresentNibbleShuffledPTopologyResidualSpnOnlyDistinguisher,
    "present_nibble_delta_topology_residual_spn_only": PresentNibbleDeltaTopologyResidualSpnOnlyDistinguisher,
}
```

Use `hidden_bits` as `base_channels` and the existing typed option helpers.
Do not modify the benchmark builder, data generation, labels, or metrics.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 2 test command. Expected: all focused tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/blockcipher_nd/models/structure/spn/__init__.py src/blockcipher_nd/models/structure/__init__.py src/blockcipher_nd/registry/model_families/spn.py tests/test_invp_topology_residual.py tests/test_project_structure.py
git commit -m "feat: register topology residual models"
```

### Task 3: Reusable Strict Four-Role Gate Core

**Files:**
- Modify: `src/blockcipher_nd/planning/invp_state_matrix_conv2d_gate.py`
- Modify: `tests/test_invp_state_matrix_conv2d_gate.py`
- Create: `src/blockcipher_nd/planning/invp_topology_residual_gate.py`
- Create: `tests/test_invp_topology_residual_gate.py`

- [ ] **Step 1: Add regression and H1 decision tests before refactoring**

First serialize the current Conv2D R0 and R1 gate reports from the preserved
artifacts in `outputs/local_smoke/` and assert the refactored public function
keeps their status, decision, models, margins, cache evidence, stopped actions,
and next action. Add H1 synthetic-row tests for this exact ordered decision
table:

```text
invalid protocol                         -> invalid_protocol
candidate <= anchor                      -> stop_topology_residual
candidate <= shuffled-P                  -> stop_true_topology_attribution
candidate <= Delta                       -> stop_invp_adapter_attribution
candidate above all but any margin < .003 -> weak_or_fragile_no_scale
all three margins >= .003                -> promote_seed1
```

Also test that readiness returns `implementation_ready`, does not interpret
metrics, and sets `research_decision_applied=false`.

- [ ] **Step 2: Run gate tests and verify RED for H1**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_invp_state_matrix_conv2d_gate.py tests/test_invp_topology_residual_gate.py -q
```

Expected: existing Conv2D tests pass and H1 import fails.

- [ ] **Step 3: Parameterize the existing strict validator minimally**

Add a frozen private specification carrying role keys, exact model options,
capacity label, semantic evidence, decision callback, stopped-action callback,
next-action strings, and claim label. Thread it only through helpers that
currently read `MODEL_ROLES` or Conv2D-specific constants:

```python
@dataclass(frozen=True)
class FourRoleGateSpec:
    model_roles: Mapping[str, str]
    anchor_options: Mapping[str, Any]
    candidate_options: Mapping[str, Any]
    capacity_label: str
    semantic_checks: Mapping[str, Any]
    readiness_next_action: str
    claim_label: str
    decide: Callable[..., tuple[str, str]]
    stopped_actions: Callable[[str], list[dict[str, str]]]
```

Create `_gate_four_role_attribution(..., spec)` containing the current public
body. Keep `gate_invp_state_matrix_conv2d(...)` as a compatibility wrapper with
the Conv2D spec. Pass `spec` to `_cache_evidence`, `_protocol_errors`,
`_rows_by_seed_and_role`, parameter reporting, and semantic reporting. Preserve
all existing strict checks: exact typed top-level/training/validation fields,
complete history replay, best checkpoint, result/progress path pairing,
`run_done.output` binding, cache-root containment, two cache creates, six
control reuses, terminal `run_done`, exact seeds, equal role capacities, and
effective `per_pair_random` schedules.

- [ ] **Step 4: Implement the H1 wrapper and exact three-margin policy**

In `invp_topology_residual_gate.py`, define the frozen H1 spec and call the
generic core. The H1 seed0 decision must require the architecture margin as
well as both control margins:

```python
if architecture_margin <= 0:
    return "stop_topology_residual", "keep_token_mixer_anchor_and_stop_adapter"
if topology_margin <= 0:
    return "stop_true_topology_attribution", "stop_adapter_without_true_topology_attribution"
if representation_margin <= 0:
    return "stop_invp_adapter_attribution", "stop_adapter_without_invp_attribution"
if min(architecture_margin, topology_margin, representation_margin) < 0.003:
    return "weak_or_fragile_no_scale", "inspect_histories_once_and_do_not_scale"
return "promote_seed1", "run_identical_seed1_local_gate"
```

For two seeds, require candidate above all on each seed, minimum architecture
margin `>=0.001`, and minimum topology and representation margins `>=0.002`
before `promote_medium_65536`. Use minima, not mean architecture margin.

- [ ] **Step 5: Run both gate suites and verify GREEN**

Run the Task 3 command. Expected: all existing Conv2D and new H1 gate tests
pass, including preserved artifact replay.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/blockcipher_nd/planning/invp_state_matrix_conv2d_gate.py src/blockcipher_nd/planning/invp_topology_residual_gate.py tests/test_invp_state_matrix_conv2d_gate.py tests/test_invp_topology_residual_gate.py
git commit -m "refactor: share strict four-role attribution gate"
```

### Task 4: CLI and Frozen Experiment Matrices

**Files:**
- Create: `src/blockcipher_nd/cli/gate_invp_topology_residual.py`
- Create: `scripts/gate-invp-topology-residual`
- Create: `configs/experiment/innovation1/innovation1_spn_present_invp_topology_residual_smoke_seed0.csv`
- Create: `configs/experiment/innovation1/innovation1_spn_present_invp_topology_residual_8192_seed0.csv`
- Modify: `tests/test_invp_topology_residual_gate.py`
- Modify: `tests/test_project_structure.py`

- [ ] **Step 1: Add failing CLI and exact-plan tests**

Assert the CLI requires matching `--results` and `--progress`, parses exact
integer seeds, writes sorted JSON plus newline, exits 0 only for gate pass, and
supports `--readiness-only`. Load both CSVs through the real planner and assert
exactly four tasks, exact role order, exact options, strict negatives,
`per_pair_random` effective metadata expectation, r7, `m=16`, seed0, CPU,
disk-backed cache root, R0 `64/class, 1 epoch, batch 32`, and H1
`8192/class, 10 epochs, batch 256`.

- [ ] **Step 2: Run CLI/plan tests and verify RED**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_invp_topology_residual_gate.py tests/test_project_structure.py -q
```

Expected: missing CLI/script/config assertions fail.

- [ ] **Step 3: Add thin CLI and executable wrapper**

Mirror the argument surface of `gate_invp_state_matrix_conv2d.py`, but call
`gate_invp_topology_residual`. Keep experiment logic out of the script.

```python
report = gate_invp_topology_residual(
    args.results,
    progress_paths=args.progress,
    expected_seeds=args.expected_seeds,
    samples_per_class=args.samples_per_class,
    epochs=args.epochs,
    readiness_only=args.readiness_only,
)
```

- [ ] **Step 4: Create the two exact four-row CSV matrices**

Clone only the protocol columns from the approved Conv2D matrices. Replace the
three candidate/control model keys, architecture labels, model options,
run/output/cache identifiers, and evidence text. Do not alter data, labels,
negative definition, cipher, round, difference, pair count, key placeholders,
validation construction, optimizer, scheduler, checkpoint selection, or metric.

- [ ] **Step 5: Run CLI/plan tests and verify GREEN**

Run the Task 4 test command. Expected: tests pass; the planner-backed tests
load each CSV into exactly four valid tasks with the frozen runtime parameters.

- [ ] **Step 6: Commit Task 4**

```bash
git add src/blockcipher_nd/cli/gate_invp_topology_residual.py scripts/gate-invp-topology-residual configs/experiment/innovation1/innovation1_spn_present_invp_topology_residual_smoke_seed0.csv configs/experiment/innovation1/innovation1_spn_present_invp_topology_residual_8192_seed0.csv tests/test_invp_topology_residual_gate.py tests/test_project_structure.py
git commit -m "experiment: add topology residual H1 matrices"
```

### Task 5: R0 Readiness Execution

**Files:**
- Modify after passed gate: `docs/experiments/innovation1-present-invp-state-matrix-conv2d-design.md`
- Generate ignored artifacts: `outputs/local_smoke/i1_present_invp_topology_residual_smoke_seed0/`
- Generate ignored cache: `outputs/local_cache/i1_present_invp_topology_residual_smoke_seed0/`

- [ ] **Step 1: Run the exact R0 matrix from an empty run-owned cache**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_invp_topology_residual_smoke_seed0.csv \
  --epochs 1 \
  --batch-size 32 \
  --hidden-bits 32 \
  --device cpu \
  --dataset-cache-root outputs/local_cache/i1_present_invp_topology_residual_smoke_seed0 \
  --dataset-cache-chunk-size 64 \
  --dataset-cache-workers 1 \
  --progress-output outputs/local_smoke/i1_present_invp_topology_residual_smoke_seed0/progress.jsonl \
  --output outputs/local_smoke/i1_present_invp_topology_residual_smoke_seed0/results.jsonl
```

Expected: four rows complete, one train and one validation cache are created,
then reused by all controls; metrics are not interpreted.

- [ ] **Step 2: Generate evaluation artifacts and validate row alignment**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results --plan configs/experiment/innovation1/innovation1_spn_present_invp_topology_residual_smoke_seed0.csv --results outputs/local_smoke/i1_present_invp_topology_residual_smoke_seed0/results.jsonl --expected-rows 4 --output outputs/local_smoke/i1_present_invp_topology_residual_smoke_seed0/validation.json
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-results --results outputs/local_smoke/i1_present_invp_topology_residual_smoke_seed0/results.jsonl --output outputs/local_smoke/i1_present_invp_topology_residual_smoke_seed0/curves.svg --history-csv outputs/local_smoke/i1_present_invp_topology_residual_smoke_seed0/history.csv --title i1_present_invp_topology_residual_smoke_seed0
```

Expected: `status=pass`, four aligned rows, SVG and history CSV exist.

- [ ] **Step 3: Run the strict readiness gate**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/gate-invp-topology-residual --results outputs/local_smoke/i1_present_invp_topology_residual_smoke_seed0/results.jsonl --progress outputs/local_smoke/i1_present_invp_topology_residual_smoke_seed0/progress.jsonl --expected-seeds 0 --samples-per-class 64 --epochs 1 --readiness-only --output outputs/local_smoke/i1_present_invp_topology_residual_smoke_seed0/readiness_gate.json
```

Expected: `status=pass`, `decision=implementation_ready`,
`research_decision_applied=false`. Any failure means repair and rerun the same
R0; do not interpret AUC.

- [ ] **Step 4: Record only verified R0 evidence and commit**

Update the design status and add run id, artifact paths, readiness decision,
capacity equality, cache/progress evidence, claim boundary, and next action
`run H1 seed0`. Then verify and commit:

```bash
git diff --check
git add docs/experiments/innovation1-present-invp-state-matrix-conv2d-design.md
git commit -m "experiment: record topology residual readiness"
```

### Task 6: H1 `8192/class` Seed0 Adjudication

**Files:**
- Modify after passed gate: `docs/experiments/innovation1-present-invp-state-matrix-conv2d-design.md`
- Generate ignored artifacts: `outputs/local_smoke/i1_present_invp_topology_residual_8192_seed0/`
- Generate ignored cache: `outputs/local_cache/i1_present_invp_topology_residual_8192_seed0/`

- [ ] **Step 1: Run the exact seed0 matrix from an empty run-owned cache**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_invp_topology_residual_8192_seed0.csv \
  --epochs 10 \
  --batch-size 256 \
  --hidden-bits 32 \
  --device cpu \
  --dataset-cache-root outputs/local_cache/i1_present_invp_topology_residual_8192_seed0 \
  --dataset-cache-chunk-size 512 \
  --dataset-cache-workers 4 \
  --progress-output outputs/local_smoke/i1_present_invp_topology_residual_8192_seed0/progress.jsonl \
  --output outputs/local_smoke/i1_present_invp_topology_residual_8192_seed0/results.jsonl
```

Expected: four complete local CPU rows, 10 epochs unless the frozen early-stop
policy fires, durable progress, and exact disk-cache reuse.

- [ ] **Step 2: Generate validation and plotting artifacts**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results --plan configs/experiment/innovation1/innovation1_spn_present_invp_topology_residual_8192_seed0.csv --results outputs/local_smoke/i1_present_invp_topology_residual_8192_seed0/results.jsonl --expected-rows 4 --output outputs/local_smoke/i1_present_invp_topology_residual_8192_seed0/validation.json
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-results --results outputs/local_smoke/i1_present_invp_topology_residual_8192_seed0/results.jsonl --output outputs/local_smoke/i1_present_invp_topology_residual_8192_seed0/curves.svg --history-csv outputs/local_smoke/i1_present_invp_topology_residual_8192_seed0/history.csv --title i1_present_invp_topology_residual_8192_seed0
```

- [ ] **Step 3: Run the strict H1 seed0 gate**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/gate-invp-topology-residual --results outputs/local_smoke/i1_present_invp_topology_residual_8192_seed0/results.jsonl --progress outputs/local_smoke/i1_present_invp_topology_residual_8192_seed0/progress.jsonl --expected-seeds 0 --samples-per-class 8192 --epochs 10 --output outputs/local_smoke/i1_present_invp_topology_residual_8192_seed0/topology_residual_gate.json
```

Expected: protocol pass and exactly one frozen decision. If protocol fails,
repair and rerun without interpreting metrics. Run seed1 only when the gate says
`promote_seed1`; never launch remote directly from seed0.

- [ ] **Step 4: Document the evidence-backed decision and next plan**

Record exact AUCs, three margins, histories/checkpoints, protocol/cache evidence,
decision, stopped actions, claim scope, and recommended next action. Explicitly
state that `8192/class` is diagnostic. Commit the result document:

```bash
git diff --check
git add docs/experiments/innovation1-present-invp-state-matrix-conv2d-design.md
git commit -m "experiment: adjudicate topology residual seed0"
```

### Task 7: Final Verification, Review, Push, and Conditional Continuation

**Files:**
- Verify all task-scoped source, tests, configs, scripts, and documentation.

- [ ] **Step 1: Run focused verification**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_invp_topology_residual.py tests/test_invp_topology_residual_gate.py tests/test_invp_state_matrix_conv2d.py tests/test_invp_state_matrix_conv2d_gate.py tests/test_project_structure.py -q
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check src/blockcipher_nd/models/structure/spn/present_invp_topology_residual.py src/blockcipher_nd/planning/invp_topology_residual_gate.py src/blockcipher_nd/planning/invp_state_matrix_conv2d_gate.py src/blockcipher_nd/cli/gate_invp_topology_residual.py tests/test_invp_topology_residual.py tests/test_invp_topology_residual_gate.py
git diff --check
```

Expected: focused tests and Ruff pass, and no whitespace errors.

- [ ] **Step 2: Run the full suite and classify only evidence-backed baseline failures**

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

Expected baseline as of plan creation: 30 known failures (29 Matplotlib 3.11
global-state failures and one JSON route-alignment failure), with no H1 or
Conv2D regression. Any different/new failure must be diagnosed before merge.

- [ ] **Step 3: Perform whole-feature spec and quality review**

Review the final diff against every frozen H1 requirement: single changed
hypothesis, common initialization, identical hybrid capacities, strict data
protocol, typed options, cache/progress provenance, exact seed0 gate, artifact
completeness, and claim boundary. Fix every material finding and rerun its
covering verification.

- [ ] **Step 4: Push the complete branch and integrate only verified work**

```bash
git status --short --branch
git push -u origin experiment/invp-topology-residual
```

After review approval, fast-forward merge to `main`, rerun focused verification
on merged main, and push `main`. Do not use scp or a dirty remote overlay if push
is rejected.

- [ ] **Step 5: Follow the gate, not preference**

If seed0 says `promote_seed1`, create the identical seed1 local matrix and
repeat the same strict workflow. For every other valid decision, stop H1 and
use its controls/histories to design a genuinely new single-variable hypothesis.
Do not mechanically run `65536/class`, remote GPU, pure Conv2D seed1, DDT, or E1
graph scaling. Even two-seed promotion only authorizes a planned
`65536/class` medium diagnostic; formal SPN claims still require completed,
retrieved, plan-aligned `>=1000000/class` multi-seed evidence.
