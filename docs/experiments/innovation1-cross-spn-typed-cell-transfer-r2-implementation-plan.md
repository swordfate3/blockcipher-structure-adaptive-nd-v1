# Innovation 1 Cross-SPN Typed Cell Transfer R2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` in the current workspace and `superpowers:test-driven-development` for every source change. Project instructions prohibit sub-agent dispatch unless explicitly requested.

**Goal:** Run the authorized E4-R2 seed0 gate that tests whether full PRESENT typed checkpoints improve GIFT beyond scratch and source/target mapping controls under an identical target budget.

**Architecture:** Extend the existing matrix runner with an optional, fail-closed checkpoint-initialization manifest. The loader verifies the source result row, checkpoint payload, SHA-256, metadata, and a strict full state-dict load before target training. Three thin GIFT transfer-role aliases keep result identities unique while retaining the exact shared typed architecture.

**Tech Stack:** Python 3.10, PyTorch, JSON manifests, CSV matrices, existing disk-cache/training/result infrastructure, pytest.

---

## Frozen Question And Gate

Research question:

```text
Does a full PRESENT true-InvP typed checkpoint improve the GIFT true-InvP
target beyond the old GIFT anchor, GIFT typed scratch, source-shuffled
pretraining, and target-shuffled fine-tuning under one identical target budget?
```

R2 uses only:

```text
GIFT-64 r6
8192/class train, 4096/class validation
4 independent pairs/sample
strict encrypted-random-plaintext negatives
fixed train/validation split keys
seed0
10 target epochs
CPU
same optimizer/checkpoint protocol as E4-R1
same R1 GIFT disk cache
```

Required roles and thresholds remain exactly those frozen in the E4 design:

```text
gift_anchor
gift_typed_scratch
true_to_true
shuffled_to_true
true_to_shuffled

true_to_true AUC >= 0.52
true_to_true - gift_anchor >= +0.003
true_to_true - gift_typed_scratch >= +0.005
true_to_true - shuffled_to_true >= +0.003
true_to_true - true_to_shuffled >= +0.003
```

No remote run, sample increase, seed1, optimizer change, classifier reset,
partial state load, or source checkpoint substitution is authorized.

## File Map

- Create `src/blockcipher_nd/engine/checkpoint_initialization.py`: manifest parsing, source-result validation, SHA-256, payload validation, strict load, provenance report.
- Modify `src/blockcipher_nd/engine/matrix_runner.py`: optional `--initialization-manifest` argument.
- Modify `src/blockcipher_nd/engine/task_runner.py`: apply initialization after deterministic model construction and attach provenance to the result row.
- Modify `src/blockcipher_nd/models/structure/spn/cross_spn_typed_cell.py`: three role aliases with no parameter changes.
- Modify SPN exports/registry: expose the role aliases.
- Create `configs/experiment/innovation1/innovation1_spn_cross_spn_typed_transfer_seed0_sources.json`: lock source results/checkpoints and expected source identities.
- Create `configs/experiment/innovation1/innovation1_spn_gift64_cross_spn_typed_transfer_smoke_seed0.csv`: five-role readiness matrix.
- Create `configs/experiment/innovation1/innovation1_spn_gift64_cross_spn_typed_transfer_8192_seed0.csv`: five-role R2 matrix.
- Create `src/blockcipher_nd/planning/cross_spn_typed_transfer_gate.py`: five-role protocol/provenance/cache/history gate.
- Create `src/blockcipher_nd/cli/gate_cross_spn_typed_transfer.py` and `scripts/gate-cross-spn-typed-transfer`: gate entrypoint.
- Create `tests/test_checkpoint_initialization.py` and `tests/test_cross_spn_typed_transfer_gate.py`.
- Update the E4 design and route verdict after R2 completes.

### Task 1: Fail-Closed Checkpoint Initialization

**Files:**
- Create: `src/blockcipher_nd/engine/checkpoint_initialization.py`
- Test: `tests/test_checkpoint_initialization.py`

- [ ] Write tests that create a real typed model checkpoint/result pair and require a successful strict load to return:

```text
kind = checkpoint
source_checkpoint
source_checkpoint_sha256
source_results
source_model
source_cipher
source_rounds
source_seed
source_samples_per_class
source_epochs
source_mapping
target_model
target_mapping
strict_state_dict_load = true
state_dict_key_count
```

- [ ] Require exact tensor equality between the loaded target state and the source payload, and exact target logits on a fixed raw input after separately reloading the same source into another compatible alias.

- [ ] Require fail-closed errors for missing manifest role, missing source result, duplicate source model rows, wrong seed/cipher/rounds/samples/epochs, mismatched checkpoint path, malformed payload, missing/extra/wrong-shaped tensor, payload/result metadata mismatch, and SHA mismatch when an expected digest is present.

- [ ] Implement:

```python
def initialize_model_from_manifest(
    model: nn.Module,
    *,
    target_model: str,
    target_mapping: str,
    manifest_path: Path | None,
) -> dict[str, Any]:
    ...
```

When no mapping exists for a target role, return a scratch record with the
deterministic initial state SHA-256. When a mapping exists, use
`model.load_state_dict(state_dict, strict=True)` and never catch or downgrade
missing/unexpected-key failures.

### Task 2: Integrate Initialization Into The Existing Runner

**Files:**
- Modify: `src/blockcipher_nd/engine/matrix_runner.py`
- Modify: `src/blockcipher_nd/engine/task_runner.py`
- Test: `tests/test_checkpoint_initialization.py`

- [ ] Add `--initialization-manifest` as an optional `Path`-like CLI value. Existing runs with no manifest must remain behaviorally unchanged.

- [ ] After `torch.manual_seed` and `build_model`, but before optional pretraining/optimizer construction, call the loader using the selected model key and a frozen target mapping derived from the role alias.

- [ ] Attach the returned record to `row["initialization"]`. Emit an `initialization_ready` progress event containing kind, source model, source SHA, strict-load status, and target model. Do not include tensor payloads in JSONL.

- [ ] Run existing matrix/checkpoint tests plus the new integration test. Expected: scratch runs are unchanged except for the new auditable result/progress field when a manifest is supplied.

### Task 3: Add Unique Transfer Role Aliases

**Files:**
- Modify: `src/blockcipher_nd/models/structure/spn/cross_spn_typed_cell.py`
- Modify: `src/blockcipher_nd/models/structure/spn/__init__.py`
- Modify: `src/blockcipher_nd/models/structure/__init__.py`
- Modify: `src/blockcipher_nd/registry/model_families/spn.py`
- Test: `tests/test_checkpoint_initialization.py`

- [ ] Add aliases:

```text
gift_cross_spn_typed_cell_true_from_present_true
gift_cross_spn_typed_cell_true_from_present_shuffled
gift_cross_spn_typed_cell_shuffled_from_present_true
```

The first two subclass the GIFT true alias; the third subclasses the GIFT
shuffled alias. They add no parameter or persistent buffer and accept no source
information in the model itself.

- [ ] Under one seed, require all three role aliases and GIFT typed scratch to have identical state-dict keys/shapes/tensors before checkpoint loading.

### Task 4: Freeze Source Manifest And Five-Role Matrices

**Files:**
- Create: `configs/experiment/innovation1/innovation1_spn_cross_spn_typed_transfer_seed0_sources.json`
- Create: `configs/experiment/innovation1/innovation1_spn_gift64_cross_spn_typed_transfer_smoke_seed0.csv`
- Create: `configs/experiment/innovation1/innovation1_spn_gift64_cross_spn_typed_transfer_8192_seed0.csv`
- Test: `tests/test_cross_spn_typed_transfer_gate.py`

- [ ] Freeze the manifest against:

```text
source results:
  outputs/local_smoke/i1_present_cross_spn_typed_cell_r1_seed0/results.jsonl
true source checkpoint:
  outputs/local_smoke/i1_present_cross_spn_typed_cell_r1_seed0/checkpoints/row0002_present_cross_spn_typed_cell_true_seed0.pt
shuffled source checkpoint:
  outputs/local_smoke/i1_present_cross_spn_typed_cell_r1_seed0/checkpoints/row0003_present_cross_spn_typed_cell_shuffled_seed0.pt
source identity:
  PRESENT-80 r7, seed0, 8192/class, 10 epochs, strict Case2, per_pair_random
```

Record the actual SHA-256 of both checkpoint files in the manifest.

- [ ] Freeze each matrix in this order:

```text
gift_cross_spn_aligned_token_mixer_raw_anchor
gift_cross_spn_typed_cell_true
gift_cross_spn_typed_cell_true_from_present_true
gift_cross_spn_typed_cell_true_from_present_shuffled
gift_cross_spn_typed_cell_shuffled_from_present_true
```

All five rows use exact E4-R1 GIFT data/training fields. R0 uses `64/class`,
one epoch; R2 uses `8192/class`, ten epochs.

### Task 5: Implement The Five-Role Transfer Gate

**Files:**
- Create: `src/blockcipher_nd/planning/cross_spn_typed_transfer_gate.py`
- Create: `src/blockcipher_nd/cli/gate_cross_spn_typed_transfer.py`
- Create: `scripts/gate-cross-spn-typed-transfer`
- Test: `tests/test_cross_spn_typed_transfer_gate.py`

- [ ] Require exactly five rows, one seed0 row per role, complete 10-epoch histories and restored best checkpoints, exact GIFT protocol fields, one shared non-empty cache root, and ten terminal cache-reuse/create events covering train/validation for every role.

- [ ] Require scratch initialization on anchor/scratch rows and exact manifest-backed provenance on all three transfer rows. Require true source SHA equality across `true_to_true` and `true_to_shuffled`, shuffled source SHA only on `shuffled_to_true`, and strict state load for every transfer row.

- [ ] R0 must return `implementation_ready` without interpreting AUC. R2 decisions are:

```text
all five thresholds pass       -> promote_e4_transfer_seed1
true_to_true best but near gate -> weak_transfer_no_scale
shuffled_to_true matches/wins   -> generic_pretraining_not_typed_transfer
true_to_shuffled matches/wins   -> target_topology_not_attributed
anchor or scratch matches/wins  -> reject_e4_transfer
any evidence mismatch           -> invalid_e4_protocol
```

- [ ] The CLI writes newline-terminated sorted JSON and returns nonzero only for invalid protocol.

### Task 6: Run R2 Readiness And Diagnostic

**Outputs:**
- `outputs/local_smoke/i1_gift64_cross_spn_typed_transfer_r0_seed0/`
- `outputs/local_smoke/i1_gift64_cross_spn_typed_transfer_r2_seed0/`

- [ ] Run focused tests, ruff, and `git diff --check`.

- [ ] Run R0 locally with the frozen source manifest, separate `64/class` cache, one epoch, CPU, five checkpoint outputs, validation, SVG/CSV, and neutral gate replay. Require exact source SHA and strict load evidence before R2.

- [ ] Commit/push the implementation after R0 passes.

- [ ] Run R2 locally with the frozen source manifest and the existing R1 GIFT cache root:

```text
outputs/local_cache/i1_gift64_cross_spn_typed_cell_r1_seed0
```

Use 10 epochs, batch 256, hidden bits 32, cache chunk 512, four workers,
train-eval interval 1, CPU, and five target checkpoint outputs.

- [ ] Validate five plan-aligned rows, generate/parse SVG and CSV, run the strict transfer gate, and report all five AUCs and four true-to-true margins.

### Task 7: Adjudicate And Document

**Files:**
- Modify: `docs/experiments/innovation1-cross-spn-typed-cell-transfer-design.md`
- Modify: `docs/experiments/innovation1-route-verdict-2026-07-09.md`

- [ ] Record run IDs, source/result/checkpoint paths and SHA-256, exact target protocol, metrics, margins, gate, artifacts, claim scope, and one evidence-backed next action.

- [ ] If `promote_e4_transfer_seed1`, freeze only an identical local seed1 transfer repeat before considering larger diagnostics. If any control fails, stop transfer and preserve E4-R1 as positive within-cipher typed representation evidence. Never launch remote or increase samples from seed0 alone.

- [ ] Run the full test suite, `git diff --check`, make a scoped adjudication commit, and push normally.

## Self-Review

- The plan changes only initialization/mapping role on the GIFT target; benchmark, labels, negative samples, metric, optimizer, target epochs, and target cache remain frozen.
- Full state-dict transfer includes the classifier and uses strict loading; no tensor is dropped or reinitialized.
- The source manifest closes the checkpoint filename/identity gap found in the existing payload format.
- Scratch and transfer rows have unique result identities without introducing new trainable architectures.
- R2 remains a single-seed `8192/class` local diagnostic. It cannot support formal, paper-scale, SOTA, ceiling, or breakthrough claims.
