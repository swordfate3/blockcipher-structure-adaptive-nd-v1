# AutoND Typed InvP Local Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. Subagent execution is not selected
> because the active repository instructions do not authorize delegation.

**Goal:** Build and execute the frozen four-row local gate that tests whether
PRESENT `InvP(DeltaC)` beats AutoND, shuffled-P, and DeltaC-only controls under
the exact AutoND public-code data protocol.

**Architecture:** Reuse the existing exact-total dataset mode, curriculum
runner, final evaluator, and four registered models without changing training
behavior. Add one focused gate module that validates protocol integrity and
computes control margins, one thin CLI wrapper, and one four-row CSV plan. Run
the matrix locally with a shared disk cache, then validate, plot, gate, and
record the decision in the experiment plan.

**Tech Stack:** Python 3.13, PyTorch, NumPy, JSONL/CSV, `uv`, pytest, Ruff.

---

## File Map

- Create `src/blockcipher_nd/planning/autond_typed_invp_gate.py`: load four
  result rows, validate r5-r9 protocol evidence, and compute the frozen typed
  representation decision.
- Create `src/blockcipher_nd/cli/gate_autond_typed_invp.py`: parse CLI options,
  call the gate, and write deterministic JSON.
- Create `scripts/gate-autond-typed-invp`: thin executable wrapper only.
- Create `tests/test_autond_typed_invp_gate.py`: gate and CLI regression tests.
- Create
  `configs/experiment/innovation1/innovation1_spn_present_autond_typed_invp_local_gate_seed0.csv`:
  frozen four-row experiment matrix.
- Modify `tests/test_autond_public_protocol.py`: lock the plan fields and model
  rows.
- Modify
  `docs/experiments/innovation1-present-autond-typed-invp-local-gate-plan.md`:
  record implementation readiness, run artifacts, metrics, decision, and next
  action.

### Task 1: Protocol-Aware Typed Gate

**Files:**

- Create: `src/blockcipher_nd/planning/autond_typed_invp_gate.py`
- Create: `tests/test_autond_typed_invp_gate.py`

- [ ] **Step 1: Write the failing strong-support test**

Create a helper that emits one synthetic result row with the real result
schema: target training metadata for r9, four curriculum stages for r5-r8, and
three final-evaluation repeats. Then add:

```python
def test_typed_invp_gate_supports_candidate_above_all_controls(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    write_rows(
        results,
        {
            "autond_dbitnet2023": [0.52, 0.53, 0.51],
            "present_nibble_invp_only_spn_only": [0.57, 0.58, 0.56],
            "present_nibble_shuffled_paligned_spn_only": [0.53, 0.54, 0.52],
            "present_nibble_delta_only_spn_only": [0.54, 0.53, 0.53],
        },
    )

    report = gate_autond_typed_invp(results)

    assert report["status"] == "pass"
    assert report["decision"] == "strong_local_support"
    assert report["candidate_margin_vs_best_control_auc"] >= 0.01
    assert report["candidate_above_all_controls_by_repeat"] is True
    assert report["next_action"] == "run_identical_seed1_local_gate"
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/test_autond_typed_invp_gate.py::test_typed_invp_gate_supports_candidate_above_all_controls -q
```

Expected: collection fails because
`blockcipher_nd.planning.autond_typed_invp_gate` does not exist.

- [ ] **Step 3: Implement result loading and protocol validation**

Implement these public constants and function:

```python
MODEL_ROLES = {
    "autond": "autond_dbitnet2023",
    "candidate": "present_nibble_invp_only_spn_only",
    "shuffled_p": "present_nibble_shuffled_paligned_spn_only",
    "delta_only": "present_nibble_delta_only_spn_only",
}


def gate_autond_typed_invp(
    results_path: Path,
    *,
    expected_rows: int = 4,
    required_margin: float = 0.01,
    train_rows: int = 16_384,
    validation_rows: int = 4_096,
    final_repeats: int = 3,
    final_rows: int = 4_096,
) -> dict[str, Any]:
    rows = _load_jsonl(results_path)
    errors = _row_set_errors(rows, expected_rows=expected_rows)
    by_model = {str(row.get("selected_model")): row for row in rows}
    for model in MODEL_ROLES.values():
        if model not in by_model:
            errors.append(f"missing_model={model}")
            continue
        errors.extend(
            _protocol_errors(
                by_model[model],
                train_rows=train_rows,
                validation_rows=validation_rows,
                final_repeats=final_repeats,
                final_rows=final_rows,
            )
        )
    if errors:
        return {
            "status": "fail",
            "decision": "invalid_protocol",
            "errors": errors,
            "next_action": "repair_protocol_and_rerun_same_matrix",
        }
    return _decision_report(by_model, required_margin=required_margin)
```

`_protocol_errors` must check exact top-level data fields, all five round-stage
row counts and class counts, `val_loss` checkpoints, carried optimizer state,
continuous/increasing step numbers, three distinct final seeds, exact final
row counts, both labels, and recomputed accuracy/AUC mean plus population
standard deviation.

- [ ] **Step 4: Run the strong-support test and verify GREEN**

Run the command from Step 2. Expected: `1 passed`.

- [ ] **Step 5: Add weak, stop, outlier, and invalid-protocol tests**

Add these assertions with the same helper:

```python
def test_typed_invp_gate_marks_ordered_submargin_result_weak(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    write_rows(
        results,
        {
            "autond_dbitnet2023": [0.54, 0.54, 0.54],
            "present_nibble_invp_only_spn_only": [0.55, 0.55, 0.55],
            "present_nibble_shuffled_paligned_spn_only": [0.545, 0.545, 0.545],
            "present_nibble_delta_only_spn_only": [0.54, 0.54, 0.54],
        },
    )
    assert gate_autond_typed_invp(results)["decision"] == "weak_or_fragile"


def test_typed_invp_gate_stops_when_delta_control_matches_candidate(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    write_rows(
        results,
        {
            "autond_dbitnet2023": [0.53, 0.53, 0.53],
            "present_nibble_invp_only_spn_only": [0.55, 0.55, 0.55],
            "present_nibble_shuffled_paligned_spn_only": [0.54, 0.54, 0.54],
            "present_nibble_delta_only_spn_only": [0.56, 0.56, 0.56],
        },
    )
    assert gate_autond_typed_invp(results)["decision"] == "stop_public_typed_adapter"


def test_typed_invp_gate_downgrades_outlier_driven_mean(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    write_rows(
        results,
        {
            "autond_dbitnet2023": [0.54, 0.54, 0.54],
            "present_nibble_invp_only_spn_only": [0.60, 0.55, 0.55],
            "present_nibble_shuffled_paligned_spn_only": [0.54, 0.56, 0.54],
            "present_nibble_delta_only_spn_only": [0.54, 0.54, 0.54],
        },
    )
    report = gate_autond_typed_invp(results)
    assert report["decision"] == "weak_or_fragile"
    assert report["candidate_above_all_controls_by_repeat"] is False


def test_typed_invp_gate_rejects_broken_optimizer_continuity(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    write_strong_rows(results, break_step_before_round=7)
    report = gate_autond_typed_invp(results)
    assert report["status"] == "fail"
    assert report["decision"] == "invalid_protocol"
    assert any("optimizer_step_continuity" in error for error in report["errors"])
```

- [ ] **Step 6: Run the complete gate test file**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_autond_typed_invp_gate.py -q
```

Expected: all gate tests pass.

### Task 2: Thin Gate CLI

**Files:**

- Create: `src/blockcipher_nd/cli/gate_autond_typed_invp.py`
- Create: `scripts/gate-autond-typed-invp`
- Modify: `tests/test_autond_typed_invp_gate.py`

- [ ] **Step 1: Write the failing CLI test**

```python
def test_typed_invp_gate_cli_writes_report(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    output = tmp_path / "gate.json"
    write_strong_rows(results)

    status = main(["--results", str(results), "--output", str(output)])

    assert status == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["decision"] == "strong_local_support"
```

- [ ] **Step 2: Run the CLI test and verify RED**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/test_autond_typed_invp_gate.py::test_typed_invp_gate_cli_writes_report -q
```

Expected: import fails because the CLI module does not exist.

- [ ] **Step 3: Implement the CLI and wrapper**

The CLI arguments are fixed to:

```python
parser.add_argument("--results", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--expected-rows", type=int, default=4)
parser.add_argument("--required-margin", type=float, default=0.01)
parser.add_argument("--train-rows", type=int, default=16_384)
parser.add_argument("--validation-rows", type=int, default=4_096)
parser.add_argument("--final-repeats", type=int, default=3)
parser.add_argument("--final-rows", type=int, default=4_096)
```

`main` calls `gate_autond_typed_invp`, creates the output parent directory,
writes sorted/indented JSON plus a trailing newline, prints the compact report,
and returns `0` for a protocol-valid decision or `1` for `invalid_protocol`.

The script wrapper must contain only the established project bootstrap pattern:

```python
#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blockcipher_nd.cli.gate_autond_typed_invp import main


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the CLI test and verify GREEN**

Run the command from Step 2. Expected: `1 passed`.

- [ ] **Step 5: Run focused lint and tests**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check \
  src/blockcipher_nd/planning/autond_typed_invp_gate.py \
  src/blockcipher_nd/cli/gate_autond_typed_invp.py \
  tests/test_autond_typed_invp_gate.py
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/test_autond_typed_invp_gate.py \
  tests/test_autond_public_protocol.py -q
```

Expected: Ruff passes and all focused tests pass.

- [ ] **Step 6: Commit and push the gate implementation**

```bash
git add \
  src/blockcipher_nd/planning/autond_typed_invp_gate.py \
  src/blockcipher_nd/cli/gate_autond_typed_invp.py \
  scripts/gate-autond-typed-invp \
  tests/test_autond_typed_invp_gate.py
git commit -m "experiment: add AutoND typed InvP gate"
git push origin main
```

### Task 3: Frozen Four-Row Plan

**Files:**

- Create:
  `configs/experiment/innovation1/innovation1_spn_present_autond_typed_invp_local_gate_seed0.csv`
- Modify: `tests/test_autond_public_protocol.py`

- [ ] **Step 1: Write the failing plan-lock test**

Add a test that loads the CSV with:

```python
plan = Path(
    "configs/experiment/innovation1/"
    "innovation1_spn_present_autond_typed_invp_local_gate_seed0.csv"
)
tasks = build_tasks(parse_args(["--plan", str(plan)]))
```

Then check:

```python
assert [task["model_key"] for task in tasks] == [
    "autond_dbitnet2023",
    "present_nibble_invp_only_spn_only",
    "present_nibble_shuffled_paligned_spn_only",
    "present_nibble_delta_only_spn_only",
]
assert len(tasks) == 4
assert all(task["rounds"] == 9 for task in tasks)
assert all(task["seed"] == 0 for task in tasks)
assert all(task["samples_per_class"] == 8_192 for task in tasks)
assert all(task["train_samples_total"] == 16_384 for task in tasks)
assert all(task["validation_samples_total"] == 4_096 for task in tasks)
assert all(task["final_test_samples_total"] == 4_096 for task in tasks)
assert all(task["final_test_repeats"] == 3 for task in tasks)
assert all(task["dataset_label_mode"] == "random_labels_total" for task in tasks)
assert all(task["negative_mode"] == "random_ciphertext" for task in tasks)
assert all(task["key_rotation_interval"] == 1 for task in tasks)
assert all(task["pretrain_round_sequence"] == (5, 6, 7, 8) for task in tasks)
assert all(task["pretrain_epochs"] == 3 for task in tasks)
assert all(task["checkpoint_metric"] == "val_loss" for task in tasks)
assert all(task["optimizer_state_transition"] == "carry_across_stages" for task in tasks)
```

- [ ] **Step 2: Run the plan test and verify RED**

Expected: failure because the plan CSV does not exist.

- [ ] **Step 3: Create the CSV**

Use the existing AutoND public-code CSV header. Create exactly four rows with
the frozen fields above, `pairs_per_sample=1`,
`feature_encoding=ciphertext_pair_bits`, `sample_structure=independent_pairs`,
`loss=mse`, `learning_rate=0.001`, `optimizer=adam`, and model-specific
`model_options`: `{}` for AutoND and
`{"spn_mixer_depth":2,"activation":"relu","norm":"layernorm"}` for the
three typed rows.

- [ ] **Step 4: Run plan and focused regression tests**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/test_autond_public_protocol.py \
  tests/test_autond_typed_invp_gate.py \
  tests/test_autond_dbitnet2023.py -q
git diff --check
```

Expected: all focused tests pass and no whitespace errors.

- [ ] **Step 5: Commit and push the plan**

```bash
git add \
  configs/experiment/innovation1/innovation1_spn_present_autond_typed_invp_local_gate_seed0.csv \
  tests/test_autond_public_protocol.py
git commit -m "experiment: plan AutoND typed InvP local gate"
git push origin main
```

### Task 4: Execute the Local Matrix

**Files:**

- Generate under:
  `outputs/local_smoke/i1_present_autond_typed_invp_local_gate_seed0/`
- Generate cache under:
  `outputs/local_cache/i1_present_autond_typed_invp_local_gate_seed0/`

- [ ] **Step 1: Run the frozen four-row matrix**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_autond_typed_invp_local_gate_seed0.csv \
  --epochs 3 \
  --batch-size 256 \
  --hidden-bits 32 \
  --device cpu \
  --learning-rate 0.001 \
  --optimizer adam \
  --amsgrad \
  --optimizer-state-transition carry_across_stages \
  --weight-decay 0 \
  --loss mse \
  --lr-scheduler none \
  --checkpoint-metric val_loss \
  --restore-best-checkpoint \
  --early-stopping-patience 0 \
  --early-stopping-min-delta 0.0 \
  --train-eval-interval 0 \
  --pretrain-round-sequence "[5,6,7,8]" \
  --pretrain-epochs 3 \
  --train-samples-total 16384 \
  --validation-samples-total 4096 \
  --final-test-samples-total 4096 \
  --final-test-repeats 3 \
  --dataset-label-mode random_labels_total \
  --sample-structure independent_pairs \
  --negative-mode random_ciphertext \
  --key-rotation-interval 1 \
  --dataset-cache-root outputs/local_cache/i1_present_autond_typed_invp_local_gate_seed0 \
  --dataset-cache-chunk-size 2048 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_autond_typed_invp_local_gate_seed0/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_autond_typed_invp_local_gate_seed0/progress.jsonl
```

Expected: four result rows and no exception. This remains a local diagnostic
even though each training split has `16384` total rows.

- [ ] **Step 2: Validate plan alignment**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_autond_typed_invp_local_gate_seed0.csv \
  --results outputs/local_smoke/i1_present_autond_typed_invp_local_gate_seed0/results.jsonl \
  --expected-rows 4 \
  --output outputs/local_smoke/i1_present_autond_typed_invp_local_gate_seed0/validation.json
```

Expected: `status=pass`, `result_rows=4`, `errors=[]`.

- [ ] **Step 3: Generate curves and history**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-results \
  --results outputs/local_smoke/i1_present_autond_typed_invp_local_gate_seed0/results.jsonl \
  --output outputs/local_smoke/i1_present_autond_typed_invp_local_gate_seed0/curves.svg \
  --history-csv outputs/local_smoke/i1_present_autond_typed_invp_local_gate_seed0/history.csv \
  --title i1_present_autond_typed_invp_local_gate_seed0
```

Expected: SVG and CSV are non-empty.

- [ ] **Step 4: Run the typed gate**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/gate-autond-typed-invp \
  --results outputs/local_smoke/i1_present_autond_typed_invp_local_gate_seed0/results.jsonl \
  --output outputs/local_smoke/i1_present_autond_typed_invp_local_gate_seed0/typed_gate.json
```

Expected: protocol status passes and decision is exactly one of
`strong_local_support`, `weak_or_fragile`, or
`stop_public_typed_adapter`.

- [ ] **Step 5: Audit within-matrix cache reuse**

Parse `progress.jsonl` and require all 13 train/validation/final splits for the
first row to be reused by each of the remaining three model rows:

```text
4 target/curriculum train splits + 1 target train split = 5 train splits
4 target/curriculum validation splits + 1 target validation split = 5 validation splits
3 final test splits
expected reused splits = 13 * 3 = 39
```

If the observed reuse count is below 39, report the exact missing
`(rounds, split, seed)` identities and stop before interpreting metrics.

### Task 5: Verify and Record the Result

**Files:**

- Modify:
  `docs/experiments/innovation1-present-autond-typed-invp-local-gate-plan.md`

- [ ] **Step 1: Run fresh focused verification**

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest \
  tests/test_autond_typed_invp_gate.py \
  tests/test_autond_public_protocol.py \
  tests/test_autond_dbitnet2023.py \
  tests/test_dataset_cache_workers.py \
  tests/test_training_metrics.py -q
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check \
  src/blockcipher_nd/planning/autond_typed_invp_gate.py \
  src/blockcipher_nd/cli/gate_autond_typed_invp.py \
  tests/test_autond_typed_invp_gate.py
git diff --check
```

Expected: all focused tests and Ruff pass; no whitespace errors.

- [ ] **Step 2: Update the experiment plan from artifacts**

Record:

```text
run id and artifact paths
four per-model validation metrics
three fresh-test accuracy/AUC values per model
fresh-test means and population standard deviations
candidate margins versus each control
protocol integrity and cache reuse count
decision and claim scope
one concrete next action selected from the frozen branches
```

Do not describe the run as formal, paper-scale, strict-negative, or a novelty
result.

- [ ] **Step 3: Commit and push the completed result record**

```bash
git add docs/experiments/innovation1-present-autond-typed-invp-local-gate-plan.md
git commit -m "experiment: adjudicate AutoND typed InvP local gate"
git push origin main
```

- [ ] **Step 4: Continue only through the frozen branch**

```text
strong_local_support -> plan and run identical seed1 local gate
weak_or_fragile      -> run seed1 only as bounded variance adjudication
stop                 -> do not scale or redesign this public-protocol adapter
invalid_protocol     -> repair and rerun the same seed0 matrix
```

The active paper-scale AutoND watcher remains independent throughout all four
branches.
