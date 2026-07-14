# Innovation 1 Cross-SPN Typed Cell Transfer Seed1 Implementation Plan

**Status:** completed and jointly adjudicated on 2026-07-14

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replicate the E4-R2 PRESENT-to-GIFT typed checkpoint-transfer gate on an independent GIFT target seed, then apply a frozen two-seed adjudication without changing model, source checkpoints, or benchmark semantics.

**Architecture:** Parameterize the existing one-run transfer gate for exactly one frozen target seed (`0` or `1`) while retaining its source-checkpoint and five-role checks. Add a small joint gate that invokes the single-seed gate separately for seed0 and seed1 and advances only when both reports pass all four attribution margins.

**Tech Stack:** Python 3.13, PyTorch, project matrix runner, JSONL/CSV/SVG artifacts, pytest, ruff.

---

## Frozen Question and Scope

```text
Question:
  Does the E4-R2 true PRESENT -> true GIFT transfer margin reproduce on an
  independent GIFT target seed?

Same-budget anchor and controls:
  gift_anchor
  gift_typed_scratch
  true_to_true
  shuffled_to_true
  true_to_shuffled

Only variable:
  target seed 0 -> 1 and an independently generated, parameter-matched GIFT
  disk cache.

Unchanged:
  PRESENT source result/checkpoints and SHA-256 values, GIFT-64 r6, fixed
  input difference, four independent pairs/sample, strict encrypted-random-
  plaintext negatives, 8192/class train, 4096/class validation, 10 epochs,
  MSE/Adam/lr=1e-4/weight-decay=1e-5, best-val-AUC restoration, CPU.
```

The gate thresholds are frozen before seed1 execution:

```text
true_to_true AUC                >= 0.52
true_to_true - gift_anchor      >= +0.003
true_to_true - typed_scratch    >= +0.005
true_to_true - shuffled_to_true >= +0.003
true_to_true - true_to_shuffled >= +0.003
```

No `65536/class`, `262144/class`, remote GPU, formal-scale claim, DDT/trail
route, source checkpoint replacement, or architecture change belongs to this
experiment.

## Files

| File | Purpose |
| --- | --- |
| `src/blockcipher_nd/planning/cross_spn_typed_transfer_gate.py` | Accept a frozen singleton target seed and add a two-seed composition gate. |
| `src/blockcipher_nd/cli/gate_cross_spn_typed_transfer.py` | Parse one target seed for the single-run gate. |
| `src/blockcipher_nd/cli/gate_cross_spn_typed_transfer_joint.py` | Parse explicit seed0/seed1 result/progress paths for the joint gate. |
| `scripts/gate-cross-spn-typed-transfer-joint` | Thin executable wrapper for the joint CLI. |
| `tests/test_cross_spn_typed_transfer_gate.py` | Prove seed1 validation, seed1 pass behavior, and two-seed decisions. |
| `configs/experiment/innovation1/innovation1_spn_gift64_cross_spn_typed_transfer_smoke_seed1.csv` | Five-row 64/class readiness matrix. |
| `configs/experiment/innovation1/innovation1_spn_gift64_cross_spn_typed_transfer_8192_seed1.csv` | Frozen five-row target-seed1 diagnostic matrix. |
| `docs/experiments/innovation1-cross-spn-typed-cell-transfer-design.md` | Record launch protocol and later adjudication. |

## Task 1: Write Seed-Aware Gate Tests

**Files:**
- Modify: `tests/test_cross_spn_typed_transfer_gate.py`

- [x] Add a `seed` argument to `_write_transfer_run`. Set every synthetic
  result row and every initialization/cache progress event to that exact seed.

- [x] Add a failing test that writes a seed1 five-role report with all margins
  passing, invokes:

```python
report = gate_cross_spn_typed_transfer(
    [results],
    progress_paths=[progress],
    expected_seeds=(1,),
)
assert report["status"] == "pass"
assert report["decision"] == "promote_e4_transfer_joint_gate"
assert report["expected_seeds"] == [1]
```

- [x] Add a failing test that passes seed0 and seed1 fixtures to the joint
  gate and expects:

```python
assert report["status"] == "pass"
assert report["decision"] == "two_seed_transfer_signal_confirmed"
assert report["next_action"] == "design_e4_r3_same_protocol_medium_diagnostic"
```

- [x] Add a failing test in which seed1 fails the source-topology margin and
  expects `two_seed_transfer_unstable_no_scale` with no remote or formal
  advance.

- [x] Run the three new tests before changing production code:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_cross_spn_typed_transfer_gate.py -k 'seed1 or two_seed'
```

Expected: failure because `(1,)` is rejected and the joint function is absent.

## Task 2: Implement Seed-Aware Single and Joint Gates

**Files:**
- Modify: `src/blockcipher_nd/planning/cross_spn_typed_transfer_gate.py`

- [x] Change `_argument_errors` to accept exactly `(0,)` or `(1,)` for a
  single-run gate. Preserve the existing `64/class, 1 epoch` readiness budget
  and `8192/class, 10 epochs` diagnostic budget.

- [x] Thread `expected_seed=expected_seeds[0]` into `_result_errors` and
  `_progress_errors`; replace each hard-coded target `seed: 0` comparison with
  `seed: expected_seed`.

- [x] Preserve the seed0 success result:

```python
return "promote_e4_transfer_seed1", "freeze_identical_e4_r2_seed1_repeat"
```

  For seed1 success return:

```python
return "promote_e4_transfer_joint_gate", "run_frozen_e4_r2_joint_gate"
```

- [x] Add `gate_cross_spn_typed_transfer_joint`, accepting exactly two result
  and two progress paths in seed order `(0, 1)`. It must call the single-run
  gate twice, return both reports under `per_seed`, and decide:

```python
if both successful decisions are the frozen seed0/seed1 promotions:
    return "two_seed_transfer_signal_confirmed", (
        "design_e4_r3_same_protocol_medium_diagnostic"
    )
return "two_seed_transfer_unstable_no_scale", (
    "stop_e4_transfer_scale_after_two_seed_variance"
)
```

  Both outcomes prohibit remote and formal claims. The unstable outcome also
  prohibits sample scale.

- [x] Re-run the tests from Task 1 and the full transfer-gate unit file:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_cross_spn_typed_transfer_gate.py
```

Expected: all pass.

## Task 3: Add CLIs and Frozen Seed1 Matrices

**Files:**
- Modify: `src/blockcipher_nd/cli/gate_cross_spn_typed_transfer.py`
- Create: `src/blockcipher_nd/cli/gate_cross_spn_typed_transfer_joint.py`
- Create: `scripts/gate-cross-spn-typed-transfer-joint`
- Create: `configs/experiment/innovation1/innovation1_spn_gift64_cross_spn_typed_transfer_smoke_seed1.csv`
- Create: `configs/experiment/innovation1/innovation1_spn_gift64_cross_spn_typed_transfer_8192_seed1.csv`

- [x] Add `--expected-seed` with default `0` to the single-run CLI and forward
  it as `expected_seeds=(args.expected_seed,)`.

- [x] Create the joint CLI with required `--seed0-results`, `--seed0-progress`,
  `--seed1-results`, `--seed1-progress`, and `--output`; write sorted,
  newline-terminated JSON and return nonzero only for invalid protocol.

- [x] Copy the five exact seed0 rows into each seed1 CSV, changing only:

```text
seed = 1
family = gift64_cross_spn_typed_transfer_smoke_seed1
network labels = E4-R2-R0-Seed1
```

  for readiness, and:

```text
seed = 1
family = gift64_cross_spn_typed_transfer_8192_seed1
network labels = E4-R2-Seed1
```

  for the diagnostic. Keep all model keys, rank order, source manifest,
  difference profile, optimizer, loss, data fields, and options byte-for-byte
  equivalent to seed0.

- [x] Add CLI and matrix tests asserting all five rows have seed1 and the
  target source manifest remains unchanged.

- [x] Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q \
  tests/test_cross_spn_typed_transfer_gate.py \
  tests/test_checkpoint_initialization.py
UV_CACHE_DIR=/tmp/uv-cache uv run ruff check \
  src/blockcipher_nd/planning/cross_spn_typed_transfer_gate.py \
  src/blockcipher_nd/cli/gate_cross_spn_typed_transfer.py \
  src/blockcipher_nd/cli/gate_cross_spn_typed_transfer_joint.py \
  tests/test_cross_spn_typed_transfer_gate.py
git diff --check
```

## Task 4: Run Seed1 Readiness and Publish It

**Files:**
- Modify: `docs/experiments/innovation1-cross-spn-typed-cell-transfer-design.md`

- [x] Run five `64/class`, one-epoch CPU rows with a new seed1 cache and the
  unchanged seed0 source manifest. Write result/progress/checkpoint/CSV/SVG
  artifacts under `outputs/local_smoke/i1_gift64_cross_spn_typed_transfer_r0_seed1/`.

- [x] Validate five rows, parse the SVG XML, and run the seed-aware gate with
  `--expected-seed 1 --readiness-only`. Require `implementation_ready`; never
  interpret its metrics.

- [x] Commit only task-scoped code, tests, scripts, configs, and the plan,
  then push `origin/main`. Do not run the 8192/class diagnostic from
  unpublished code.

## Task 5: Run and Adjudicate Seed1 Diagnostic

**Files:**
- Modify: `docs/experiments/innovation1-cross-spn-typed-cell-transfer-design.md`
- Modify: `docs/experiments/innovation1-route-verdict-2026-07-09.md`

- [x] Run the five seed1 `8192/class` rows locally on CPU using a new cache:

```text
outputs/local_cache/i1_gift64_cross_spn_typed_cell_r1_seed1
```

  and output root:

```text
outputs/local_smoke/i1_gift64_cross_spn_typed_transfer_r2_seed1
```

- [x] Generate validation JSON, history CSV, SVG, parse the SVG, run the
  seed1 gate, then run the joint gate over the completed seed0 and seed1
  result/progress artifacts.

- [x] Document both five-role metrics, four margins per seed, source SHA-256,
  cache roots, validators, gates, claim scope, and the joint verdict.

- [x] Commit/push the adjudication. Report a concrete next action. A positive
  two-seed gate may authorize only an E4-R3 medium-diagnostic plan; it does not
  itself authorize remote scale or a formal/SOTA claim.

## Review

- The task changes only the target random seed and gate bookkeeping.
- All data, labels, negative construction, transfer source tensors, model
  capacity, optimizer, and selection metric are frozen.
- The joint gate prevents a seed0-only positive from being scaled as if it were
  replicated evidence.
- The result remains diagnostic until a future plan supplies larger completed,
  retrieved, multi-seed evidence.

## Completion Record

```text
seed1 true-to-true AUC             = 0.575072139502
seed1 true-to-true accuracy        = 0.551513671875
seed1 calibrated accuracy          = 0.557128906250
seed1 best epoch                   = 7
seed1 vs anchor                    = +0.023235499859
seed1 vs scratch                   = +0.011130839586
seed1 vs source-shuffled           = +0.015329629183
seed1 vs target-shuffled           = +0.057054758072

joint status                       = pass
joint decision                     = two_seed_transfer_signal_confirmed
joint next_action                  = design_e4_r3_same_protocol_medium_diagnostic
```

The unchanged source SHA-256 values are recorded in the parent E4 design.
Seed1 validation, SVG parsing, the 50-row history export, the seed-aware gate,
and the two-seed joint gate all passed. Evidence is under:

```text
outputs/local_smoke/i1_gift64_cross_spn_typed_transfer_r2_seed1/
outputs/local_smoke/i1_gift64_cross_spn_typed_transfer_r2_joint_seed0_seed1/gate.json
```

The completed implementation and gate were published in commit `5eb59a2`.
The next work is a separately frozen E4-R3 same-protocol medium diagnostic;
this implementation record does not authorize remote or formal-scale claims.
