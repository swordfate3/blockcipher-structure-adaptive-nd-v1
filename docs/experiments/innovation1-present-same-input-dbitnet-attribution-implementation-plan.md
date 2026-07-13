# PRESENT Same-Input DBitNet Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. Do not dispatch subagents for this
> project task.

**Goal:** Implement and adjudicate E3-R1, a strict four-role comparison of the
current InvP Token-Mixer against a DBitNet-2023 learner over the same mapped
16-pair PRESENT Case2 differences.

**Architecture:** Add one thin PRESENT mapped-delta wrapper around the existing
`AutoNDDBitNet2023Distinguisher`, three fixed mapping variants, and one
four-role strict gate built on the shared Case2 attribution protocol. Reuse the
existing matrix runner, disk cache, plotting, result validation, and generic
four-role gate infrastructure.

**Tech Stack:** Python 3.10, PyTorch, pytest, CSV experiment matrices, JSONL
result/progress artifacts, repository evaluation CLIs.

**Status:** Completed and pushed to `origin/main` through commit `a127546`.

---

### Task 1: Add mapped-delta DBitNet behavior tests

**Files:**
- Create: `tests/test_present_same_input_dbitnet.py`
- Create: `src/blockcipher_nd/models/structure/spn/present_same_input_dbitnet.py`

- [x] Write failing tests that reconstruct true, shuffled, and raw mapped
  1024-bit views directly from 2048 raw bits.
- [x] Verify the tests fail because the new module and classes do not exist:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_present_same_input_dbitnet.py
```

- [x] Implement `PresentMappedDeltaDBitNet2023Distinguisher` with strict input
  validation, a non-persistent mapping buffer, a `mapped_delta_view` method,
  delegation to `AutoNDDBitNet2023Distinguisher(input_bits=1024)`, and delegated
  dense-kernel auxiliary loss.
- [x] Add fixed subclasses for true InvP, shuffled-P, and raw Delta mappings.
- [x] Verify exact views, shape errors, finite forward/backward, equal capacity,
  equal common initialization, and differing mapping buffers pass.

### Task 2: Register the three fixed models

**Files:**
- Modify: `src/blockcipher_nd/models/structure/spn/__init__.py`
- Modify: `src/blockcipher_nd/models/structure/__init__.py`
- Modify: `src/blockcipher_nd/registry/model_families/spn.py`
- Modify: `tests/test_present_same_input_dbitnet.py`

- [x] Add failing registry tests for:

```text
present_invp_dbitnet2023
present_shuffled_p_dbitnet2023
present_raw_delta_dbitnet2023
```

- [x] Verify the tests fail with `unsupported model`.
- [x] Export and register the three models with no optional architecture
  expansion; forward only the fixed raw-pair geometry and existing DBitNet
  defaults.
- [x] Verify registry construction and model metadata tests pass.

### Task 3: Implement the strict E3-R1 gate

**Files:**
- Create: `tests/test_present_same_input_dbitnet_gate.py`
- Create: `src/blockcipher_nd/planning/present_same_input_dbitnet_gate.py`
- Create: `src/blockcipher_nd/cli/gate_present_same_input_dbitnet.py`
- Create: `scripts/gate-present-same-input-dbitnet`

- [x] Write failing tests for exact roles, R0 neutrality, seed0 promotion,
  weak-margin stop, comparator loss, wrong effective key schedule, missing row,
  unequal DBitNet capacity, incomplete histories, and CLI JSON writing.
- [x] Verify failures are caused by the absent gate module.
- [x] Implement a `FourRoleGateSpec` using
  `PRESENT_CASE2_ATTRIBUTION_PROTOCOL`, the `+0.003` seed0 margins, and the
  documented seed1/stop decisions.
- [x] Implement the thin CLI and executable script following the H2 gate
  pattern.
- [x] Verify focused gate tests pass.

### Task 4: Freeze R0 and R1 matrices

**Files:**
- Create: `configs/experiment/innovation1/innovation1_spn_present_same_input_dbitnet_smoke_seed0.csv`
- Create: `configs/experiment/innovation1/innovation1_spn_present_same_input_dbitnet_8192_seed0.csv`
- Create: `configs/experiment/innovation1/innovation1_spn_present_same_input_dbitnet_8192_seed1.csv`
- Modify: `tests/test_present_same_input_dbitnet_gate.py`

- [x] Write failing matrix-lock tests for four exact roles, r7, 16 pairs,
  strict negatives, Case2 official MCND data, frozen optimizer/checkpoint
  settings, `64/class` R0, and `8192/class` R1.
- [x] Add the three exact CSV matrices. Seed1 is prepared but must not execute
  unless seed0 returns `promote_seed1`.
- [x] Verify matrix parsing and plan-alignment tests pass.

### Task 5: Run R0 readiness

**Artifacts:**
- `outputs/local_smoke/i1_present_same_input_dbitnet_smoke_seed0/`
- `outputs/local_cache/i1_present_same_input_dbitnet_smoke_seed0/`

- [x] Run the four-row one-epoch R0 matrix on CPU with disk cache.
- [x] Validate four exact result rows, cache create/reuse, histories,
  checkpoints, progress, and effective key metadata.
- [x] Generate `history.csv` and `curves.svg` and parse the SVG.
- [x] Replay the strict gate with `--readiness-only`; require
  `decision=implementation_ready` and no research interpretation.

### Task 6: Run and adjudicate R1 seed0

**Artifacts:**
- `outputs/local_smoke/i1_present_same_input_dbitnet_8192_seed0/`
- `outputs/local_cache/i1_present_same_input_dbitnet_8192_seed0/`

- [x] Run the frozen `8192/class`, `4096/class` validation, seed0, 10-epoch
  matrix on CPU.
- [x] Validate four exact rows and generate CSV/SVG artifacts.
- [x] Run the strict E3-R1 gate and record all AUCs and three candidate margins.
- [x] If and only if the decision is `promote_seed1`, execute the prepared
  seed1 matrix and run a joint two-seed gate. Otherwise stop seed1 and every
  larger or remote run.

### Task 7: Record verdict and close the failed residual-focus retry

**Files:**
- Modify: `docs/experiments/innovation1-present-same-input-dbitnet-attribution-design.md`
- Modify: `docs/experiments/innovation1-present-r8-residual-bucket-axis-spectrum-plan.md`
- Modify: `docs/experiments/innovation1-route-verdict-2026-07-09.md`

- [x] Record the bounded remote evidence that residual-focus retry1 has no
  active experiment process, no done marker, and failed at command 4 because
  checkpoint classifier widths `64` and `128` mismatch. Mark original retry as
  stopped and do not repair/relaunch it.
- [x] Record E3-R1 run IDs, artifacts, exact AUCs, margins, gate decision,
  evidence scope, stopped actions, and one concrete next action.
- [x] Keep `8192/class` language diagnostic-only.

### Task 8: Verify, commit, and push

- [x] Run focused tests and the full suite:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

- [x] Run Ruff on changed Python files, `git diff --check`, exact plan/result
  validation, SVG parsing, and strict gate replay.
- [x] Inspect the scoped diff and ensure no unrelated files are staged.
- [x] Commit the implementation/result documentation and push `main` using the
  configured remote. Report any platform rejection exactly; do not use a dirty
  overlay or alternate transfer path.
