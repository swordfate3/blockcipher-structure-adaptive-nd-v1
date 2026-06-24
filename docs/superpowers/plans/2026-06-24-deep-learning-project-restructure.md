# Deep Learning Project Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the repository layout as a clean deep-learning project with config-driven experiments, package-level runners/tasks, and thin command entrypoints.

**Architecture:** Move executable experiment logic out of `experiments/` and out of `blockcipher_nd.experiments`. Use `engine/` for training orchestration, `tasks/` for concrete experiment tasks, `planning/` for plans/matrices/alignment, `registry/` for ciphers/models/features, `remote/` for remote script generation and monitoring, and `cli/` for user-facing commands. Put experiment definitions under root-level `configs/`.

**Tech Stack:** Python, PyTorch, NumPy, pytest, uv.

---

### Task 1: Create New Package Skeleton

**Files:**
- Create: `src/blockcipher_nd/engine/__init__.py`
- Create: `src/blockcipher_nd/tasks/__init__.py`
- Create: `src/blockcipher_nd/tasks/innovation1/__init__.py`
- Create: `src/blockcipher_nd/planning/__init__.py`
- Create: `src/blockcipher_nd/registry/__init__.py`
- Create: `src/blockcipher_nd/remote/__init__.py`
- Create: `src/blockcipher_nd/cli/__init__.py`
- Create: `configs/README.md`
- Create: `configs/experiment/innovation1/README.md`
- Create: `configs/remote/README.md`
- Create: `scripts/README.md`

- [ ] Create the target directories.
- [ ] Add `__init__.py` files that document each package boundary.
- [ ] Add config README files defining root-level config ownership.

### Task 2: Move Runner And Task Logic

**Files:**
- Move: `src/blockcipher_nd/experiments/matrix_runner.py` -> `src/blockcipher_nd/engine/matrix_runner.py`
- Move: `src/blockcipher_nd/experiments/smoke_runner.py` -> `src/blockcipher_nd/tasks/smoke.py`
- Move: `src/blockcipher_nd/experiments/innovation1/spn_candidate_evidence_baseline.py` -> `src/blockcipher_nd/tasks/innovation1/spn_candidate_evidence.py`
- Move: `src/blockcipher_nd/experiments/innovation1/spn_active_pattern_baseline.py` -> `src/blockcipher_nd/tasks/innovation1/spn_active_pattern.py`
- Move: `src/blockcipher_nd/experiments/innovation1/spn_feature_separation_audit.py` -> `src/blockcipher_nd/tasks/innovation1/spn_feature_audit.py`
- Move: `src/blockcipher_nd/experiments/innovation1/zhang_wang_official_checkpoint.py` -> `src/blockcipher_nd/tasks/innovation1/zhang_wang_checkpoint.py`
- Move: `src/blockcipher_nd/experiments/innovation1/result_plan_alignment.py` -> `src/blockcipher_nd/planning/result_alignment.py`

- [ ] Move files mechanically.
- [ ] Update imports from `blockcipher_nd.experiments.*` to the new packages.
- [ ] Keep `main(argv=None)` APIs for CLI modules.

### Task 3: Split Registries From Experiment Namespace

**Files:**
- Move: `src/blockcipher_nd/experiments/factories.py` -> `src/blockcipher_nd/registry/factories.py`
- Move: `src/blockcipher_nd/experiments/difference_profiles.py` -> `src/blockcipher_nd/registry/difference_profiles.py`
- Modify: `src/blockcipher_nd/experiments/__init__.py`
- Modify imports throughout `src/`.

- [ ] Move factory and difference profile code into `registry/`.
- [ ] Update package exports.
- [ ] Leave no new code importing `blockcipher_nd.experiments`.

### Task 4: Move CLI To Package And Scripts

**Files:**
- Create: `src/blockcipher_nd/cli/train.py`
- Create: `src/blockcipher_nd/cli/smoke.py`
- Create: `src/blockcipher_nd/cli/audit_spn_features.py`
- Create: `src/blockcipher_nd/cli/spn_candidate_evidence.py`
- Create: `src/blockcipher_nd/cli/spn_active_pattern.py`
- Create: `src/blockcipher_nd/cli/validate_results.py`
- Create: `src/blockcipher_nd/cli/evaluate_zhang_wang_checkpoint.py`
- Replace root scripts in `scripts/` with thin Python entrypoints or shell-neutral Python wrappers.

- [ ] Move all primary executable entrypoints to package `cli/`.
- [ ] Keep root `scripts/` small and human-readable.
- [ ] Remove old `experiments/` Python wrapper directory.

### Task 5: Move Config Assets And Clean Generated Scripts

**Files:**
- Move: `experiments/innovation1/plans/*.csv` -> `configs/experiment/innovation1/*.csv`
- Move: `experiments/innovation1/configs/remote/*.json` -> `configs/remote/*.json`
- Delete: `scripts/generated/`
- Delete: `experiments/`

- [ ] Move current config assets to root `configs/`.
- [ ] Remove generated remote scripts from source tree.
- [ ] Ensure README explains generated scripts belong under `outputs/` or `runs/`.

### Task 6: Update Tests And Docs

**Files:**
- Modify: `tests/test_experiments_thin_cli_refactor.py` or replace with `tests/test_project_structure.py`
- Modify: `README.md`
- Modify: `MIGRATION_MINIMAL.md`

- [ ] Rewrite tests around the new structure.
- [ ] Assert there is no root `experiments/` directory and no `scripts/generated/`.
- [ ] Assert importable package APIs come from `engine`, `tasks`, `planning`, and `registry`.
- [ ] Update docs to show the new deep-learning project layout.

### Task 7: Verify

**Commands:**
- `uv run pytest -q`
- `uv run python scripts/train --help`
- `uv run python scripts/spn-candidate-evidence --help`
- `uv run python scripts/validate-results --help`

- [ ] Fix import and path failures.
- [ ] Remove `__pycache__` and `.pytest_cache`.
- [ ] Report that this workspace has no usable Git repository, so commit/push cannot be completed here.
