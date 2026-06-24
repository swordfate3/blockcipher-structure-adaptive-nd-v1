# Experiments Thin CLI Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `experiments/` a thin CLI and experiment-asset directory by moving reusable training, dataset, audit, and validation logic into `src/blockcipher_nd/`.

**Architecture:** CLI scripts in `experiments/` parse arguments and call importable package APIs. Core matrix execution lives under `blockcipher_nd.experiments`, while Innovation 1 route-specific logic lives under `blockcipher_nd.experiments.innovation1`. Tests import package modules directly except when explicitly exercising CLI subprocess behavior.

**Tech Stack:** Python, NumPy, PyTorch, pytest, uv-managed project environment.

---

### Task 1: Move Matrix Runner Core

**Files:**
- Create: `src/blockcipher_nd/experiments/matrix_runner.py`
- Modify: `experiments/run_innovation_one_matrix.py`
- Modify: `tests/test_experiment_matrix_runner.py`

- [ ] Move all non-trivial matrix runner functions from `experiments/run_innovation_one_matrix.py` into `matrix_runner.py`.
- [ ] Expose `parse_args(argv=None)` and `main(argv=None)` from the package module.
- [ ] Replace the experiment script with a thin wrapper that imports and calls `main`.
- [ ] Update tests that import helper functions to import from `blockcipher_nd.experiments.matrix_runner`.
- [ ] Keep subprocess tests pointed at the CLI script to verify the executable entry remains usable.

### Task 2: Move Candidate Evidence Baseline Core

**Files:**
- Create: `src/blockcipher_nd/experiments/innovation1/spn_candidate_evidence_baseline.py`
- Create: `src/blockcipher_nd/experiments/innovation1/__init__.py`
- Modify: `experiments/innovation1/run_spn_candidate_evidence_baseline.py`
- Modify: `tests/test_spn_candidate_evidence.py`

- [ ] Move dataset generation, feature cache, pair generation, baseline model, training, metrics, and result assembly into the package module.
- [ ] Expose `make_candidate_dataset`, `make_cached_candidate_dataset`, `binary_accuracy`, `binary_auc`, `parse_args(argv=None)`, and `main(argv=None)`.
- [ ] Replace the experiment script with a thin wrapper.
- [ ] Update tests to import `make_candidate_dataset` from the package module.

### Task 3: Move Audit And Alignment Core

**Files:**
- Create: `src/blockcipher_nd/experiments/innovation1/spn_feature_separation_audit.py`
- Create: `src/blockcipher_nd/experiments/innovation1/result_plan_alignment.py`
- Modify: `experiments/innovation1/audit_spn_feature_separation.py`
- Modify: `experiments/innovation1/validate_result_plan_alignment.py`
- Modify: `tests/test_spn_feature_separation_audit.py`
- Modify: `tests/test_result_plan_alignment.py`

- [ ] Move SPN feature audit reusable functions into the package module and keep the script as a wrapper.
- [ ] Move result-plan alignment reusable functions into the package module and keep the script as a wrapper.
- [ ] Update tests to import package modules directly.

### Task 4: Document The Boundary

**Files:**
- Modify: `README.md`
- Modify: `experiments/innovation1/README.md`

- [ ] State that `experiments/` is for CLI wrappers and plan/config assets only.
- [ ] Point reusable APIs to `src/blockcipher_nd/experiments/`.
- [ ] Mention that training/data/cache/feature logic belongs in `src`, not in CLI scripts.

### Task 5: Verify

**Commands:**
- `uv run pytest tests/test_spn_candidate_evidence.py tests/test_spn_feature_separation_audit.py tests/test_result_plan_alignment.py -q`
- `uv run pytest tests/test_experiment_matrix_runner.py -q`
- `uv run pytest -q`

- [ ] Fix failures caused by import paths, duplicate parser behavior, or moved module names.
- [ ] Confirm the remaining `experiments/**/*.py` files are thin wrappers or small CLI utilities.
