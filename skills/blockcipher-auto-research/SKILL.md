---
name: blockcipher-auto-research
description: Project-local autonomous research workflow for this blockcipher neural distinguisher repository. Use when working on controlled experiments, baselines, feature/model/training ablations, JSONL/SVG result generation, SPN/PRESENT evidence interpretation, remote GPU handoff rules, or any request to run/compare/improve blockcipher_nd research experiments.
---

# Blockcipher Auto Research

## Overview

Use this skill to run a bounded, reproducible research loop for `blockcipher-structure-adaptive-nd-v1`: establish a baseline, change one hypothesis at a time, train with fixed settings, record JSONL/CSV/SVG artifacts, apply evidence gates, and commit/push code changes after verification.

This skill adapts the general `auto-research` idea to this repository. Do not assume a single editable `train.py`; this project uses thin CLI wrappers in `scripts/` and implementation under `src/blockcipher_nd/`.

## First Checks

Before editing or running experiments:

1. Read `AGENTS.md`, `.learnings/LEARNINGS.md`, and `.learnings/ERRORS.md` if they exist.
2. Check git state with `git status --short --branch`.
3. Use `UV_CACHE_DIR=/tmp/uv-cache uv run ...` for project commands in this environment.
4. Keep generated artifacts under `outputs/`, `runs/`, or another ignored result directory.
5. Do not include unrelated dirty files in commits.

## Project Map

Use these boundaries when choosing what to inspect or edit:

- `scripts/`: thin CLI entrypoints only.
- `src/blockcipher_nd/cli/`: argument parsing and command orchestration.
- `src/blockcipher_nd/engine/`: task/matrix execution, dataset/model assembly, result row construction.
- `src/blockcipher_nd/data/`: differential data generation, validation, and disk cache.
- `src/blockcipher_nd/features/`: input organization and cipher-structure feature encoders.
- `src/blockcipher_nd/models/`: neural distinguisher architectures.
- `src/blockcipher_nd/training/`: PyTorch training loop, optimizer, metrics, and typed results.
- `src/blockcipher_nd/evaluation/`: JSONL summaries, validation, SVG plots, and CSV export.
- `configs/experiment/innovation1/`: experiment plans and matrices.
- `docs/`: human-facing research notes and project documentation.

## Experiment Workflow

Follow this loop for local research work:

1. Define the question: cipher, rounds, model, feature route, samples per class, seeds, epochs, and primary metric.
2. Run or identify the baseline with the same budget.
3. Change one idea at a time. Keep diffs attributable to one hypothesis.
4. Run the fixed-budget training command.
5. Generate or update plot artifacts from JSONL when useful.
6. Compare against the baseline using the same metric and evidence scale.
7. Keep the change only if it improves the metric, simplifies code without harming quality, or fixes correctness/reproducibility.
8. Run relevant tests.
9. Commit scoped source/doc/test/config changes and push when a remote exists.

Default local smoke command shape:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --ciphers speck32 \
  --models mlp \
  --rounds 1 \
  --seeds 0 \
  --samples-per-class 64 \
  --pairs-per-sample 1 \
  --epochs 3 \
  --batch-size 16 \
  --hidden-bits 16 \
  --device cpu \
  --output outputs/<run_name>/results.jsonl
```

Default plot command shape:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-results \
  --results outputs/<run_name>/results.jsonl \
  --output outputs/<run_name>/curves.svg \
  --history-csv outputs/<run_name>/history.csv \
  --title <run_name>
```

## Metrics

For binary neural distinguishers, treat these as the normal result signals:

- Primary local comparison: `val_auc`, then `val_accuracy`.
- Supporting signals: `val_loss`, `train_auc`, `train_accuracy`, and calibration/best-accuracy fields.
- Overfitting check: compare train metrics against validation metrics across epochs.
- Report per-epoch history when discussing training behavior, not just final metrics.

Never improve a result by changing validation data, labels, negative-sample definition, metric computation, or plan-alignment logic unless the user explicitly asks to redesign the benchmark.

## Evidence Gates

Use precise language for experiment scale:

- Tiny or small runs are smoke tests or diagnostics, not formal training.
- For SPN/PRESENT, distinguish total rows from `samples_per_class`.
- Do not call `8k`, `16k`, `32k`, or `65k samples_per_class` SPN/PRESENT runs definitive failures.
- Before claiming a SPN/PRESENT ceiling, require completed and retrieved plan-aligned scale evidence.
- A reasonable SPN/PRESENT ladder is `65536/class -> 262144/class`; formal claims should use at least `1000000/class` and multiple seeds.
- Keep ARX/SPECK evidence separate from SPN/PRESENT evidence.
- Strict negative samples must be encrypted random plaintexts. Random ciphertext negatives are only ablation/control evidence.
- Multi-query aggregation is application-level evidence, not raw single-sample SOTA evidence.

When evidence is incomplete, say exactly whether it is planned, running, completed remotely, fallback-retrieved, retrieved from a verified result branch, and plan-aligned.

## Remote GPU Rules

For remote Windows GPU work:

- Remote alias: `lxy-a6000`.
- Project-owned files and run artifacts must stay under `G:\lxy`.
- Prefer `G:\lxy\blockcipher-structure-adaptive-nd` for code and `G:\lxy\blockcipher-structure-adaptive-nd-runs` for outputs.
- Generated Windows launch commands must use `cmd.exe /c`, not `cmd.exe /k`.
- Medium or larger remote jobs must have disk-backed dataset/feature cache, metadata, progress logging, and parameter-matched reuse/resume behavior before launch.
- Remote experiments should normally launch from a pushed GitHub commit.
- Do not SSH-poll from the main thread after a remote launch or handoff is recorded; use supported local monitors/retrieval workflows.

## Commit Discipline

After code, config, docs, tests, generated-script, or learning-rule edits:

1. Run the smallest relevant verification, normally:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q
```

2. Inspect scoped diff:

```bash
git diff -- <paths>
```

3. Stage only task-related files.
4. Commit with a concise message.
5. Push to the configured remote when available.

Ignored result artifacts under `outputs/` do not need commits unless the user explicitly asks to version them.

## Reporting

For each experiment or research change, report:

- Hypothesis tested.
- Exact command or plan used.
- Output artifact paths.
- Baseline metric versus new metric.
- Status: keep, discard, crash, or diagnostic only.
- Verification command and result.
- Commit hash when code/doc/test/config changes were made.

Use Chinese for user-facing summaries in this project unless the user asks otherwise.
