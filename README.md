# Block Cipher Structure-Adaptive Neural Distinguisher

面向毕业论文“创新一”的分组密码结构适配神经差分区分器项目。

当前研究主线：

```text
面向分组密码结构与输入组织联合适配的神经差分区分器方法
```

本项目按深度学习工程结构组织：配置描述实验，`src/` 提供可复用组件和训练引擎，
`scripts/` 只作为薄命令入口。

## Install

```bash
uv sync
uv run pytest -q
```

GPU quick check:

```bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

## Project Layout

```text
configs/
  experiment/innovation1/       Innovation 1 experiment matrices/config assets
  remote/                       remote launch specs

src/blockcipher_nd/
  ciphers/                      reduced-round block cipher implementations
  data/                         differential dataset specs, generation, cache
    differential/               config, metadata, validation, row samplers, key helpers
    cache/                      disk-backed dataset materialization/reuse
  features/                     feature facade, encoder families, structure-aware extractors
    encoders/                   bitwise, ARX, SPN/PRESENT feature encoders
  models/                       baseline and structure-adaptive distinguishers
    structure/spn/              concrete SPN/PRESENT pair-set architectures
      present_inception_*.py    MCND blocks and architecture variants
  training/                     PyTorch trainer, metrics, loaders, optimizers, config types
  evaluation/                   result summaries
  registry/                     cipher/model factories, HPO options, cipher/difference profiles
    model_families/             baseline, pairset, ARX, SPN, MoE model builders
  engine/                       runner orchestration, dataset cache hooks, progress events
    task_config.py              task-to-training/dataset config adapters
    pretraining.py              optional curriculum pretraining stage
    results.py                  JSONL result assembly
  tasks/                        concrete train/audit/eval tasks
    innovation1/spn_candidate/  candidate-evidence dataset/cache and baseline internals
  planning/                     matrix plan loading and result-plan alignment
  remote/                       remote execution support
  cli/                          package-backed command modules

scripts/
  train                         main matrix training CLI
  smoke                         small local smoke run
  spn-candidate-evidence        PRESENT candidate-evidence route
  spn-active-pattern            PRESENT active-pattern route
  audit-spn-features            SPN feature separation audit
  validate-results              result-plan alignment check
  evaluate-zhang-wang-checkpoint

docs/
  experiments/                  experiment notes
  research/                     literature notes

outputs/, runs/                 generated artifacts, not source
```

There is intentionally no root `experiments/` source directory. Experiment
definitions live under `configs/`; implementation lives under
`src/blockcipher_nd/`; generated remote scripts and run artifacts belong
under ignored output/run directories.

## Main Commands

Show the main training CLI:

```bash
uv run python scripts/train --help
```

Run a tiny local smoke:

```bash
uv run python scripts/smoke \
  --rounds 1 \
  --samples-per-class 16 \
  --epochs 1 \
  --batch-size 8 \
  --hidden-bits 8 \
  --output outputs/smoke_result.json
```

Run the matrix runner with a config matrix:

```bash
uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_active_pattern_r7_screen.csv \
  --epochs 1 \
  --batch-size 8 \
  --hidden-bits 8 \
  --device cpu \
  --output outputs/example_matrix.jsonl
```

Run the SPN candidate-evidence route with disk-backed feature cache:

```bash
uv run python scripts/spn-candidate-evidence \
  --output outputs/spn_candidate_evidence_smoke.jsonl \
  --rounds 7 \
  --seed 0 \
  --samples-per-class 8 \
  --pairs-per-sample 2 \
  --feature-cache-root outputs/feature_cache/spn_candidate_evidence_smoke \
  --feature-cache-chunk-size 2 \
  --progress-output outputs/spn_candidate_evidence_smoke_progress.jsonl \
  --epochs 1 \
  --device cpu
```

Validate a completed matrix result against a plan:

```bash
uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_active_pattern_r7_screen.csv \
  --results outputs/example_matrix.jsonl \
  --output outputs/example_matrix_alignment.json
```

## Evidence Rules

- Do not call SPN/PRESENT `8k`, `16k`, `32k`, or `65k` samples-per-class runs
  formal training or definitive failures. They are smoke/screen or medium
  diagnostics.
- For SPN/PRESENT, distinguish total rows from `samples_per_class`.
- Strict negative samples must be encrypted random plaintexts. Random ciphertext
  negatives are ablation/control evidence only.
- PRESENT-80 r7 reference: Zhang/Wang 2022 Case2 `m=16`, accuracy `0.7205`.
  State clearly when this has not been reproduced locally.
- Remote medium/scale runs that generate datasets or feature matrices must write
  disk-backed cache, metadata, progress logs, and support parameter-matched reuse.

## Development Rules

- Use `uv run pytest ...`, not bare `pytest`.
- Keep generated artifacts out of source directories.
- New reusable code should go under `src/blockcipher_nd/`, not `scripts/`.
- New experiment definitions should go under `configs/`, not Python source.
