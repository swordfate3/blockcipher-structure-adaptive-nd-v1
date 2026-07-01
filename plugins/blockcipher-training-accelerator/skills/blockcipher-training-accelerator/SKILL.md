---
name: blockcipher-training-accelerator
description: Use when planning or running opt-in training speed probes for blockcipher-structure-adaptive-nd-v1 without changing the main trainer.
---

# Blockcipher Training Accelerator

Use this skill for project-local training acceleration work that must preserve experiment
protocols and avoid changing the main `src/blockcipher_nd` trainer.

## Rules

- Treat acceleration as an opt-in side path.
- Do not change cipher, rounds, sample structure, negative samples, labels, metric computation,
  validation key, or checkpoint metric.
- Use `bench-command` for timing existing commands before claiming speed changes.
- Use `split-matrix` to divide independent matrix rows across GPUs before considering DDP.
- Record speed evidence separately from cryptanalytic accuracy evidence.
- Promote AMP, DataLoader tuning, `torch.compile`, or CUDA Graphs only after a quality-drift gate.

## Commands

From the repository root:

```bash
PYTHONPATH=plugins/blockcipher-training-accelerator/src \
  python -m blockcipher_training_accelerator --help
```

Timing wrapper:

```bash
PYTHONPATH=plugins/blockcipher-training-accelerator/src \
  python -m blockcipher_training_accelerator bench-command \
  --label smoke \
  --report outputs/speed_bench/smoke_timing.json \
  -- \
  uv run python scripts/train --help
```

Matrix splitter:

```bash
PYTHONPATH=plugins/blockcipher-training-accelerator/src \
  python -m blockcipher_training_accelerator split-matrix \
  --plan configs/experiment/innovation1/example.csv \
  --shards 2 \
  --output-dir outputs/speed_bench/example_shards
```
