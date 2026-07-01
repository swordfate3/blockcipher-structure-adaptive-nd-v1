# Blockcipher Training Accelerator

Optional training-speed utilities for `blockcipher-structure-adaptive-nd-v1`.

This plugin is deliberately separate from `src/blockcipher_nd`. It does not modify the
current trainer, model code, data generation, metric computation, or experiment protocol.
The first version provides two low-risk capabilities:

1. `bench-command`: run an existing command and write a timing report.
2. `split-matrix`: split a CSV experiment matrix into shards for GPU-level parallel launch.

Run from the repository root:

```bash
PYTHONPATH=plugins/blockcipher-training-accelerator/src \
  python -m blockcipher_training_accelerator --help
```

Example timing probe:

```bash
PYTHONPATH=plugins/blockcipher-training-accelerator/src \
  python -m blockcipher_training_accelerator bench-command \
  --label smoke-train \
  --report outputs/speed_bench/smoke_train_timing.json \
  -- \
  uv run python scripts/train --ciphers speck32 --models mlp --rounds 1 \
  --seeds 0 --samples-per-class 64 --epochs 1 --batch-size 16 \
  --device cpu --output outputs/speed_bench/smoke_train.jsonl
```

Example matrix split:

```bash
PYTHONPATH=plugins/blockcipher-training-accelerator/src \
  python -m blockcipher_training_accelerator split-matrix \
  --plan configs/experiment/innovation1/innovation1_spn_present_topology_aware_network_r7_262k.csv \
  --shards 2 \
  --output-dir outputs/speed_bench/topology_matrix_shards
```

Future speed profiles such as DataLoader tuning, BF16 AMP, `torch.compile`, and CUDA
Graphs must pass a quality-drift gate before they are used for meaningful Innovation 1
experiments.
