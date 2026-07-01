# Blockcipher Training Accelerator

Optional training-speed utilities for `blockcipher-structure-adaptive-nd-v1`.

This plugin is deliberately separate from `src/blockcipher_nd`. It does not modify the
current trainer, model code, data generation, metric computation, or experiment protocol.
The first version provides two low-risk capabilities:

1. `bench-command`: run an existing command and write a timing report.
2. `split-matrix`: split a CSV experiment matrix into shards for GPU-level parallel launch.
3. `build-launch-plan`: turn shard manifests into per-GPU train commands.
4. `run-accelerated`: run a matrix through the plugin trainer with an opt-in speed profile.
5. `quality-gate`: compare baseline and accelerated JSONL files for protocol alignment
   and metric drift.

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

Example accelerated tiny run:

```bash
PYTHONPATH=plugins/blockcipher-training-accelerator/src:src \
  python -m blockcipher_training_accelerator run-accelerated -- \
  --ciphers speck32 --models mlp --rounds 1 --seeds 0 \
  --samples-per-class 64 --pairs-per-sample 1 --epochs 1 \
  --batch-size 16 --hidden-bits 16 --device cpu \
  --output outputs/speed_bench/accelerated_smoke.jsonl \
  --speed-profile baseline
```

Available speed profiles:

```text
baseline
dataloader
amp-bf16
compile
amp-bf16-compile
full
```

AMP and compile profiles only become effective on CUDA. CPU runs stay protocol-compatible
and record that the CUDA-only knobs were not effective.

Example quality gate:

```bash
PYTHONPATH=plugins/blockcipher-training-accelerator/src \
  python -m blockcipher_training_accelerator quality-gate \
  --baseline outputs/speed_bench/baseline.jsonl \
  --candidate outputs/speed_bench/accelerated.jsonl \
  --output outputs/speed_bench/quality_gate.json
```
