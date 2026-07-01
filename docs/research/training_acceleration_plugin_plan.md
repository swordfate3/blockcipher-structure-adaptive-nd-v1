# 训练加速插件研究计划

**日期:** 2026-07-01

**状态:** 插件首版实现中

**范围:** 在不修改现有 `src/blockcipher_nd` 训练代码的前提下，为 blockcipher 神经区分器实验提供可选训练加速插件。

## 背景

当前项目的训练主线已经具备磁盘数据集 cache 和多进程数据生成能力，但训练 loop 仍是标准 PyTorch eager 执行。源码核对结果：

- `src/blockcipher_nd/training/trainer.py` 目前没有 AMP/BF16、`torch.compile`、CUDA Graphs 或 DDP。
- `src/blockcipher_nd/training/data.py` 的 `DataLoader` 使用默认配置，没有 `num_workers`、`pin_memory`、`persistent_workers` 或 `prefetch_factor`。
- batch 搬运到 GPU 使用普通 `.to(device)`，没有 `non_blocking=True`。
- `src/blockcipher_nd/training/metrics.py` 的大规模验证会把分数拉回 CPU，再做排序 AUC 和阈值搜索。
- `src/blockcipher_nd/data/cache/disk.py` 已有磁盘 cache 和 `dataset_cache_workers`，所以插件首版不重复实现数据缓存。

训练加速必须和 cryptanalytic 证据分离：速度提升本身不证明模型更强，任何加速 profile 不能改变数据、标签、负样本、验证 key、metric 或 checkpoint 规则。

## 外部依据

本计划参考以下官方/权威资料：

- PyTorch Performance Tuning Guide：建议使用异步数据加载、`pin_memory`，并说明 `torch.compile` 可通过 kernel fusion 和减少 eager overhead 提速。<https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html>
- PyTorch AMP Recipe：说明 autocast/GradScaler 混合精度训练路径，可利用 Tensor Core 降低训练时间和显存。<https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html>
- PyTorch `torch.compile` Tutorial：说明编译模式通过优化 PyTorch 代码执行来加速，但需要按模型和 shape 实测。<https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>
- PyTorch DataLoader 文档：`num_workers`、`pin_memory`、`persistent_workers`、`prefetch_factor` 是标准数据加载调优入口。<https://docs.pytorch.org/docs/stable/data.html>
- PyTorch CUDA Graphs 官方博客：CUDA Graphs 适合静态 shape、CPU launch overhead 明显的场景，应作为后期实验项。<https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/>
- NVIDIA Mixed Precision Training Guide：混合精度可利用 Tensor Cores，但需要数值稳定和质量漂移验证。<https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html>

## 插件目标

新建项目内插件：

```text
plugins/blockcipher-training-accelerator/
```

插件目标分三层：

1. **观测层:** 给现有训练命令加 timing/profile，不改变训练行为。
2. **调度层:** 把独立 matrix rows/seeds/models 拆分到不同 GPU 运行，优先获得 wall-clock 提速。
3. **执行层:** 后续再以 opt-in profile 加入 DataLoader tuning、BF16 AMP、`torch.compile` 等训练执行加速。

首版只实现观测层和调度层，因为它们不碰现有 trainer，风险最低。

## 插件边界

插件允许：

- 调用现有 `scripts/train` 或其他 CLI。
- 记录命令耗时、return code、输出 report。
- 拆分 CSV plan，生成 shard CSV 和 manifest。
- 后续生成远程多 GPU launch plan。

插件不允许默认改变：

- cipher、rounds、sample structure。
- `negative_mode`，尤其不能把 strict `encrypted_random_plaintexts` 换成其他负样本。
- train/validation key。
- labels、validation rows 或 metric computation。
- checkpoint metric 和 result schema。
- 现有 `src/blockcipher_nd/training/trainer.py`。

## 首版实现

首版提供两个 CLI：

```bash
PYTHONPATH=plugins/blockcipher-training-accelerator/src \
  python -m blockcipher_training_accelerator bench-command ...
```

`bench-command` 运行任意现有命令并生成 JSON timing report，字段包括：

- label
- command
- cwd
- started_at / finished_at
- duration_seconds
- returncode
- status

```bash
PYTHONPATH=plugins/blockcipher-training-accelerator/src \
  python -m blockcipher_training_accelerator split-matrix ...
```

`split-matrix` 读取 CSV plan，按 `round-robin` 或 `contiguous` 策略输出 shard CSV 和 manifest。后续每个 shard 可以独立传给 `scripts/train --plan <shard>`，从而在 GPU0/GPU1 并行跑不同模型或 seed。

## 后续 Speed Profiles

后续 profile 必须逐个打开、逐个验证。

### DataLoader Profile

候选参数：

```text
num_workers = 2 / 4 / 8
pin_memory = true
persistent_workers = true
prefetch_factor = 2 / 4
non_blocking = true
```

风险：Windows spawn、多进程 mmap 读取、worker 初始化开销可能抵消收益，所以需要远程实测。

### BF16 AMP Profile

A6000 属于 Ampere，优先测试 BF16 autocast。目标是利用 Tensor Core 提升训练吞吐，同时避免 FP16 GradScaler 的额外复杂度。

质量 gate：

```text
loss 无 NaN/Inf
val_auc / calibrated_accuracy 无异常下降
samples/sec 或 epoch wall time 有明确收益
```

### `torch.compile` Profile

只在模型 shape 稳定、warmup 后有收益时启用。报告必须区分：

```text
cold compile time
warmup time
steady-state step/sec
full-run wall-clock
```

### CUDA Graphs Profile

后期实验项。只有在 static shape、固定 batch、模型无明显 graph break 且 CPU launch overhead 成为主要瓶颈时再试。

## 质量漂移 Gate

任何训练执行加速 profile 用于 meaningful 实验前，必须通过以下 gate：

1. same protocol。
2. same samples_per_class。
3. same seed。
4. same train/validation key。
5. same `negative_mode=encrypted_random_plaintexts`。
6. same checkpoint metric。
7. same metric computation。
8. result row count 一致。
9. `val_auc` 和 `calibrated_accuracy` 没有超过预设阈值的异常下降。
10. loss 无 NaN/Inf。

建议阈值：

```text
smoke: 只证明路径可跑，不评价指标
diagnostic: val_auc 漂移 <= 0.002，或处于已知 seed variance 内
正式路线: 先在 262144/class 通过，再允许用于 1000000/class
```

## 优先级

短期优先级：

1. Matrix row/seed 级并行调度。
2. Timing/profile 报告标准化。
3. DataLoader tuning。
4. BF16 AMP。
5. `torch.compile`。
6. CUDA Graphs / DDP。

对当前项目，矩阵行级并行比立即 DDP 更合适，因为多数实验是多个独立 row 的 same-budget 对比。把不同模型/seed 分给不同 GPU，可以在不改变单个训练语义的情况下减少总等待时间。

## 验证计划

首版插件验证：

```bash
PYTHONPATH=plugins/blockcipher-training-accelerator/src \
  UV_CACHE_DIR=/tmp/uv-cache uv run pytest plugins/blockcipher-training-accelerator/tests -q
```

后续接入远程前，还需要：

```bash
PYTHONPATH=plugins/blockcipher-training-accelerator/src \
  python -m blockcipher_training_accelerator bench-command --help

PYTHONPATH=plugins/blockcipher-training-accelerator/src \
  python -m blockcipher_training_accelerator split-matrix --help
```

## 下一步

1. 使用首版插件对一个 CPU tiny smoke 命令做 timing probe。
2. 使用 `split-matrix` 拆分一个 2-3 row 的 Innovation 1 plan。
3. 生成远程 GPU0/GPU1 shard launch 方案，但仍使用现有 remote readiness、`G:\lxy`、`cmd.exe /c`、watcher 和 result validation 规则。
4. 如果 row-level 并行稳定，再实现 DataLoader/BF16 profile 的 isolated accelerated runner。
