# Innovation 1 PRESENT r10 条件式 Weak-Probe 计划

**日期：** 2026-07-05

**状态：** planned / no remote assets / wait for r9 gate

**研究蓝图：** `docs/research/innovation1-present-higher-round-strategy.md`

## 1. 研究问题

如果 r9 在 strict encrypted-random-plaintext negatives、Zhang/Wang Case2
`m=16`、同协议同预算下仍有可检测弱信号，那么下一步才问：

```text
PRESENT-80 r10 是否还有可被 SPN-aware pair-set evidence pooling 或
curriculum / transfer 放大的微弱 real-vs-random 神经信号？
```

这个计划只是条件分支，不是当前可启动实验。

## 2. 为什么 r10 不立即启动

当前 active watcher 已经有：

```text
i1_present_r9_weak_probe_262k_seed0_gpu0_20260705
i1_present_r8_pairset_1m_seed0_gpu1_20260705
```

r10 不能在 r9 未 retrieved / validated / gate-noted 前启动，原因是：

```text
1. 如果 r9 from-scratch 已近随机，r10 from-scratch 大概率只会浪费 GPU；
2. 如果 r9 只有 pair-set 或 curriculum 信号，r10 首轮模型应继承该路线；
3. 如果 r9 的问题来自输入差分，r10 应先走 difference-screen，而不是改轮数硬跑。
```

## 3. 启动条件

只有以下任一条件满足，才允许创建 r10 CSV / remote config：

| r9 gate 状态 | r10 动作 |
|---|---|
| r9 best AUC `> 0.55` 且超过 baseline `+0.005` | 准备 r10 65536/class weak screen，模型用 r9 最强候选 |
| r9 AUC `0.52-0.55` 且 pair-set strongest | 先做 r9 seed1 或 pair-mixer，再决定 r10 |
| r9 from-scratch near-random 但 curriculum 明显提升 | 准备 r10 curriculum/transfer screen |
| r9 difference screen 找到非 Zhang/Wang 强候选 | 用该差分做 r10 65536/class screen |
| r9 全部 near-random，且无差分/curriculum 正信号 | 不启动 r10，转新数据结构或理论路线 |

## 4. 首轮 r10 设计原则

首轮 r10 只能是 screen / diagnostic：

```text
samples_per_class = 65536
pairs_per_sample = 16
negative_mode = encrypted_random_plaintexts
sample_structure = zhang_wang_case2_official_mcnd
checkpoint_metric = val_auc
dataset_cache = disk-backed under G:\lxy
```

矩阵必须 lean：

```text
1. same-protocol baseline 或 r9 最强 anchor
2. 一个唯一变化候选
```

不得同时混改：

```text
rounds
model
difference_profile
sample_structure
negative_mode
validation key
metric
```

## 5. 证据语言

r10 的 `65536/class` 只能写：

```text
r10 weak screen
r10 no detectable signal under this protocol
r10 route candidate for 262144/class confirmation
```

不能写：

```text
r10 成功
r10 SOTA
PRESENT 高轮突破
```

如果 r10 screen 有候选，下一步必须是：

```text
262144/class confirmation -> 1000000/class single-seed -> multi-seed route evidence
```

当前状态哨兵：

```text
no r10 remote assets
no r10 GPU launch
no r10 success/failure claim
```

## 6. 当前下一步

等待以下结果之一：

```text
1. r9 weak-probe watcher 自动拉回并 postprocess；
2. r8 pairset 1M watcher 自动拉回并 postprocess；
3. r9 difference-screen 或 r9 curriculum 被 gate 选中并完成。
```

在这些条件之前，本计划保持 planned，不创建 r10 远程资产，不占用 GPU。
