# Innovation 1 PRESENT Pair-Evidence Pooling Screen 计划

**日期：** 2026-07-05

**状态：** smoke prepared / 65536-class screen planned / not launched

**研究蓝图：** `docs/research/innovation1-present-higher-round-strategy.md`

## 1. 研究问题

r8 round-extension 的中等诊断显示，当前正向信号主要来自：

```text
present_nibble_invp_pair_consistency_spn_only
```

而不是单独的 InvP-only。这个结果说明高轮 SPN/PRESENT 可能不是“单个 pair 很强”，而是“多个 pair 中存在弱但一致的证据”。

因此本 screen 问一个更具体的问题：

```text
高轮 pair-set 证据更像少数强 pair 的 sparse evidence，
还是多个 pair 一起贡献的 dense weak evidence？
```

## 2. 唯一变化假设

固定：

```text
cipher = PRESENT-80
rounds = 8
seed = 0
pairs_per_sample = 16
feature_encoding = ciphertext_pair_bits
negative_mode = encrypted_random_plaintexts
sample_structure = zhang_wang_case2_official_mcnd
difference_profile = present_zhang_wang2022_mcnd
checkpoint_metric = val_auc
validation_key = 0x11111111111111111111
```

变化：

```text
pair evidence pooling mode
```

候选：

| Row | Model | Pooling | Hypothesis |
|---:|---|---|---|
| 0 | `present_nibble_invp_pair_consistency_spn_only` | `topk_logsumexp` | 当前 anchor |
| 1 | `present_nibble_invp_pair_mixer_consistency_spn_only` | `topk_logsumexp` | pair mixer + sparse soft top-k |
| 2 | `present_nibble_invp_pair_mixer_consistency_spn_only` | `logsumexp` | pair mixer + dense all-pair weak evidence |
| 3 | `present_nibble_invp_pair_mixer_consistency_spn_only` | `topk_mean` | pair mixer + sparse non-soft top-k evidence |

这不是 benchmark 变化，也不是负样本变化。

## 3. 配置

Smoke：

```text
configs/experiment/innovation1/innovation1_spn_present_pair_evidence_pooling_screen_smoke.csv
```

Screen：

```text
configs/experiment/innovation1/innovation1_spn_present_pair_evidence_pooling_screen_r8_65k_seed0.csv
```

## 4. 证据等级

```text
smoke = 只证明路径能跑
65536/class = screen / diagnostic only
262144/class = medium diagnostic
1000000/class = paper-scale single-seed diagnostic
```

这个 screen 即使某个 pooling 模式更好，也只能说明：

```text
pair-evidence pooling 方向值得进入 262144/class confirmation
```

不能写：

```text
正式优于 baseline
高轮突破
SOTA
```

## 5. 启动条件

当前不启动远程，因为已有 active watcher：

```text
i1_present_r9_weak_probe_262k_seed0_gpu0_20260705
i1_present_r8_pairset_1m_seed0_gpu1_20260705
```

后续只有当以下条件之一成立，才考虑把本 screen 做成远程 65536/class：

| 条件 | 动作 |
|---|---|
| r8 pairset 1M 保留正向 | 用 pooling screen 判断 pair-mixer 的下一步具体形态 |
| r9 weak-probe 中 pair-consistency 有弱信号 | 准备 r9 版本 pooling screen |
| r9 weak-probe 近随机但 r8 pairset 仍正向 | 优先 r8 pooling/aggregation attribution |
| r8 pairset 1M 失败且 r9 近随机 | 暂缓 pooling screen，转 difference/curriculum 或新数据结构 |

## 6. Gate

若未来启动 65536/class screen：

| 结果 | 决策 |
|---|---|
| best pair-mixer pooling AUC `>= anchor + 0.005` | 准备 262144/class confirmation |
| best pair-mixer pooling `0 < delta < 0.005` | 弱正，等 active gates 或重复 screen |
| 所有 pair-mixer pooling `<= anchor` | 暂停 pooling screen |
| validate / plan alignment 失败 | 不看指标，先修复 |

## 7. 当前动作

本轮只做：

```text
1. 准备 smoke 和 65536/class screen CSV；
2. 本地 CPU smoke 验证路径；
3. 提交推送；
4. 不启动远程。
```
