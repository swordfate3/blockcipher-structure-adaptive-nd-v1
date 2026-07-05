# Innovation 1 PRESENT r9 Pair-Evidence Pooling Screen 计划

**日期：** 2026-07-05

**状态：** smoke prepared / 65536-class screen planned / gate-locked / not launched

**研究蓝图：** `docs/research/innovation1-present-higher-round-strategy.md`

## 1. 研究问题

当前 r9 weak-probe 已经在远程 watcher 中运行：

```text
i1_present_r9_weak_probe_262k_seed0_gpu0_20260705
```

如果它的 `present_nibble_invp_pair_consistency_spn_only` 行保留弱信号，下一步不应立刻扩大矩阵或盲目 1M，而应先问：

```text
r9 的 pair-set 弱证据更像少数强 pair 的 sparse evidence，
还是多个 pair 累积形成的 dense weak evidence？
```

这个计划只测试 pair pooling 形态，不改变 benchmark、负样本、输入差分或验证 key。

## 2. 唯一变化假设

固定：

```text
cipher = PRESENT-80
rounds = 9
seed = 0
samples_per_class = 65536
pairs_per_sample = 16
feature_encoding = ciphertext_pair_bits
negative_mode = encrypted_random_plaintexts
sample_structure = zhang_wang_case2_official_mcnd
difference_profile = present_zhang_wang2022_mcnd
difference_member = 0
checkpoint_metric = val_auc
validation_key = 0x11111111111111111111
```

唯一变化：

```text
pair evidence pooling mode / pair mixer interaction
```

| Row | Model | Pooling | Hypothesis |
|---:|---|---|---|
| 0 | `present_nibble_invp_pair_consistency_spn_only` | `topk_logsumexp` | r9 weak-probe pair-set anchor |
| 1 | `present_nibble_invp_pair_mixer_consistency_spn_only` | `topk_logsumexp` | pair mixer + sparse soft top-k |
| 2 | `present_nibble_invp_pair_mixer_consistency_spn_only` | `logsumexp` | pair mixer + dense all-pair weak evidence |
| 3 | `present_nibble_invp_pair_mixer_consistency_spn_only` | `topk_mean` | pair mixer + sparse non-soft top-k evidence |

## 3. 配置

Smoke：

```text
configs/experiment/innovation1/innovation1_spn_present_pair_evidence_pooling_screen_r9_smoke.csv
```

Screen：

```text
configs/experiment/innovation1/innovation1_spn_present_pair_evidence_pooling_screen_r9_65k_seed0.csv
```

Prepared remote assets：

```text
run_id = i1_present_r9_pair_evidence_pooling_screen_65k_seed0_gpu0_20260705
configs/remote/innovation1_spn_present_pair_evidence_pooling_screen_r9_65k_seed0_gpu0_20260705.json
configs/remote/generated/run_i1_present_r9_pair_evidence_pooling_screen_65k_seed0_gpu0_20260705.cmd
configs/remote/generated/monitor_i1_present_r9_pair_evidence_pooling_screen_65k_seed0_gpu0_20260705.sh
```

## 4. 启动条件

当前不启动远程，因为已有 active watcher：

```text
i1_present_r9_weak_probe_262k_seed0_gpu0_20260705
i1_present_r8_pairset_1m_seed0_gpu1_20260705
```

只有以下条件之一满足，才考虑启动本 screen：

| 条件 | 动作 |
|---|---|
| r9 weak-probe 中 pair-consistency 是 best candidate，且 AUC `> 0.505` | 启动 r9 pooling screen，判断 pair evidence 形态 |
| r9 weak-probe 是 weak trace，postprocess 分支进入 `r9_variance_or_aggregation_review` | 用本 screen 作为 application-level aggregation 诊断候选 |
| r8 pairset 1M 正向但 r9 from-scratch 很弱 | 本 screen 可作为 r8->r9 pooling/curriculum 前的结构诊断 |
| r9 baseline 最好或全部 near-random | 不启动本 screen，优先 difference screen 或 curriculum |

## 5. 证据等级

```text
smoke = 只证明路径能跑
65536/class = screen / diagnostic only
262144/class = medium diagnostic
1000000/class = paper-scale single-seed diagnostic
```

本 screen 即使有正向结果，也只能写：

```text
r9 pair-evidence pooling diagnostic signal
application-level aggregation candidate
candidate for 262144/class confirmation
```

不能写：

```text
r9 正式突破
r9 SOTA
raw single-sample formal evidence
```

## 6. Gate

若未来启动 65536/class screen：

| 结果 | 决策 |
|---|---|
| best pair-mixer pooling AUC `>= anchor + 0.005` | 准备 r9 262144/class pooling confirmation |
| best pair-mixer pooling `0 < delta < 0.005` | 弱正，只作为 aggregation/curriculum 候选 |
| 所有 pair-mixer pooling `<= anchor` | 暂停 r9 pooling screen，回到 curriculum/difference |
| validate / plan alignment 失败 | 不看指标，先修复 |

## 7. 当前动作

本轮只做：

```text
1. 准备 r9 smoke 和 65536/class screen CSV；
2. 准备 remote config / launcher / monitor；
3. 本地 smoke 和 readiness 验证；
4. 提交推送；
5. 不启动远程。
```
