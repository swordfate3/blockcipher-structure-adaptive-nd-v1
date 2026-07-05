# Innovation 1 PRESENT Cross-Pair Mixer Consistency 计划

**日期：** 2026-07-05

**状态：** smoke passed / 262144-class remote assets prepared / not launched

**当前配置：**

```text
configs/experiment/innovation1/innovation1_spn_present_pair_mixer_consistency_smoke.csv
configs/experiment/innovation1/innovation1_spn_present_pair_mixer_consistency_r8_262k_seed0.csv
configs/remote/innovation1_spn_present_pair_mixer_consistency_r8_262k_seed0_gpu0_20260705.json
configs/remote/generated/run_i1_present_r8_pair_mixer_consistency_262k_seed0_gpu0_20260705.cmd
configs/remote/generated/monitor_i1_present_r8_pair_mixer_consistency_262k_seed0_gpu0_20260705.sh
```

## 1. 研究问题

当前 r8 诊断里，`present_nibble_invp_pair_consistency_spn_only` 是最强内部信号之一。它的核心思路是：

```text
每个 pair 先编码成 InvP/P-layer aligned SPN embedding，
再通过 top-k/logsumexp evidence pooling 做弱证据聚合。
```

但该模型在聚合前没有显式让 16 个 pair embedding 互相交互。若高轮 PRESENT 的信号表现为“多个 pair 之间的一致弱证据”，那么单独打分再池化可能不够。

本计划测试一个单独假设：

```text
在不改变输入差分、负样本、sample_structure、训练协议的情况下，
给 InvP pair embeddings 增加轻量 cross-pair token mixer，
是否比当前 pair-consistency anchor 更能捕捉 pair-set 一致性。
```

## 2. 唯一变化假设

固定：

```text
feature_encoding = ciphertext_pair_bits
sample_structure = zhang_wang_case2_official_mcnd
negative_mode = encrypted_random_plaintexts
difference_profile = present_zhang_wang2022_mcnd
pairs_per_sample = 16
checkpoint_metric = val_auc
```

唯一变化：

```text
present_nibble_invp_pair_consistency_spn_only
-> present_nibble_invp_pair_mixer_consistency_spn_only
```

新模型只在 pair embedding 层加入 cross-pair mixer：

```text
InvP pair encoder -> cross-pair mixer -> mean/max/min/std/evidence pooling -> classifier
```

## 3. 当前 Smoke

Smoke 只验证：

```text
1. model alias 能 build；
2. forward shape 正确；
3. CSV 能被 matrix runner 解析；
4. tiny CPU training path 能跑；
5. strict negatives / Zhang-Wang Case2 / fixed validation key 没被改变。
```

Smoke 不评价准确率，不作为模型能力证据。

## 4. 后续 Gate

当前远程 GPU 仍由以下任务占用：

```text
i1_present_r9_weak_probe_262k_seed0_gpu0_20260705
i1_present_r8_pairset_1m_seed0_gpu1_20260705
```

因此本路线暂不启动远程。等 active watcher 返回后，再按 gate 选择：

| 条件 | 动作 |
|---|---|
| r8 1M pair-consistency 明显强于 baseline | 准备 r8 262144/class pair-mixer vs pair-consistency anchor |
| r9 weak-probe 中 pair-consistency 有弱正信号 | 准备 r9 262144/class pair-mixer vs pair-consistency anchor |
| r9 near-random 且 difference screen 优先 | 暂缓 pair-mixer，先做输入差分筛选 |
| active tasks 出现失败/不对齐 | 先修复主线，不启动本路线 |

## 4.1 262144/class 诊断资产

本轮已准备 r8 pair-mixer 262144/class 诊断资产，但不启动远程：

```text
run_id = i1_present_r8_pair_mixer_consistency_262k_seed0_gpu0_20260705
scale = 262144/class
expected_rows = 2
models = pair-consistency anchor vs pair-mixer candidate
status = prepared / not launched
```

固定协议：

```text
rounds = 8
seed = 0
pairs_per_sample = 16
sample_structure = zhang_wang_case2_official_mcnd
negative_mode = encrypted_random_plaintexts
difference_profile = present_zhang_wang2022_mcnd
checkpoint_metric = val_auc
lr_scheduler = official_cyclic
dataset_cache = G:\lxy disk-backed cache
```

唯一变化：

```text
present_nibble_invp_pair_consistency_spn_only
-> present_nibble_invp_pair_mixer_consistency_spn_only
```

Gate：

| 结果 | 决策 |
|---|---|
| pair-mixer AUC `>= anchor + 0.003` | 支持 pair-mixer route，准备 seed1 或 r9 诊断 |
| `0 < delta < 0.003` | 弱正，先等 r8 1M / r9 weak-probe 再决定 |
| `delta <= 0` | 暂停 pair-mixer，不扩大 |
| validate / plan alignment 失败 | 不看指标，先写 failure/repair |

证据范围：

```text
262144/class single seed = medium diagnostic only
not formal evidence
not high-round breakthrough
```

## 5. 证据等级

```text
smoke = path/readiness only
262144/class = medium diagnostic
1000000/class single seed = strong diagnostic
multi-seed 1M = route evidence
```

没有 `>=1000000/class` 多 seed 前，不写突破或正式优于。
