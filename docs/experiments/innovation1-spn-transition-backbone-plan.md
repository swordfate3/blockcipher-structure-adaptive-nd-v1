# Innovation1 N2 SPN Transition-Aware Backbone 实验计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `skills/blockcipher-auto-research/SKILL.md` for evidence gates, and use Karpathy-style coding discipline for implementation. This plan is an experiment record under `docs/experiments/`, not a broad research blueprint.

## 1. 背景

上一轮结构消融：

```text
run_id = i1_spn_n1v2_ablation_r7_262k_seed0_gpu0_20260627
scale = 262144/class
negative_mode = encrypted_random_plaintexts
sample_structure = zhang_wang_case2_official_mcnd
checkpoint_metric = val_auc
status = completed remotely, fallback-retrieved locally, plan-aligned
```

关键结果：

| model | role | accuracy | calibrated_accuracy | AUC | loss |
|---|---|---:|---:|---:|---:|
| `present_zhang_wang_keras_mcnd` | N0 baseline | 0.676277 | 0.711231 | 0.784541 | 0.612181 |
| `present_nibble_paligned_mcnd` | N1-v1 late fusion | 0.673466 | 0.710648 | 0.784299 | 0.611594 |
| `present_nibble_paligned_spn_only` | N1-S SPN-only | 0.716358 | 0.716434 | 0.791488 | 0.547521 |
| `present_nibble_paligned_gated_mcnd` | N1-v2 true-P gated | 0.702484 | 0.710701 | 0.784897 | 0.566700 |
| `present_nibble_shuffled_paligned_gated_mcnd` | shuffled-P control | 0.668278 | 0.710266 | 0.784281 | 0.622337 |

判断：

```text
N1-v2 gated = weak positive but failed continuation gates.
SPN-only = strongest row in the 262k seed0 ablation.
```

因此下一步不直接放大 current gated-MCND route，而是研究：

```text
Can a PRESENT/SPN transition-aware backbone preserve or improve the SPN-only signal?
```

## 2. 实验问题

主问题：

```text
SPN-only 的信号是简单 Delta/InvP nibble token 统计，还是可以通过更贴近 SPN transition 的网络结构进一步增强？
```

可归因子问题：

```text
1. pair-level evidence pooling 是否优于当前 mean/max pooling？
2. true PRESENT P-layer transition residual 是否优于 shuffled-P 控制？
3. N2 是否能超过当前最强 anchor `present_nibble_paligned_spn_only`？
```

## 3. 不变量

所有 N2 诊断保持上一轮协议：

```text
cipher = PRESENT-80
rounds = 7
seed = 0
samples_per_class = 262144
pairs_per_sample = 16
feature_encoding = ciphertext_pair_bits
negative_mode = encrypted_random_plaintexts
sample_structure = zhang_wang_case2_official_mcnd
train_key = 0x00000000000000000000
validation_key = 0x11111111111111111111
loss = mse
optimizer = adam
weight_decay = 0.00001
lr_scheduler = official_cyclic
learning_rate = 0.0001
max_learning_rate = 0.002
checkpoint_metric = val_auc
restore_best_checkpoint = true
early_stopping_patience = 8
early_stopping_min_delta = 0.0001
batch_size = 1024
hidden_bits = 32
```

禁止事项：

```text
不改 validation data
不改 labels
不改 negative sample definition
不改 metric computation
不把 random-ciphertext negatives 和 strict negatives 混报
不把 multi-query aggregation 当 raw single-sample SOTA
```

## 4. 模型族

### Anchor A: N0 Zhang/Wang MCND

```text
model_key = present_zhang_wang_keras_mcnd
role = same-budget external baseline anchor
```

### Anchor B: N1-S current SPN-only

```text
model_key = present_nibble_paligned_spn_only
role = strongest prior 262k diagnostic anchor
```

### N2-a: SPN transition evidence backbone

```text
model_key = present_nibble_paligned_transition
```

设计：

```text
reuse DeltaC + InvP(DeltaC) nibble view
encode each pair into pair embedding
apply evidence pooling across 16 pairs
classifier(attention/evidence pooled pair embeddings)
```

与 N1-S 的关键区别：

```text
N1-S = mean/max over pair embeddings
N2-a = learned MIL/evidence pooling over pair embeddings
```

目的：

```text
测试区分信号是否集中在少数强 pair，而不是平均分布在所有 pair。
```

### N2-b: SPN transition residual backbone

```text
model_key = present_nibble_paligned_transition_residual
```

设计：

```text
reuse DeltaC + InvP(DeltaC) nibble view
split 32 nibbles into 16 DeltaC nibbles and 16 InvP(DeltaC) nibbles
learn source tokens, aligned target tokens, and target-source transition tokens
mix the 16 transition tokens with a token mixer
pool pair embeddings with learned evidence pooling
```

目的：

```text
显式建模 PRESENT P-layer 前后同一 nibble position 的 transition residual，
而不只是把 DeltaC 与 InvP(DeltaC) 拼接后交给普通 mixer。
```

### N2-c: shuffled transition residual control

```text
model_key = present_nibble_shuffled_transition_residual
```

设计：

```text
与 N2-b 相同
唯一差异：InvP alignment 使用固定 deterministic shuffled bit permutation
```

目的：

```text
如果 true-P residual <= shuffled residual，则不能声称 P-layer topology 有贡献。
```

## 5. 262k 判定门槛

Primary metric：

```text
AUC
```

Secondary：

```text
calibrated_accuracy
loss
best_epoch
train/val gap
```

继续条件：

```text
best N2 AUC >= N1-S SPN-only AUC + 0.001
and best N2 calibrated_accuracy >= N1-S calibrated_accuracy
and true-P residual AUC >= shuffled residual AUC + 0.001
```

弱继续条件：

```text
best N2 AUC > N1-S SPN-only AUC
but margin < 0.001
and true-P residual > shuffled residual
```

动作：

```text
repeat 262k seed1 before 1M
```

停止或降级条件：

```text
all N2 <= N1-S SPN-only
or true-P residual <= shuffled residual
```

动作：

```text
do not scale N2 to 1M
analyze why SPN-only works by input ablation: DeltaC-only, InvP-only, true/shuffled, nibble grouped vs flat
```

## 6. 产物要求

每个非 smoke run 必须保留：

```text
plan CSV
remote config JSON
source commit
remote run dir
local retrieved dir
results JSONL
history CSV
curves SVG
result_gate.txt or result_gate.json
progress JSONL
stdout/stderr
git/gpu/torch evidence
```

完成后自动更新本文件，包含：

```text
run_id
artifact paths
gate result
metrics table
deltas vs N0 and N1-S
true-P vs shuffled-P attribution
claim scope
next action
```

## 7. 执行清单

- [x] Task 1: 增加 N2 transition backbone 模型构建/forward 测试。
- [x] Task 2: 实现 N2-a transition evidence backbone。
- [x] Task 3: 实现 N2-b true-P transition residual backbone。
- [x] Task 4: 实现 N2-c shuffled transition residual control。
- [x] Task 5: 注册新 model keys。
- [x] Task 6: 加 smoke CSV 和 262k CSV。
- [x] Task 7: CPU smoke 验证所有 N2 模型可训练 1 epoch。
- [x] Task 8: 提交并推送代码/config/docs。
- [ ] Task 9: 远程启动 262k N2 消融并挂本地 tmux monitor 自动拉回。
- [ ] Task 10: 结果完成后自动更新本文件，再决定是否 seed1 或 1M。
