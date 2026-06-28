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
- [x] Task 9: 远程启动 262k N2 消融并挂本地 tmux monitor 自动拉回。
- [x] Task 10: 结果完成后自动更新本文件，再决定是否 seed1 或 1M。

## 8. 262k seed0 结果记录

```text
run_id = i1_spn_n2_transition_r7_262k_seed0_gpu1_20260627
status = completed remotely, fallback-retrieved locally, plan-aligned
remote = lxy-a6000 GPU1
source_commit = a0c05edc16fd4ad1ac2167d681bad857a1799fda
scale = 262144/class
rounds = 7
seed = 0
pairs_per_sample = 16
negative_mode = encrypted_random_plaintexts
sample_structure = zhang_wang_case2_official_mcnd
checkpoint_metric = val_auc
local_dir = outputs/remote_results/i1_spn_n2_transition_r7_262k_seed0_gpu1_20260627/
```

远程 gate：

```text
result_lines = 5
expected_rows = 5
train_exit_code = 0
gate_exit_code = 0
stderr = empty
```

本地 gate：

```text
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_n2_transition_backbone_r7_262k.csv \
  --results outputs/remote_results/i1_spn_n2_transition_r7_262k_seed0_gpu1_20260627/i1_spn_n2_transition_r7_262k_seed0_gpu1_20260627.jsonl \
  --expected-rows 5 \
  --output outputs/remote_results/i1_spn_n2_transition_r7_262k_seed0_gpu1_20260627/i1_spn_n2_transition_r7_262k_seed0_gpu1_20260627_local_result_gate.json
```

结果：

```text
status = pass
plan_rows = 5
result_rows = 5
missing_result_keys = []
unexpected_result_keys = []
field_mismatches = []
```

本地重绘：

```text
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-results \
  --results outputs/remote_results/i1_spn_n2_transition_r7_262k_seed0_gpu1_20260627/i1_spn_n2_transition_r7_262k_seed0_gpu1_20260627.jsonl \
  --output outputs/remote_results/i1_spn_n2_transition_r7_262k_seed0_gpu1_20260627/i1_spn_n2_transition_r7_262k_seed0_gpu1_20260627_curves.svg \
  --history-csv outputs/remote_results/i1_spn_n2_transition_r7_262k_seed0_gpu1_20260627/i1_spn_n2_transition_r7_262k_seed0_gpu1_20260627_history.csv \
  --title i1_spn_n2_transition_r7_262k_seed0_gpu1_20260627
```

指标：

| model | role | accuracy | calibrated_accuracy | AUC | loss | epochs_ran | best_epoch |
|---|---|---:|---:|---:|---:|---:|---:|
| `present_zhang_wang_keras_mcnd` | N0 baseline | 0.704327 | 0.709827 | 0.783952 | 0.564545 | 14 | 6 |
| `present_nibble_paligned_spn_only` | N1-S SPN-only anchor | 0.715187 | 0.715763 | 0.790665 | 0.548985 | 20 | 16 |
| `present_nibble_paligned_transition` | N2-a evidence pooling | 0.694069 | 0.694187 | 0.763328 | 0.573728 | 18 | 10 |
| `present_nibble_paligned_transition_residual` | N2-b true-P residual | 0.697327 | 0.698929 | 0.767918 | 0.570679 | 20 | 17 |
| `present_nibble_shuffled_transition_residual` | N2-c shuffled residual | 0.691391 | 0.691719 | 0.760053 | 0.576825 | 20 | 20 |

N2 deltas：

| comparison | delta_accuracy | delta_calibrated_accuracy | delta_AUC |
|---|---:|---:|---:|
| N2-a vs N0 | -0.010258 | -0.015640 | -0.020624 |
| N2-a vs N1-S | -0.021118 | -0.021576 | -0.027337 |
| N2-b true residual vs N0 | -0.007000 | -0.010899 | -0.016034 |
| N2-b true residual vs N1-S | -0.017860 | -0.016834 | -0.022747 |
| N2-c shuffled residual vs N0 | -0.012936 | -0.018108 | -0.023899 |
| N2-c shuffled residual vs N1-S | -0.023796 | -0.024044 | -0.030612 |
| N2-b true residual vs N2-c shuffled | +0.005936 | +0.007210 | +0.007865 |

门槛判定：

```text
best N2 AUC >= N1-S AUC + 0.001       false
best N2 calibrated_accuracy >= N1-S   false
true-P residual AUC >= shuffled +0.001 true, observed +0.007865
```

结论：

```text
N2 transition-aware backbone = negative as a scaling route.
N2-b true-P residual > shuffled residual = positive attribution clue.
```

解释：

```text
N2-a evidence pooling 和 N2-b transition residual 都没有保住 SPN-only 的强信号，
并且明显低于 N0 baseline 与 N1-S anchor。
因此当前 N2 网络结构不能作为主路线继续放大到 1M。

但 N2-b true-P residual 明显高于 N2-c shuffled residual，
说明 PRESENT P-layer 对齐不是完全无效；
问题更可能是当前 residual/pooling backbone 表达能力或 pair-set 建模方式不对，
而不是 P-layer structure 本身完全没有信号。
```

当前状态：

```text
classification = medium diagnostic result, not formal evidence
decision = do not scale current N2 to 1M or seed1
keep = true-P vs shuffled attribution clue
discard = N2-a evidence pooling and current N2-b residual as scaling routes
next_action = SPN-only input ablation and pair-set consistency analysis
```

下一步建议：

```text
1. 做 SPN-only 输入消融：DeltaC-only / InvP-only / DeltaC+InvP / shuffled-InvP。
2. 做 pair-set consistency 分析：16 pairs 之间的一致性、方差、top-k pair evidence、pair-pair interaction。
3. 暂停继续堆 transition backbone；先定位 SPN-only 为什么强。
```
