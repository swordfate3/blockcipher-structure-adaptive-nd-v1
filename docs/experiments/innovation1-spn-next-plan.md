# Innovation 1 SPN 下一步执行计划

本文档是 `docs/research/spn_structured_nn_research_plan.md` 的近期执行版。研究蓝图可以保持较宽，覆盖 SPN-GNN、DDT gate、SKINNY/GIFT 迁移、多 pair 和 related-key；本执行计划只回答接下来应该怎么做，怎样避免实验不可归因。

当前原则：

- 以 `blockcipher-auto-research` 作为实验流程：同预算 baseline、单假设改动、固定训练协议、结构化 artifact、证据门槛。
- 以 Karpathy-style coding discipline 作为实现风格：先读代码，小步实现，少抽象，只改目标相关内容，用最窄验证证明改动有效。
- 不把 benchmark 改动和模型/特征改动混在一起。除非明确重设 benchmark，否则保持 validation data、labels、negative mode、sample structure、metric、plan alignment 不变。

## 1. 当前锚点

短期 SPN 主线不从空白网络开始，而是以 Zhang/Wang 2022 PRESENT-80 r7 Case2 `m=16` 为强 baseline。

严格参考值：

```text
Zhang/Wang 2022 Table 4
PRESENT-80 r7 Case2 m=16
accuracy = 0.7205
```

本项目已确认的锚点：

```text
official checkpoint eval:
  raw_pair_count = 1000000
  grouped_eval_rows = 62500
  accuracy = 0.721536
  status = official checkpoint reproduction, not from-scratch training

PyTorch 64k/class Keras-layout:
  accuracy = 0.611740
  calibrated_accuracy = 0.614655
  AUC = 0.658023
  status = medium diagnostic

PyTorch 262k/class official-cyclic:
  accuracy = 0.710911
  calibrated_accuracy = 0.711933
  AUC = 0.786293
  status = successful medium diagnostic, not formal reproduction

PyTorch 1M/class seed0 official-cyclic:
  run_id = zhang_wang_present_r7_1m_official_cyclic_seed0_20260625
  accuracy = 0.715281
  calibrated_accuracy = 0.718555
  AUC = 0.793897025948
  best_epoch = 18
  status = near-reference single-seed baseline; enough as Innovation1 baseline anchor
```

短期判断：`1M/class seed0` 已经接近 Zhang/Wang `0.7205` 参考值，差距约 `0.001945` calibrated accuracy。它足以作为 Innovation1 后续同预算对照 baseline，但仍不能写成多 seed 正式复现。

## 2. 下一步决策门槛

当前远程 1M baseline 完成后，先做结果判定，再决定是否开创新实验。

### 2.1 如果 1M 达到或接近参考值

条件：

```text
accuracy 或 calibrated_accuracy >= 0.7205 附近
或 accuracy/calibrated_accuracy >= 0.715 且 AUC 与曲线稳定
```

动作：

1. 记录 `1M/class seed0` 已达到 near-reference reproduction anchor。
2. 不再单独追 baseline seed1/seed2，除非后续 Innovation1 1M 结果需要方差解释。
3. 直接推进 `I1-SPN-001-paper-scale seed0`：同一 1M/class 矩阵内比较 Zhang/Wang baseline 与 `present_nibble_paligned_mcnd`。
4. 若 1M 创新仍正向，再开多 seed 正式实验。

### 2.2 如果 1M 略低但延续 262k 趋势

条件：

```text
accuracy/calibrated_accuracy 约 0.710 - 0.715
AUC 高且 validation 曲线稳定
```

动作：

1. 优先补 `seed1/seed2`，判断是否是 seed0 方差。
2. 检查训练曲线是否有过拟合、early stopping 是否过早、best checkpoint 是否合理。
3. 不急着改 benchmark。只有多 seed 仍系统偏低时，再做训练协议诊断。

### 2.3 如果 1M 明显低于 262k 预期

条件：

```text
accuracy/calibrated_accuracy 明显低于 0.710
或 AUC/validation curve 异常
```

动作：

1. 暂停创新网络扩展。
2. 先诊断 baseline 协议：LR cycle、batch size、checkpoint metric、early stopping、validation size、数据缓存一致性、官方训练细节差异。
3. 不在不稳定 baseline 上做创新结论。

## 3. 网络选择路线

网络选择不从“哪个模型更先进”出发，而从“要证明哪个结构假设”出发。

### N0: Zhang/Wang Keras MCND

作用：强 baseline，不是创新点。

```text
model_key = present_zhang_wang_keras_mcnd
feature_encoding = ciphertext_pair_bits
sample_structure = zhang_wang_case2_official_mcnd
negative_mode = encrypted_random_plaintexts
pairs_per_sample = 16
lr_scheduler = official_cyclic
checkpoint_metric = val_auc
```

所有创新网络都先和 N0 在同预算下比较。

### N1: Present Nibble/P-layer Aligned MCND

这是第一优先创新网络。

暂定名称：

```text
present_nibble_paligned_mcnd
```

核心假设：

```text
PRESENT 的 SPN 结构由 4-bit S-box cell 和 P-layer bit permutation 组成。
在 Zhang/Wang MCND backbone 旁边加入 nibble/P-layer aligned view，
可能提升 r7 的样本效率、calibration 或训练稳定性。
```

最小实现：

```text
raw branch:
  复用 Zhang/Wang Keras MCND 的 pair-set bit backbone

SPN branch:
  从 C, C' 派生 Delta C
  计算 InvP(Delta C)
  按 16 个 nibble/cell 构造 token
  shared cell encoder: 小 MLP 或 Conv1D
  lightweight token mixer: 1-2 层 residual MLP/Conv1D
  pooling: mean + max

fusion:
  concat(raw_embedding, spn_embedding)
  small classifier head
```

第一版不要加入：

```text
GNN message passing
DDT gate
attention 大模块
cross-cipher generic abstraction
multi-task auxiliary loss
```

保留这些不变：

```text
negative_mode = encrypted_random_plaintexts
sample_structure = zhang_wang_case2_official_mcnd
pairs_per_sample = 16
lr_scheduler = official_cyclic
checkpoint_metric = val_auc
loss = mse
validation protocol 不变
```

判断：

- 如果 N1 比 N0 同预算提升，说明 SPN-aligned view 有价值。
- 如果 N1 没提升，说明 Zhang/Wang raw MCND 可能已经学到了足够多的局部结构，或者该 SPN view 设计不够有效。

### N2: DDT/Active Auxiliary MCND

第二优先，只有 N1 有趋势或 baseline 稳定后再做。

原则：

```text
active/DDT 不作为 standalone 24-dim 主路线。
此前 active-only 24-dim screen 已接近 chance，不能继续主推同类粗特征。
```

正确用法：

```text
main head:
  real-vs-random binary classification

auxiliary head:
  active nibble pattern
  或 DDT legality / DDT score bucket
  或 trail consistency proxy

loss:
  main_loss + lambda * aux_loss
  first lambda = 0.1
```

判断：

- 目标不是辅助头本身准确率，而是主任务 AUC、calibrated accuracy 或稳定性是否提升。
- 若辅助指标高但主任务不升，不算成功。

### N3: P-layer Graph Message Passing

第三优先，不作为第一步。

核心假设：

```text
真实 P-layer 拓扑作为固定 message passing edge，
比普通 token mixer 或 CNN 邻域更贴合 SPN 差分扩散。
```

必要消融：

```text
true P-layer edges
random shuffled edges
fully connected edges
no edge / token MLP
same input CNN or MLP
```

判断：

- 只有 true edges 明显优于 shuffled/random edges，才能声称 P-layer topology 有贡献。
- 如果 true edges 无优势，SPN-GNN 路线应降级为负结果或后续探索，不进入主贡献。

## 4. 近期实验阶梯

### 4.1 Baseline track

```text
B0: 1M/class seed0 Zhang/Wang official-cyclic
  status: completed, fallback-retrieved, near-reference single-seed anchor
  metrics: accuracy 0.715281, calibrated_accuracy 0.718555, AUC 0.793897025948
  action: use as Innovation1 same-budget baseline anchor; no standalone rerun needed now

B1: 1M/class seed1 Zhang/Wang official-cyclic
  launch condition: only if Innovation1 1M result requires baseline variance analysis

B2: 1M/class seed2 Zhang/Wang official-cyclic
  launch condition: only after B1 if preparing formal multi-seed reproduction statistics
```

### 4.2 Innovation track

```text
I1-SPN-001-smoke:
  model = present_nibble_paligned_mcnd
  samples_per_class = 64 or 256
  device = cpu
  purpose = shape/forward/train smoke only

I1-SPN-001-screen:
  samples_per_class = 65536
  remote = lxy-a6000
  compare = same-budget Zhang/Wang N0 baseline
  status language = medium diagnostic only

I1-SPN-001-strong:
  samples_per_class = 262144
  launch condition = 64k screen improves AUC or calibrated_accuracy

I1-SPN-001-paper-scale:
  samples_per_class = 1000000
  plan = configs/experiment/innovation1/innovation1_spn_present_nibble_paligned_mcnd_r7_1m_seed0.csv
  remote_config = configs/remote/innovation1_spn_present_nibble_paligned_mcnd_r7_1m_seed0_gpu1_20260626.json
  run_id = innovation1_spn_present_nibble_paligned_mcnd_r7_1m_seed0_gpu1_20260626
  compare = same-budget Zhang/Wang baseline row in the same matrix
  launch condition = satisfied: 64k and 262k positive signals survived, baseline anchor is near reference
  status language = paper-scale single-seed diagnostic only
```

## 5. Keep / Discard 标准

### Keep

保留一个创新模块需要满足至少一条：

```text
同预算 AUC 明显高于 N0 baseline
同预算 calibrated_accuracy 明显高于 N0 baseline
accuracy 接近或超过 Zhang/Wang reference gap
相同指标下训练更稳定、overfitting 更小、方差更小
实现明显简化且不损失指标
```

### Discard

丢弃或降级一个路线的条件：

```text
提升来自 benchmark 改动而非模型/特征
negative_mode 或 sample_structure 不一致导致不可比
只在 tiny/smoke 尺度有效，64k/262k 消失
多 seed 方差覆盖全部提升
参数量/FLOPs 大幅增加但指标无稳定收益
真实 P-layer topology 不优于 shuffled/random topology
辅助任务好看但主任务无提升
```

## 6. Artifact 和文档要求

每个非 smoke 实验必须保留：

```text
plan CSV
remote config JSON
source commit
remote run dir
local retrieved dir
results JSONL
history CSV
curves SVG
result_gate.txt
progress JSONL
stdout/stderr
git/gpu/torch evidence
```

每次结果进入文档时必须说明：

```text
planned / running / completed remotely / fallback-retrieved / verified-branch retrieved
是否 plan-aligned
是否 strict negative
是否 single-seed
是否 formal
```

## 7. 暂不做的事

近期不做：

```text
不直接实现 full SPN-GNN 作为第一创新网络
不直接上 Transformer/MoE/大而全 adaptive router
不把 related-key 结果和 ordinary E=R 结果混报
不把 multi-pair 提升直接当 raw single-sample SOTA
不把 single-seed 1M innovation 结果写成正式结论；若正向，必须进入多 seed
不为了提升指标修改 validation data、labels、metric 或 result gate
```

## 8. 立即行动清单

当前立即推进：

1. 提交并推送 `I1-SPN-001-paper-scale seed0` plan CSV 与 remote config。
2. 远程启动 `innovation1_spn_present_nibble_paligned_mcnd_r7_1m_seed0_gpu1_20260626`。
3. 使用本地 tmux monitor 自动等待、拉回 logs/results。
4. 完成后只按 paper-scale single-seed diagnostic 汇报；若正向，再准备 seed1/seed2。

当前最推荐的下一步创新任务名称：

```text
I1-SPN-001: Present Nibble/P-layer Aligned MCND
```

一句话目标：

```text
在不改变 Zhang/Wang official protocol 的前提下，验证显式 nibble/P-layer aligned SPN view 是否能在同预算下提升 PRESENT r7 MCND 的 AUC、calibrated accuracy 或训练稳定性。
```
