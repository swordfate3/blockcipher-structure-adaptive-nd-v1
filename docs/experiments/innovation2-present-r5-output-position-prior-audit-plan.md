# 创新2 E6：输出位置边际先验归因审判

**日期：** 2026-07-16
**状态：** 已完成；输出位置先验解释 E5 富集
**执行位置：** 本地 CPU
**远程 GPU：** 禁止

## 触发原因

E5 按预注册门槛通过，但结构 MLP 的 8 个 `4096-key` 零观察失衡候选全部
属于 `output_nibble=0`。随后只重建冻结训练 split 的审计发现：512 个训练
geometry、16 把训练密钥中，`output_nibble=0` 的经验平衡率为 `1.0`。

因此 E5 证明了 MLP 能富集候选，但没有排除一个更简单的解释：网络主要
学会了输出位置的边际先验。E6 在不改网络和 fresh keys 的条件下加入强位置
对照，判断是否仍有可归因于结构交互的残差价值。

## 冻结数据

```text
ranking/source = E4 geometry-disjoint seed0 冻结产物
train reconstruction = seed0，512 structures，16 keys
test candidates = 同一 128 个 E4 test geometries
fresh evaluation = 与 E5 相同的 4096 keys，key_seed=2026071601
top_k = 16
matched-random seed = 2026071602
training performed = no
```

重建必须逐项验证四组 geometry id 和 train key 列表与 E4
`dataset_summary.json` 完全相同，否则协议失败。

## 四种选择器

1. `structure_mlp`：原 E4 MLP top-16。
2. `train_output_position_prior`：只用 E4 train split 标签，计算 16 个
   output nibble 各自的 q1 边际均值；按均值升序、structure id 升序选择
   test top-16，不读取 test/E5 标签。
3. `position_matched_linear`：保持 MLP top-16 的 output-nibble 数量分布完全
   相同，在每个位置内按 E4 linear rank 选择相同数量。
4. `position_matched_random`：保持同一位置数量分布，在每个位置内用固定
   seed 无放回随机选择，再按 structure id 排序。

第二种检查“简单位置先验能否解释全局富集”，后两种检查“固定位置分布后，
MLP 是否还学到活动位置、掩码和固定上下文交互”。

## Readiness

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/audit-innovation2-integral-position-prior \
  --run-id i2_present_r5_integral_position_prior_readiness_seed0 \
  --ranking-root outputs/local_diagnostic/i2_present_r5_integral_parity_geometry_holdout_ranking_seed0 \
  --source-root outputs/local_diagnostic/i2_present_r5_integral_parity_geometry_holdout_seed0 \
  --top-k 4 --fresh-keys 64 --key-seed 2026071601 --matched-random-seed 2026071602 \
  --experiment-seed 0 --gate-mode position-prior-smoke \
  --output-root outputs/local_smoke/i2_present_r5_integral_position_prior_readiness_seed0
```

readiness 必须验证：E4 gate/geometry 有效、训练结构与密钥精确重建、每种
选择器数量相同、两个 matched selector 的位置直方图与 MLP 完全相同、
fresh keys 历史互斥、批量 parity 与标量实现一致、所有产物完整。

## 完整 E6

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/audit-innovation2-integral-position-prior \
  --run-id i2_present_r5_integral_position_prior_audit_4096_seed0 \
  --ranking-root outputs/local_diagnostic/i2_present_r5_integral_parity_geometry_holdout_ranking_seed0 \
  --source-root outputs/local_diagnostic/i2_present_r5_integral_parity_geometry_holdout_seed0 \
  --top-k 16 --fresh-keys 4096 --key-seed 2026071601 --matched-random-seed 2026071602 \
  --experiment-seed 0 --gate-mode position-prior-audit \
  --output-root outputs/local_diagnostic/i2_present_r5_integral_position_prior_audit_4096_seed0
```

## 冻结门槛

```text
MLP - train output-position prior mean balance   >= +0.03
MLP - position-matched linear mean balance       >= +0.02
MLP - position-matched random mean balance       >= +0.03
```

- 三门全过：`innovation2_integral_neural_interaction_residual_supported`。
- 第一门失败、后两门全过：
  `innovation2_integral_position_prior_dominant_with_conditional_residual`；论文
  必须把主要全局收益归于位置先验，只把位置内残差归于 MLP。
- 任一位置匹配门失败：
  `innovation2_integral_position_prior_explains_enrichment`；不得把 E5 的全局
  富集写成神经结构交互优势，E0-E5 作为受位置先验限制的诊断结果。

## 后续

E6 后停止为同一 PRESENT r5 数据追加模型或选择器。无论裁决分支，都将
真实边界写入毕业论文：正结果写方法贡献，条件/负结果写强基线揭示的适用
范围。不得根据 E6 结果重选 seed、top-k、fresh keys 或门槛。

## 完成结果

readiness：

```text
run_id = i2_present_r5_integral_position_prior_readiness_seed0
status = pass
decision = innovation2_integral_position_prior_audit_ready
readiness checks = 16/16 true
```

训练 split 的位置先验重建显示：

```text
output_nibble 0 train q1 rate = 0.000000000
output_nibble 1 train q1 rate = 0.106481481
output_nibble 2 train q1 rate = 0.131048387
next-lowest nonzero position  = output_nibble 3, q1 0.123046875
```

完整 E6：

```text
run_id = i2_present_r5_integral_position_prior_audit_4096_seed0
status = hold
decision = innovation2_integral_position_prior_explains_enrichment

selector                         mean balance
structure MLP                    0.956604004
train output-position prior      0.941787720
position-matched linear          0.950653076
position-matched random          0.919967651

MLP - output-position prior      +0.014816284  < +0.03
MLP - position-matched linear    +0.005950928  < +0.02
MLP - position-matched random    +0.036636353  >= +0.03
```

MLP 的 output-position 直方图为 `0:8;1:4;2:2;12:1;15:1`。位置匹配线性
和随机严格复用该直方图。MLP 只胜位置匹配随机，没有胜训练位置先验和
位置匹配线性的冻结门，因此神经交互残差不成立。

最终裁决：停止在同一 PRESENT r5 数据和 4096 fresh keys 上增加选择器、
调网络或改门槛。毕业论文应把 E0-E6 写成一个完整的强基线审计案例：弱
对照下神经排序看似有效，加入训练位置先验后主要优势被解释。该结果支持
“评测框架与失效边界”，不支持“MLP 优于简单位置规则”。

产物：

```text
outputs/local_smoke/i2_present_r5_integral_position_prior_readiness_seed0/
outputs/local_diagnostic/i2_present_r5_integral_position_prior_audit_4096_seed0/
```
