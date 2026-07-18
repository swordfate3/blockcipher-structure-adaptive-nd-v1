# 创新2 E71：PRESENT四轮显式轮序平衡谱算子计划

日期：2026-07-18

状态：已完成 / hold / 正向轮序未超过反向轮序控制，停止正式训练

## 1. 研究问题

E68的`Prefix-Guided Nodewise Profile Operator`已在PRESENT-80四轮、8-bit活动cube的严格
unit-output标签上获得双seed平均AUC `0.957222`，但它把r1/r2/r3的三个13维ANF前缀直接拼成
39维输入。网络知道列位置，却没有显式执行“按轮推进”的共享状态更新。

E71只测试一个新假设：

```text
r1 13维前缀 -> shared recurrent node update -> shared S/P message block
r2 13维前缀 -> shared recurrent node update -> shared S/P message block
r3 13维前缀 -> shared recurrent node update -> shared S/P message block
                                                -> 64个unit logits
```

候选名为`Round-Recurrent Prefix-Guided Profile Operator`（RR-PGPO）。创新变量是显式轮序和
共享轮处理器，不是增加网络深度、Transformer容量或标签信息。

## 2. 冻结数据与输入

```text
cipher/rounds       = PRESENT-80 / 4
label source        = E65严格unit balance profile
train/validation    = 50 / 18 structure-disjoint groups
observed edges      = 356 / 120
input               = E65逐值重放的r1/r2/r3，各13维
target/loss         = 64个unit标签 / masked BCE
negative/unknown    = E43 concrete witness / unknown不训练
```

不得读取第四轮certificate、witness、parity、full-cube或标签派生特征。数据、split、target、
checkpoint选择、AUC计算和optimizer与E67/E68保持不变。

## 3. 最小公平矩阵

三行模型使用完全相同的hidden、参数、初始化、batch顺序、optimizer和epoch：

| 行 | 轮序 | P-layer | 作用 |
|---|---|---|---|
| `true_order_true_P` | r1,r2,r3 | 正确P | 候选 |
| `wrong_order_true_P` | r3,r2,r1 | 正确P | 轮序归因控制 |
| `true_order_corrupted_P` | r1,r2,r3 | fair-corrupted P | 拓扑归因控制 |

E68同seed true-P checkpoint只读作为同预算方法锚点，不作为第四个训练行。RR-PGPO固定13维输入、
hidden22、GRU式共享node update、一个共享S/P residual block、dropout0.10；三行参数必须相同，
且参数量与E68的`5679`相差不超过10%。

## 4. Phase A readiness

```text
run_id = i2_present_r4_round_recurrent_profile_operator_readiness_seed0_20260718
output = outputs/local_smoke/i2_present_r4_round_recurrent_profile_operator_readiness_seed0_20260718
epochs/batch/seed = 2 / 8 / 0
device = local CPU
```

协议门：source/hash/prefix重放、输出`batch x 64`、masked loss、有限梯度、三行参数公平、
cell重标号等变、正确/错误轮序logit不同、正确/错误P logit不同、禁用字段不存在，全部通过。

readiness信号门：

```text
candidate validation AUC                 >= 0.70
candidate - wrong-order validation AUC   >= 0.02
candidate - corrupted-P validation AUC   >= 0.02
```

只在全部通过时自动进入Phase B；失败则停止RR-PGPO，不增加hidden、epoch或训练行。

## 5. Phase B正式归因

Phase B保持同一三行矩阵，只改为30 epochs。seed0冻结锚点是E67 true-P `0.953056`：

```text
candidate validation AUC                 >= 0.93
candidate train - validation AUC         <= 0.15
candidate - E67 true-P AUC               >= -0.02
candidate - wrong-order                  >= 0.03
candidate - corrupted-P                  >= 0.03
```

seed0全部通过才运行相同seed1；seed1锚点为E68 true-P `0.961389`并使用同一门。双seed都通过，
才保留RR-PGPO为“显式轮序有效且不损失E68质量”的候选；只有当双seed平均AUC超过E68平均
`0.957222`时才更新第一名，否则E68保持第一，RR-PGPO只作为轮序归因结果。

## 6. 证据范围与停止项

E71仍是PRESENT-80四轮、8-bit活动cube、严格unit-output balance profile的本地结构方法实验，
不是高轮积分区分器、新攻击、远程规模或SOTA。E70已经证明当前provider不能支持4/12-bit
迁移，因此E71不得宣称跨维度泛化，也不得用有限key投票补标签。

## 7. 2026-07-18 Phase A实际结果

E65/E43 source、SHA-256、`96 x 64 x 39`前缀、476个observed edges、`50/18`
structure-disjoint split均重放通过。模型contract：

```text
parameter count                    = 5,461（三行完全相同）
parameter ratio to E68             = 0.961613
cell relabel max absolute error    = 2.384e-7
masked BCE error                   = 0
round-order logit difference       = 0.811260
topology logit difference          = 0.243392
```

两轮最佳验证结果：

| 模式 | best epoch | train AUC | validation AUC |
|---|---:|---:|---:|
| 正确轮序r1->r2->r3、正确P | 2 | `0.710990` | `0.716667` |
| 反向轮序r3->r2->r1、正确P | 1 | `0.832250` | `0.867222` |
| 正确轮序r1->r2->r3、错误P | 2 | `0.687287` | `0.697222` |

正确轮序达到绝对readiness门，但：

```text
candidate - reversed order = -0.150556   (要求 >= +0.02)
candidate - corrupted P    = +0.019444   (要求 >= +0.02)
```

因此：

```text
status   = hold
decision = innovation2_present_round_recurrent_readiness_not_passed
formal 30 epochs = no
remote = no
```

39维布局已用`anf_prefix_features`源码和E65 CSV核对，确实是连续的r1/r2/r3三个13维段；反向
轮序领先不是切片错误。但它是控制行的事后强结果，不能重新命名为候选后直接扩训练。

## 8. 裁决与推荐下一步

停止原定义RR-PGPO，不增加hidden、epoch或seed，不把两轮`0.867222`写成正式方法结果。
E68继续保持当前第一名。

下一步先执行无新训练的E72方向归因审计：在E67/E68双seed true-P checkpoint上分别遮蔽
r1/r2/r3 13维切片，并用同一E65 split拟合single-round train-only ridge，判断验证信号是否
确实集中在靠近输出的r3，以及双seed模型对三段的依赖是否一致。若r3主导和切片遮蔽方向在
双seed稳定成立，才预注册一个“output-to-input backward recurrent”新候选；否则把反向轮序
高分视为两轮优化偶然并停止轮递归路线。

最终`curves.svg`经`visual-qa-redraw`渲染为`1900 x 1013`像素检查；三行训练/验证AUC、
反向轮序名称、两项差值、`0.70/0.02`门、停止裁决和范围均无重叠、裁切或图例歧义。
