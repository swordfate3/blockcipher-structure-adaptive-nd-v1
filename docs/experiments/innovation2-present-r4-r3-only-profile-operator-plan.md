# 创新2 E73：PRESENT四轮r3-only平衡谱算子压缩计划

日期：2026-07-18

状态：已完成 / pass / r3-only双seed确认为更简洁的方法

## 1. 研究问题

E72确认r3是E68稳定方法的主导前缀：r3-only ridge AUC=`0.799444`，中和r3使双seed
checkpoint AUC下降`0.355556/0.271389`；r2影响近零，r1只下降`0.025556/0.009722`。

E73测试一个最小化假设：删除r1/r2的26维输入，只把r3的13维ANF support/degree前缀送入
E68同一个`Prefix-Guided Nodewise Profile Operator`。不改变64-node输出、共享消息步、P-layer
关系、标签、split、loss或optimizer。

## 2. 冻结数据与三行矩阵

```text
cipher/rounds       = PRESENT-80 / 4
label source        = E65严格unit profile
train/validation    = 50 / 18 structure-disjoint groups
observed edges      = 356 / 120
input               = anf_prefix_26..38（r3 13维）
hidden/steps        = 32 / 2
dropout/batch       = 0.10 / 8
optimizer           = AdamW(lr=1e-3, weight_decay=1e-4)
```

三行只改变关系：

```text
independent node
true-P r3-only candidate
fair-corrupted-P control
```

三行参数必须完全相同；候选参数量必须至少比E68的`5679`减少10%。不得读取r1/r2、第四轮
certificate、witness、parity、full-cube或标签派生特征。

## 3. Phase A readiness

```text
run_id = i2_present_r4_r3_only_profile_operator_readiness_seed0_20260718
output = outputs/local_smoke/i2_present_r4_r3_only_profile_operator_readiness_seed0_20260718
epochs/seed/device = 2 / 0 / local CPU
```

协议门：E65/E43 source、hash、r3切片逐值重放、输出shape、masked BCE、有限梯度、参数公平、
cell重标号等变、正确/错误P logits不同、禁用字段不存在，全部通过。

信号门：

```text
true-P validation AUC                >= 0.75
true-P - independent                 >= 0.03
true-P - corrupted                   >= 0.03
```

通过才自动进入Phase B；失败则保留完整39维E68，不增加容量或epoch。

## 4. Phase B正式比较

Phase B保持同一三行矩阵，30 epochs seed0。与E67 seed0锚点比较：

```text
r3-only true-P validation AUC        >= 0.93
train - validation gap               <= 0.15
r3-only - E67 full-prefix            >= -0.02
r3-only - independent/corrupted      each >= 0.03
```

全部通过才运行seed1，并用E68 seed1 `0.961389`重复同门。双seed保持质量且参数减少至少10%，
则r3-only成为首选简化方法；若双seed平均超过E68 `0.957222`才更新性能第一，否则只更新为更
简洁的等效方法。

## 5. 证据范围

E73仍是PRESENT-80四轮、8-bit活动cube严格unit-output balance profile的本地方法实验，不是
高轮积分区分器、跨维度证据、攻击、远程规模或SOTA。

## 6. Phase A readiness结果

E65/E43 source、hash、r3切片、split、masked loss、cell等变、参数公平和训练流程全部通过。
三行均为`4795`参数，相对E68的`5679`为`0.844339`，减少`15.57%`。

| 模式 | best epoch | train AUC | validation AUC |
|---|---:|---:|---:|
| independent | 2 | `0.567558` | `0.598194` |
| true-P | 1 | `0.814607` | `0.834167` |
| fair-corrupted-P | 2 | `0.628109` | `0.650278` |

```text
true - independent = +0.235972
true - corrupted   = +0.183889
decision           = innovation2_present_r3_only_profile_readiness_passed
```

绝对`0.75`和两项`+0.03`门均通过，因此按计划进入30轮seed0。

## 7. 30轮seed0正式结果

| 模式 | best epoch | train AUC | validation AUC |
|---|---:|---:|---:|
| independent | 24 | `0.604169` | `0.657500` |
| true-P | 28 | `0.964746` | `0.945556` |
| fair-corrupted-P | 29 | `0.728033` | `0.820000` |

```text
true - independent        = +0.288056
true - corrupted          = +0.125556
true - E67 full 39-d      = -0.007500
train - validation gap    = +0.019190
decision                  = innovation2_present_r3_only_neural_gain_attributed
```

绝对`0.93`、过拟合、`-0.02`完整前缀容差和两项拓扑归因门全部通过，进入seed1。

## 8. seed1与双seed联合结果

seed1三行validation AUC：

```text
independent       = 0.671944
true-P            = 0.947778
fair-corrupted-P  = 0.879444

true - independent = +0.275833
true - corrupted   = +0.068333
true - E68 seed1   = -0.013611
train-val gap      = +0.017883
```

联合比较：

| 方法 | seed0 true AUC | seed1 true AUC | mean true AUC | 参数量 |
|---|---:|---:|---:|---:|
| 完整39维E68 | `0.953056` | `0.961389` | `0.957222` | 5679 |
| r3-only E73 | `0.945556` | `0.947778` | `0.946667` | 4795 |
| 差值 | `-0.007500` | `-0.013611` | `-0.010556` | `-15.57%` |

两颗seed均通过全部冻结门：

```text
status   = pass
decision = innovation2_present_r3_only_two_seed_confirmed
remote   = no
```

## 9. 裁决与推荐下一步

E73没有超过E68的AUC，所以完整39维E68仍是性能第一；但E73用少15.6%的参数和少66.7%的
输入维度，把双seed平均损失控制在`0.010556`，同时逐seed显著超过独立node与错误P。这使
r3-only成为当前首选的简洁、可解释方法版本，可用于毕业论文中的压缩/机制归因结果。

停止在同一E65 split上继续枚举hidden、steps、attention或更多前缀组合，也不做远程规模。
下一阶段若继续神经结构探索，应先做E74第二真实SPN（优先GIFT-64或SKINNY-64）的严格unit
profile标签readiness：同样要求sound positive、concrete negative、unknown不训练、
structure-disjoint split和强确定性基线。只有标签门通过，才比较r3-only true-P、同参数错误P和
确定性ridge；不得把PRESENT四轮结果直接宣称跨密码泛化。

readiness、seed0正式与双seed最终三张`curves.svg`均经`visual-qa-redraw`渲染为
`1900 x 1013`像素检查；AUC、拓扑差值、完整前缀窄范围对比、门槛、裁决和范围均无重叠、
裁切或图例歧义。
