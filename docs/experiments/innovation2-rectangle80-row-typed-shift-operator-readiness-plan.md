# 创新2 E92：RECTANGLE Row-Typed Shift Operator两轮readiness计划

日期：2026-07-19

状态：已完成 / `hold` / row类型独立增益不足

## 1. 研究问题

E91确定性审计证明row-typed true表示相对无类型true、同维错误P和同维错误row分别提高
`+0.014282/+0.037325/+0.017224`，三项机制门通过。E92只检验：

```text
在不增加参数的条件下，把RECTANGLE row类型编码进P层消息，
能否比无类型算子、错误P和错误row类型产生可归因的两轮神经增益？
```

## 2. 新结构与容量约束

沿用E89的`4,795`参数共享node operator。对hidden=32的P层前驱消息，按row类型执行固定通道循环
置换：

```text
row 0 shift = 0
row 1 shift = 8
row 2 shift = 16
row 3 shift = 24
```

置换没有可学习参数；同一个96->32更新矩阵因此以参数共享方式获得row-specific有效权重。cell均值、
输入投影、更新块、输出头、hidden、steps均不变。

四行矩阵：

```text
untyped_true       = 真实P，不做row通道置换
row_typed_true     = 真实P，使用真实row 0/1/2/3
row_typed_corruptP = 错误P，使用真实row类型
wrong_row_trueP    = 真实P，使用wrong_row=(row+1+cell mod 3) mod 4
```

四行必须均为`4,795`参数、相同初始化seed和训练预算。错误row不读结构标签或最终平衡状态。

## 3. 冻结来源与训练

```text
profile source = E88
formal boundary= E90 hold
mechanism gate = E91 pass
input           = 13维第3轮ANF前缀
hidden / steps  = 32 / 2
dropout         = 0.10
batch size      = 8 structures
epochs          = 2
optimizer       = AdamW, lr=1e-3, weight_decay=1e-4
seed / device   = 0 / local CPU
```

重放E88/E90/E91来源、hash、cell-major适配和3192个坐标。不得恢复E89/E90 checkpoint。

## 4. 协议与readiness门

协议门：四行参数量均`4795`、64-logit、masked BCE、真实/错误P不同、真实/错误row channel map不同、
cell重标号等变、两轮完成、指标/梯度有限、模型不读取certificate/witness/parity/label状态。

质量与归因门：

```text
row_typed_true validation AUC >= 0.65
row_typed_true - untyped_true >= +0.01
row_typed_true - row_typed_corruptP >= +0.03
row_typed_true - wrong_row_trueP >= +0.01
row_typed_true - E91 typed true ridge >= -0.03
```

全部通过：

```text
decision = innovation2_rectangle80_row_typed_shift_operator_readiness_passed
next     = 相同四行30轮seed0正式归因
```

任一门失败：关闭当前Row-Typed Shift Operator，不调shift、hidden、steps、epoch或追加row embedding。
协议失败先修复实现，不解释AUC。

## 5. 运行与产物

```text
run_id = i2_rectangle80_row_typed_shift_operator_readiness_seed0_20260719
output = outputs/local_smoke/i2_rectangle80_row_typed_shift_operator_readiness_seed0_20260719
```

生成`results.jsonl`、`history.csv`、四个checkpoint、`gate.json`、`summary.json`、`metadata.json`、
`progress.jsonl`和中文`curves.svg`，刷新最近结果索引并执行`visual-qa-redraw`。E92不是正式神经
收益、7轮论文复现、高轮、攻击、远程规模或SOTA结论。

## 6. 实际结果与裁决

E88/E90/E91来源、cell-major适配、四行固定通道图、参数公平、masked loss、cell重标号和两轮训练
协议全部通过。四行均为`4795`参数，没有row embedding或额外可学习容量；candidate的cell重标号
误差为`2.3842e-7`。

两轮最佳validation AUC：

```text
untyped true       = 0.805034
row-typed true     = 0.812825
row-typed corruptP = 0.775674
wrong-row trueP    = 0.806581
E91 typed ridge    = 0.838964
```

readiness差值：

```text
typed true - untyped true = +0.007792  (< +0.01)
typed true - corruptP     = +0.037152  (pass)
typed true - wrong row    = +0.006244  (< +0.01)
typed true - typed ridge  = -0.026139  (pass, gate >= -0.03)
```

绝对质量、错误P和ridge防退化门通过，但真实row类型相对无类型与错误row的独立增益均不足。正式裁决：

```text
status   = hold
decision = innovation2_rectangle80_row_typed_shift_operator_not_ready
formal   = no
remote   = no
```

固定通道置换保留了部分row信号，却不足以证明神经收益来自正确row语义；因此不运行30轮，不追加
learnable row embedding、lane专属网络、hidden、steps或epoch。E91确定性机制通过与E92神经门失败并不
矛盾：线性117维row交互可用，但当前参数零增量的通道置换未充分实现该交互。

推荐下一步停止RECTANGLE架构枚举，形成E87-E92的边界综合：第三密码严格标签成立，E89/E90显示
真实P神经质量和贴线拓扑信号，E91证明row类型的确定性价值，但E92否定当前最小神经实现。创新2
正式神经方法排名仍由PRESENT/GIFT双密码r3-only operator保持；新的训练实验应等待另一种sound
标签或独立文献/机制证据，而不是继续在RECTANGLE上调结构。

`curves.svg`已通过`visual-qa-redraw`最终2156x1141像素检查，无文字重叠、裁剪、遮挡、缺字、
图例冲突、阈值歧义或不可读内容。
