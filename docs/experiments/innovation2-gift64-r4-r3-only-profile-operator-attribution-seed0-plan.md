# 创新2 E78：GIFT-64四轮r3-only平衡谱算子30轮seed0正式归因计划

日期：2026-07-19

状态：已完成 / pass / 允许完全相同协议seed1确认

## 1. 研究问题

E77已确认E76的单节点ridge门信息范围错配：公平true-P topology-expanded ridge达到`0.743496`，
领先最强same-family错误P `+0.041363`；同权重checkpoint反事实margin为`+0.055151`。

E78检验：在不改变E75数据、r3输入、模型容量和控制的条件下，30轮训练后真实GIFT P-layer
是否仍稳定领先独立node、错误P和公平确定性拓扑ridge。

## 2. 冻结来源与矩阵

```text
profile source = E75 192结构严格unit profile
readiness      = E76两轮hold结果（只作历史来源，不恢复checkpoint）
readjudication = E77 pass
```

三行从相同seed0随机初始化重新训练，不从E76恢复权重：

```text
independent = 4,795参数，本节点更新
corrupted   = 4,795参数，同cell + destination-cell shift1错误P
true        = 4,795参数，同cell + 真实GIFT P
```

训练协议：

```text
input dimensions = 13 (r3-only)
hidden / steps    = 32 / 2
dropout           = 0.10
batch size        = 8 structures
epochs            = 30
optimizer         = AdamW, lr=1e-3, weight_decay=1e-4
seed / device     = 0 / local CPU
```

标签、split、620个observed坐标和checkpoint选择继续冻结；每行按validation AUC选择最佳epoch。

## 3. 正式门

协议门：E75/E77来源与hash、三行同参数、masked loss、cell等变、true/corrupted拓扑不同、30轮完成、
指标有限，且无证书/witness/parity/label状态进入模型。

候选质量门：

```text
true-P validation AUC >= 0.80
true-P train - validation AUC <= 0.15
true-P - E77 true-P topology-expanded ridge >= +0.03
```

拓扑归因门：

```text
true-P - independent >= +0.03
true-P - corrupted   >= +0.03
```

全部通过：

```text
decision = innovation2_gift64_r3_only_neural_gain_attributed
next     = 完全相同30轮seed1确认
```

任一候选或拓扑门失败：保留E75/E77标签与确定性拓扑证据，关闭GIFT r3-only神经正式路线；不得
增加hidden、steps、epochs、attention或更换较弱错误P。

## 4. 运行与产物

```text
run_id = i2_gift64_r4_r3_only_profile_operator_attribution_seed0_20260719
output = outputs/local_diagnostic/i2_gift64_r4_r3_only_profile_operator_attribution_seed0_20260719
```

必须生成`results.jsonl`、`history.csv`、三个checkpoint、`gate.json`、`summary.json`、
`metadata.json`、`progress.jsonl`和中文`curves.svg`，刷新最近结果索引并执行真实像素
`visual-qa-redraw`。即使通过也只构成GIFT-64四轮seed0正式归因，不是双seed、跨密码泛化、高轮、
攻击、远程规模或SOTA结论。

## 5. 实际结果

E75/E77来源、全部hash与gate、620坐标、三行参数公平、masked loss、cell等变和30轮协议全部
通过。三行均为`4795`参数，真实P最佳checkpoint出现在epoch 28。

30轮最佳validation AUC：

```text
independent node = 0.571280  (best epoch 25)
corrupted GIFT P = 0.774714  (best epoch 27)
true GIFT P      = 0.913111  (best epoch 28)
E77 true-P ridge = 0.743496
```

正式差值：

```text
true - independent = +0.341831
true - corrupted   = +0.138398
true - E77 ridge   = +0.169615
true train - validation = -0.004748
```

候选质量门、拓扑归因门和协议门全部通过：

```text
status   = pass
decision = innovation2_gift64_r3_only_neural_gain_attributed
next     = 完全相同30轮seed1
remote   = no
```

这是当前首个在真实GIFT-64严格输出性质标签上，30轮正式超过独立node、same-family错误P和公平
确定性拓扑ridge的神经结构seed0结果。它仍不是双seed稳定结论，不能称跨密码泛化或高轮突破。

`curves.svg`已执行`visual-qa-redraw`。首次1600x848像素检查发现错误P数值`0.775`与0.80门线
碰撞；重画后把全部验证值移入蓝色柱体，最终图无文字重叠、裁剪、缺字、图例冲突或阈值歧义。

## 6. 推荐下一步

只运行E79：相同E75数据、三行、13维输入、4795参数、30轮与优化器，唯一改变训练seed为1。
seed1必须独立满足true AUC、相对independent/corrupted、相对E77 ridge和过拟合门；通过后才做双seed
联合裁决。不得先调整模型或重新选择checkpoint规则。
