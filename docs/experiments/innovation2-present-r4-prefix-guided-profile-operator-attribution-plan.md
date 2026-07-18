# 创新2 E67：PRESENT四轮prefix引导平衡谱算子30轮seed0正式归因计划

日期：2026-07-18

状态：已完成 / pass / 允许同矩阵seed1复核

## 1. 研究问题

E66两轮readiness中，正确P profile mixer验证AUC为`0.799167`，相对同参数independent和
fair-corrupted-P分别为`+0.081389/+0.106944`。E67在不改变任何数据或模型结构的情况下，
正式判断该差异能否在30轮训练后同时满足：

1. 绝对预测质量；
2. 超过同容量独立node；
3. 依赖正确P-layer；
4. 超过E65最强安全确定性ANF-prefix ridge。

## 2. 冻结协议

```text
cipher/rounds      = PRESENT-80 / 4
source             = E65 strict unit profile + E66 readiness
train/validation   = 50 / 18 structure-disjoint groups
observed edges     = 356 / 120
prefix             = r1/r2/r3 39维ANF support/degree，逐行重放
model              = PrefixGuidedProfileOperator
hidden/steps       = 32 / 2
dropout            = 0.10
batch              = 8 structures
epochs             = 30
seed               = 0
optimizer          = AdamW(lr=1e-3, weight_decay=1e-4)
checkpoint         = best validation observed-edge AUC
device             = local CPU
```

禁止第4轮certificate、witness、key/offset parity、label-derived feature、validation拟合、
绝对output embedding或新增网络容量。

## 3. 冻结矩阵

```text
0. E65 ANF-prefix ridge                 validation AUC=0.793611，只读
1. independent shared node block       同容量非关系控制
2. true-P profile mixer                 候选
3. fair-corrupted-P profile mixer       错误拓扑控制
```

三行神经模型参数量、初始化、batch顺序、optimizer、epoch和checkpoint协议完全一致。

## 4. 正式门

协议门重新核验E43/E65/E66 run id、decision、hash、数组、split、prefix replay、模型contract、
参数公平、cell重标号等变、masked loss和30轮完成。

候选门：

```text
true-P validation AUC >= 0.78
true-P train - validation AUC <= 0.15
```

关系归因门：

```text
true-P - independent >= 0.03
true-P - corrupted   >= 0.03
```

方法增益门：

```text
true-P - E65 ANF-prefix ridge >= 0.02
```

全部通过才得到`profile_operator_neural_gain_attributed`，允许同矩阵seed1。候选和关系门通过但
方法增益门失败，只能保留“正确拓扑神经处理器有效但未超过确定性前缀”的方法结果，不开放
seed1。候选或关系门失败则停止该结构。任何hold都不增加hidden、steps、epochs或远程规模。

## 5. 产物与范围

```text
run_id = i2_present_r4_prefix_guided_profile_operator_attribution_seed0_20260718
output = outputs/local_diagnostic/i2_present_r4_prefix_guided_profile_operator_attribution_seed0_20260718
```

E67仍是PRESENT-80四轮严格输出性质预测的本地方法实验，不是高轮积分区分器、新攻击、远程
规模结果或SOTA。

## 6. 2026-07-18实际结果

E43/E65/E66 source、hash、严格标签、split、39维prefix逐行重放、cell等变、masked loss、
参数量和三行30轮训练全部通过。最佳checkpoint：

| 模式 | best epoch | train AUC | validation AUC | validation accuracy |
|---|---:|---:|---:|---:|
| independent node | 22 | `0.740910` | `0.762500` | `0.666667` |
| true-P profile mixer | 28 | `0.984061` | `0.953056` | `0.875000` |
| fair-corrupted-P mixer | 29 | `0.822781` | `0.800833` | `0.708333` |

与冻结锚点比较：

```text
true - independent      = +0.190556
true - corrupted        = +0.152222
true - E65 prefix ridge = +0.159444
true train - validation = +0.031006
```

绝对候选、过拟合、同容量、错误拓扑和ANF ridge增益门全部通过：

```text
status   = pass
decision = innovation2_present_profile_operator_neural_gain_attributed
remote   = no
```

## 7. 裁决与推荐下一步

执行同协议seed1复核，唯一变化为随机种子。数据、三行矩阵、39维prefix、hidden32、steps2、
dropout0.10、batch8、30 epochs、optimizer、checkpoint和全部正式门不变。seed1必须独立满足
`true AUC >= 0.78`、过拟合gap、领先independent/corrupted各`0.03`和领先ANF ridge`0.02`；
之后以两seed联合裁决决定是否保留为创新2真实PRESENT四轮结构方法证据。

最终`curves.svg`经`visual-qa-redraw`渲染为`2200 x 1163`像素检查；绝对AUC、三项差值、
0.78/0.03/0.02门、过拟合说明、裁决和证据范围均无重叠、裁切或含混。
