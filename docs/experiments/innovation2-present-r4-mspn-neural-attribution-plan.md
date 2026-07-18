# 创新2 E47：PRESENT r4 MSPN 30轮seed0正式归因计划

日期：2026-07-18

状态：完成 / hold / MSPN正式候选未过门

## 1. 研究问题

E45选择ANF-prefix路线，E46确认MSPN实现满足64-bit、cell等变、true/corrupted敏感、有限
梯度、无预计算证书输入和参数预算。E47正式判断：MSPN能否在E43严格checkerboard标签上
明显超过E44 pair-state，并将收益归因到正确P-layer。

## 2. 冻结来源与模型

```text
data source      = E43 matched_contrast.csv
deterministic    = E45 ANF-prefix ridge, AUC 0.686082
neural anchor    = E44 triangle, AUC 0.561979
model source     = E46 MSPN readiness passed

hidden           = 32
degree channels  = 9
shared steps     = 4
epochs           = 30
batch            = 32
dropout          = 0.10
AdamW            = lr 1e-3, weight_decay 1e-4
checkpoint       = best validation AUC
seed             = 0
device           = local CPU
```

不得改变E43 split、label、structure、mask、E46模型结构或预处理。不得输入E45手工特征、
final oracle、certificate字段或witness信息。

## 3. 冻结矩阵

```text
1. E45 ANF-prefix ridge anchor（只读）
2. E44 triangle anchor（只读）
3. MSPN true P-layer seed0
4. MSPN fair-corrupted P-layer seed0
5. MSPN true P-layer + train-label shuffle seed0
```

三行MSPN使用相同参数、seed、训练预算与checkpoint协议；shuffle只打乱train标签，保持
validation标签真实。

## 4. 正式门

流程门沿用E46并重新核验source hash、cell等变、参数、禁用输入、有限metric和三行30 epochs。

候选门：

```text
MSPN true validation AUC       >= 0.62
MSPN true - E44 triangle       >= 0.04
MSPN true - E45 prefix ridge   >= -0.04
MSPN true - label shuffle      >= 0.05
```

拓扑归因门：

```text
MSPN true - fair-corrupted P   >= 0.03
```

全部通过：

```text
decision = innovation2_present_mspn_topology_attributed
next     = 同矩阵seed1本地确认
```

候选未过：停止MSPN，不加hidden/epoch；比较训练曲线与E45单特征，判断压缩状态是否丢失
变量身份。候选过门但拓扑差不足：保留ANF传播候选，但撤回P-layer归因；下一步测试player
transport与term combination的分离消融。shuffle异常只修协议。任何情况都不直接迁移r5、
seed1或远程GPU。

## 5. 产物

```text
outputs/local_diagnostic/i2_present_r4_mspn_neural_attribution_seed0_20260718/

results.jsonl
history.csv
gate.json
summary.json
metadata.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

声明范围：PRESENT-80四轮、本地seed0、严格标签上的MSPN候选与拓扑归因；不是高轮积分
区分器、远程规模结果、新攻击或SOTA。

## 6. 2026-07-18实际结果

权威run：

```text
i2_present_r4_mspn_neural_attribution_seed0_20260718
```

E43--E46 source、hash、标签、split、MSPN contract和三行30 epochs流程检查全部通过。结果：

| 行 | 最佳epoch | train AUC | validation AUC |
|---|---:|---:|---:|
| E45 ANF-prefix ridge | 0 | 不适用 | `0.686082` |
| E44 triangle | 27 | `0.611363` | `0.561979` |
| MSPN true P | 29 | `0.794375` | `0.518673` |
| MSPN fair-corrupted P | 30 | `0.782613` | `0.560830` |
| MSPN train-label shuffle | 10 | `0.572519` | `0.527291` |

冻结差值：

```text
MSPN true - E44 triangle       = -0.043307
MSPN true - E45 prefix ridge   = -0.167409
MSPN true - fair-corrupted P   = -0.042157
MSPN true - label shuffle      = -0.008618
```

true MSPN训练AUC接近`0.79`但组外validation只有`0.519`，存在明显结构组过拟合；错误P-layer
反而达到`0.561`。因此所有候选门和拓扑归因门均失败：

```text
status   = hold
decision = innovation2_present_mspn_candidate_not_ready
seed1    = no
remote   = no
```

## 7. 推荐下一步

停止当前MSPN，不增加hidden、epoch或层数。E45 prefix有效而E47失败，说明仅保留每bit的
匿名degree/support压缩状态不足：它没有显式保留“哪些活动变量共同组成单项式”的身份，
term mean/product也可能把不同support集合压成相同表示。

下一审判应是E48“support-state collision与变量身份必要性审计”：

```text
1. 对E43每个structure计算r1/r2/r3精确support集合签名；
2. 比较MSPN degree-only签名碰撞率和标签冲突率；
3. 构造固定维随机XOR/minhash support sketch，检查在不读取full-cube oracle时能否降低碰撞；
4. 用相同train-only ridge比较degree-only、identity-sketch、true/corrupted transport。
```

只有identity sketch显著减少跨标签碰撞并在validation超过E45 prefix或至少稳定接近，才设计
下一网络`Identity-Sketch Monomial Propagator`。否则停止神经近似证书路线，保留E45确定性
方法学结果。E48前NBFNet、seed1、r5迁移和远程GPU继续关闭。

最终`curves.svg`按`visual-qa-redraw`渲染为`1385×729`像素检查；标题、说明、30轮曲线、
锚点柱图、数值、阈值、裁决和导出边界均无重叠、裁切或缺字，已记录
`visual_qa_passed.marker`。
