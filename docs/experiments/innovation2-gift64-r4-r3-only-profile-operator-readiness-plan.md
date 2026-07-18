# 创新2 E76：GIFT-64四轮r3-only平衡谱算子readiness计划

日期：2026-07-19

状态：已完成 / hold / 单节点ridge门失败但神经拓扑控制全部通过

## 1. 研究问题

E75已经证明GIFT-64四轮、192结构的严格64维unit-balance profile具备足够正负宽度，且一元
structure/output边际AUC均为0.5。E76只检验一个神经结构假设：

```text
在同一个结构的64个输出bit上，
使用真实GIFT P-layer共享消息，
能否比同参数独立node和错误P-layer更好地利用第3轮ANF前缀？
```

本实验不从PRESENT加载checkpoint，只迁移E73已确认的归纳偏置与13维r3输入格式。两轮结果只作
实现与防退化readiness，不作正式神经收益裁决。

## 2. 冻结数据与基线

```text
source = outputs/local_audits/i2_gift64_r4_unit_balance_profile_192_structures_20260719
train / validation matched edges = 496 / 124
train / validation structures    = 110 / 33
output                            = one masked 64-logit profile per structure
unknown coordinates              = excluded by profile_observed mask
```

训练前必须重放E75 gate、shape、620个matched坐标、结构互斥、targets/observed与CSV逐项一致、
前缀有限性和文件hash。

确定性基线使用训练集均值/方差标准化和`lambda=1e-3` ridge：

```text
full-prefix ridge = r1+r2+r3, 39 dimensions
r3-only ridge     = r3, 13 dimensions
```

它们只读取公开可计算的前三轮ANF support统计，不读取最终证书、negative witness、parity或标签状态。

## 3. 三行同预算神经矩阵

```text
independent = 每个output node只使用自身状态；无跨node拓扑消息
true        = 同nibble聚合 + 真实GIFT P-layer消息
corrupted   = 同nibble聚合 + destination-cell rotation合成后的错误GIFT P-layer
```

三行使用完全相同的参数量、初始化seed和训练预算：

```text
input dimensions = 13 (only r3)
hidden dimension = 32
shared steps     = 2
dropout          = 0.10
batch size       = 8 structures
epochs           = 2
optimizer        = AdamW, lr=1e-3, weight_decay=1e-4
seed             = 0
device           = local CPU
```

模型必须保持64-logit输出、masked BCE、cell重标号等变、true/corrupted同参数并能产生不同logit。

## 4. 预注册门

确定性r3信息门：

```text
r3-only ridge validation AUC >= 0.60
r3-only ridge - full-prefix ridge >= -0.03
```

两轮神经readiness门：

```text
true-P validation AUC >= 0.65
true-P - independent >= +0.03
true-P - corrupted   >= +0.03
true-P - r3 ridge    >= -0.03
```

协议无效：修复实现，不解释AUC。确定性r3门失败但full-prefix有信号：停止r3-only，下一步只允许
完整39维GIFT profile operator readiness。神经门失败：停止GIFT r3-only正式训练，不后验调整
hidden、steps、attention或epoch。

全部通过：

```text
decision = innovation2_gift64_r3_only_profile_readiness_passed
next     = 同一三行30轮seed0正式归因
remote   = no
```

## 5. 运行与产物

```text
run_id = i2_gift64_r4_r3_only_profile_operator_readiness_seed0_20260719
output = outputs/local_smoke/i2_gift64_r4_r3_only_profile_operator_readiness_seed0_20260719
```

必须生成`results.jsonl`、`history.csv`、`checkpoints/`、`gate.json`、`summary.json`、
`metadata.json`、`progress.jsonl`和中文`curves.svg`。图像必须执行真实像素`visual-qa-redraw`，结果
必须刷新最近结果索引。E76不构成高轮、跨密码泛化、攻击、远程规模或SOTA结论。

## 6. 实际结果

E75来源、620个matched坐标、shape、CSV/NumPy重放、hash、GIFT真实/错误P置换、三行参数公平、
masked loss、cell重标号与两轮训练协议全部通过。三行参数均为`4795`，cell重标号最大误差
`2.3842e-7`，同权重true/corrupted初始logit差为`0.223726`。

确定性ridge：

```text
full39 validation AUC = 0.477888
r3-only validation AUC = 0.507804
r3 - full39           = +0.029917
```

三行两轮最佳validation AUC：

```text
independent node = 0.560874
corrupted GIFT P = 0.704475
true GIFT P      = 0.760666
```

控制差值：

```text
true - independent = +0.199792
true - corrupted   = +0.056191
true - r3 ridge    = +0.252862
```

神经readiness四个门全部通过，但预注册确定性门`r3-only ridge >= 0.60`失败。因此不能根据两轮
神经AUC直接进入30轮，正式裁决为：

```text
status   = hold
decision = innovation2_gift64_r3_only_prefix_not_sufficient
formal   = no
remote   = no
```

这不等于“神经结构没有信号”。当前ridge只读取单个output node自己的13维r3特征，而候选读取
本节点、同cell和P-layer前驱的跨node交互；因此ridge门和神经结构的可见信息范围并不公平。
结果同时显示真实P优于same-family错误P，不能在不审计该基线错位的情况下简单关闭。

## 7. 推荐下一步

E77只做无新训练的拓扑交互归因：

```text
deterministic local ridge      = local r3 13维（E76锚点）
deterministic true-P ridge     = local + same-cell mean + true-P predecessor，39维
deterministic corrupted-P ridge= local + same-cell mean + corrupted-P predecessor，39维
```

同时冻结E76 true-P checkpoint参数，在正确GIFT P和三个destination-cell rotation错误P上只做
validation推理。若true拓扑展开ridge达到`0.60`且领先corrupted至少`0.03`，并且同权重checkpoint
在正确P上领先所有错误P至少`0.03`，则说明E76原单节点ridge门错配，允许另立计划重新审判正式
训练；否则关闭GIFT r3-only路线，不增加epoch、hidden、steps或attention。

`curves.svg`已执行`visual-qa-redraw`。首次像素检查发现ridge数值与AUC=0.5虚线碰撞，重画后将
数值移入柱体；最终1600x808像素图无文字重叠、裁剪、缺字、图例冲突或坐标歧义。
