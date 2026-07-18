# 创新2 E46：Monomial Support Propagation Network readiness计划

日期：2026-07-18

状态：完成 / pass / 允许E47正式seed0计划

## 1. 研究问题

E45证明E43严格标签的ANF 1--3轮前缀复杂度validation AUC为`0.686082`，高于正确
P-layer可达`0.648161`和E44 triangle`0.561979`。E46实现一个新的密码结构专用网络：

```text
Monomial Support Propagation Network（MSPN）
```

核心问题不是MSPN是否在两轮smoke中达到最终性能，而是它能否在不读取手工prefix特征或
final-certificate oracle的前提下，安全地执行64-bit可微ANF项传播、产生有限梯度、区分
正确/错误P-layer并从E43 matched标签开始学习。

## 2. 模型结构

输入仍为：

```text
PRESENT S-box ANF terms + P-layer + active bits + output linear mask
```

网络维护`batch × 64 × hidden`状态。初始每bit使用9维degree-support种子：固定/密钥常数
对应degree 0，活动变量额外激活degree 1。每轮共享执行：

```text
1. constant/key token injection
2. 对每个4-bit S-box输出坐标，按真实ANF monomial term集合聚合输入bit状态
3. term mean/product gate + shared MLP，压缩为hidden状态
4. true或fair-corrupted P-layer搬运
```

四轮后用output-mask query、active-set pool、global pool和交互pool产生二分类logit。

禁止输入：

```text
E45 features.csv
任何预计算support size或degree histogram前缀
full-cube candidate count
E43 positive/negative certificate字段
witness key/offset/parity word
```

MSPN必须自己从活动位、mask、S-box ANF项和P-layer学习近似传播。

## 3. Readiness smoke预算

```text
source             = E43 matched_contrast.csv
rounds             = 4 shared steps
hidden             = 32
degree seed width  = 9
epochs             = 2
batch              = 32
dropout            = 0.10
optimizer          = AdamW(lr=1e-3, weight_decay=1e-4)
seed               = 0
device             = local CPU
```

矩阵：

```text
1. E45 ANF-prefix ridge anchor（只读结果，不喂给网络）
2. E44 triangle anchor（只读结果）
3. MSPN true P-layer
4. MSPN fair-corrupted P-layer
5. MSPN true P-layer + train-label shuffle
```

三个MSPN行参数、初始化seed、batch、epoch和训练协议相同；label shuffle只打乱train标签，
validation标签保持真实，用于检查训练管线泄漏。

## 4. Readiness门

```text
E43/E44/E45 source run id、decision、hash与metric匹配
initial state shape                    = 8 × 64 × 32
shared propagation steps               = 4
PRESENT S-box ANF 16输入重构           = pass
true/corrupted P-layer均为permutation且不同
cell relabel max logit error            <= 1e-6
true/corrupted same-weight logit delta  >= 1e-5
forward logit、loss、gradient            finite
MSPN parameter count                    in [0.5, 2.0] × E44 pair-state 10725
禁止的预计算feature输入                absent
三行均完成2 epochs且metric finite
label-shuffle validation AUC            in [0.35, 0.65]
```

readiness通过只允许另建E47正式seed0计划；smoke AUC不用于选择true/corrupted或声称网络
有效。若等变、有限性或禁用输入失败，只修实现。若参数过界，调整hidden/MLP而不改数据。
若shuffle异常，先修训练协议。禁止从2 epoch结果直接运行seed1、r5或远程GPU。

## 5. E47预告门

readiness通过后，E47冻结30 epochs、seed0，比较E45 prefix ridge、E44 triangle、MSPN true、
MSPN fair-corrupted和label-shuffle。候选门建议：

```text
MSPN true validation AUC            >= 0.62
MSPN true - E44 triangle            >= 0.04
MSPN true - fair-corrupted          >= 0.03
MSPN true接近E45 prefix ridge       >= -0.04
```

这些正式门需在E47计划中再次冻结；E46不后验修改。

## 6. 产物

```text
outputs/local_smoke/i2_present_r4_mspn_readiness_smoke_seed0_20260718/

results.jsonl
history.csv
gate.json
summary.json
metadata.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

声明范围：PRESENT-80四轮MSPN实现和训练readiness；不是有效神经预测结果、高轮积分区分器、
远程规模证据或SOTA攻击。

## 7. 2026-07-18实际结果

权威run：

```text
i2_present_r4_mspn_readiness_smoke_seed0_20260718
```

E43/E44/E45 source run id、decision、hash、E44 triangle `0.561979`和E45 prefix ridge
`0.686082`全部重新核验通过。模型contract：

```text
initial state                         = 8 x 64 x 32
shared step modules                   = 1
execution rounds                      = 4
parameters                            = 17788
parameter ratio to E44                = 1.658555
cell relabel max logit error          = 5.2154e-08
true/corrupted initial logit delta    = 0.054705
forward/loss/gradient finite          = true/true/true
S-box ANF reconstruction              = pass
precomputed prefix/oracle buffers     = absent
```

两轮smoke AUC：

```text
MSPN true P-layer       = 0.506500
MSPN fair-corrupted P   = 0.507325
MSPN train-label shuffle= 0.504668
```

这些数值不用于候选性能裁决。三行均完成两轮，shuffle在冻结区间，所有readiness检查通过：

```text
status   = pass
decision = innovation2_present_mspn_readiness_passed
```

## 8. 推荐下一步

建立E47 30轮seed0正式归因，保持hidden32、degree9、batch32、相同checkerboard split和
相同MSPN代码。矩阵固定为E45 prefix ridge、E44 triangle、MSPN true、MSPN fair-corrupted
与MSPN label-shuffle。正式门使用第5节预告值；只有MSPN true同时达到`0.62`、领先E44
`0.04`、领先错误拓扑`0.03`且不落后prefix ridge超过`0.04`，才开放seed1。E47仍为本地
CPU，不使用远程GPU。

最终`curves.svg`按`visual-qa-redraw`渲染为`1385×729`像素检查；标题、说明、两轮曲线、
锚点柱图、数值、readiness范围和导出边界均无重叠、裁切或缺字，已记录
`visual_qa_passed.marker`。
