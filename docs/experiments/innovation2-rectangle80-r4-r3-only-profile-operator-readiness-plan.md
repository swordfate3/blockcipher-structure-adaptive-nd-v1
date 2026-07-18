# 创新2 E89：RECTANGLE-80四轮r3-only平衡谱算子readiness计划

日期：2026-07-19

状态：已完成 / `pass` / 允许冻结30轮seed0本地归因

## 1. 研究问题

E88已经证明RECTANGLE-80四轮、192结构的严格64维unit-balance profile具备sound、宽、
structure-disjoint且无一元位置捷径的数据门。E89只检验一个神经结构假设：

```text
在同一个结构的64个输出bit上，
使用真实RECTANGLE ShiftRow P-layer共享消息，
能否比同参数独立node和same-family错误P-layer更好地利用第3轮ANF前缀？
```

不加载PRESENT/GIFT checkpoint；只迁移已经在两种真实SPN验证过的归纳偏置、13维r3输入格式和
公平控制。两轮结果只决定是否进入30轮seed0，不作正式神经收益或新颖性结论。

## 2. 冻结数据与来源门

```text
source = outputs/local_audits/i2_rectangle80_r4_unit_balance_profile_192_structures_20260719
train / validation matched edges = 2416 / 776
train / validation structures    = 144 / 48
output                            = one masked 64-logit profile per structure
unknown coordinates              = excluded by profile_observed mask
```

训练前必须重放E88 run id、`pass`裁决、全部协议门、192结构、`192 x 64` targets/observed、
`192 x 64 x 39`前缀、3192个matched坐标、CSV/NumPy逐项一致、结构互斥和来源文件SHA-256。

## 3. 公平确定性基线

E89不重复GIFT E76的单节点ridge错配。对每个matched坐标使用训练集标准化、`lambda=1e-3` ridge：

```text
local13       = 目标输出node的第3轮13维前缀
cell13        = 同4-bit cell四个node的第3轮前缀均值
predecessor13 = 由指定P-layer进入目标node的源node前缀
expanded39    = concat(local13, cell13, predecessor13)
```

固定比较：

```text
local13
true RECTANGLE-P expanded39
corrupted-P expanded39
```

错误P只把真实P的目标cell循环移动一格，保持64位置换、4-bit cell和lane结构。确定性门：

```text
true-P ridge validation AUC >= 0.60
true-P ridge - local ridge >= +0.03
true-P ridge - corrupted-P ridge >= +0.03
```

## 4. 三行同预算神经矩阵

```text
independent = 每个output node只使用自身状态
true        = 同cell聚合 + 真实RECTANGLE P-layer前驱消息
corrupted   = 同cell聚合 + 目标cell移动一格的错误P-layer消息
```

冻结预算：

```text
input dimensions = 13 (only r3)
hidden dimension = 32
shared steps     = 2
parameters       = 4,795 each
dropout          = 0.10
batch size       = 8 structures
epochs           = 2
optimizer        = AdamW, lr=1e-3, weight_decay=1e-4
seed             = 0
device           = local CPU
```

必须通过64-logit、masked BCE、三行同参数、真实/错误P不同、cell重标号等变、有限梯度和禁止读取
certificate/witness/parity/full-cube/label状态等协议门。

## 5. 神经readiness门与裁决

```text
true-P validation AUC >= 0.65
true-P - independent >= +0.03
true-P - corrupted   >= +0.03
true-P - fair true-P ridge >= -0.03
```

协议失败：修复实现，不解释AUC。公平确定性拓扑门失败：保留E88标签证据并关闭当前RECTANGLE
r3-only神经路线，不更换更弱基线。神经门失败：不调hidden、steps、attention或epoch。

全部通过：

```text
status   = pass
decision = innovation2_rectangle80_r3_only_profile_readiness_passed
next     = 同一三行30轮seed0正式归因
remote   = no
```

## 6. 运行与产物

```text
run_id = i2_rectangle80_r4_r3_only_profile_operator_readiness_seed0_20260719
output = outputs/local_smoke/i2_rectangle80_r4_r3_only_profile_operator_readiness_seed0_20260719
```

必须生成`results.jsonl`、`history.csv`、`checkpoints/`、`gate.json`、`summary.json`、
`metadata.json`、`progress.jsonl`和中文`curves.svg`，刷新最近结果索引并执行真实像素
`visual-qa-redraw`。E89不是7轮论文复现、正式神经收益、高轮区分器、跨密码checkpoint迁移、攻击、
远程规模或SOTA结论。

## 7. 实际结果与裁决

E88来源、3192个matched坐标、CSV/NumPy重放、hash、cell-major可逆重排、RECTANGLE真实/错误P
置换、三行参数公平、masked loss、cell重标号和两轮训练协议全部通过。三行参数均为`4795`，
cell重标号最大误差`1.7881e-7`，同权重true/corrupted初始logit差`0.157643`。

信息范围对齐的ridge validation AUC：

```text
local13                    = 0.687869
corrupted-P expanded39     = 0.774292
true RECTANGLE-P expanded39= 0.824682

true ridge - local ridge     = +0.136814
true ridge - corrupted ridge = +0.050391
```

三行两轮最佳validation AUC：

```text
independent node = 0.662404
corrupted P      = 0.757646
true RECTANGLE P = 0.805034
```

控制差值：

```text
true - independent = +0.142629
true - corrupted   = +0.047388
true - fair ridge  = -0.019649
```

全部公平确定性门、神经门和协议门通过：

```text
status   = pass
decision = innovation2_rectangle80_r3_only_profile_readiness_passed
formal   = seed0 allowed
remote   = no
```

这说明真实RECTANGLE P层在两轮内同时超过同参数独立节点和same-family错误P，并接近更强的公平
确定性ridge；正向结果不是由弱基线、错误S盒cell分组或位置边际制造。E89仍只是readiness，不能
作为正式神经收益或7轮论文复现。

推荐下一步只运行冻结30轮seed0正式归因：完全重放E88数据、cell-major适配、三行、13维输入、
4795参数、AdamW和seed0；以E89 true-P公平ridge `0.824682`作为安全锚点。必须同时要求true-P
绝对质量、相对独立/错误P、相对公平ridge和train-validation gap通过。失败则关闭RECTANGLE神经
分支；成功才允许相同协议seed1，不增加容量或启动远程GPU。

`curves.svg`已执行`visual-qa-redraw`。首次像素检查发现负margin数值与`-0.03`阈值线碰撞，调整
标签位置后重新渲染；最终2156x1102像素无文字重叠、裁剪、遮挡、缺字、图例冲突或坐标歧义。
