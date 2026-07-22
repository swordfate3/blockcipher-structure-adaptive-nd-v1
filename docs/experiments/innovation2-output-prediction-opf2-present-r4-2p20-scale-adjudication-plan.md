# 创新2 OPF2：PRESENT四轮输出预测 `2^20` 训练规模审判计划

日期：2026-07-22

状态：计划冻结 / 实现与本地readiness通过 / 远程运行中

## 1. 为什么OPF1还不能关闭四轮

OPF1在固定seed7密钥、八个预注册真实密文输出bit、`2^17/2^16`训练/测试对、100 epochs和五模型
同协议下得到：

```text
真实P位置头平均AUC = 0.513755358
标签打乱平均AUC    = 0.500934143
真实P - 标签打乱   = +0.012821215
通过逐bit门        = 0 / 8
```

它严格证明当前预算下三轮到四轮出现经验边界，但不是数据规模上限。Kimura等人的Table 12对完整PRESENT
只实测到三轮：`2^17`训练对、密文预测成功概率`2^-1.30`；四轮是`N/A`。论文正文把额外轮留作需要
`>2^19`训练对的未决问题，不是已经取得的四轮结果。相邻Jeong输出仿真工作使用`2^20--2^22`训练对和
最多300 epochs，但没有PRESENT或显式P层模型。因此OPF2只借用其规模动机，不声称复现论文四轮。

## 2. 唯一改变的变量

```text
OPF1 train rows = 131072  (2^17)
OPF2 train rows = 1048576 (2^20)
```

以下保持不变：

```text
cipher / rounds     = PRESENT-80 / 4
key                 = OPF1同一seed7固定未知密钥
test                = OPF1原65536条测试明文，逐值完全相同
input / target      = 64个MSB-first明文bit / 同八个真实密文输出bit
models              = OPF1原五模型
epochs / batch      = 100 / 250
loss / optimizer    = raw-output MSE / RMSprop 0.001
selection           = final epoch
sample class        = false
```

## 3. 防止旧测试集泄漏进扩展训练集

同一个seed7唯一明文RNG流按以下索引切分：

```text
train = [0, 131072) U [196608, 1114112) = 1048576条
test  = [131072, 196608)                 = 65536条
```

也就是说保留OPF1原训练前缀和原测试段，只从原测试段之后追加训练明文。正式缓存必须冻结：

```text
OPF1 train plaintext raw SHA256 = eca0f5705c2d9a6b4f0475bfb90e55d2bfa2d5e4d7b8c380b10ab55778a4555a
OPF1 test plaintext raw SHA256  = 5c5410d4c0761f729f5f705d43a7392bf90f6ae0bee65a57321760d515b82fec
```

所有`1114112`条明文必须唯一，实际训练与测试索引必须零重合。磁盘缓存继续使用
`plaintexts.npy/features.npy/full_targets.npy/cache_metadata.json`，metadata额外记录非连续训练段、旧测试段、
生成进度和参数匹配复用；不得先整批放进内存再生成。

## 4. 五模型与裁决门

五模型、初始化、容量和控制与OPF1完全相同：

```text
全局头ResCNN
无P位置头ResCNN
真实P位置头SPN-ResCNN
错误P位置头SPN-ResCNN
真实P位置头SPN-ResCNN + 训练标签打乱
```

四轮输出预测主门不改变：

```text
真实P平均AUC >= 0.55
真实P - 标签打乱平均AUC >= 0.03
真实P平均accuracy-majority >= 0.005
至少4/8 bit同时满足：AUC >= 0.55、对shuffle >= 0.015、accuracy-margin >= 0.005
```

真实P相对全局、无P和错误P继续作为归因诊断，不阻断“输出是否可预测”的主结论。另报告相对OPF1
`0.513755358`的规模增益，但不根据中间epoch或单个bit事后改门。

## 5. 执行路径和下一步

本地只跑`64/64`、1 epoch CPU readiness，验证非连续训练索引、旧测试保留逻辑、缓存恢复、五模型、
40条结果、5条history、5个checkpoint和SVG；随机小样本AUC不作性能结论。

readiness通过后从推送提交在A6000 GPU0运行正式`2^20/2^16`五模型矩阵，并由本地tmux watcher自动回收。
远程结果完成后必须验证`40/500/5`产物门、全部SHA256、结果索引和正式SVG像素质量。

若主门通过：原样换新固定未知密钥确认四轮，确认前不进五轮。若主门未通过：停止`2^22`、300 epochs等
机械扩展，下一步只允许预注册一个新四轮架构假设，并把OPF2作为同规模锚点。不得把任一失败解释为
所有网络、所有密钥或数学上的四轮不可预测。

## 6. 本地readiness完成

本地CPU readiness已按冻结的`64/64`、1 epoch协议完成：

```text
run_id      = i2_output_prediction_opf2_present_r4_position_bound_spn_rescnn_2p20_smoke_seed7_20260722
status      = pass
decision    = innovation2_position_bound_r4_scale_local_readiness_passed
results     = 40/40
history     = 5/5
checkpoints = 5/5
```

实现门验证了五模型矩阵、OPF1测试段保留式非连续切分、磁盘缓存、checkpoint和结果产物。当前门控中的
AUC来自64条训练和64条测试的随机小样本，仅证明代码路径可运行，不构成四轮性能证据，也不用于调整正式门槛。

产物：

```text
outputs/local_readiness/
  i2_output_prediction_opf2_present_r4_position_bound_spn_rescnn_2p20_smoke_seed7_20260722/
```

`scripts/index-results`已通过，当前记录为`outputs/00_RECENT_RESULTS.md`中的`001`。最终`curves.svg`已按
2264x1340像素渲染并完成`visual-qa-redraw`检查：标题、说明、热力图、四组差值柱图、阈值线、标签和证据
边界均无非预期重叠、裁切、缺字或结构歧义，无需重绘。

下一可执行动作保持不变：提交并推送冻结实现后，从精确推送提交在A6000 GPU0启动正式
`2^20/2^16`、100 epochs五模型矩阵；启动后只做一次有界确认，随后交给本地tmux watcher监控、验证、
回收、绘图和索引。正式结果未回收前不得把本地readiness写成四轮恢复信号。

## 7. 远程启动

正式矩阵已从冻结并推送的实现启动：

```text
source commit = 0d2670782f92aa4665b955fcaaf75dceed242f19
run_id        = i2_output_prediction_opf2_present_r4_position_bound_spn_rescnn_2p20_key7_gpu0_20260722
GPU           = A6000 physical GPU0
remote root   = G:\lxy\blockcipher-structure-adaptive-nd-runs\i2_opf2_r4_poshead_2p20_k7_20260722
source clone  = 独立干净的run-owned clone
local monitor = tmux i2_opf2_r4_2p20_monitor
```

旧的`G:\lxy\blockcipher-structure-adaptive-nd`克隆存在历史修改，本次没有重置或复用。启动后唯一一次有界
远程确认已发现精确revision记录、GPU/torch日志、`started.marker`和`readiness=status=pass`。后续等待、
失败检测、验证结果分支回收、SHA256校验、正式绘图和结果索引由本地tmux watcher负责；主任务不得用SSH
循环轮询。当前状态仅为`running`，不是远程完成或已回收结果。

## 8. 结果未知时冻结的条件架构分支

为避免在OPF2揭盲后根据指标临时挑模型，唯一允许的hold分支已经盲预注册：

```text
docs/experiments/
  innovation2-output-prediction-opf3-present-r4-round-recurrent-spn-plan.md
```

OPF3只测试共享权重的四步SPN轮递推器：每步先做4-bit cell局部混合，再做公开P层路由，并以轮次位置
上下文吸收固定未知轮密钥差异；数据、密钥、八个输出bit、`2^20/2^16`、100 epochs、RMSprop、MSE和
final-epoch选择全部继承OPF2。它不是并行任务，当前没有实现或启动。

```text
OPF2通过 -> OPF3关闭；原样更换全新固定未知密钥确认OPF2
OPF2未通过且完整来源门通过 -> 才执行OPF3本地readiness；冻结模型代码可在OPF2指标未知时盲态准备
```

LSTM、Transformer、更多wrong-P、失败的全局头SPN-ResCNN混合、通用扩容、`2^22`和300 epochs均未被
选为该条件分支，理由和完整停止门见OPF3计划。
