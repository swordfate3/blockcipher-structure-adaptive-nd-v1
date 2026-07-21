# 创新2 OP9：PRESENT三轮Kimura式完整密文输出预测校准

日期：2026-07-21

状态：远程单密钥校准已验证回收 / 完整输出路线暂缓

## 1. 唯一研究问题

在固定未知PRESENT-80密钥和未见明文上，Kimura论文族的完整输出LSTM协议能否在三轮恢复当前
小预算MLP/parity诊断没有覆盖的真实64-bit密文输出信号？

```text
C = PRESENT_K^3(P)
input  = P的64个MSB-first bit
target = C的64个MSB-first真实bit
```

网络预测64个输出值，没有真假样本、正负类别、积分平衡标签、kernel或ATM关系标签。

用户于远程训练进行中明确：创新2不要求同一样本的64个bit全部同时命中，主目标是发现哪些真实密文
输出bit容易预测。OP9-B因此保留为共享骨干的64位置扫描和Kimura论文族校准；完整64-bit exact-match
降为文献参照，不能再单独裁决创新2成功或失败。逐bit候选必须进入OP10的独立fresh-holdout确认。

## 2. 文献锚点

PRESENT r3 ciphertext prediction按Kimura 2022 Table 7与Table 12/C.2冻结：

```text
model          = stacked LSTM + Dense(64)
hidden         = 300
layers         = 6
loss           = mean squared error on raw Dense outputs
optimizer      = RMSprop
learning rate  = 0.001
epochs         = 100
batch size     = 250
train rows     = 2^17 total plaintext/ciphertext pairs
test rows      = 2^16 total disjoint pairs
key            = one fixed unknown key per trained model
primary metric = rounded 64-bit exact match
```

这是单密钥论文协议校准。Kimura Experiment 2对100把独立密钥分别训练并汇总；OP9不冒充该范围。

## 3. 精简矩阵

远程正式校准只保留三行：

```text
kimura_lstm_true_output       论文式6层LSTM，真实64-bit密文输出
matched_mlp_true_output       约400万参数两层MLP，同MSE/RMSprop/epoch/数据
kimura_lstm_label_shuffle     同一论文LSTM，仅打乱训练输出行；测试仍是真实密文
```

参数量匹配MLP使用hidden `1936`，避免把约400万参数LSTM与OP1的约3万参数MLP直接解释为纯架构差异。
除模型拓扑与标签打乱外，三行共享固定密钥、明文、输出、训练步数、优化器和评价集合。

## 4. 两阶段规模

### OP9-A 本地实现门

```text
train/test = 64 / 64 total rows
epochs     = 1
batch      = 32
model      = 完整hidden300/layers6 LSTM与hidden1936 MLP
device     = CPU
```

只要求数据、MSB-first编码、完整输出重放、三行训练、checkpoint和指标有效。任何性能数字都不作为
研究结果。通过后自动提交、推送并启动远程OP9-B。

### OP9-B 远程单密钥校准

```text
train/test = 131072 / 65536 total rows
epochs     = 100
batch      = 250
device     = lxy-a6000 CUDA
```

数据必须分块写入磁盘缓存，包含`plaintexts.npy`、`features.npy`、`full_targets.npy`、metadata、
progress和参数匹配复用；不得在训练前纯内存一次性生成全部数据。

## 5. 指标与门控

论文主指标：

```text
raw Dense output -> numpy.rint -> 64-bit exact match
Kimura Table 12 PRESENT r3 CP reference = 2^-1.30 ≈ 0.406126
reference count on 2^16 tests          ≈ 26616 complete matches
```

支持指标：逐bit match、macro AUC、MSE、舍入到非`0/1`的cell比例、逐bit majority baseline、
训练历史和checkpoint SHA-256。AUC不替换论文exact-match。

本地实现门通过条件：

```text
固定密钥、互斥明文、MSB-first输入/输出和标量PRESENT重放全部通过
三行各完成1 epoch且指标有限
真实与打乱行测试标签均为同一真实密文输出
三个最终checkpoint与manifest存在且hash匹配
```

远程路线继续门：

```text
kimura_lstm exact-match至少观察到1个真实64-bit完整命中
kimura_lstm bit match >= 0.505
kimura_lstm bit match - shuffled bit match >= +0.005
kimura_lstm macro AUC - shuffled macro AUC >= +0.010
```

上述门只裁决完整输出论文族是否存在信号，不裁决创新2的逐bit主目标，也不裁决论文水平。最终报告还必须列出：

```text
observed exact-match rate / count
paper reference exact-match = 2^-1.30 ≈ 0.406126
observed / paper-reference ratio
observed与论文参考的log2差距（若observed非零）
```

Table 12是多把独立固定密钥模型的平均结果，而OP9-B只有一把固定密钥。即使路线继续门通过，若
exact-match远低于`0.406126`，只能写“检测到单密钥真实输出信号但与论文报告有差距”，不能写成
论文协议复现或论文水平校准通过。这一外部参照在远程结果揭盲前冻结，不改变正在运行的训练协议。

参数量匹配MLP只用于说明LSTM是否带来序列架构增益，不作为论文协议有效性的必要条件。

## 6. 下一步与停止项

无论完整64-bit exact-match门是否通过，下一步先读取三行模型的64个逐bit指标：在OP9测试集上只做
候选发现，再用同一固定密钥、全新且不重合的`2^16`明文确认冻结候选。只有确认bit才能进入专用输出头
和第二固定秘密密钥复验；在此之前不扩到r4、不加入parity、不增加网络族。完整输出路线若未通过，
只停止把“全64-bit同时命中”作为目标，不能据此宣布单bit输出预测失败。

无论结果如何，OP9都不是100密钥论文复现、正式攻击轮数、PRESENT高轮突破或SOTA结果。OP9也不直接
给出“最容易bit”的无偏结论，因为从64个位置中选择最大值必须经过OP10独立holdout确认。

## 7. OP9-A本地实现门结果

本地完整架构实现门已完成：

```text
run_id   = i2_output_prediction_op9_present_r3_kimura_lstm_smoke_20260721
scale    = 64 train / 64 test total rows, 1 epoch
status   = pass
decision = innovation2_output_prediction_kimura_lstm_local_smoke_passed
```

十项协议检查和五项执行检查全部通过，包括MSB-first明文/密文重放、固定密钥、互斥明文、三行训练、
真实测试输出、checkpoint存在与SHA-256。完整架构参数量为：

```text
Kimura LSTM       = 3,994,864
matched MLP       = 3,999,840
relative gap      < 0.13%
```

实现门的LSTM只训练1 epoch，原始Dense输出有`90.625%`舍入为非`0/1`，因此bit match仅`0.0505`；
这个数字不作性能解释，也不和OP1--OP8指标比较。门只证明论文式raw-MSE、舍入exact-match和远程
缓存/checkpoint通路可执行。研究性能只能由OP9-B的100 epochs结果裁决。

`curves.svg`已由`visual-qa-redraw`渲染为`1800 x 1226`检查，标题、四面板、缩放轴、中文标签、
exact-match主指标和本地证据边界无重叠、裁切、缺字或结构歧义。

## 8. OP9-B远程执行标识

```text
run_id = i2_output_prediction_op9_present_r3_kimura_lstm_2p17_seed0_gpu0_20260721
config = configs/remote/innovation2_output_prediction_op9_present_r3_kimura_lstm_2p17_seed0_gpu0_20260721.json
plan   = configs/experiment/innovation2/innovation2_output_prediction_op9_present_r3_kimura_lstm_2p17_seed0.json
```

远程包使用GPU0、固定提交、独立干净克隆、Task Scheduler `cmd.exe /c`、分块落盘数据缓存、逐epoch
覆盖式checkpoint和本地tmux自动回收。预计耗时`6--14小时`，以monitor回收的门控产物为准。

## 9. OP9-B启动记录

远程任务已于`2026-07-21 18:35 +08:00`从已推送提交启动：

```text
source commit = 714c9942f25810b850cb31573eaaad369f18538e
remote state  = running
remote root   = G:\lxy\blockcipher-structure-adaptive-nd-runs\i2_output_prediction_op9_present_r3_kimura_lstm_2p17_seed0_gpu0_20260721
monitor       = tmux:i2-op9-kimura-monitor
```

第一次调度从提交`5bdd8a5`进入训练入口后立即失败，错误为远程直接Python未发现`blockcipher_nd`。
提交`714c994`为run脚本增加`PYTHONPATH=%SOURCE_ROOT%\src`，定向回归`7 passed`；重新调度后
`started.marker`、Git revision、GPU、torch和readiness证据存在，旧`failed.marker`已被runner清除。

本地monitor负责稀疏同步进度；只有远程结果分支推送、SHA-256校验、协议门和缓存完整性全部通过后，
任务才可从`running`改为`retrieved from verified result branch`。该回收门现已完成，正式结果见下一节。

## 10. OP9-B正式结果与裁决

远程三行矩阵已完成、推送验证结果分支并回收，本地SHA-256、缓存、checkpoint manifest和协议门全部
通过。恢复后的固定执行提交为`06f7638`；远程环境缺少`matplotlib`，因此SVG按修复后的流程在本地
验证回收后生成，并通过`visual-qa-redraw`像素检查，无文字重叠、裁切或语义歧义。

```text
Kimura LSTM true output:
  bit match   = 0.499915838
  macro AUC   = 0.500007607
  exact match = 0 / 65536
  test MSE    = 0.250080714

matched MLP true output:
  bit match   = 0.498848200
  macro AUC   = 0.508066504
  exact match = 0 / 65536
  test MSE    = 0.392121862

Kimura LSTM label shuffle:
  bit match   = 0.500027418
  macro AUC   = 0.500013948
  exact match = 0 / 65536
```

OP9冻结的完整输出LSTM路线门未通过：LSTM没有超过随机或标签打乱控制，完整命中为0，与Kimura
Table 12的`2^-1.30 ≈ 0.406126`多密钥平均结果存在数量级差距。因此停止继续增加完整输出LSTM的
数据、epoch、层数、seed和轮数；这不是PRESENT三轮所有单bit输出预测失败。

MLP的宏AUC为`0.5081`，高于两个LSTM，但总体bit match仍低于逐bit多数类均值，且测试MSE暴露明显
泛化问题。按用户冻结的新主目标，OP9不再用全64-bit平均或exact-match裁决创新2，而是把两个真实
输出模型交给OP10逐bit扫描。OP10必须在发现集选择具体`(bit, model)`，再用全新不重合明文确认；
只有确认位置才能进入专用输出头和第二固定密钥复验。

```text
status              = hold
decision            = innovation2_output_prediction_kimura_lstm_single_key_not_supported
remote full-output  = stop
next adjudication   = OP10 easy true-output-bit fresh confirmation
```
