# 创新2 H0：PRESENT 高轮积分神经主流锚点计划

**日期：** 2026-07-16
**状态：** paper-reference seed0 round-reach-only / identical seed1 package prepared
**先导证据：** PRESENT-80 r5 E0-E6
**主流锚点：** Wu/Guo 2024 PRESENT integral-neural r8

## 1. 研究问题

在 keyed PRESENT-80 上，项目能否建立一个与公开积分神经工作语义一致的
multiset 区分任务，并在 r5、r7、r8 轮数梯上得到高于强基线的可重复
信号？如果 r8 成立，项目候选表示或网络能否在同一数据与预算下超过主流
anchor，并为后续 r9 探针提供依据？

本计划不把 r5 结构条件概率任务机械扩轮。它建立新的高轮 benchmark，
因为现有预审计中 r6/r7 的 16-point parity 标签已接近随机，E6 又证明
r5 的主要 top-k 收益可由输出位置先验解释。

## 2. 文献轮数参照

| 路线 | 密码 | 轮数 | 数据/结果 | 证据类型 |
|---|---|---:|---|---|
| 本项目 r5 先导任务 | PRESENT-80 | 5 | 4096 fresh keys；E6 hold | 结构条件经验预测 |
| Wu/Guo | PRESENT-80 | 5/6/7/8 | accuracy `0.9958/0.9913/0.8203/0.5732` | 积分神经区分 |
| 既有 division-property 等结果 | PRESENT | 9 | data `2^60`/`2^63` | 确定性 key-independent 积分 |
| Split-and-Cancel | PRESENT | 10 | data `2^60`/`2^63`，portion `>=2^-2` | 确定性弱密钥/条件积分 |
| Split-and-Cancel | GIFT-64 | 11（含标准输入侧扩展） | data `2^62`/`2^63`，portion `>=2^-2` | 确定性弱密钥/条件积分 |
| Split-and-Cancel | GIFT-128 | 13（含标准输入侧扩展） | data `2^127`，portion `>=2^-2` | 确定性弱密钥/条件积分 |

轮数可以用于宏观对比，但神经经验区分、完整密钥空间确定性积分和弱密钥
积分不得合并成同一强度的结论。

## 3. 冻结 benchmark

### 3.1 密码与轮数

```text
cipher            = PRESENT-80
round ladder      = r5, r7, r8
stretch round     = r9, only after r8 passes two-seed gate
keys              = train/validation/test mutually disjoint
```

r5 是协议校准点，r7 是强信号中间点，r8 是主流目标。首轮矩阵不加入 r6，
避免增加不能改变裁决的机械行。

### 3.2 一条样本

每个 multiset 包含 16 个明文。正类使用满足冻结五轮积分输入结构的活动
nibble 集合；负类使用 16 个不受该结构约束的随机明文。两类都必须用真实
PRESENT-80 和随机独立密钥加密，不能用随机密文替代负类。

参考数据表示按原文逐项对拍：

```text
C_j                         16 ciphertexts
InvP(C_j xor C_0)           previous-round aligned view
InvS(InvP(C_j xor C_0))     inverse-S-box derived view
multisets_per_sample        2 for the main reference row
```

实现前必须从原文或公开代码确认确切 bit/nibble 顺序、`C_0` 处理、两个
multiset 的拼接方式和标签采样。任何未确认项都写入 protocol audit，不能
用“合理猜测”冒充复现。

### 3.3 原文协议审计冻结（2026-07-16）

原文第 4--10 页和图 5--8 可以确认：

```text
positive multiset    = 随机固定高 60 bit，最低 4 bit 遍历 0..15
negative multiset    = 16 个不受上述结构约束的随机明文
encryption           = 正负类均由同一条样本的随机 PRESENT-80 master key 加密
reference            = 每个 multiset 内的 C_0
invP_j               = P^-1(C_j xor C_0)
invS_j               = S^-1(invP_j)
one-multiset order   = invP_0..invP_15, invS_0..invS_15
headline multisets   = 2
train/validation/test= 2^21 / 2^17 / 2^17 total samples
batch size           = 2000
loss                 = MSE + L2(weight=1e-5)
optimizer            = Adam
epochs               = 50（图 8 横轴）
selection            = 10 次独立训练，报告最高 accuracy
```

原始 PDF 第 5--6 页进一步确认数据生成顺序：先生成包含 `r` 个合法积分
multiset 的集合 `M`，再为该样本随机生成一个 master key `K` 和二元标签 `Y`；
`Y=1` 时直接用同一个 `K` 加密整个 `M`，`Y=0` 时把 `M` 中所有明文替换为
不受约束的随机明文，再仍用同一个 `K` 加密。随后每个 multiset 独立以自己的
`C_0` 形成 `InvP/InvS`，输出顺序按 multiset 依次拼接。这与当前正负类、
同样本共享 key 和两组 multiset 的语义一致。

当前实现为提高可恢复性做了两项分布近似但非字节级复现的控制：

```text
paper label sampling      = 每样本随机 Bernoulli Y
project label sampling    = row_index 奇偶交替，严格 50/50 平衡
paper key sampling        = 每样本随机生成 80-bit K
project key sampling      = seed/split/row 派生的无碰撞伪随机 80-bit K
```

在 `2^21` 量级相对 80-bit key 空间时，随机 key 碰撞概率可忽略，严格平衡标签
也不改变条件类分布；但这两项仍必须作为 protocol-control approximation 报告，
不能把 bridge 或未来 paper-scale run 称为逐随机流 exact reproduction。

原文还区分了三组不能混报的 r8 数值：

| 设置 | multiset 数 | r8 accuracy |
|---|---:|---:|
| 主结果，完整网络 | 2 | `0.5732` |
| 只比较数据格式 | 1 | `0.5369` |
| 只比较网络结构 | 1 | `0.5323` |

原文对单 multiset 给出 TensorFlow reshape：`[w, 2LM/(sw), s]`，其中
`w=16, s=8, L=64, M=16`，即 `[16,16,8]`。它没有明确说明两个
multiset 如何进入同一张量。本项目冻结的可审计假设是保持每个 multiset
内部 Eq. 6 顺序，并沿第二空间轴拼接为 `[16,32,8]`；结果必须标记
`paper_tensor_concat_assumption=spatial_axis_1`。

论文图 7 给出了 `1x1 Conv -> MBConv -> Flatten -> FC2048 x3 -> sigmoid`，
但没有报告 `Nf`、dropout rate、MBConv/Dense block 精确数量、Adam 初始
学习率或衰减上下界。公开作者代码也尚未从权威来源核验。因此，在这些参数
补齐前，本项目只能称为 **Wu/Guo paper-family protocol anchor**，不能称为
exact reproduction。readiness 和 local diagnostic 使用较小 head 只验证数据
路径；remote reference-scale 才使用 `FC2048 x3` 并完整记录推定参数。

对原始 PDF 第 9 页直接进行版面核对后，可以进一步冻结以下边界：

```text
Fig. 7 initial layer       = 1x1 Conv(Nf) -> BN -> ReLU
Fig. 7 MBConv              = 1x1 Conv(2*Nf) -> BN -> ReLU
                              -> 3x3 depthwise Conv(2*Nf) -> BN -> ReLU
                              -> Dropout -> 1x1 Conv(Nf) -> BN -> residual add
Fig. 7 Dense output        = 再经过 1x1 Conv(Nf) -> BN -> ReLU，并与 block 输入连接
Fig. 7 prediction header   = FC2048 -> BN -> ReLU
                              -> FC2048 -> BN -> ReLU -> Dropout
                              -> FC2048 -> BN -> Sigmoid
depicted MBConv blocks     = 图中画出 1 个；正文没有明确声明是否重复
Algorithm 1 inputs         = total training batches n, current batch i,
                              learning-rate range (min,max)
Algorithm 1 printed rule   = i <= n/2 时按 max/min/i 下降，否则固定为 min
```

因此，`FC2048 x3`、两个 Dropout 的位置和图中单个 MBConv 数据流已经确认；
真正仍缺的是 `Nf`、两个 Dropout rate、`min/max` 数值、公式排版中可能丢失的
归一化项，以及图中单个 MBConv 是否在实现中重复。图 8 横轴明确为 `50 epochs`。
这些缺口不影响当前只改变数据量的 5-epoch bridge，但在 bridge 通过后必须先
冻结，不能把当前小 head/no-scheduler 实现直接称为论文规模复现。

### 3.4 模型矩阵

首个有意义矩阵只保留四行：

| 角色 | 输入 | 目的 |
|---|---|---|
| paper-family anchor | Wu/Guo `InvP+InvS` reshape，MBConv | 校准原文轮数斜率 |
| project candidate | 同一 bit 数据，显式 `[multiset, view, text, nibble, bit]` SPN 网格 | 只改变张量组织与网络归纳偏置 |
| same-input linear | 同一展平输入 | 排除线性捷径 |
| shuffled-label candidate | 与 candidate 同架构 | 排除训练/类不平衡伪信号 |

另外冻结不训练的 `negative parity weight` 基线：分别对 `InvP` 和 `InvS`
跨 16 个文本求 bit parity，以 parity-one 数量的相反数作为正类分数。如果
它已经解释主要准确率，神经结果只能报告为数据结构效应，不得归因于网络。

## 4. 规模与运行路径

### 4.1 Readiness smoke

```text
train total rows       = 256
validation total rows  = 128
test total rows        = 128
rounds                 = r5 only
seeds                  = 0
device                 = local CPU
multisets per sample   = 2
paper head width       = 64（仅 shape/readiness，不用于论文比较）
```

只验证生成器、标签、张量、训练、JSONL/CSV/SVG/gate 和缓存恢复，不评价
准确率。

### 4.2 Local diagnostic

```text
train total rows       = 8192
validation total rows  = 2048
test total rows        = 4096
rounds                 = r5, r7, r8
seeds                  = 0
device                 = local CPU or local available GPU
multisets per sample   = 2
paper/candidate rows   = 同一 cache、同一 split、同一 epochs
epochs                 = 5
batch size             = 128
base channels          = 16
paper/candidate head   = 256
MBConv/residual blocks = 2
loss / optimizer       = MSE / Adam
learning rate / L2     = 1e-3 / 1e-5
```

这是轮数斜率和泄漏检查，不是正式失败或论文复现。
r5 必须先单独通过校准门，才运行完全同预算的 r7、r8；r5 不过时立即停止
后两轮并审计 bit order、双 multiset 拼接、训练动态或网络近似。

### 4.3 Remote reference-scale gate

Wu/Guo 原文报告：

```text
training samples       = 2^21 total
validation samples     = 2^17 total
test samples           = 2^17 total
batch size             = 2000
repetitions            = 10, best model reported
```

原文的 `sample` 是否一条包含两个 multiset、标签是否严格均衡，必须在协议
审计中确认。确认后报告总行数和约每类行数，不能把 `2^21 total` 写成
`2^21/class`。

毕业论文范围先运行 seed0；只有 r8 seed0 过门，才运行完全独立 seed1。
若要声称精确复现原文，仍需满足其重复次数和 checkpoint 选择方式；两个
seed 只支持项目级高轮证据。

远程运行前必须具备参数匹配的 disk-backed cache：

```text
features.npy / labels.npy or equivalent chunks
metadata with cipher/round/seed/label/data-format identity
durable generation progress before training
resume/reuse only on exact metadata match
```

## 5. 冻结门槛

### 5.1 Readiness

- PRESENT-80 已知向量和逐轮实现通过；
- 标量和向量化 multiset 生成逐样本一致；
- train/validation/test 密钥互斥；
- 正负类各占一半，负类为加密随机明文集合；
- `InvP/InvS` 与手算 fixture 一致；
- parity/统计基线、线性和 shuffled 控制均产出；
- cache、progress、results、history、SVG 和 gate 均可恢复生成。

### 5.2 Local advance gate

```text
r5 paper-anchor accuracy            >= 0.90
r8 candidate/anchor AUC             finite
no deterministic baseline explains >= 95% accuracy unless explicitly reported
no label, key, split, duplicate or ordering leakage
```

r5 校准不过时先修协议，不扩样。r8 在本地接近随机不自动判死，但只有在 r5
与 r7 斜率符合原文且不存在实现错位时，才允许一次数据稀缺诊断例外。

### 5.3 Remote r8 gate

```text
primary metric                         test accuracy and ROC-AUC
minimum useful r8 accuracy             >= 0.53
binomial 95% lower confidence bound    > 0.50
paper-reference proximity              report delta to 0.5732, no forced pass
candidate advantage over anchor        >= +0.005 accuracy or AUC
candidate advantage over strong stats  >= +0.01 accuracy or AUC
```

“达到主流神经积分轮数”只要求 r8 两个独立 seed 的置信区间高于随机；“超过
主流方法”还要求同预算 candidate 超过 paper anchor 和强统计基线。若只有
anchor 过门，成果是复现与平台建立，不是模型创新。

## 6. Advance / stop

```text
r8 seed0 fails after protocol-calibrated scale:
  stop r9 and GIFT; audit data format and deterministic baselines

r8 seed0 passes:
  run identical seed1

r8 seed0+seed1 pass, candidate does not beat anchor:
  claim mainstream round reach only; stop architecture claim

r8 seed0+seed1 pass, candidate beats anchor and strong stats:
  run one frozen r9 probe; keep r9 as exploratory until multi-seed confirmation
```

禁止路线：

- 不把删除 AddRoundKey 的 31-round public transform emulation 当成 keyed 攻击；
- 不用 small PRESENT-[4] 的轮数替代 PRESENT-80；
- 不把随机密文负类当作严格证据；
- 不在 r8 未通过时机械启动 r9/r10；
- 不在 PRESENT anchor 未站稳时并行扩 AES、GIFT、SKINNY；
- 不把经验神经区分与确定性 SAT/MILP/Split-and-Cancel 证明混报。

## 7. 产物

每个完成运行必须生成：

```text
results.jsonl
progress.jsonl
history.csv
curves.svg
gate.json
validation.json
dataset metadata/cache manifest
```

并刷新 `outputs/00_RECENT_RESULTS.md`。正式结果记录必须列明总训练、验证、
测试行数、约每类行数、multiset 数量、每个 multiset 的 16 个明文、epochs、
seed、密钥协议、负类定义和 checkpoint 选择。

## 8. 下一动作

实现独立 Innovation 2 high-round multiset task，不复用 Innovation 1 的
`plaintext_integral_nibble` pair-set 标签。先完成以下 readiness：

1. 固定明文/密钥手算 fixture 对拍 `C_j`、`InvP`、`InvS` 和 bit 顺序；
2. 参数匹配的 `features.npy/labels.npy/metadata.json` 分块 cache 能恢复与复用；
3. 四模型行和固定 parity baseline 共享完全相同的 train/validation/test；
4. 输出 `results.jsonl/progress.jsonl/history.csv/curves.svg/gate.json/validation.json`；
5. readiness 通过后运行 `8192/2048/4096 total rows` 的 r5/r7/r8 本地诊断。

只有 r5 anchor 校准通过、r7 到 r8 的斜率方向合理且控制有效，才提交远程
reference-scale 包。不得把 readiness 的小 head 结果与论文 `0.5732` 比较。

## 9. 主要来源

- Wu and Guo, *Improved integral neural distinguisher model for lightweight cipher PRESENT*，本地原文：`papers/innovation_one/grobid_md/improved-integral-neural-distinguisher-model-for-lightweight-cipher-present.md`。
- Wang, Hadipour, and Gerhalter, *On Extending Integral Distinguishers*，本地原文：`papers/innovation_two/text/on_extending_integral_distinguishers.txt`。
- Kimura et al., *Output Prediction Attacks on Block Ciphers Using Deep Learning*，本地原文：`papers/innovation_two/text/2021_kimura_output_prediction_block_ciphers.txt`。
- Singh, *PRESENT Full Round Emulation: Structural Flaws and Predictable Outputs*，本地原文：`papers/innovation_two/text/2025_singh_present_full_round_emulation.txt`。

## 10. H0 实现与本地结果

### 10.1 Readiness

```text
run_id = i2_present_high_round_integral_r5_readiness_seed0
train/validation/test total rows = 256 / 128 / 128
multisets per sample = 2
input bits = 4096
status = pass
decision = innovation2_high_round_integral_readiness_passed
recent result index = 001（完成当时）
```

readiness 验证了：严格正负类、真实 PRESENT-80、每样本唯一 master key、
split key domain 分离、`InvP(C0 xor C0)=0`、
`InvS(InvP(C0 xor C0))=0x5555555555555555`、LSB-first bit 顺序、
四模型共享 cache、`features.npy/labels.npy/metadata.json` 参数匹配复用与续生成、
以及完整 JSONL/CSV/SVG/gate/validation 产物。

### 10.2 Shuffled-control 修复审计

首个 r5 diagnostic：

```text
run_id = i2_present_high_round_integral_r5_local_8192_seed0
decision = innovation2_high_round_integral_invalid_control
```

它只打乱 train labels，却使用真实 validation AUC 选 checkpoint。真实 validation
AUC 在五个 epoch 为 `0.5403/0.6461/0.5770/0.5527/0.4702`，错误地选择了偶然
最高的 epoch 2。修复后 train/validation labels 独立打乱，并用打乱 validation
选择 checkpoint；candidate/control 使用相同初始化。

修复后仍观察到同初始化、完全不训练的 structured candidate 在 r5 上
`AUC=0.7183`，说明随机结构网络天然响应极强的 r5 数据统计，不能要求其必然
为 `0.5`。冻结的有效控制改为：shuffled-fit validation 必须接近随机，并且
真实标签 candidate 必须超过 untrained/shuffled architecture prior。该修复不
改变真实 test 标签或 candidate/anchor 训练数据。

### 10.3 同预算轮数梯

所有有效行使用：`8192/2048/4096 total rows`、2 multisets、5 epochs、
batch 128、base channels 16、head 256、2 blocks、MSE/Adam、seed0。

| 轮数 | paper-family anchor AUC | structured candidate AUC | linear AUC | fixed parity AUC | untrained/shuffled oriented prior | candidate-prior margin |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | `1.000000` | `1.000000` | `0.999999` | `1.000000` | `0.718323` | `+0.281677` |
| 7 | `0.530674` | `0.696898` | `0.613023` | `0.539603` | `0.523522` | `+0.173376` |
| 8 | `0.511205` | `0.511633` | `0.507319` | `0.503694` | `0.504342` | `+0.007291` |

r5 anchor accuracy `0.999268`，通过协议校准，但固定 parity baseline 同样完美，
所以它只是经典积分结构的实现检查。r7 candidate 有明确非线性训练增量，并
超过同输入 linear 和固定统计；这是当前最强的 Innovation 2 高轮本地信号。
r8 在本地小数据下仍近随机，不能称为达到或超过主流 r8。

有效运行：

```text
outputs/local_diagnostic/i2_present_high_round_integral_r5_local_8192_seed0_controlfix2/
outputs/local_diagnostic/i2_present_high_round_integral_r7_local_8192_seed0/
outputs/local_diagnostic/i2_present_high_round_integral_r8_local_8192_seed0/
```

## 11. 下一步：r8 远程数据稀缺桥接

### 11.1 研究问题

本地 r8 近随机是路线无效，还是相对论文 `2^21` train total 的 256 倍数据
缺口所致？r7 candidate `0.6969` 和 r5 校准通过，使一次数据稀缺诊断例外
合理；它不是机械宣告远程大规模成功。

### 11.2 冻结变量

只增加数据量，保持本地有效实现的协议、模型和训练设置：

```text
cipher / round           = PRESENT-80 / r8 only
train total rows         = 262144（约 131072/class）
validation total rows    = 32768（约 16384/class）
test total rows          = 65536（约 32768/class）
multisets per sample     = 2
epochs / batch           = 5 / 128
base/head/blocks         = 16 / 256 / 2
dropout                  = 0.1
loss / optimizer         = MSE / Adam
learning rate / L2       = 1e-3 / 1e-5
seed                     = 0
models                   = paper-family anchor / structured candidate /
                           same-input linear / same-init shuffled-label candidate
fixed controls           = parity weight / untrained same-init candidate
device                   = remote A6000
gate mode                = bridge
```

该规模仍是 bridge diagnostic，不是论文 `2^21` total train、50 epochs、10 次
重复的精确复现或 paper-scale 结论。

### 11.3 Advance / stop gate

```text
readiness/cache/artifact checks all pass
shuffled-fit validation AUC within 0.03 of 0.5
candidate test AUC >= 0.53
candidate test AUC >= architecture prior oriented AUC + 0.01
candidate test AUC >= strongest oriented fixed parity AUC + 0.01
```

若 signal gate 通过，才准备 `2^21 total train` 的 paper-reference run，并补充
独立 seed。若仍低于 `0.53`，停止继续加样本，优先审计论文缺失的 `Nf`、
双 multiset 拼接、block 数和学习率调度；不得直接跑 r9 或扩 GIFT。

远程启动前必须先完成范围提交并成功推送；平台拒绝 push 时不得使用 SCP、
dirty overlay 或旧远程 clone 启动未发布实现。

### 11.4 可执行远程包（2026-07-16）

```text
run_id = i2_present_r8_high_round_integral_bridge_262144_seed0_gpu0_20260716
physical GPU = 0（通过 CUDA_VISIBLE_DEVICES 隔离）
source = 用户传入的已推送 commit，run-owned clean detached clone
source root = G:\lxy\blockcipher-structure-adaptive-nd-runs\<run_id>\source
run root = G:\lxy\blockcipher-structure-adaptive-nd-runs\<run_id>
cache root = G:\lxy\blockcipher-structure-adaptive-nd-runs\<run_id>\dataset_cache
result root = G:\lxy\blockcipher-structure-adaptive-nd-runs\<run_id>\results
result branch = results/<run_id>
local retrieval = outputs/remote_results/<run_id>
local monitor = tmux i2-r8-integral-bridge-262144
```

冻结计划：

```text
configs/experiment/innovation2/
  innovation2_present_r8_high_round_integral_bridge_262144_seed0.json
```

远程配置和执行脚本：

```text
configs/remote/
  innovation2_present_r8_high_round_integral_bridge_262144_seed0_gpu0_20260716.json
configs/remote/generated/
  launch_i2_present_r8_high_round_integral_bridge_262144_seed0_gpu0_20260716.cmd
  run_i2_present_r8_high_round_integral_bridge_262144_seed0_gpu0_20260716.cmd
  monitor_i2_present_r8_high_round_integral_bridge_262144_seed0_gpu0_20260716.sh
```

远程训练命令由 run script 冻结为：

```text
F:\Anaconda\envs\DWT\torch310\python.exe
  scripts\run-innovation2-high-round-integral
  --rounds 8
  --train-rows 262144
  --validation-rows 32768
  --test-rows 65536
  --multiset-count 2
  --epochs 5
  --batch-size 128
  --base-channels 16
  --head-bits 256
  --block-count 2
  --dropout 0.1
  --learning-rate 0.001
  --weight-decay 0.00001
  --seed 0
  --device cuda
  --gate-mode bridge
```

Task Scheduler 的 `/TR` 使用 `cmd.exe /c`。run script 在训练前核对 clean
source 和 pinned commit，并写入 `started.marker`；成功推送验证结果分支后写入
`result_branch_pushed.marker` 和 `done.marker`；任何 source、CUDA、运行、行数、
产物或 bridge-plan 失败都写入 `failed.marker`。主线程只做一次启动确认，之后
由本地 tmux watcher 每 120 秒同步轻量日志/结果，完成后自动校验 `SHA256SUMS`、
四行产物、三份完整 cache metadata，刷新 `outputs/00_RECENT_RESULTS.md/json`。

缓存 feature 原始大小约 `1.375 GiB`，另有标签、日志和小型结果文件。依据本地
r8 同实现生成/训练速度，保守预计远程全流程 `1--3 小时`；实际进度以 watcher
同步的 `progress.jsonl` 为准，不以 GPU 显存下降推断完成。

### 11.5 首次启动失败与修复

首次启动使用提交 `81c8cb12a804742a56d77188318f9458ff21e90a`。远程证据确认：

```text
source revision = match
remote readiness = pass
torch = 2.5.1+cu118
CUDA = 11.8 / available true / visible A6000 count 1
started.marker = present
dataset progress = not started
failure = ModuleNotFoundError: No module named 'blockcipher_nd'
```

原因是新 CLI wrapper 在远程直接使用环境 Python 时没有把仓库 `src/` 放入
`sys.path`；本地 `uv run` 的已安装项目环境掩盖了这个入口差异。修复后 wrapper
与仓库其他 thin entrypoint 一致，显式插入 `<repo>/src`。run script 还增加了
fail-closed retry 规则：已有 `done.marker` 时拒绝重跑；仅在未完成时清除旧的
`failed/started/failure_reason` 运行标记，缓存和首次失败日志不删除。重新启动
必须来自包含该修复的已推送新提交。

修复提交的远程 `--help` 预检随后发现 `torch310` 没有安装可选的
`matplotlib`。旧 CLI 在模块顶层导入绘图库，导致训练也被无关的绘图依赖
阻塞。处理原则是不改变远程环境和训练协议：绘图改为训练后的惰性导入；缺少
可选依赖时写入显式 `plot_deferred.marker`、有效占位 SVG 和简要 CSV，仍要求
训练、四行结果、dataset/cache/gate/validation 全部通过。verified result branch
取回并校验哈希后，本地 watcher 使用仓库标准 `scripts/plot-results` 覆盖生成
完整中文 `curves.svg/history.csv`，再执行本地验证和最近结果索引刷新。

### 11.6 双 multiset anchor layout 修复（结果揭盲前冻结）

对 Eq. 6 的扁平顺序做索引张量审计时发现：提交 `4b3a2c3` 的 paper-family
anchor 把两组 multiset 的 4096 bit 直接 reshape 为 `[16,32,8]`。在 row-major
顺序下，第二组 multiset 的首元素落在错误位置；例如预期右半区首元素索引应为
`2048`，旧实现实际为 `128`。正确的 `spatial_axis_1` 假设必须先把每组独立
reshape 为 `[16,16,8]`，再拼接成 `[16,32,8]`。

影响边界在远程结果产生前冻结如下：

```text
affected       = paper-family anchor tensor layout only
unaffected     = raw cached data, labels, keys, structured candidate,
                 shuffled candidate, flat linear, fixed parity controls
active run     = 继续作为 candidate data-scarcity bridge，不丢弃已生成 cache
old remote gate= 仅作原始运行记录，不能由 anchor-only AUC 触发 advance
local gate     = candidate test AUC 自身必须 >= 0.53，并通过两项 +0.01 control margin
follow-up      = repaired anchor 必须复用同一数据协议重新校准，才可比较论文结构
```

这是结果揭盲前的 correctness 修复，不是看到 AUC 后调整门槛。当前远程完成后
必须使用修复提交重新本地裁决；若旧 gate 因 anchor 单独达到 `0.53` 而显示
advance，本项目仍按 `hold` 处理。不得把旧 anchor 行用于 Wu/Guo 架构比较。

修复提交 `28751e8` 后，使用原 cache、原 seed 和完全相同训练预算重跑了
r5/r7/r8 四行矩阵。唯一变化是 anchor tensor join：

| 轮数 | 旧 anchor AUC | 修复 anchor AUC | 修复差值 | candidate AUC（前/后） |
|---:|---:|---:|---:|---:|
| 5 | `1.000000000` | `0.999995708` | `-0.000004292` | `1.000000000 / 1.000000000` |
| 7 | `0.530674458` | `0.523216486` | `-0.007457972` | `0.696897984 / 0.696897984` |
| 8 | `0.511205196` | `0.506536245` | `-0.004668951` | `0.511633039 / 0.511633039` |

linear 和 shuffled-control AUC 也逐位相同，证明修复只影响 anchor。r5 仍完成
协议校准，但 parity baseline 同样为 `1.0`；修复后的 r7/r8 anchor 都低于
`0.53`。因此旧布局确有轻微乐观偏差，不能用于论文结构比较；当前远程结果
只按 candidate 数据规模桥接解释是必要且充分的隔离措施。

```text
recent index 001 = i2_present_high_round_integral_r8_local_8192_seed0_anchorlayoutfix
recent index 002 = i2_present_high_round_integral_r7_local_8192_seed0_anchorlayoutfix
recent index 003 = i2_present_high_round_integral_r5_local_8192_seed0_anchorlayoutfix
```

远程回收后不手工改写原始 `gate.json`。使用以下本地工具从四行结果、
`dataset_summary.json`、`fixed_baselines.json` 和归档的 remote config 重新执行
当前 candidate-only gate：

```text
scripts/readjudicate-innovation2-high-round-integral
  --artifacts outputs/remote_results/<run_id>
  --remote-config outputs/remote_results/<run_id>/remote_config.json
  --invalidate-anchor-layout
  --expected-source-commit 4b3a2c33cc323b5586533f0fffb78edbe70e0adf
  --output outputs/remote_results/<run_id>/gate.local.json
```

`gate.local.json` 会记录 policy version 和被排除的 anchor 角色；原始远程
`gate.json` 保留作运行历史。最近结果索引优先读取 `gate.local.json`，因此不会
展示旧 anchor-only gate 的错误 advance，但仍可分别检查两份门控文件。
如果归档的 `git_revision.txt` 不等于上述冻结提交，本地重裁决直接返回
`innovation2_high_round_integral_readjudication_source_mismatch`，不解释指标。

### 11.7 seed0 bridge 完成与正式裁决（2026-07-16）

```text
run_id = i2_present_r8_high_round_integral_bridge_262144_seed0_gpu0_20260716
training source = 4b3a2c33cc323b5586533f0fffb78edbe70e0adf
recovery source = c9607661216bd5d23d00e9701b45d0428c34491a
retrieval = verified result branch + SHA256SUMS pass
recent result index = 001（完成时）
```

训练、四行结果、固定基线、三份 cache 和远程原始 gate 均已完成。首次归档在
生成 `SHA256SUMS` 时失败，原因是启用 delayed expansion 的 batch 内仍包含
`p.name!='SHA256SUMS'`；CMD 吞掉了 `!`。恢复脚本只重新验证、重裁决、哈希、
归档和推送现有结果，没有重新生成数据或训练模型。恢复后的全部归档文件通过
normalized SHA-256 校验，本地重新生成了中文 `curves.svg/history.csv`。

冻结四行 test 结果：

| 角色 | test accuracy | test AUC | 解释 |
|---|---:|---:|---|
| paper-family anchor | `0.526321411` | `0.537351800` | 旧双 multiset layout，仅保留历史，不参与结构比较 |
| structured candidate | `0.527618408` | `0.545408788` | 正式 candidate 证据 |
| same-input linear | `0.508087158` | `0.520985522` | 同输入可训练线性基线 |
| same-init shuffled candidate | `0.500000000` | `0.501000870` | 标签控制；fit-validation AUC `0.507042095` |

固定控制及正式 margin：

```text
untrained same-init candidate AUC            = 0.500800185
oriented architecture prior AUC              = 0.501000870
strongest oriented parity AUC                = 0.504599789
candidate - linear                           = +0.024423266
candidate - oriented architecture prior      = +0.044407918
candidate - strongest oriented parity        = +0.040808999
candidate - local 8192-total candidate        = +0.033775749
```

`gate.local.json` 的四项正式检查全部通过：

```text
shuffled-fit validation AUC 距 0.5 <= 0.03     pass
candidate test AUC >= 0.53                     pass
candidate - architecture prior >= 0.01         pass
candidate - strongest oriented parity >= 0.01  pass
decision = innovation2_high_round_integral_bridge_advance
```

因此 seed0 证据支持：在冻结 Wu/Guo-family multiset 协议下，PRESENT-80 r8
存在不能由 flat linear、随机初始化、shuffled label 或固定 parity 解释的神经信号，
且本地 `8192 total` 的近随机结果确有明显数据稀缺成分。它仍不能支持以下声明：

- 不是论文 `2^21 total train / 50 epochs / 10 repeats` 规模；
- 不是 exact reproduction；论文仍未报告 `Nf`、dropout rate、可执行 LR 范围
  和精确 block count；
- 不是多 seed 正式确认；
- 不能用旧 layout anchor 声称 candidate 超过 Wu/Guo 网络；
- 不是 r9、GIFT 或确定性积分结果，也不是突破声明。

权威本地产物：

```text
outputs/remote_results/
  i2_present_r8_high_round_integral_bridge_262144_seed0_gpu0_20260716/
    results.jsonl
    gate.json
    gate.local.json
    validation.local.json
    cache_metadata.json
    curves.svg
    history.csv
    SHA256SUMS
```

## 12. 下一步：独立 seed1 同规模确认

### 12.1 研究问题与单变量

seed0 的 `0.545409` candidate AUC 是否能在完全独立的 plaintext、master-key、
model initialization 和 train/validation/test 随机流上复现？唯一研究变量是
`seed: 0 -> 1`。规模、数据定义、四行模型、epochs、batch、优化器和 gate 不变。
由于 seed1 从包含 `28751e8` 修复的 source 启动，其 anchor 使用正确的逐 multiset
reshape；这同时补充同规模可用 paper-family 对照，但不改变 candidate 假设。

```text
run_id = i2_present_r8_high_round_integral_bridge_262144_seed1_gpu0_20260716
train total / per class = 262144 / 131072
validation total / per class = 32768 / 16384
test total / per class = 65536 / 32768
multisets / texts = 2 / 16 each
input bits = 4096
epochs / batch = 5 / 128
base / head / blocks / dropout = 16 / 256 / 2 / 0.1
seed = 1
device = remote A6000 GPU0
```

同预算矩阵仍为四行：修复后 paper-family anchor、structured candidate、flat
linear、same-init shuffled-label candidate；固定控制仍为 InvP/InvS/total parity 和
same-init untrained candidate。不得增加第五个网络或同时改变学习率。

### 12.2 seed1 gate 与后续决策

```text
source/cache/artifact/negative/split checks all pass
shuffled-fit validation AUC 距 0.5 <= 0.03
candidate test AUC >= 0.53
candidate - oriented architecture prior >= 0.01
candidate - strongest oriented parity >= 0.01
```

- seed1 全过：生成 seed0/seed1 joint gate；创新2可报告“两颗 seed 均达到
  PRESENT-80 r8 有用神经信号”。随后才准备 `2^21 total train / 50 epochs`
  paper-reference，并将论文未公开参数冻结为显式 approximation；
- seed1 candidate `0.52--0.53` 或 margin 不过：判为 seed-sensitive bridge，先做
  同协议训练动态/初始化审计，不直接放大到 `2^21`；
- control/source/cache 失败：判 invalid，修控制而不解释 AUC；
- 无论哪种结果，当前都不启动 r9、GIFT、AES，也不把 seed0 单点与论文
  `0.5732 accuracy` 做突破比较。

可执行包：

```text
configs/experiment/innovation2/
  innovation2_present_r8_high_round_integral_bridge_262144_seed1.json
configs/remote/
  innovation2_present_r8_high_round_integral_bridge_262144_seed1_gpu0_20260716.json
configs/remote/generated/
  launch_i2_present_r8_high_round_integral_bridge_262144_seed1_gpu0_20260716.cmd
  run_i2_present_r8_high_round_integral_bridge_262144_seed1_gpu0_20260716.cmd
  monitor_i2_present_r8_high_round_integral_bridge_262144_seed1_gpu0_20260716.sh
```

### 12.3 双 seed 联合裁决冻结（seed1 揭盲前）

联合门槛在 seed1 结果回收前固定，避免根据第二颗 seed 的 AUC 修改标准。联合
过程不重新训练，只读取两份 verified artifact root 中的 `gate.local.json` 和
`results.jsonl`：

```text
joint run_id = i2_present_r8_high_round_integral_bridge_262144_joint_seed0_seed1_20260716
exact seeds = {0, 1}
same protocol except seed = required
both source revisions match expected = required
seed0 anchor_layout_invalidated = true
seed1 anchor_layout_invalidated = false
```

每颗 seed 独立重复同一组冻结门槛：

```text
candidate test AUC >= 0.53
candidate - oriented architecture prior >= 0.01
candidate - strongest oriented fixed parity >= 0.01
abs(shuffled-fit validation AUC - 0.5) <= 0.03
source gate status/decision = pass/bridge_advance
```

联合裁决只有三种：

- `confirmed`：两颗 seed 的 source、协议、控制和信号全部有效；下一步仅准备
  `2^21 total train / 50 epochs` paper-reference approximation；
- `not_confirmed`：source 有效但至少一颗 seed 的信号门未过；停止机械放大，
  审计 seed sensitivity 与 checkpoint dynamics；
- `invalid`：seed pair、冻结协议、source revision 或 anchor 失效规则不匹配；
  先修证据链，不解释 AUC。

执行命令及固定产物：

```text
scripts/gate-innovation2-high-round-integral-joint
  --run-id i2_present_r8_high_round_integral_bridge_262144_joint_seed0_seed1_20260716
  --source-artifacts outputs/remote_results/<seed0-run> outputs/remote_results/<seed1-run>
  --output-root outputs/local_diagnostic/<joint-run-id>

outputs:
  results.jsonl
  seed_metrics.csv
  curves.svg
  gate.json
  progress.jsonl
```

无论联合结果如何，当前阶段都不并行启动 r9、GIFT 或 AES。

### 12.4 seed1 verified 结果与双 seed 正式裁决（2026-07-16）

seed1 已从 verified remote archive 回收，source、四行结果、三份磁盘 cache、
原始归档 21 个 SHA256、严格负类、split 和 local re-adjudication 均通过：

```text
source commit = 1290275fdeef0e43b1bc14a55e0e03af0d3cc45b
result rows = 4 / 4
train cache = 262144 / 262144
validation cache = 32768 / 32768
test cache = 65536 / 65536
anchor layout invalidated = false
```

正式 test：

| seed1 角色 | accuracy | AUC | 说明 |
|---|---:|---:|---|
| repaired paper-family anchor | `0.530212402` | `0.541776528` | 正确逐 multiset reshape |
| structured candidate | `0.533843994` | `0.546991938` | 本项目候选 |
| same-input linear | `0.516311646` | `0.519512238` | 同输入线性基线 |
| same-init shuffled candidate | `0.500000000` | `0.501086449` | fit-validation AUC `0.500786804` |

seed1 candidate 差值：

```text
candidate - repaired anchor AUC        = +0.005215411
candidate - linear AUC                 = +0.027479701
candidate - architecture prior AUC     = +0.043738822
candidate - strongest parity AUC       = +0.045577290
```

因此 seed1 单独通过四项 bridge signal gate，并在这一颗 seed 上达到冻结的
`+0.005` 同预算 paper-family anchor 差值。seed0 anchor 因历史双 multiset
reshape 错误继续排除，所以不能把 anchor 优势夸大成双 seed 架构结论。

双 seed 联合结果：

```text
run_id = i2_present_r8_high_round_integral_bridge_262144_joint_seed0_seed1_20260716
status = pass
decision = innovation2_high_round_integral_two_seed_bridge_confirmed
candidate AUC mean / min / max = 0.546200363 / 0.545408788 / 0.546991938
candidate-linear AUC delta mean = +0.025951483
candidate-prior AUC delta mean = +0.044073370
candidate-parity AUC delta mean = +0.043193145
shuffled-fit validation AUC mean = 0.503914449
```

联合 validity 和 signal checks 全过。允许的结论是：在每颗 seed `262144 total`
训练行（约 `131072/class`）的 bridge 上，PRESENT-80 r8 候选神经信号已由两颗
独立 seed 确认，并超过 linear、架构先验和固定 parity 控制。这仍不是
`2^21 total / 50 epochs / 10 repeats` 论文规模复现，也不是突破声明。

权威产物：

```text
outputs/remote_results/
  i2_present_r8_high_round_integral_bridge_262144_seed1_gpu0_20260716/
outputs/local_diagnostic/
  i2_present_r8_high_round_integral_bridge_262144_joint_seed0_seed1_20260716/
outputs/00_RECENT_RESULTS.md entry = 001 joint / 002 seed1
```

证据支持的下一步是执行第 13 节 `2^21 total / 50 epochs` paper-reference
approximation seed0；保持 r9、GIFT、AES 停止。若该规模只确认 round reach，
只报告达到 r8；只有 candidate 同时超过 anchor 与强控制才讨论架构收益。

## 13. 双 seed 通过后的论文参考规模近似契约

本节只冻结可执行契约，**不授权在第 12.3 节联合门控确认前启动远程训练**。
独立 `paper_reference` gate 已与 bridge gate 分离，防止把不同规模、epoch 或
网络宽度的运行错误解释成 bridge 或 exact reproduction。

### 13.1 同预算问题与矩阵

研究问题：在论文公开的总数据量、训练轮数、batch 和 `FC2048 x3` 宽度下，
PRESENT-80 r8 是否仍有高于随机和强控制的神经信号；同输入 structured
candidate 是否优于 Wu/Guo paper-family anchor？唯一模型变量仍是张量组织与
归纳偏置，数据、split、训练预算和 checkpoint 选择保持一致。

```text
cipher / rounds             = PRESENT-80 / 8
train total / approx class  = 2^21 / 2^20
validation total / class    = 2^17 / 2^16
test total / class          = 2^17 / 2^16
multisets / texts           = 2 / 16 each
input bits                  = 4096
epochs / batch              = 50 / 2000
loss / optimizer / L2       = MSE / Adam / 1e-5
checkpoint                  = best validation AUC, restored before test
seed                         = 0 first
device                       = remote A6000
gate mode                    = paper_reference
```

四行矩阵保持精简且同预算：paper-family anchor、structured candidate、flat
linear、same-init shuffled-label candidate。固定 parity 与 same-init untrained
candidate 不训练，但必须使用完全相同的 test split。第一颗 paper-reference
seed 通过后才准备完全相同的独立 seed；不执行论文的 10 次独立训练取最好值，
因此只能支持项目级 paper-reference-scale approximation。

### 13.2 显式近似参数

论文没有公开以下数值，本项目冻结一组可审计假设，不允许在结果揭盲后调参：

```text
Nf / base channels           = 16（假设）
MBConv block count           = 1（按图 7 画出的单 block）
FC width / count             = 2048 / 3（原文明确）
MBConv and dense dropout     = 0.1 / 0.1（共享假设）
Adam initial learning rate   = 1e-3（假设）
learning-rate scheduler      = none（Algorithm 1 缺失 min/max 与归一化项）
label sampling               = deterministic alternating 50/50
key sampling                 = deterministic unique 80-bit key per sample
two-multiset tensor join     = per-multiset [16,16,8], then spatial-axis concat
independent repetitions      = 1 first，非论文 10-repeat best selection
per-epoch full train eval     = disabled；最终 checkpoint 仅完整评估一次 train
CUDA memory preflight        = batch 2000 forward/backward/Adam，先于 cache 生成
```

这些假设、原文缺失项和 `exact_reproduction=false` 必须写入 gate、结果行、远程
配置和最终报告。后续若找到作者代码，只能另开 protocol audit，不能静默改写
已冻结运行。

### 13.3 Advance / stop gate

计划对齐必须逐项满足第 13.1--13.2 节参数。控制有效还要求 cache/split/fixture/
strict negative 全过，且 shuffled-fit validation AUC 距 `0.5 <= 0.03`。

PRESENT-80 r8 round reach 对 anchor 与 candidate 分别检查：

```text
test accuracy >= 0.53
test AUC >= 0.53
test accuracy Wilson 95% lower confidence bound > 0.50
```

任一神经模型同时满足三项，只能确认 paper-reference-scale r8 round reach。
candidate 架构优势还必须同时满足：

```text
candidate - oriented architecture prior AUC >= 0.01
candidate - strongest oriented fixed parity AUC >= 0.01
max(candidate-anchor accuracy delta, AUC delta) >= 0.005
```

裁决：

- round reach 与 candidate 优势均过：确认候选在该近似协议的单 seed 优势，
  下一步只运行同协议独立 seed；
- 仅 round reach 过：只报告达到 r8，不声明超过 paper-family anchor；随后运行
  同协议独立 seed；
- round reach 未过：停止机械放大，审计 `Nf/dropout/block/LR/tensor join`；
- source/cache/control/plan 失效：修证据链，不解释指标。

paper-reference seed 集合冻结为 `{0, 1}`：seed0 是当前首颗运行，seed1 只能在
seed0 完整回收且裁决为 `candidate_advantage` 或 `round_reach_only` 后启用。
seed2 及更多重复不属于当前计划，必须由新的实验问题和计划另行授权。

联合 bridge 已确认，远程包可以生成；但 launcher 只能在本节实现与配置完成
测试、范围提交并成功推送后，从该精确 commit 启动。启动前先完成 batch 2000
CUDA 显存预检，失败时不得生成 9 GiB cache。不启动 r9、GIFT、AES 或其他
密码扩展。

### 13.4 远程启动与 watcher 交接（2026-07-16）

论文参考规模包在 `f64aab9` 完成并推送。首次计划任务在正式训练、显存预检和
cache 生成前 fail-closed：SYSTEM 账户执行旧 `for /f (git status --porcelain)`
clean gate 时返回 `dirty_source`，但交互账户对同一 run-owned clone 的
`git status --porcelain` 为空。该次没有生成 cache 或实验结果，不能计为一次
训练运行。

修复 `694fcc2` 不放宽 clean gate，而是：

```text
GIT_CONFIG_KEY_0 / VALUE_0 = 本次 run-owned source 的 safe.directory
git status stdout/stderr = 持久化到独立日志
git command failure = source_status_failed / exit 8
porcelain 文件非空 = dirty_source / exit 2
porcelain 文件为 0 byte = 继续 exact source revision gate
```

修复后同一 run id 从精确推送 commit 重新启动并通过一次 bounded confirmation：

```text
run_id = i2_present_r8_high_round_integral_paper_reference_2pow21_seed0_gpu0_20260716
remote source = 694fcc2b36aa3af0df3a9dc9fc4074d1623596e6
remote root = G:\lxy\blockcipher-structure-adaptive-nd-runs\<run_id>
started.marker = present
readiness output = present
memory_preflight.json = present
progress.jsonl = present
state = running
```

后续不由主线程 SSH 轮询。自动回收交给：

```text
tmux = i2-r8-paper-reference-2pow21-seed0
monitor root = outputs/remote_results_incomplete/
  i2_present_r8_high_round_integral_paper_reference_2pow21_seed0_gpu0_20260716_monitor/
verified destination = outputs/remote_results/
  i2_present_r8_high_round_integral_paper_reference_2pow21_seed0_gpu0_20260716/
watcher workflow revision = dc2d16adbe8e89ce68b5bbc5cd1bb8250d00700e
```

watcher 只在 result branch marker、原始 SHA、四行结果、三份 cache、source、
paper-reference gate 和 local validation 全部通过后回收并刷新最近结果索引。
重绘 SVG 后必须留下 `visual_qa_pending.marker`；后续 agent 使用
`visual-qa-redraw` 完成像素检查、必要重绘并写入 pass，才能把结果处理称为完整。
运行中任务没有完成 gate，因此当前不进入 `00_RECENT_RESULTS`。

`2026-07-16 23:24 +08:00` 的有界本地健康检查发现 tmux socket、monitor 和
scp 进程均已消失，但本地同步目录没有 remote failed/done/result-branch marker；
最后一份进度仍是连续的 train cache chunk。没有 SSH 轮询或重启远程训练，只
按规则重启本地 `i2-r8-paper-reference-2pow21-seed0` watcher。新会话在
`23:25:14 sync`、`23:25:15 running`，随后回收到 `rows_done=718336 / 2097152`
的 train cache 进度。该事件是本地监控恢复，不是新的训练 run，也不改变远程
source `694fcc2` 或实验协议。

`12a4fe1` 后的后续启动安全审计还移除了 launcher 对
`git config --global --add safe.directory` 的写入，改用仅对本次 launcher
进程及其子进程生效的 `GIT_CONFIG_*`。这不改变已经从 `694fcc2` 启动的
seed0 训练；它只约束未来的重启或条件独立 seed，避免生成的实验流程修改
`G:\lxy` 之外的账户级 Git 配置。

### 13.5 条件独立 seed1 计划（未启动）

独立确认计划已冻结在：

```text
configs/experiment/innovation2/
  innovation2_present_r8_high_round_integral_paper_reference_2pow21_seed1.json
```

该计划当前是 `blocked_until_seed0_retrieved_gate_and_visual_qa_pass`，没有 remote
config、launcher 或运行目录。只有 seed0 从 verified result branch 回收、本地
readjudication 为 `candidate_advantage` 或 `round_reach_only`，并且最终曲线通过
`visual-qa-redraw` 后，才生成远程包。唯一实验变量是 `seed: 0 -> 1`；其余
训练/验证/测试总量、四行矩阵、网络参数、50 epochs、batch 2000、严格负样本、
key sampling 与显式论文近似假设必须逐字段相同。

联合裁决预先冻结为：

- 两颗都为 `candidate_advantage`：确认双 seed 候选优势；
- 两颗都达到 round reach、但不都满足候选优势：只确认双 seed r8 round reach；
- seed1 未确认：hold，审计 seed 方差与冻结参数假设，不机械追加 seed；
- 任一 source/cache/control/plan/visual 失效：拒绝联合解释，先修证据链。

上述规则已在 seed0 指标揭盲前实现为
`adjudicate_joint_paper_reference`，联合 run id 冻结为：

```text
i2_present_r8_high_round_integral_paper_reference_2pow21_joint_seed0_seed1
```

纯 gate 要求 exact seed0/seed1、逐字段同协议、四角色完整、source revision
匹配、本地重裁决有效、计划与 readiness 全过、shuffled-fit 控制有效，并且两颗
最终图均有 `visual_qa_passed.marker`。联合 CLI 和真实图只在 seed1 完成后加载
artifact；不得修改这里已冻结的 decision 分支和阈值。

联合 artifact CLI 使用现有入口：

```text
scripts/gate-innovation2-high-round-integral-joint --mode paper_reference
```

该模式会从两个 verified source 目录读取 `gate.local.json`、`results.jsonl` 和
`visual_qa_passed.marker`，执行预注册纯 gate，并生成 `results.jsonl`、
`gate.json`、`seed_metrics.csv`、`progress.jsonl` 与专用中文 `curves.svg`。
每次绘图后 CLI 会删除旧 pass 并写 `visual_qa_pending.marker`。真实联合图生成后
仍必须单独调用 `visual-qa-redraw`，不能用两颗 source 已通过视觉 QA 代替联合图
自身的像素验收；验收完成后才删除 pending 并写 `visual_qa_passed.marker`。

seed1 包生成前置条件使用只读入口：

```text
scripts/check-innovation2-paper-reference-seed1-precondition \
  --plan configs/experiment/innovation2/innovation2_present_r8_high_round_integral_paper_reference_2pow21_seed1.json \
  --source-artifacts outputs/remote_results/i2_present_r8_high_round_integral_paper_reference_2pow21_seed0_gpu0_20260716 \
  --output <precondition-report.json>
```

该 gate 检查 verified retrieval、本地 seed0 gate、允许的单 seed decision、
source revision、四行 seed0 结果、三份 complete cache、数据摘要、本地 artifact
validation、batch-2000 显存预检、visual pass 且无 pending。通过时只返回
`should_generate_remote_package=true` 和 `should_launch_remote=false`；随后仍须生成
精确 seed1 config/launcher/monitor，运行 readiness 与测试，范围提交并推送，再从
推送 commit 启动。任一证据缺失或无效时保持 seed1 blocked。

### 13.6 seed0 verified 结果与单 seed 裁决（2026-07-17）

seed0 已从 verified result branch 回收。远程 source、四行结果、三份磁盘 cache、
严格负类、逐样本唯一 PRESENT-80 key、split、CUDA 显存预检、原始 archive 哈希、
本地 artifact validation 和独立 readjudication 均通过：

```text
run_id = i2_present_r8_high_round_integral_paper_reference_2pow21_seed0_gpu0_20260716
source commit = 694fcc2b36aa3af0df3a9dc9fc4074d1623596e6
train cache = 2,097,152 / 2,097,152
validation cache = 131,072 / 131,072
test cache = 131,072 / 131,072
result rows = 4 / 4
validation.local = pass
gate.local = pass
```

正式 test 结果：

| seed0 角色 | accuracy | AUC | 最佳 validation AUC |
|---|---:|---:|---:|
| Wu/Guo paper-family anchor | `0.539436340` | `0.553644314` | `0.553490084` @ epoch 5 |
| structured candidate | `0.529075623` | `0.548082726` | `0.549722653` @ epoch 19 |
| same-input linear | `0.524902344` | `0.535075686` | `0.536630982` @ epoch 22 |
| same-init shuffled control | `0.499877930` | `0.498149598` | `0.501544337` @ epoch 5 |

关键差值与冻结门：

```text
candidate - anchor accuracy = -0.010360718
candidate - anchor AUC      = -0.005561587
candidate - linear AUC      = +0.013007040
candidate - architecture prior AUC = +0.043532458
candidate - strongest fixed parity AUC = +0.046145304

anchor accuracy / AUC / Wilson lower bound =
  0.539436340 / 0.553644314 / 0.536736813
candidate accuracy / AUC / Wilson lower bound =
  0.529075623 / 0.548082726 / 0.526372547
```

anchor 同时通过 `accuracy >= 0.53`、`AUC >= 0.53` 和 Wilson 下界高于随机，
因此单 seed 的 PRESENT-80 r8 round reach 成立。candidate 的 AUC 有信号并超过
linear、架构先验和 fixed parity，但 accuracy 比冻结的 `0.53` 低约
`0.000924377`，且 accuracy/AUC 都没有比 anchor 高 `0.005`。正式裁决是：

```text
decision = innovation2_high_round_integral_paper_reference_round_reach_only
candidate_architecture_advantage = not confirmed
```

这支持“在项目的论文参考规模近似协议下，PRESENT-80 8 轮可区分”，不支持
structured candidate 优于 Wu/Guo paper-family anchor。anchor accuracy 比论文
headline `0.5732` 低约 `0.033764`；论文使用 10 次独立训练取最好，本项目 seed0
只有一次训练，且 `Nf/dropout/block/LR/tensor join` 含显式假设，因此不是 exact
reproduction、SOTA 或突破证据。

真实 `curves.svg` 已使用 `visual-qa-redraw` 在 `1800 x 1281` 像素全幅和顶部/
底部细节视图检查。标题、中文 glyph、轴、动态尺度、图例、曲线、表格和边界均无
重叠或裁切；`visual_qa_pending.marker` 已替换为 `visual_qa_passed.marker`。最近结果
索引已刷新，verified seed0 是 `outputs/00_RECENT_RESULTS.md` 的 `001`。

权威产物：

```text
outputs/remote_results/
  i2_present_r8_high_round_integral_paper_reference_2pow21_seed0_gpu0_20260716/
```

### 13.7 seed1 包准备与下一步

seed0 结果、verified retrieval 和 visual QA 全部通过后，机器前置门返回：

```text
decision = innovation2_paper_reference_seed1_precondition_ready
should_generate_remote_package = true
should_launch_remote = false
```

因此已生成完全同预算的 seed1 包：

```text
run_id = i2_present_r8_high_round_integral_paper_reference_2pow21_seed1_gpu0_20260717
config = configs/remote/innovation2_present_r8_high_round_integral_paper_reference_2pow21_seed1_gpu0_20260717.json
run = configs/remote/generated/run_i2_present_r8_high_round_integral_paper_reference_2pow21_seed1_gpu0_20260717.cmd
launcher = configs/remote/generated/launch_i2_present_r8_high_round_integral_paper_reference_2pow21_seed1_gpu0_20260717.cmd
monitor = configs/remote/generated/monitor_i2_present_r8_high_round_integral_paper_reference_2pow21_seed1_gpu0_20260717.sh
```

remote readiness 已通过。回归测试逐字段确认训练/验证/test 总量、四行矩阵、网络、
50 epochs、batch 2000、严格负类、key sampling、缓存、显存预检和全部论文近似
假设不变，唯一研究变量是 `seed: 0 -> 1`。范围提交并推送后，从该精确 commit
启动 seed1 并交给本地 tmux watcher。seed1 verified 回收且真实图通过 QA 后，执行
预注册联合 gate；不调参、不加 seed2，不启动 r9、GIFT 或 AES。

首次从 `3d3f417` 调用 launcher 后，计划任务在 cache、显存预检和训练前按 clean
gate fail-closed，exit code 为 `2`。SYSTEM 首次 `git status --porcelain` 将一个与
本实验无关的历史 `.pt` checkpoint 报为 modified；但该文件的 index、HEAD、
worktree raw hash 和 path-filtered hash 均为
`6a34b35e1628ae32886d525d7c2e36cd3c7f5410`，交互账户再次执行 porcelain 为空。
因此这是新 clone 首次状态扫描的瞬时 stat-cache 假阳性，不是允许忽略的真实修改。

修复 `a2883cc` 保留 fail-closed：首次 porcelain 非空时执行一次
`git update-index -q --refresh` 并重新生成 porcelain；只有第二次为空才继续，第二次
仍非空仍进入 `dirty_source / exit 2`。seed0/seed1 两份模板应用相同逻辑，回归测试
继续证明实验参数唯一变化是 seed。首次失败没有生成 dataset cache、memory preflight
或训练结果，不计为一次实验运行；修复推送后允许使用同一 run id 从新的精确 commit
重启。
