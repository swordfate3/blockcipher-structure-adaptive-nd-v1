# 创新2 H0：PRESENT 高轮积分神经主流锚点计划

**日期：** 2026-07-16
**状态：** readiness + local r5/r7/r8 diagnostic completed / remote data-scarcity bridge planned
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
candidate or anchor test AUC >= 0.53
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
