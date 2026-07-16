# 创新2 H0：PRESENT 高轮积分神经主流锚点计划

**日期：** 2026-07-16
**状态：** planned / implementation-required / no remote launch
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

### 3.3 模型矩阵

首个有意义矩阵只保留四行：

| 角色 | 输入 | 目的 |
|---|---|---|
| paper anchor | Wu/Guo `InvP+InvS`, DenseNet/MBConv | 直接轮数锚点 |
| project candidate | 同一输入，项目 SPN-aware 候选 | 测模型增益 |
| same-input linear | 同一展平输入 | 排除线性捷径 |
| shuffled-label candidate | 同一输入 | 排除训练/类不平衡伪信号 |

另外计算不训练的 parity/低维统计基线。如果它们已经解释主要准确率，神经
结果只能报告为数据结构效应，不得归因于网络。

## 4. 规模与运行路径

### 4.1 Readiness smoke

```text
train total rows       = 256
validation total rows  = 128
test total rows        = 128
rounds                 = r5 only
seeds                  = 0
device                 = local CPU
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
```

这是轮数斜率和泄漏检查，不是正式失败或论文复现。

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

先做实现审计，不直接训练：确认仓库当前没有把 Innovation 1 的
`plaintext_integral_nibble` pair-set 误当成 Wu/Guo 纯 multiset 协议，并从
原文/公开代码冻结 exact data format fixture。审计完成后再实现独立 task、
测试和 readiness smoke。

## 9. 主要来源

- Wu and Guo, *Improved integral neural distinguisher model for lightweight cipher PRESENT*，本地原文：`papers/innovation_one/grobid_md/improved-integral-neural-distinguisher-model-for-lightweight-cipher-present.md`。
- Wang, Hadipour, and Gerhalter, *On Extending Integral Distinguishers*，本地原文：`papers/innovation_two/text/on_extending_integral_distinguishers.txt`。
- Kimura et al., *Output Prediction Attacks on Block Ciphers Using Deep Learning*，本地原文：`papers/innovation_two/text/2021_kimura_output_prediction_block_ciphers.txt`。
- Singh, *PRESENT Full Round Emulation: Structural Flaws and Predictable Outputs*，本地原文：`papers/innovation_two/text/2025_singh_present_full_round_emulation.txt`。
