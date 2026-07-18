# 创新2 E50：PRESENT四轮证书引导pair-state残差网络readiness计划

日期：2026-07-18

状态：计划冻结 / 待实现

## 1. 研究问题

E45/E48证明可安全计算的1--3轮ANF/degree前缀在E43严格标签上达到约`0.69` AUC；E47/E49
证明让MSPN从输入重新学习这套传播在structure-disjoint split上失败。E50不再逼神经网络
重建证书，而是测试一种神经符号混合结构：

```text
Certificate-Guided Pair-State Residual（CGPR）
```

冻结确定性ANF-prefix base，只让E44的64-bit directed pair-state处理器学习base尚未解释的
残差。E50只判断实现、零残差等价、参数公平、拓扑敏感和短训练是否就绪，不判断正式性能。

## 2. 数据、确定性base与禁用信息

```text
cipher             = PRESENT-80
rounds             = 4
source             = E43 matched checkerboard
train              = 800 rows（400/400）
validation         = 236 rows（118/118）
split              = structure-disjoint，沿用E43--E49
label              = universal balance 0/1
base features      = E45 r1/r2/r3 39维ANF-prefix
base fit           = train-only standardization + ridge lambda 1e-3
expected base AUC  = 0.686082
```

39维特征只允许由active bits、output mask、PRESENT S-box ANF和所选P-layer的1--3轮prefix
计算。禁止第4轮full-cube计数、最终certificate status、witness、key/offset parity、label
派生输入和validation拟合。

## 3. CGPR结构

先复现E45 ridge score并冻结。残差分支使用E44的64-bit pair-state表示与参数预算：

```text
active/mask -> 64 x 64 directed pair state
             -> pair-local/triangle processor
             -> query-conditioned pooled embedding
             -> bounded residual = 0.25 * tanh(MLP(embedding))

final score = frozen ridge score + bounded residual
```

残差head最后一层零初始化，因此初始化时final score必须与ridge逐样本一致。确定性39维特征
不得进入pair-state processor；pair-state embedding也不得改变冻结ridge权重。true与
fair-corrupted P-layer使用完全相同的参数形状、初始化和训练预算。

prefix-only容量控制使用相同宽度/深度/参数量的MLP从39维标准化prefix产生bounded residual，
不读取pair-state。必要时用无信息零输入padding匹配参数；不得增加隐藏层来补参数。

## 4. 两轮本地readiness矩阵

```text
0. E45 ANF-prefix ridge anchor                  只读/重新核验
1. ridge + prefix-only residual                非线性容量控制
2. ridge + true-P pair-state triangle residual CGPR候选
3. ridge + fair-corrupted-P triangle residual  错误transport控制
```

三行训练保持：

```text
epochs       = 2
batch        = 32
seed         = 0
dropout      = 0.10
optimizer    = AdamW(lr=1e-3, weight_decay=1e-4)
checkpoint   = best validation AUC
device       = local CPU
```

readiness的两轮AUC不得用于选择processor、调残差上限或声称超过ridge。

## 5. Readiness门

协议与实现门：

```text
E43/E44/E45/E49 run id、decision、hash与关键metric匹配
重新计算E45 prefix validation AUC             = 0.686082 ± 1e-12
39维标准化只使用train rows                     = pass
第4轮oracle/certificate/witness输入             = absent
zero-residual final score与ridge max error      <= 1e-7
ridge weights在训练前后max delta                = 0
true/corrupted同权重初始score delta             >= 1e-5
prefix-only/true/corrupted残差参数差             <= 1%
logit、loss、gradient finite                    = pass
三行均完成2 epochs                              = pass
每行validation AUC                              in [0.35, 0.80]
```

全部通过：

```text
decision = innovation2_present_cgpr_readiness_passed
next     = 另建E51 30轮seed0正式残差与拓扑归因计划
```

任何source、泄漏、零等价、冻结base或参数门失败只修实现。若短训练metric异常，先审计score
标度与checkpoint，不调数据或门。readiness通过也不开放seed1、r5或远程GPU。

## 6. E51预告与停止边界

正式计划至少比较E45 ridge、prefix-only residual、true-P CGPR和fair-corrupted-P CGPR。
正式候选应同时满足：

```text
true CGPR validation AUC                 >= 0.70
true CGPR - E45 ridge                    >= 0.02
true CGPR - prefix-only residual          >= 0.02
true CGPR - fair-corrupted-P residual     >= 0.03
```

阈值需在E51实现前再次冻结，不得根据E50两轮结果修改。正式门失败则停止E43四轮新网络枚举，
保留E45确定性方法和E44神经锚点，不机械增加容量、epoch、seed或远程规模。

## 7. 产物与声明范围

```text
outputs/local_smoke/i2_present_r4_cgpr_readiness_seed0_20260718/

results.jsonl
history.csv
gate.json
summary.json
metadata.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

声明范围：PRESENT-80四轮、E43严格标签、两轮本地CGPR实现readiness；不是有效预测、高轮
积分区分器、新攻击、远程规模证据或SOTA。
