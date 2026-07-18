# 创新2 E87：RECTANGLE-80四轮严格unit-balance profile标签readiness计划

日期：2026-07-19

状态：已完成 / `pass` / 无神经训练 / 本地标签门

## 1. 研究问题

E73/E79已经确认r3-only Profile Operator在PRESENT和GIFT两种真实P-layer型SPN上分别有效；
E86又表明完全共享参数不能替代两套独立模型。下一步不继续调共享权重，而是回答：

```text
第三种真实P-layer型64-bit SPN RECTANGLE，
能否形成与PRESENT/GIFT同语义、sound、宽且抗位置捷径的严格unit-output平衡谱标签？
```

标签门通过前禁止训练神经网络。

## 2. RECTANGLE版本与规范来源

使用2015最终版`RECTANGLE-80`，不是2014初始版`REC-0`：

```text
block bits    = 64
key bits      = 80
full rounds   = 25
state         = 4 x 16 bit rows
S-box         = [6,5,C,A,1,E,7,9,B,0,3,D,8,F,4,2]
ShiftRow      = row rotations [0,1,12,13]
round order   = AddRoundKey -> SubColumn -> ShiftRow
final step    = AddRoundKey K25
```

主规范是NIST Lightweight Cryptography Workshop托管的设计者论文与演示文稿：

```text
session8-wentao-paper.pdf
session8-huang-wentao.pdf
```

论文列出完整S-box、轮函数、80-bit密钥调度、25个轮常量和bit-slice公式。独立交叉实现使用
`SalaQ/RECTANGLE` commit `2a71673e1479365cd11c9d6bb75f84f6c7a4926a`中的`Source.c`；其
全零80-bit key、全零plaintext最终输出为：

```text
0874E8B1E3542D96
```

该GitHub实现不是作者官方仓库，只作为第二实现；论文规范是权威来源。E87必须同时通过标量实现、
向量化实现、bit-slice/S-box重放和上述全零向量，避免错用REC-0或行序。

## 3. 文献边界

Zahednejad--Lyu 2022把积分神经框架应用于RECTANGLE，报告四活动bit、7轮、约`83.8%`测试准确率；
Xiang et al. 2016和Wang et al. 2020还给出更高轮经典积分结果。这些工作证明RECTANGLE是有意义的
积分研究对象，但任务是密文multiset分类或确定性division-property搜索，不等于本项目的：

```text
input  = cipher topology + active 8-bit cube structure + output unit mask
target = 该指定输出bit是否对所有密钥保持XOR平衡
```

因此不能把文献轮数或准确率直接作为E87 AUC基线，也不能预先宣称新颖性或高轮突破。

## 4. 冻结预算与same-budget锚点

E87对齐GIFT E74标签readiness，只改变密码规范：

```text
rounds                = 4
active dimension      = 8
structures            = 96
witness keys          = 16
offsets per structure = 8
checkerboard attempts = 64
structure seed        = 20260718
```

这不是论文7轮密文分类复现。四轮用于先检查严格标签是否处在正负过渡区；若标签不就绪，不训练网络。

## 5. Sound标签语义

对每个`structure x output bit`：

```text
positive:
  通过RECTANGLE S-box ANF支持传播证明8维full-cube monomial不可能出现；
  因而指定输出bit对所有密钥与所有固定offset的XOR必为0。

negative:
  找到具体80-bit key与固定offset，使256个明文加密后的指定输出bit XOR为1；
  保存并用独立标量实现重放该反例。

unknown:
  既无sound positive certificate，也未找到negative witness；不得强行标0或标1。
```

轮密钥XOR只改变常数项，不改变变量support；positive证明不得读取validation标签或有限密钥投票。

## 6. Structure-disjoint checkerboard

沿用PRESENT/GIFT协议：

```text
validation structures = index % 4 == 0
train structures      = others
```

只从已解析positive/negative中选择checkerboard，使每个选中structure和output bit的正负数严格相等。
报告global、output-bit和active-bit train-only边际在validation上的AUC，禁止位置或结构ID捷径。

## 7. 协议门

1. 最终版RECTANGLE S-box、轮常量、ShiftRow和80-bit密钥调度逐项匹配论文；
2. 25轮全零测试向量等于`0874E8B1E3542D96`；
3. 向量化四轮在多key、多plaintext上逐项等于独立标量实现；
4. S-box ANF重建全部16个输入，ShiftRow是64-bit permutation；
5. 抽样negative witness使用标量实现重放；
6. 所有positive有support-absence certificate，所有negative有具体key/offset；
7. positive与negative证据无冲突，unknown保持`-1`；
8. train/validation structure严格不相交。

## 8. 标签宽度与反捷径门

对齐GIFT E74：

```text
raw positive >= 256
raw negative >= 256
resolved positive prevalence in [0.10, 0.90]
mixed structures >= 32
distinct ternary signatures >= 4

matched train positive/negative >= 150/150
matched validation positive/negative >= 50/50
matched total structures >= 32
matched validation structures >= 8
matched validation output bits >= 16

strongest unary validation AUC <= 0.65
duplicate edges = 0
structure class delta = 0
output-bit class delta = 0
```

## 9. 裁决与下一步

```text
协议失败:
  status = fail
  next   = 修复最终版规范、行序、向量化、support或witness重放

raw正负/mixed标签本身过窄:
  status = hold
  next   = E88只改变rounds 4 -> 5，仍为96结构无训练标签门

raw标签足够但checkerboard容量不足:
  status = hold
  next   = E88只改变structures 96 -> 192，其他协议不变

全部通过:
  status = pass
  next   = E88扩展到192结构后，再开放RECTANGLE r3-only三行readiness
```

不得降低门槛、把unknown当negative、使用有限key多数投票、直接训练网络或启动远程GPU。

## 10. 计划产物与边界

```text
run_id = i2_rectangle80_r4_unit_balance_profile_readiness_20260719
output = outputs/local_audits/i2_rectangle80_r4_unit_balance_profile_readiness_20260719

structures.json
atlas.jsonl
profile_targets.npy
profile_observed.npy
prefix_features.npy
matched_unit_contrast.csv
results.jsonl
gate.json
summary.json
metadata.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

证据范围仅为RECTANGLE-80四轮严格unit-profile标签readiness；不是神经网络增益、论文7轮复现、
高轮积分区分器、攻击、跨密码迁移、远程规模或SOTA。

## 11. E87实际结果与裁决

E87按冻结协议完成：

```text
run_id = i2_rectangle80_r4_unit_balance_profile_readiness_20260719

raw positive = 4941
raw negative = 952
raw unknown  = 251
resolved positive prevalence = 0.8384524012

mixed structures = 96 / 96
distinct ternary signatures = 95
```

structure-disjoint checkerboard得到：

```text
train      = 602 positive / 602 negative / 72 structures
validation = 182 positive / 182 negative / 24 structures
validation output bits = 37

global AUC     = 0.500000
output-bit AUC = 0.500000
active-bit AUC = 0.500000
strongest AUC  = 0.500000
```

最终版零向量、S-box ANF、25个轮常量、P-layer双射、16项向量/标量对拍和24个抽样negative
witness重放全部通过。所有positive均有full-cube support缺失证书，所有negative均有具体key/offset
反例，train与validation结构不相交。

```text
status   = pass
decision = innovation2_rectangle80_unit_profile_ready
```

这证明RECTANGLE-80已成为PRESENT、GIFT之后第三种通过同语义严格标签readiness的真实SPN，
但E87没有训练神经网络。下一步只运行E88：把结构数从96扩到192，保持四轮、8-bit cube、
16 witness keys、8 offsets、structure split、标签语义和全部门槛不变。E88通过后，才开放
RECTANGLE r3-only的`independent / true P-layer / same-family corrupted P-layer`三行两轮
readiness；E88失败则不训练网络，也不启动远程GPU。
