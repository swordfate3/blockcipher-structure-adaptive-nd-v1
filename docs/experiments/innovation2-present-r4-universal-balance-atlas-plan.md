# 创新2 E43：PRESENT r4 全称平衡证书/反例标签 atlas 计划

日期：2026-07-18

状态：完成 / pass / 允许E44本地seed0神经归因

## 1. 研究问题

E42确认64-bit有向pair-state实现可运行，但现有PRESENT/SKINNY经验kernel标签没有一个
同时通过fresh-key稳定性、结构宽度和边际捷径门。E43不继续增加经验key、context或网络
复杂度，而是测试一种语义严格的三态标签提供者：

```text
input  = PRESENT拓扑 + rounds + 8-bit coordinate cube + linear output mask
target = 该mask是否对所有80-bit key和所有inactive-bit offset保持XOR平衡

positive = sound ANF-support over-approximation证明最高cube单项式不可能出现
negative = 至少一个具体(key, offset)的实际加密给出masked XOR=1
unknown  = 当前方法既不能证明positive，也没有找到negative反例
```

`unknown`不得写成负类。正类是充分证书，负类是反例证书；两者都不依赖“有限key零失败
近似证明”。

## 2. 文献与现有证据

Todo的division property通过传播ANF单项式的充分条件识别平衡输出；Hwang等将线性mask的
平衡明确写为密文选定位的XOR和为0，并指出monomial prediction可判断组合输出是否具有
key-independent积分性质。E31同时证明ATM的unknown constant和不完备搜索结果不能直接
变成0/1标签。

E43采用更保守但本地可执行的版本：传播活动变量单项式支撑的超集。若完整8变量单项式
不在所选输出位的任一超集中，则真实ANF中也不可能存在，因此对任意key和offset的cube
和为0。反方向只接受实际XOR=1的反例，不把“可能存在单项式轨迹”当作negative。

本地临时探针已确认PRESENT r4存在混合宽度；r5在同类探针中没有得到positive证书，
因此E43冻结r4作为真实密码训练锚点，不把它包装成高轮区分器。

## 3. 冻结 atlas

```text
cipher                         = PRESENT-80
rounds                         = 4
active dimension               = 8 coordinate bits
structures                     = 96
  coordinate nibble pairs      = 24
  deterministic random cubes   = 72
output masks                   = 300 unique linear masks
  unit / nibble / P / same-nibble / adjacent-nibble pairs
witness keys                   = 16 deterministic 80-bit keys
witness inactive offsets       = 8 per structure
structure split                = index mod 4; 72 train / 24 validation
```

正类证书对所有key和offset成立；witness bank只用于寻找负类，不限制正类scope。输出mask
可以是单bit或多bit线性组合，标签始终是所选密文位XOR后的一个bit。

## 4. 反捷径 matched benchmark

原始atlas必须先报告mask-only、mask-family和active-bit边际。不得把原始表直接交给神经
网络。训练和验证split分别构造2x2 checkerboard：

```text
             mask A   mask B
structure X     1        0
structure Y     0        1
```

每个rectangle同时平衡两个structure和两个mask；每条`(structure, mask)`边最多使用一次。
这保证精确structure、mask、mask family和active-bit一元边际不能单独解释标签。split在
匹配前冻结，train/validation structure不重叠。使用标签做分层匹配属于benchmark构造，
必须记录；它不是神经模型选择或结果后筛选。

## 5. Readiness门

全部通过才允许另建E44本地神经矩阵：

```text
official PRESENT vector和vectorized/scalar fixture通过
PRESENT S-box ANF重构16个输入全部通过
raw structures                         = 96
raw masks                              = 300
raw positive / negative                >= 1000 / >= 1000
raw resolved positive prevalence       in [0.10, 0.90]
同时含positive和negative的structure   >= 32
distinct ternary label signatures      >= 4
sampled negative witness scalar复验    = 100%
matched train positive / negative       >= 250 / >= 250
matched validation positive / negative  >= 100 / >= 100
matched独立structure总数                >= 32
matched validation structures           >= 8
matched strongest unary marginal AUC    <= 0.65
每个matched structure和mask内部正负计数相等
```

## 6. 裁决与下一网络实验

通过时：

```text
decision = innovation2_present_universal_balance_atlas_ready
next     = E44 64-bit pair-state真实PRESENT本地seed0归因

1. train-only unary marginal baseline
2. pair-local true P-layer
3. triangle true P-layer
4. 最强候选的fair-corrupted P-layer control
```

E44固定同一matched dataset、参数预算、split、epochs和checkpoint metric。先要求神经候选
超过AUC `0.60`且`true - fair-corrupted >= 0.03`，才运行seed1。E43本身不训练，不使用
远程GPU。

若原始标签宽度不足，扩大的是coordinate cube结构族或更紧的sound证书，不增加witness
key伪造正类。若matched宽度或捷径门失败，重新设计benchmark，不直接训练。r5没有正类
证书时只记作当前证明器边界，不写成PRESENT r5不存在积分性质。

## 7. 产物

```text
outputs/local_audits/i2_present_r4_universal_balance_atlas_20260718/

atlas.jsonl
matched_contrast.csv
structures.json
masks.json
metadata.json
summary.json
results.jsonl
gate.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

声明范围：真实PRESENT-80 r4全称平衡标签benchmark readiness；不是新高轮积分区分器、
不是神经性能、不是对全部候选的完备分类，也不是SOTA攻击。

## 8. 2026-07-18实际结果

权威run：

```text
i2_present_r4_universal_balance_atlas_20260718
```

冻结的`96 × 300 = 28800`个structure-mask候选全部完成：

```text
positive sound certificates = 3572
negative counterexamples    = 18590
unknown                     = 6638
resolved positive prevalence= 0.161177
mixed structures            = 72 / 96
distinct ternary signatures = 65
```

32个跨atlas抽样的negative均用标量`Present80.encrypt`重新计算完整256点cube，
`32/32`复现masked XOR=1。PRESENT官方向量、S-box ANF 16输入重构、证书字段、反例字段和
train/validation structure互斥检查全部通过。

原始atlas确有严重一元捷径：

```text
mask-only AUC    = 0.967092
active-bit AUC   = 0.666412
mask-family AUC  = 0.620014
```

因此原始表仍禁止训练。checkerboard匹配后的权威benchmark为：

```text
train       = 400 positive + 400 negative, 53 structures, 105 masks
validation  = 118 positive + 118 negative, 18 structures,  67 masks
total       = 1036 rows, 71 structure groups

global / mask / family / active-bit validation AUC = 0.5 / 0.5 / 0.5 / 0.5
duplicate edges                                  = 0
maximum per-structure class delta                = 0
maximum per-mask class delta                     = 0
```

所有readiness、width和shortcut门通过：

```text
status   = pass
decision = innovation2_present_universal_balance_atlas_ready
```

这不是PRESENT r4新积分区分器声明；贡献是得到首个语义严格、正负均有证书且一元边际被
显式平衡的真实PRESENT神经标签benchmark。r5探针没有positive证书只表示当前sound
over-approximation证明能力到达边界，不表示r5不存在积分性质。

## 9. 推荐下一步

立即建立E44本地seed0归因矩阵，固定本E43的`matched_contrast.csv`：

```text
1. train-only unary marginal baseline
2. 64-bit pair-local true P-layer
3. 64-bit triangle true P-layer
4. 最强真拓扑候选的fair-corrupted P-layer control
```

单变量是pair processor/topology；不得改变标签、split、mask、structure或checkpoint协议。
readiness门冻结为候选验证AUC至少`0.60`、领先一元基线至少`0.05`；归因门要求最强真拓扑
领先fair-corrupted至少`0.03`。只有seed0同时通过才运行seed1，E44仍为本地小规模网络
readiness，不使用远程GPU。

最终`curves.svg`按`visual-qa-redraw`渲染为`1444×729`像素检查；中文标题、解释、柱图、
数值、图例、AUC阈值线、裁决和导出边界均无重叠、裁切或缺字，已记录
`visual_qa_passed.marker`。
