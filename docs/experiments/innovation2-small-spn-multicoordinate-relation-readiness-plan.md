# 创新2 E62：小型SPN严格多坐标relation标签readiness计划

日期：2026-07-18

状态：已完成 / 通过 / DeepSets与RCCA训练门开放

## 1. 研究问题

E61证明PRESENT两轮ATM完整key-polynomial支撑不能在冻结预算内扩展。E62不降低E61标准或继续
提高SAT cap，而是使用E37已完成的16-bit small-SPN全密钥精确parity cache，判断两坐标关系：

```text
coordinate = (active structure, linear output mask)
relation   = two distinct coordinates at the same cipher variant and round
label 1    = both coordinate parity vectors XOR to zero for all 256 master keys
label 0    = at least one concrete master key has odd relation parity
```

E62只建立严格多坐标benchmark并审计捷径，不训练模型。通过后才比较DeepSets与RCCA。

## 2. 冻结来源

```text
source run = i2_small_spn_expanded_topology_4s16p_256keys_20260718
variants   = 4 S-boxes x 16 P-layers = 64
rounds     = 2, 3, 4, 5
structures = 14 coordinate structures
masks      = 64 nonzero linear masks
keys       = all 256 values of the frozen 8-bit master-key space
coordinates per round = 14 x 64 = 896
```

必须复核E37 gate、array shape/dtype、metadata和全部cache completion。E62不重新生成或改变密钥、
S-box、P-layer、结构、mask、轮数或split。

## 3. 预注册relation模板池

从`C(896,2)`个不同坐标对中，用固定seed `62001`在读取标签前生成65,536个唯一无序pair：

```text
left < right
left != right
relation size = 2
candidate pair count = 65536
same pair pool reused for all rounds and all variants
```

每个坐标由`structure_index * 64 + mask_index`唯一编码。pair生成不得使用parity、label或heldout
拓扑信息。

## 4. 严格标签与证书

对每个`variant x round x pair`，从E37 `parity_words.npy`计算两个线性mask的256-bit parity
向量。两向量完全相等则relation对全部主密钥恒为0，标positive；否则标negative，并保存第一把
parity不同的master-key index作为严格反例。

所有positive必须直接复算256-bit XOR为全零；所有negative witness必须重放为1。没有random、
unknown、有限密钥外推或`not in basis`标签。

## 5. Train-only选择与split

沿用E37拓扑split：

```text
train         = 3 S-box x 12 P = 36 variants
unseen-S      = 1 S-box x 12 P = 12 variants
unseen-P      = 3 S-box x 4 P  = 12 variants
dual-unseen   = 1 S-box x 4 P  = 4 variants
```

对每个`round x relation template`只统计36个train标签，选择：

```text
9 <= train positives <= 27
maximum selected per round = 1024，按冻结candidate顺序截取
```

heldout标签不得参与选择。相同relation template跨variant复用是有意设计：任务要求模型根据
S-box/P-layer判断同一坐标关系何时成立。必须报告relation-ID train-rate强基线，防止把模板记忆
误写成拓扑推理。

## 6. Readiness与反捷径门

```text
selected relation templates                        >= 256
supported rounds with >=64 templates               >= 2
train positive / negative                          each >= 3000
unseen-S positive / negative                       each >= 768
unseen-P positive / negative                       each >= 768
dual positive / negative                           each >= 192
distinct 64-topology label patterns                >= 128
all positives replay all-256-key zero               pass
all negative witnesses replay odd                   pass
train P-sensitive any-S fraction                   >= 0.75
dual P-effect relation fraction                    >= 0.40
train SxP interaction fraction                     >= 0.50
full SxP interaction fraction                      >= 0.70
unseen-S strongest topology-free marginal AUC      <= 0.80
unseen-P strongest topology-free marginal AUC      <= 0.80
dual strongest topology-free marginal AUC          <= 0.75
```

强基线至少包含global、relation-ID train positive rate、relation structural-feature group和
coordinate marginal。exact 256-key vectors只属于label oracle，不能成为模型输入。

## 7. 同预算锚点与唯一变量

```text
evidence anchor = E37 exact all-key topology benchmark
architecture anchor for later = coordinate-set DeepSets
one change = single (structure,mask) property -> two-coordinate GF(2) relation property
fixed = variants, rounds, keys, splits, cipher topology, cache, heldout ownership
```

E39 SPN-PRR的dual AUC `0.716651`属于单坐标输出性质任务，只作方法背景，不能与E62 AUC直接
横向宣称胜负。

## 8. 推进与停止

readiness通过后，下一实验才实现：

```text
1. coordinate-set DeepSets
2. Relation-Cipher Cross-Attention (RCCA), true P-layer
3. RCCA wrong-P-layer control
4. label-shuffle process control
```

固定hidden64、2个relation token、2层cross-attention、40 epochs、batch128、seed0/1、本地CPU。
RCCA必须逐seed超过DeepSets和最强边际，mean dual至少领先DeepSets与wrong-P各0.03，label-shuffle
不超过0.60，才保留该架构。否则关闭RCCA，不增加容量、epoch、seed或远程GPU。

若readiness本身失败，则停止该small-SPN relation benchmark并重新审视目标，不训练网络。任何通过
结果只属于16-bit合成SPN方法证据，不是PRESENT/GIFT高轮结果、攻击或SOTA。

## 9. 预注册运行

```text
run_id = i2_small_spn_multicoordinate_relation_readiness_20260718
output = outputs/local_audits/i2_small_spn_multicoordinate_relation_readiness_20260718
execution = local CPU, no neural training, no remote GPU
```

## 10. 正式结果

冻结65,536个candidate pair在约5秒内完成向量化全密钥审计：

```text
selected relation templates = 2048
per round selected           = r2 0 / r3 1024 / r4 1024 / r5 0
supported rounds             = 2
distinct 64-topology patterns= 1370

train       positive / negative = 41531 / 32197
unseen-S    positive / negative = 16376 / 8200
unseen-P    positive / negative = 13723 / 10853
dual-unseen positive / negative = 6158 / 2034
```

来源、candidate唯一性、train-only选择、cache completion和split契约全部通过。每个positive的
256-bit relation parity向量均为全零；每个negative都保存第一把odd master-key index并通过重放。

拓扑敏感性：

```text
train P-sensitive any-S relation fraction = 1.000000
dual P-effect relation fraction            = 0.985352
train SxP interaction fraction             = 0.989258
full SxP interaction fraction              = 1.000000
```

最强拓扑无关边际AUC：

```text
unseen-S    = 0.668047
unseen-P    = 0.652909
dual-unseen = 0.685895
```

dual最强项是结构特征group，relation-ID train-rate为`0.654275`，coordinate marginal为
`0.626163`；均未越过冻结`0.75`上限。因此标签不由relation ID、结构维数或mask重量完全解释。

```text
status   = pass
decision = innovation2_small_spn_multicoordinate_relation_training_ready
training = not performed in E62
remote   = no
```

权威产物：

```text
outputs/local_audits/i2_small_spn_multicoordinate_relation_readiness_20260718/gate.json
outputs/local_audits/i2_small_spn_multicoordinate_relation_readiness_20260718/results.jsonl
outputs/local_audits/i2_small_spn_multicoordinate_relation_readiness_20260718/relation_labels.npy
outputs/local_audits/i2_small_spn_multicoordinate_relation_readiness_20260718/witness_key_indices.npy
outputs/local_audits/i2_small_spn_multicoordinate_relation_readiness_20260718/selected_relations.csv
outputs/local_audits/i2_small_spn_multicoordinate_relation_readiness_20260718/curves.svg
```

## 11. 裁决与推荐下一步

E62通过，下一实验E63按第8节固定预算实现两类模型：DeepSets只编码两个relation coordinate；
RCCA让两个coordinate query token与真实small-SPN S-box/P-layer token做cross-attention。必要控制是
wrong-P和label-shuffle，不同时加入GraphGPS、SCGT、TokenGT或增大relation size。

E63先跑hidden32、8 epochs、seed0的本地readiness，验证token permutation不变性、true/wrong-P
输出差异、四行训练和checkpoint。通过后自动进入hidden64、40 epochs、seed0/1正式矩阵。
正式门保持：RCCA逐seed超过DeepSets和dual边际`0.685895`，mean dual领先DeepSets与wrong-P各
至少0.03，label-shuffle dual不超过0.60。失败立即关闭RCCA；通过也只能声称小型SPN方法证据。
