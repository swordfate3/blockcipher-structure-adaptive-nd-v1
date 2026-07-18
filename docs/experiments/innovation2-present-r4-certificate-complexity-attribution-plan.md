# 创新2 E45：PRESENT r4证书复杂度与拓扑特征归因计划

日期：2026-07-18

状态：完成 / pass / MSPN路线开放

## 1. 研究问题

E44在E43严格checkerboard标签上得到：

```text
triangle true P validation AUC       = 0.561979
triangle fair-corrupted validation AUC= 0.549698
true - corrupted                     = +0.012281
```

网络超过一元边际，但没有通过候选或P-layer归因门。E45不训练新网络，而是用train-only
确定性特征回答这个弱信号主要来自哪类关系：

1. 静态`active set × output mask`几何，不需要P-layer；
2. 正确P-layer的1--4步可达关系；
3. ANF支撑传播的1--3轮前缀复杂度；
4. 最终四轮证书本身。

最后一类是标签构造oracle，只作为上限和语义自检，不作为可比较的神经baseline。

## 2. 冻结来源与split

```text
source atlas = i2_present_r4_universal_balance_atlas_20260718
source neural= i2_present_r4_pair_state_neural_attribution_seed0_20260718
train        = 800 rows, 400/400, 53 structures
validation   = 236 rows, 118/118, 18 structures
```

必须重新核验两个source run id、decision、SHA256、行数、类别、structure互斥、edge唯一以及
每个structure/mask内部正负平衡。不得读取E43 unknown行作为负类。

## 3. 冻结特征族

所有特征只从`active_bits`、`output_mask_bits`、PRESENT P-layer与sound ANF支撑传播计算。

### 3.1 static-set

不使用P-layer：

```text
mask weight
active/mask cell count与lane histogram
active-mask bit overlap
same-cell active-to-mask pair count
bit-index mean/std与cyclic span
active/mask cell occupancy交集
```

### 3.2 topology-reachability

分别使用true P-layer和E44同定义fair-corrupted P-layer：

```text
P^1 ... P^4(active) 与 mask 的bit overlap
P^1 ... P^4(active cell closure) 与 mask cell overlap
最早命中step
四步累计命中数
```

true与corrupted特征维数、归一化和拟合器必须相同。

### 3.3 ANF-prefix-complexity

仅使用正确P-layer，但不得使用第4轮完整cube单项式是否存在：

```text
round 1/2/3 selected-output support size的mean/max/sum
round 1/2/3 selected-output support saturation ratio
round 1/2/3 selected-output support union size
round 1/2/3各degree层的占比摘要
```

它模拟“可学习单项式支撑传播网络”可利用的中间状态复杂度。

### 3.4 final-certificate oracle

```text
selected output bits中包含full 8-variable cube monomial的bit数量
```

E43 positive按该数量为0得到，negative必然大于0；因此预期接近或等于完美分类。它只验证
数据与证书语义一致，不能与神经候选宣称公平性能比较。

## 4. 拟合器与防泄漏

每个非oracle族固定使用同一个train-only ridge-linear score：

```text
standardization = train mean/std only
intercept       = yes
ridge lambda    = 1e-3
selection       = none
metric          = validation AUC
```

同时报告每个单特征的方向无关AUC `max(AUC, 1-AUC)`。不得根据validation选择特征、lambda
或符号。oracle直接使用确定性计数，不拟合。

## 5. 裁决门

流程门：source与特征协议全部通过，所有矩阵有限，train-only标准化，true/corrupted维数
一致，oracle与标签语义一致。

方向排序相对E44 triangle `0.561979`：

```text
certificate_prefix_route:
  prefix ridge AUC >= 0.60
  prefix - static >= 0.03
  prefix - true topology >= 0.02
  -> 下一网络 = Monomial Support Propagation Network (MSPN)

topology_route:
  true topology AUC >= 0.60
  true - corrupted >= 0.03
  true >= prefix - 0.01
  -> 下一网络 = query-conditioned NBFNet-style relation reasoner

static_route:
  static AUC >= max(prefix, true topology) - 0.01
  static AUC >= 0.60
  -> 暂停拓扑网络，重构benchmark或采用set-interaction baseline

unresolved:
  所有非oracle AUC < 0.60
  -> 不开发新网络；先增加不泄漏的证书状态监督或分析unknown边界
```

多个路线同时满足时，优先选择validation AUC最高者；差小于`0.01`时优先更简单的
`static < topology < prefix`。E45只选择方向，不把确定性特征AUC写成神经结果。

## 6. 下一网络的最小边界

若MSPN解锁，首个实现只能显式维护每bit的压缩degree/support state，按PRESENT
`AddRoundKey -> S-box ANF combine -> P-layer`共享更新4步，并用output-mask query池化；
必须比较true/fair-corrupted与final-oracle-free deterministic prefix baseline。

若NBFNet解锁，使用query-conditioned path messages和4步共享Bellman-Ford更新；不得同时
加入ANF prefix特征。若static路线成立，先训练小型DeepSets/Set Transformer同预算锚点，
不声称拓扑创新。

E45及其后续smoke均为本地任务，不使用远程GPU。任何路线在seed0正确拓扑归因前不得进入
seed1、r5迁移或远程规模。

## 7. 产物

```text
outputs/local_audits/i2_present_r4_certificate_complexity_attribution_20260718/

features.csv
results.jsonl
gate.json
summary.json
metadata.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

声明范围：E43/E44四轮严格标签的确定性特征归因与下一架构路由；不是新积分区分器、神经
模型结果、高轮结论或SOTA攻击。

## 8. 2026-07-18实际结果

权威run：

```text
i2_present_r4_certificate_complexity_attribution_20260718
```

E43/E44 source run id、decision、SHA256、800/236行、400/400与118/118类别、96个structure、
300个mask、structure互斥、edge唯一和每个structure/mask正负平衡均重新核验通过。

同一train-only ridge结果：

| 特征族 | validation AUC | 最强单特征 |
|---|---:|---|
| static-set | `0.504309` | 见`results.jsonl` |
| fair-corrupted topology | `0.459063` | 见`results.jsonl` |
| true topology reachability | `0.648161` | 见`results.jsonl` |
| ANF round1--3 prefix | `0.686082` | 见`results.jsonl` |
| final certificate oracle | `1.000000` | full-cube candidate count |

关键差值：

```text
true topology - corrupted topology = +0.189098
ANF prefix - true topology          = +0.037920
ANF prefix - static set             = +0.181772
```

正确P-layer可达特征本身明显有效，证明E44标签并非拓扑完全无关；但ANF前缀在不读取最终
四轮full-cube oracle的情况下仍进一步领先`+0.0379`，通过certificate-prefix全部冻结门。
topology路线因为`true < prefix - 0.01`没有独立获胜；static路线接近随机。

最终裁决：

```text
status         = pass
decision       = innovation2_present_mspn_route_ready
selected_route = certificate_prefix_route
remote_scale   = no
```

## 9. 推荐下一步

建立E46 `Monomial Support Propagation Network` readiness smoke。网络应显式维护64个bit的
压缩单项式支撑状态，按4轮共享执行：

```text
key/constant channel injection
-> PRESENT S-box ANF term combination
-> true/fair-corrupted P-layer transport
-> output-mask query pooling
```

首版不得直接输入E45手工prefix特征或final full-cube count，否则只是泄漏确定性baseline。
它必须从active bits、mask、S-box ANF和P-layer学习传播；同预算比较MSPN true、MSPN
fair-corrupted、E45 prefix ridge与E44 triangle。readiness smoke只检查有限前向/反向、
参数预算、true/corrupted差异和短训练可学习性；通过后另建seed0正式计划。

最终`curves.svg`按`visual-qa-redraw`渲染为`1378×729`像素检查；标题、说明、AUC柱图、
差值、0.03门槛、oracle范围和导出边界均无重叠、裁切或缺字，已记录
`visual_qa_passed.marker`。
