# 创新2 E48：support-state碰撞与变量身份必要性审计计划

日期：2026-07-18

状态：完成 / hold / 变量身份路线不支持

## 1. 研究问题

E45的ANF 1--3轮prefix ridge达到`0.686082`，但E47 MSPN true validation只有`0.518673`，
train AUC却达到`0.794375`，且错误P-layer更高。MSPN只维护匿名degree/support强度，可能把
不同活动变量组合产生的support集合压成相同状态。E48判断失败是否确实来自变量身份碰撞。

## 2. 冻结来源

```text
data = E43 matched checkerboard, 800 train / 236 validation
E45 = certificate-prefix attribution passed
E47 = MSPN candidate not ready
rounds represented = ANF prefix rounds 1, 2, 3
active variables   = each structure's sorted eight active bits, local IDs 0..7
```

重新核验E43/E45/E47 run id、decision、hash、行数、类别、structure互斥和E47 metric。

## 3. Support身份表示

对每个`(structure, output mask)`，每轮构造256维向量：第`m`维表示所选输出位的support集合
中包含8变量monomial mask `m`的次数，除以mask weight。三轮连接为768维。

```text
exact identity vector = r1[256] || r2[256] || r3[256]
degree-only vector     = each round按popcount(m)=0..8聚合，共27维
```

不得使用第4轮full-cube出现计数、E43 certificate status、witness或label作为特征。

## 4. 固定identity sketch与控制

冻结随机seed `48001`，从768维构造固定Rademacher投影：

```text
sketch16 / sketch32 / sketch64
```

primary gate只使用`sketch64`，16/32仅报告压缩曲线，不根据validation选宽度。

两个同维控制：

1. `local-id-permuted sketch64`：每个structure使用独立固定0..7变量置换，保持degree和support
   大小，但破坏跨structure局部变量身份对应；
2. `fair-corrupted-P sketch64`：用E44相同destination-cell rotation传播support，保持S-box
   ANF、活动维度、投影和宽度，只改变P-layer transport。

## 5. 碰撞与预测指标

### 5.1 碰撞

对degree-only、exact identity和GF(2) binary sketch16/32/64分别报告：

```text
unique signatures
rows in signatures containing both labels
conflicting-row rate
conflicting signature count
```

binary sketch使用相同seed的0/1投影；它只用于碰撞统计。预测使用Rademacher实值投影。

### 5.2 Train-only预测

degree、exact identity、true sketch16/32/64、permuted sketch64和corrupted sketch64全部使用：

```text
train-only standardization
ridge lambda = 1e-3
validation AUC
```

不根据validation调lambda、符号或宽度。

## 6. 裁决门

`identity_sketch_route`全部满足：

```text
sketch64 validation AUC                    >= 0.62
sketch64 - degree-only                     >= 0.03
sketch64 - local-id-permuted sketch64      >= 0.03
sketch64 - corrupted-P sketch64            >= 0.03
binary sketch64 conflicting-row rate       <= degree conflict rate - 0.20
```

通过则下一网络为`Identity-Sketch Monomial Propagator`，但网络不得直接读取E48离线sketch；
它应给8个活动变量分配可重标号token，在ANF term组合中传播变量身份sketch，再由mask query
池化。

`exact_identity_only`：exact identity ridge达到`0.62`且超过degree `0.03`，但sketch64未过。
下一候选改为稀疏`Monomial Token Set Transformer`，不强行压到64维。

`identity_not_supported`：exact identity也没有稳定超过degree，或permuted/corrupted控制不降。
停止变量身份神经路线；保留E45确定性prefix与E44 triangle作为现有结论。

E48不训练神经网络，不使用远程GPU。后续readiness必须另建计划。

## 7. 产物

```text
outputs/local_audits/i2_present_r4_support_identity_collision_20260718/

features.csv
collision.csv
results.jsonl
gate.json
summary.json
metadata.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

声明范围：PRESENT-80四轮严格标签的support身份碰撞和下一架构路由；不是神经结果、高轮
积分区分器、新攻击或SOTA。

## 8. 2026-07-18实际结果

权威run：

```text
i2_present_r4_support_identity_collision_20260718
```

E43/E45/E47 run id、decision、source hash、checkerboard行数与structure互斥、P-layer置换、
train-only标准化和固定投影检查全部通过。预测结果：

| 表示 | validation AUC |
|---|---:|
| degree-only | `0.689170` |
| exact support identity | `0.599109` |
| sketch16 | `0.632361` |
| sketch32 | `0.680480` |
| sketch64 | `0.670712` |
| local-ID permuted sketch64 | `0.407785` |
| fair-corrupted-P sketch64 | `0.599325` |

碰撞结果：

| 签名 | unique | 跨标签冲突行率 |
|---|---:|---:|
| degree-only | `765/1036` | `2.6062%` |
| exact identity | `857/1036` | `0.1931%` |
| binary sketch16 | `790/1036` | `0.3861%` |
| binary sketch32 | `792/1036` | `0.1931%` |
| binary sketch64 | `792/1036` | `0.1931%` |

关键差值：

```text
sketch64 - degree-only          = -0.018457
sketch64 - local-ID permuted    = +0.262927
sketch64 - fair-corrupted-P     = +0.071388
```

身份打乱和错误P-layer都会显著伤害sketch，说明表示确实携带变量对应和transport信息；但
精确身份与sketch64都没有超过degree-only，且degree-only本身冲突率只有`2.61%`。冻结的
identity-sketch和exact-token门因此均失败：

```text
status   = hold
decision = innovation2_present_support_identity_not_supported
```

这个结果否定的是“E47主要因为变量身份碰撞而失败”，不是证明所有神经结构均无效。
`Identity-Sketch Monomial Propagator`、`Monomial Token Set Transformer`、seed1、r5迁移和
远程GPU全部关闭。

## 9. 推荐下一步

E45的ANF prefix ridge为`0.686082`，E48的degree-only为`0.689170`，而E47 MSPN true只有
`0.518673`。最窄、仍有证据支持的问题不是再换更大的网络，而是：MSPN的中间状态是否
根本没有学会E45有效的1--3轮degree spectrum。

下一步冻结为E49本地readiness：保持E47数据、MSPN hidden32、四步传播和balance head不变，
只在训练期加入1--3轮、每轮13维ANF-prefix辅助目标；这些确定性target不得作为最终分类器
输入。与label-only E47锚点、同预算train-target shuffle安慰剂和自洽fair-corrupted-P控制
比较。只有heldout structure上的degree-target误差明显优于shuffle，且balance AUC也超过
E47 true与同预算安慰剂，才允许另建30轮正式计划。

E49只做本地两轮readiness，不开放seed1、remote、r5或更深/更宽MSPN。若辅助target无法
组外学习，或只改善target误差而不改善balance AUC，则停止当前证书传播神经路线，保留
E45确定性归因和E44 triangle神经锚点。

最终`curves.svg`按`visual-qa-redraw`渲染为`1800×914`像素，经过两次重绘后重新检查；中文
标题、柱值、阈值图例、百分比单位、横轴换行、底部裁决和导出边界均无重叠、裁切、缺字
或含义歧义，已记录`visual_qa_passed.marker`。
