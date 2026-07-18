# 创新2 E64：小型SPN多坐标relation非平凡消去分解计划

日期：2026-07-18

状态：已完成 / 暂缓 / 多坐标网络路线停止

## 1. 研究问题

E62的relation标签严格且宽，但E63中DeepSets与RCCA都没有稳定超过边际。E64检查标签问题是否
其实主要退化为两个singleton平衡状态的简单组合。

对每个`variant x relation template`，从E37全256主密钥parity向量精确分类：

```text
positive, both coordinate vectors zero      = trivial both-balanced
positive, equal and both nonzero             = nontrivial cancellation
negative, exactly one vector zero             = singleton-status mismatch
negative, both nonzero and different          = nontrivial negative
```

E64不训练、不改E62模板、不读取神经预测。

## 2. 冻结来源

```text
E37 parity cache = 64 variants x 4 rounds x 14 structures x 256 keys
E62 relations    = 2048 templates, each exactly two distinct coordinates
E62 labels       = 64 x 2048, all-key strict
split            = 36 train / 12 unseen-S / 12 unseen-P / 4 dual
```

必须逐样本复算：relation label等于两个coordinate 256-bit向量是否相等；E62 negative witness仍在
对应key index为odd。

## 3. 强基线

在完整E62样本上评估三项拓扑无关确定性分数：

```text
both balanced              = left_zero AND right_zero
same singleton status      = left_zero == right_zero
either balanced            = left_zero OR right_zero
```

报告每个heldout split最强AUC。不能只在筛选后的nontrivial子集报告0.5。

## 4. 非平凡宽度门

```text
train nontrivial cancellation positive / negative each >= 3000
unseen-S positive / negative                         each >= 768
unseen-P positive / negative                         each >= 768
dual positive / negative                             each >= 192
nontrivial positive fraction in every split          >= 0.25
full-label strongest singleton-status AUC:
  unseen-S / unseen-P <= 0.80
  dual <= 0.75
all relation labels and witnesses replay             pass
```

通过表示E62确实含宽的“两个非零key-dependent坐标互相消去”问题；这不挽救已失败的RCCA，只允许
再排名一种已有证据的关系算子。失败则E62重分类为singleton组合主导，停止多坐标网络路线。

## 5. 推荐边界

若通过，下一候选只能是复用E39已成功的低秩bit-pair path reasoner作为relation-coordinate
interaction，不新造Transformer名字；先做readiness与DeepSets同预算比较。若失败，不训练任何
新模型，创新2保留E39小型SPN结构方法证据与E62严格标签数据集。

```text
run_id = i2_small_spn_relation_decomposition_20260718
output = outputs/local_audits/i2_small_spn_relation_decomposition_20260718
execution = local exact audit / no training / no remote GPU
```

## 6. 正式结果

E37 parity cache、E62 relation标签与全部negative witness逐样本重放通过。四类分解如下：

```text
split         trivial+   nontrivial+   one-zero-   nontrivial-
train          41373          158         27171          5026
unseen-S       16346           30          7338           862
unseen-P       13673           50          9353          1500
dual            6152            6          1941            93
```

nontrivial positive占全部positive比例：

```text
train      = 0.003804
unseen-S   = 0.001832
unseen-P   = 0.003644
dual       = 0.000974
```

最强singleton-status基线均是`both_balanced`：

```text
unseen-S   AUC = 0.999084
unseen-P   AUC = 0.998178
dual       AUC = 0.999513
```

因此E62正类几乎全部是“两坐标各自都balanced”，而不是两个nonzero key-dependent parity vector
发生GF(2)消去。dual只有6条nontrivial positive，远低于192门；所有positive宽度和反捷径门失败。

```text
status   = hold
decision = innovation2_small_spn_relation_nontrivial_width_not_ready
training = no
remote   = no
```

## 7. 裁决与推荐下一步

停止当前多坐标神经网络路线，不实现pair-path relation模型，不扩大模板或模型，不启动远程GPU。
E63的失败现在有更清楚的解释：训练任务主要是singleton balance的AND组合，RCCA的复杂交互没有
必要目标；但这不改变RCCA正式双seed确实失败的裁决。

创新2保留两项可写证据：E39 SPN-PRR在合成单坐标benchmark上的可归因拓扑收益，以及E62/E64
对严格多坐标标签构造与捷径审计的方法学反例。下一研究动作应回到真实密码高轮输出性质的标签
定义或论文写作整合，而不是继续枚举小型SPN新网络。
