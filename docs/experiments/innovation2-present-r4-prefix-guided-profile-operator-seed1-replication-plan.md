# 创新2 E68：PRESENT四轮prefix引导平衡谱算子seed1复核计划

日期：2026-07-18

状态：已完成 / pass / 双seed确认

## 1. 研究问题

E67 seed0的正确P profile mixer验证AUC为`0.953056`，相对independent、corrupted和E65
ANF-prefix ridge分别为`+0.190556/+0.152222/+0.159444`。E68只改变随机种子为1，判断该
结构增益是否能独立复现。

## 2. 完全冻结协议

数据、标签、split、39维prefix、5679参数模型、hidden32、steps2、dropout0.10、batch8、
30 epochs、AdamW、checkpoint和三行矩阵全部沿用E67。唯一变化：

```text
seed = 1
```

禁止根据seed0结果修改任何超参数、门、特征或checkpoint规则。

## 3. Seed1与联合门

seed1独立满足：

```text
true validation AUC                  >= 0.78
true train - validation AUC          <= 0.15
true - independent                   >= 0.03
true - fair-corrupted                >= 0.03
true - E65 ANF-prefix ridge          >= 0.02
```

联合门要求seed0和seed1全部独立通过，并且两seed平均的三项差值仍分别达到`0.03/0.03/0.02`。
通过得到`profile_operator_two_seed_confirmed`；失败则保留seed0单seed证据但不称稳定方法。

## 4. 下一步边界

两seed通过后，不继续增加seed、epoch或容量。下一研究优先级是把已确认的unit-output nodewise
operator扩展到E43的多bit linear mask查询，先做无训练标签/边际审计，再决定是否增加一个
mask-query decoder；不得直接迁移r5，因为五轮严格positive标签提供者仍未就绪。

```text
run_id = i2_present_r4_prefix_guided_profile_operator_seed1_20260718
output = outputs/local_diagnostic/i2_present_r4_prefix_guided_profile_operator_seed1_20260718
```

## 5. 2026-07-18实际结果

E65/E67 source、hash、严格标签、split、prefix replay、模型contract和seed1三行30轮训练全部
通过。seed1最佳checkpoint：

| 模式 | best epoch | train AUC | validation AUC | validation accuracy |
|---|---:|---:|---:|---:|
| independent node | 21 | `0.750663` | `0.765000` | `0.675000` |
| true-P profile mixer | 27 | `0.982957` | `0.961389` | `0.900000` |
| fair-corrupted-P mixer | 26 | `0.822781` | `0.819444` | `0.725000` |

seed1差值：

```text
true - independent      = +0.196389
true - corrupted        = +0.141944
true - E65 prefix ridge = +0.167778
true train - validation = +0.021568
```

双seed联合：

```text
mean true AUC           = 0.957222
mean true-independent   = +0.193472
mean true-corrupted     = +0.147083
mean true-ANF ridge     = +0.163611
```

每颗seed和联合门全部通过：

```text
status   = pass
decision = innovation2_present_profile_operator_two_seed_confirmed
remote   = no
```

## 6. 裁决与推荐下一步

保留`Prefix-Guided Nodewise Profile Operator`为创新2当前最强的真实PRESENT结构方法证据：
它在四轮、严格unit-output universal-balance标签、structure-disjoint验证上，两颗seed均显著
超过同容量独立node、fair-corrupted P-layer和安全ANF-prefix ridge。

声明必须同时保留边界：四轮、小型本地benchmark、部分观察的64维unit profile；不是PRESENT
高轮积分区分器、攻击轮数突破或SOTA。下一步E69先做无训练多bit linear-mask profile审计：

```text
source       = E43 236个非unit masks
families     = nibble / player_pair / same_nibble_pair / adjacent_nibble_pair
single change= unit node输出扩展为multi-bit mask query
controls     = global/mask/family/active边际 + true/corrupted reachability + ANF prefix
gate         = 每族宽度、structure-disjoint覆盖、validation mask seen、边际AUC <= 0.55
execution    = local exact audit / no training / no remote
```

只有E69通过，才设计一个从已确认64-node表示读取任意linear mask的轻量query decoder；不过门
则停在unit profile成果，不机械增加Transformer或迁移r5。

最终`curves.svg`经`visual-qa-redraw`渲染为`2200 x 1165`像素检查；双seed分组柱、每seed
三项增益、ANF ridge线、联合均值、裁决和范围均无重叠、裁切或歧义。
