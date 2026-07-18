# 创新2 E95：RECTANGLE-80 四轮嵌套 cube 关系无训练机制门

日期：2026-07-19

状态：已完成 / `hold` / 关系未归因，嵌套 cube 神经路线关闭

## 1. 研究问题

E94 已确认 192 条 `7 ⊂ 8 ⊂ 9` cube 链的严格标签 sound、足够宽、chain-disjoint、满足单调定理且
不存在一元捷径。E95 只回答：

```text
在相同 r1-r3 prefix 特征、相同 ridge 容量和相同单调投影下，
正确的 cube 子集/超集关系是否比独立节点、打乱关系和错误超集提供可归因的验证信息？
```

E95 不训练神经网络。通过才允许两轮 `Monotone Cube-Lattice Operator` readiness；失败则关闭该路线，
不得调 ridge、关系种子、维度范围或门槛后重跑。

## 2. 冻结来源

```text
source run = i2_rectangle80_r4_nested_cube_monotonic_readiness_20260719
source root = outputs/local_audits/i2_rectangle80_r4_nested_cube_monotonic_readiness_20260719

gate.json SHA-256                    = 323c7bada04941e79dc932ab1c562cab977b0288039632557cf1556b41c9f4f9
chains.json SHA-256                  = f027b920d0a8bacfe1f2e5b3a4707444cb5b92af3b6f85896be2593bebe4c8bc
matched_nested_contrast.csv SHA-256 = 49ded50d4d5c48adf610112223d9fa34d6db85b78ca3bab4ffa4aacba03abb3b
atlas.jsonl SHA-256                  = 2fa8706232edccca5e0bd45bd362dbbc075660b55b58731a4ea155f306afbef3
```

必须重放 E94 `status=pass`、`decision=innovation2_rectangle80_nested_cube_monotonic_labels_ready`、
全部内部检查和 `visual_qa_passed.marker`。冻结样本为 9,556 个 matched edges：train 7,216，validation
2,340；拆分单位仍是整条 chain。

## 3. Prefix 特征

对每个 `(chain, dimension, output_bit)` 独立重算 RECTANGLE-80 第 1、2、3 轮 active-variable ANF
support，仅取第 3 轮固定 13 维摘要：

```text
4 个归一化 support-size 统计
9 个 degree 0..8 频率；9 次项并入最高 bucket
```

不输入第 4 轮 support、E94 label、certificate、witness、unknown 状态或 validation 统计。

每个预测边固定 44 列：

```text
own r3 prefix           = 13
predecessor r3 prefix   = 13
successor r3 prefix     = 13
predecessor/successor mask = 2
dimension one-hot       = 3
total                   = 44
```

缺失邻居填零且 mask=0。所有模式均保留 44 列及 45 个含截距 ridge 系数；
`ridge_lambda=1e-3`，只用 train 标准化和拟合。

## 4. 唯一变量：关系映射

```text
independent_dimension:
  predecessor/successor 全零；不做单调投影

true_nesting:
  使用同一 chain 的真实 d7-d8-d9 邻居；做三点递增 L2 isotonic projection

shuffled_nesting:
  train 与 validation 各自按 chain index 循环平移 1，保持 split、维度和边际分布；
  使用错误 chain 邻居；做同一 isotonic projection

wrong_superset:
  在同 split 内选择第一个确定性 chain，使四个相邻包含关系全部为假；
  做同一 isotonic projection

true_unconstrained:
  与 true_nesting 完全相同的 44 列特征和 ridge，仅不做 isotonic projection
```

shuffled/wrong 映射不得读取标签。所有映射冻结、无 seed 搜索；必须是 split-preserving derangement。

## 5. 评估与协议门

ridge 在 E94 matched train edges 拟合，但对完整 `192 × 3 × 64` 节点产生分数。单调模式按同一
`(chain, output_bit)` 的 d7/d8/d9 三点做无标签 L2 isotonic projection，最终只在 E94 matched
validation edges 计算 AUC。

协议必须满足：

```text
E94 四个来源哈希一致且全部 gate checks pass
192 chains / 9556 matched edges / 7216 train / 2340 validation
prefix shape = 192 x 3 x 64 x 13，全部 finite
所有模式 feature_count = 44，coefficient_count = 45
shuffled/wrong 均 split-preserving 且无 self mapping
wrong_superset 四类包含关系违反率 = 100%
关系映射不读取 labels
true/shuffled/wrong 投影后单调违反数 = 0
train/validation chains disjoint
```

## 6. 预注册门槛

```text
true_nesting validation AUC >= 0.70
true_nesting train-validation AUC gap <= 0.15

true - independent_dimension >= +0.03
true - shuffled_nesting      >= +0.03
true - wrong_superset        >= +0.03
true - true_unconstrained    >= -0.01
```

全部通过：

```text
status   = pass
decision = innovation2_rectangle80_nested_cube_relation_mechanism_ready
next     = 两轮 Monotone Cube-Lattice Operator readiness，local CPU，seed0
```

任一来源、关系、容量、拆分或单调协议无效：

```text
status   = fail
decision = innovation2_rectangle80_nested_cube_relation_protocol_invalid
```

协议有效但任一质量/关系 margin 未过：

```text
status   = hold
decision = innovation2_rectangle80_nested_cube_relation_not_attributed
next     = 关闭当前 nested-cube neural route
```

不得因为某一 margin 贴线而补跑映射、换 ridge lambda、添加 label-derived 特征、降低门槛或直接训练神经网络。

## 7. 计划产物

```text
run_id = i2_rectangle80_r4_nested_cube_relation_mechanism_20260719
output = outputs/local_audits/i2_rectangle80_r4_nested_cube_relation_mechanism_20260719

relation_maps.json
ridge_reports.json
results.jsonl
gate.json
summary.json
metadata.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

E95 只能支持“真实 cube nesting 在确定性 prefix/ridge 机制上可归因/不可归因”，不构成神经收益、
第三 SPN 正式神经确认、高轮区分器、攻击或 SOTA。

## 8. 实际结果

E94 四个冻结来源哈希、全部内部 gate、visual QA、192 条 chain、9,556 个 matched edges 和
`7,216/2,340` train/validation 行均重放通过。重算的 prefix shape 为 `192 x 3 x 64 x 13`，
五种模式均为 44 列、45 个含截距 ridge 系数，所有投影模式最终单调违反数为零。

| 模式 | train AUC | validation AUC | 最终单调违反数 |
|---|---:|---:|---:|
| independent dimension | 0.646827 | 0.659853 | 5,498（未投影控制） |
| true nesting | 0.682690 | 0.674691 | 0 |
| shuffled nesting | 0.655192 | 0.656339 | 0 |
| wrong superset | 0.655192 | 0.656339 | 0 |
| true unconstrained | 0.680858 | 0.671110 | 11,975（未投影控制） |

真实 nesting 的 validation margin：

```text
true - independent     = +0.014837  fail (< +0.03)
true - shuffled        = +0.018351  fail (< +0.03)
true - wrong superset  = +0.018351  fail (< +0.03)
true - unconstrained   = +0.003581  pass (>= -0.01)

true validation AUC    = 0.674691  fail (< 0.70)
train-validation gap   = +0.008000 pass (<= 0.15)
```

冻结构造还暴露一个需要明确限定的控制冗余：对当前 192 条 chain，循环平移 1 的 shuffled mapping
已经让四类相邻包含关系全部失效，因此 deterministic wrong-superset selector 在 `192/192` 个目标上
选择了同一 mapping。两行结果完全相同，不是两个独立错误关系复现。该冗余不能用于救回路线；即使只
把它们视为一个完整的 split-preserving corrupted-relation 控制，真实 margin 也只有 `+0.018351`，
仍未过门。按预注册约束不另换 shuffle 或 wrong mapping 补跑。

## 9. 裁决

```text
status   = hold
decision = innovation2_rectangle80_nested_cube_relation_not_attributed
training = no
remote   = no
```

E94 的 sound 单调标签结论保持有效，但 E95 说明正确 nesting 在安全的 r3-prefix/ridge 机制上只有小幅
信息，未达到开放新神经结构所需的独立关系 margin。不得实现或训练 `Monotone Cube-Lattice
Operator`，不得添加 dimension embedding、attention、hidden width、层数、epoch 或远程规模。

## 10. 推荐下一步：E96 研究组合边界复核

下一步不是另一组模型，而是本地无训练的架构候选复核：

```text
question = E95关闭后，现有sound标签与文献机制中是否还存在未被强基线解释的结构任务？

frozen sources:
  E80/E93  PRESENT+GIFT正式r3-only方法与架构边界
  E69      multi-bit mask被componentwise基线完全解释
  E70      4/12-bit活动维度标签provider未就绪
  E84      SKINNY residual未归因
  E86      PRESENT/GIFT共享参数质量失败
  E90-E92  RECTANGLE原算子/row机制/row神经边界
  E94/E95  nested-cube标签通过但关系机制未归因

execution = local evidence/literature audit, no labels regenerated, epochs=0, seeds=none, remote=no
```

E96 必须输出按 `formal_confirmed / label_ready_but_unattributed / provider_missing / closed` 分类的候选表。
只有候选同时具有新的 sound 标签语义、非平凡确定性机制、必要错误关系控制和同预算预神经 margin
门，才允许新计划；否则停止创新2架构枚举，保留 PRESENT/GIFT 双密码正式方法用于论文写作。
