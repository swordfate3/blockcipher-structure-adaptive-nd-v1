# 创新2 E77：GIFT-64四轮拓扑交互基线与同权重反事实再审判计划

日期：2026-07-19

状态：已完成 / pass / E76单节点ridge门确认信息范围错配

## 1. 研究问题

E76因单节点r3 ridge AUC仅`0.507804`而hold，但三行两轮神经AUC为
`independent/corrupted/true = 0.560874/0.704475/0.760666`，真实P仍领先错误P`+0.056191`。
原ridge只看目标output node自己的13维特征，而神经候选还看同cell与P-layer前驱，二者信息范围
不公平。

E77不改E76裁决、不训练新模型，只回答两个归因问题：

1. 给确定性ridge同样的local/cell/P前驱信息后，真实GIFT拓扑是否优于多个错误P；
2. 固定E76 true-P checkpoint的全部可学习参数，仅替换P-layer，正确P推理是否稳定优于错误P。

## 2. 冻结来源

```text
profile source = E75 192结构严格unit profile
checkpoint     = E76 true-P两轮最佳checkpoint
split/labels   = 完全重放E75，不重新matching
training       = none
```

必须验证E75状态/裁决/620坐标/hash和E76状态hold、裁决
`innovation2_gift64_r3_only_prefix_not_sufficient`、全部协议门、全部神经readiness门、唯一失败的
确定性门`r3_ridge_auc_at_least_0p60`以及E76 true validation AUC逐值重放。

## 3. 确定性拓扑展开ridge

对每个`(structure, output_bit)`构造：

```text
local13       = 目标node的r3前缀
cell13        = 目标node所在4-bit cell的r3前缀均值
predecessor13 = 通过指定P-layer进入目标node的源node r3前缀
expanded39    = concat(local13, cell13, predecessor13)
```

使用训练集标准化、`lambda=1e-3` ridge，比较：

```text
local13
true-P expanded39
corrupted-P shift1 expanded39
corrupted-P shift2 expanded39
corrupted-P shift3 expanded39
```

三个错误P只把GIFT目标cell循环移动1/2/3格，保持64位置换、4-bit cell与lane结构，不改变标签或
split。确定性门：

```text
true-P ridge validation AUC >= 0.60
true-P ridge - local ridge >= +0.03
true-P ridge - max(corrupted ridge) >= +0.03
```

## 4. 同权重checkpoint反事实

读取E76 true-P checkpoint；所有可学习参数逐值冻结。构造正确P和同样三个错误P的模型，只替换
`player/inverse_player` buffer，不优化任何参数。要求：

```text
correct-P checkpoint AUC == E76 recorded true AUC (abs error <= 1e-12)
correct-P - max(corrupted-P inference AUC) >= +0.03
```

## 5. 裁决与下一步

两组门全部通过：

```text
decision = innovation2_gift64_topology_interaction_gate_repaired
```

这只说明E76的单节点ridge前置门信息范围错配，并为另立E78的30轮seed0正式归因计划提供依据；
不直接修改E76 gate，也不在E77训练网络。

任一归因门失败：

```text
decision = innovation2_gift64_topology_interaction_not_confirmed
next     = 关闭GIFT r3-only，不增加epoch/hidden/steps/corruption搜索
```

协议失败则修复实现，不解释AUC。

## 6. 运行与产物

```text
run_id = i2_gift64_r4_topology_interaction_readjudication_20260719
output = outputs/local_audits/i2_gift64_r4_topology_interaction_readjudication_20260719
```

必须生成`results.jsonl`、`gate.json`、`summary.json`、`metadata.json`、`progress.jsonl`和中文
`curves.svg`，刷新最近结果索引并执行真实像素`visual-qa-redraw`。证据范围仍只限GIFT-64四轮
本地归因，不是正式神经收益、高轮、跨密码泛化、攻击或SOTA。

## 7. 实际结果

E75 profile与E76 gate/results/true checkpoint全部逐项重放。四个player均为不同64位置换，三个
错误P保持GIFT lane映射；所有ridge只使用训练集标准化，同权重correct-P checkpoint AUC与E76
记录值逐位一致。

信息范围对齐的ridge validation AUC：

```text
local r3 only       = 0.507804
corrupted P shift1 = 0.702133
corrupted P shift2 = 0.646722
corrupted P shift3 = 0.610042
true GIFT P        = 0.743496

true - local                 = +0.235692
true - max corrupted ridge  = +0.041363
```

冻结E76 true checkpoint可学习参数，只替换P-layer后的validation AUC：

```text
corrupted P shift1 = 0.705515
corrupted P shift2 = 0.635796
corrupted P shift3 = 0.619667
true GIFT P        = 0.760666

true - max corrupted inference = +0.055151
```

确定性三门、同权重反事实门和全部协议门均通过：

```text
status   = pass
decision = innovation2_gift64_topology_interaction_gate_repaired
training = no
remote   = no
```

这证明E76原`local13 ridge >= 0.60`前置门的信息范围小于神经候选；当ridge获得相同
local/cell/P前驱信息后，真实GIFT P达到`0.743496`并稳定超过三个same-family错误P。同一checkpoint
的反事实也独立确认参数确实依赖正确P，而不是仅靠训练随机性或位置边际。

## 8. 推荐下一步

E78另立计划运行同一三行30轮seed0正式归因：

```text
data/labels/split = E75冻结
models            = independent / true-P / corrupted shift1
input             = r3-only 13维
parameters        = 4795 each
epochs/seed       = 30 / 0
anchor            = E77 true-P topology-expanded ridge 0.743496
```

正式门必须要求true-P绝对AUC、相对独立/错误P、相对公平确定性ridge和train-validation gap同时通过；
失败则关闭GIFT路线，成功才允许相同协议seed1确认。不得增加模型容量或选择更容易的错误P。

`curves.svg`已通过`visual-qa-redraw`最终1600x836像素检查，无文字重叠、裁剪、缺字、图例冲突、
阈值歧义或不可读内容。
