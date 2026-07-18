# 创新2 E93：跨SPN神经结构证据与边界综合计划

日期：2026-07-19

状态：已完成 / `pass` / 当前benchmark网络枚举停止

## 1. 研究问题

E80已经确认PRESENT/GIFT双密码r3-only方法；E84、E86和E87-E92又分别给出SKINNY残差、共享参数、
RECTANGLE严格标签、原算子和row-typed新结构的边界。E93只回答：

```text
创新2当前哪些神经结构拥有正式证据、哪些仅有机制/诊断价值、哪些路线必须关闭；
下一次实验预算应投向新网络，还是投向新的sound标签/独立机制？
```

E93不训练模型、不改变历史裁决。

## 2. 冻结来源

| 分支 | 来源 | 预期边界 |
|---|---|---|
| PRESENT/GIFT独立r3-only方法 | E80 | pass，双密码双seed正式方法证据 |
| SKINNY ridge残差 | E84 | hold，真实拓扑残差未超过控制 |
| PRESENT/GIFT共享参数 | E86 | hold，GIFT质量退化超过门 |
| RECTANGLE严格标签 | E88 | pass，192结构sound标签 |
| RECTANGLE原r3-only 30轮 | E90 | hold，相对错误P少0.000354 |
| RECTANGLE row表示机制 | E91 | pass，无训练typed ridge机制 |
| RECTANGLE Row-Typed算子 | E92 | hold，正确row未超过无类型/错误row门 |

每个来源读取`gate.json`并记录SHA-256。run id、status、decision或内部门不匹配时E93为协议失败。

## 3. 排名语义

输出结构按证据强度分为：

```text
formal_confirmed = 多seed、真实密码、严格标签、同参数错误拓扑控制通过
mechanism_only   = 确定性或短训练证据支持机制，但正式神经门未通过
closed           = 预注册控制或质量门失败，不得机械扩展
deferred         = 无当前标签/机制证据，不分配训练预算
```

不得把高AUC但控制margin失败的RECTANGLE E90排在PRESENT/GIFT正式方法之前，也不得把E91确定性
ridge写成神经网络收益。

## 4. 裁决与下一预算门

如果E80正式方法未通过，整体`hold`并修复双密码证据。如果任一来源协议不匹配，整体`fail`。

若E80通过、第三密码神经正式门仍未通过：

```text
status   = pass
decision = innovation2_architecture_boundary_confirmed_third_spn_neural_not_confirmed
```

下一步：停止PRESENT/GIFT同benchmark枚举、SKINNY残差、共享参数和RECTANGLE row调参。只有以下任一
条件出现才重新开放训练：

1. 新真实密码通过同级sound、structure-disjoint、抗边际捷径标签门；
2. 新输出任务在文献与确定性审计中证明现有算子无法表达的独立机制；
3. 新架构在无训练同容量控制上先产生至少`0.03`的真实拓扑或类型margin。

## 5. 产物

```text
run_id = i2_neural_architecture_boundary_synthesis_20260719
output = outputs/local_audits/i2_neural_architecture_boundary_synthesis_20260719

results.jsonl
architecture_ranking.csv
source_hashes.json
gate.json
summary.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

完成后刷新最近结果索引并执行`visual-qa-redraw`。E93不构成新神经收益、高轮区分器、攻击或SOTA。

## 6. 实际结果与裁决

E80/E84/E86/E88/E90/E91/E92的run id、status、decision、SHA-256和对应内部门全部重放通过。
最终机器可读排名为：

| 排名 | 路线 | 证据等级 | 裁决 |
|---:|---|---|---|
| 1 | PRESENT/GIFT分别训练的r3-only Profile Operator | formal_confirmed | 保留为创新2正式神经方法 |
| 2 | RECTANGLE原r3-only算子 | mechanism_only | 30轮贴线诊断，不运行seed1 |
| 3 | RECTANGLE row-typed确定性表示 | mechanism_only | 保留解释，不是神经收益 |
| 4 | RECTANGLE Row-Typed Shift Operator | closed | 不运行30轮、不加row embedding |
| 5 | PRESENT/GIFT共享参数算子 | closed | 保留两套独立模型 |
| 6 | SKINNY true-ridge residual | closed | 只保留严格标签和确定性ridge |
| 7 | Transformer/GraphGPS/NBFNet通用变体 | deferred | 无新标签/机制门，不分配训练预算 |

关键控制margin：

```text
formal PRESENT mean true-wrong = +0.096944
formal GIFT mean true-wrong    = +0.132414

RECTANGLE原算子 true-wrong     = +0.029646
RECTANGLE row机制 true-wrongrow= +0.017224
RECTANGLE row神经 true-wrongrow= +0.006244
共享参数 GIFT-anchor           = -0.053590
SKINNY residual true-independent= -0.000457
```

正式裁决：

```text
status   = pass
decision = innovation2_architecture_boundary_confirmed_third_spn_neural_not_confirmed
training = no
remote   = no
```

创新2当前正式结论是同一r3-only归纳偏置在PRESENT/GIFT两套严格标签上分别训练、双seed均超过同参数
控制。RECTANGLE已具备第三密码sound标签，但正式神经拓扑门未确认；高AUC或确定性机制不能替代该门。

## 7. 推荐下一步

停止当前unit-profile benchmark上的网络名称枚举。只有以下任一前置证据出现才重新开放训练：

```text
new sound label family
or independent output-task mechanism
or same-capacity pre-neural true-topology margin >= +0.03
```

下一轮应先做创新2新标签/输出任务排序，优先检查比unit-output balance更有结构信息、但仍可严格证明
正类和构造具体负类的任务。不得直接把Transformer、GraphGPS、NBFNet、attention或更大hidden加入
现有数据重跑。

`curves.svg`已通过`visual-qa-redraw`最终2156x1084像素检查，无文字重叠、裁剪、遮挡、缺字、
数值误导、跨密码AUC误读或不可读内容。
