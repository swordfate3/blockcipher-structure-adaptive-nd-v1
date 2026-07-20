# 创新2 E99：PRESENT九轮ATM支撑组件互斥PU神经排序本地门

日期：2026-07-20

状态：预注册修订 / E98-C通过 / 待执行本地训练

## 1. 研究问题

E98-B已证明冻结公开ATM九轮语料可构成468个已知正例、六组各78条、关系/组件/坐标三层互斥的
positive-unlabeled排序数据门。E99只回答：

```text
在不改变正例、未标注候选、六折分组和排序指标的前提下，
神经关系编码能否稳定超过最强确定性位置捷径，
且PRESENT真实P-layer消息传递能否超过不使用密码拓扑的同任务控制？
```

本实验不预测具体积分数值，不把unlabeled当负例，不使用binary accuracy/AUC。

训练前附加的全示例精确关系重合检查发现，E98-B按support构造的六折仍会让同一个旋转
unlabeled候选出现在train和test。部分train pool在排除这些重合后候选数降为0。因此禁止按当前
E98-B fold启动训练；先执行E98-C，把候选生成的旋转轨道也纳入不可拆组件。E98-C若不通过，本计划
关闭；若通过，必须在执行前更新本节来源hash和fold语义。

E98-C已通过：共享坐标与同旋转轨道联合组件仍能形成`6×78`，train/test全部关系重合为0，最小
train/test候选为`55/51`。本计划现在冻结使用E98-C的联合组件fold，不再使用E98-B support-only fold。

## 2. 冻结来源与语义

```text
source run  = i2_present_r9_atm_support_rotation_orbit_pu_readiness_20260720
source gate = pass / innovation2_present_r9_atm_support_orbit_pu_ready
gate sha256 = ebebd137a90c53ea9a45c0f3af8a30b02803d9f1e395f38e4d822bbd31523568
rounds      = 9
cipher      = PRESENT round function
key model   = independent 64-bit round keys
positive    = corrected canonical 468-relation independent basis
unlabeled   = synchronous 64-bit rotations after known-positive and support filters
```

它不是PRESENT-80密钥调度数据，也不是独立论文复现。每个pool只有一条公开已知正关系；其余候选的
真实密码学性质未知，因此训练目标是positive-unlabeled ranking，不是严格正负分类。

## 3. 六折协议

六个E98-C支撑+旋转轨道联合组件组逐一作为test group，其余五组作为train positives。每个fold重新构造候选：

- train pool候选不得含test support坐标；
- test pool候选不得含train support坐标；
- 所有候选排除全部470个公开已知positive；
- train/test全部示例的精确relation重合必须为0；
- 每个pool至少31个unlabeled；
- 模型不接收pool位置，positive固定排在序列第0项只用于loss，排名同分按relation hash裁决。

不从test fold选择checkpoint。每个模型固定训练40 epochs并使用最后一轮权重。六折指标聚合后再跨
seed裁决。

## 4. 精简同协议矩阵

```text
absolute_position                 确定性最强E98-B捷径，不训练
summary_mlp                       只看关系项数、重量和绝对位置等统计摘要
coordinate_deepsets               读取每个(u,v)的128-bit身份，集合池化，不使用P-layer
present_topology_set              固定真实P-layer和nibble邻接消息传递，再做关系集合池化
present_topology_set_label_shuffle 同一拓扑模型/参数/预算，训练target改为确定性随机unlabeled
```

`coordinate_deepsets`是“神经网络是否只记坐标身份”的控制；`summary_mlp`检查提升能否由低维位置统计
解释；`label_shuffle`检查有限语料记忆/偶然排序。

## 5. 固定训练预算

```text
seeds        = 0, 1
folds/seed   = 6
epochs       = 40
batch pools  = 32
optimizer    = Adam
learning rate= 0.002
weight decay = 0.0001
loss         = listwise cross entropy over one known positive plus unlabeled candidates
device       = local CPU (readiness gate)
checkpoint   = final epoch, no test-selected best checkpoint
```

所有神经行使用同一train/test pool、训练轮数、batch顺序种子和排序计算。记录参数量、每fold训练时间、
loss、Recall@1、Recall@5、MRR、Top-5 enrichment和最差fold。

## 6. 冻结裁决门

E98-B最强非随机锚点是：

```text
absolute_position Recall@5 = 0.132478632
absolute_position MRR      = 0.120060420
```

`advance_remote_design`必须同时满足：

```text
source hash/status/decision和全部pool不变量通过；
present_topology_set两seed均：
  Recall@5 >= absolute_position + 0.05；
  MRR      >= absolute_position + 0.03；
  Recall@5 >= summary_mlp + 0.02；
  MRR      >= summary_mlp + 0.01；
  Recall@5 >= coordinate_deepsets + 0.01；
  MRR      >= coordinate_deepsets + 0.005；
  最差fold Recall@5 >= 0.10；
present_topology_set两seed Recall@5差 <= 0.10；
label_shuffle不得达到真实topology行门槛。
```

通过只允许撰写远程规模/外部来源确认计划；本地E99本身仍不是突破、区分器、攻击、SOTA或论文级
独立确认。

`hold_generic_neural_signal`：coordinate或summary超过确定性锚点，但真实P-layer不满足拓扑归因。
保持远程关闭，审计/重设计拓扑表示，不机械增加epoch或宽度。

`stop_public_corpus_neural_route`：所有真实神经行均未稳定超过锚点，或最差fold/双seed失败。停止当前
公开九轮语料神经路线，不进行远程放大。

`protocol_invalid`：来源、分组、候选、标签打乱、最后权重或指标协议漂移。只修协议，不解释科学
结果。

## 7. 计划产物

```text
run_id = i2_present_r9_atm_support_component_pu_neural_ranking_seed0_seed1_20260720
output = outputs/local_diagnostic/i2_present_r9_atm_support_component_pu_neural_ranking_seed0_seed1_20260720/
```

产物包括`results.jsonl`、`gate.json`、`summary.json`、`metadata.json`、`progress.jsonl`、
`fold_metrics.csv`、`history.csv`、`curves.svg`和`visual_qa_passed.marker`。完成后更新本文件正式结果与
可执行下一步，并刷新最近结果索引。

## 8. 正式结果

执行时间：2026-07-20。完整执行`2 seeds × 6 folds × 4 trained rows × 40 epochs`，使用最终epoch
权重；所有E98-C来源、候选宽度和泄漏检查通过。

```text
status   = hold
decision = innovation2_present_r9_pu_generic_neural_signal_only
remote   = no
```

六折聚合结果：

| 模型 | seed | 参数量 | Recall@1 | Recall@5 | MRR | 最差折Recall@5 |
|---|---:|---:|---:|---:|---:|---:|
| absolute_position | 0/1 | 0 | 0.049145 | 0.128205 | 0.119001 | 0.076923 |
| summary_mlp | 0 | 1033 | 0.373932 | 0.523504 | 0.448112 | 0.384615 |
| summary_mlp | 1 | 1033 | 0.393162 | 0.570513 | 0.483670 | 0.487179 |
| coordinate_deepsets | 0 | 3937 | 0.903846 | 0.995726 | 0.945691 | 0.987179 |
| coordinate_deepsets | 1 | 3937 | 0.888889 | 0.993590 | 0.935544 | 0.987179 |
| present_topology_set | 0 | 2041 | 0.198718 | 0.585470 | 0.367639 | 0.525641 |
| present_topology_set | 1 | 2041 | 0.202991 | 0.568376 | 0.365750 | 0.538462 |
| topology label shuffle | 0 | 2041 | 0.059829 | 0.143162 | 0.116671 | 0.038462 |
| topology label shuffle | 1 | 2041 | 0.051282 | 0.136752 | 0.107987 | 0.064103 |

坐标集合网在两seed和全部六折几乎满分，远超位置规则、摘要MLP和标签打乱，证明九轮公开正关系
相对旋转unlabeled候选存在强且稳定的可学习坐标结构。因为fold同时按positive support和候选旋转
轨道互斥，这不是相同relation跨train/test的记忆泄漏。

但当前`present_topology_set`没有保留可学习的绝对bit身份：它虽然明显超过位置规则和标签打乱，
却比`coordinate_deepsets`低约`0.41..0.43 Recall@5`和`0.57..0.58 MRR`，MRR也低于摘要MLP。
所以不能把强结果归因于PRESENT真实P-layer，冻结拓扑advance门失败，远程保持关闭。

产物位于：

```text
outputs/local_diagnostic/i2_present_r9_atm_support_component_pu_neural_ranking_seed0_seed1_20260720/
```

`curves.svg`经`visual-qa-redraw`渲染为2500×1348像素检查，无文字重叠、裁切、缺字、含糊标题、
不可辨曲线或误导坐标轴。

## 9. 推荐下一步

下一问题不是增加样本/epoch，而是验证真实P-layer能否在强坐标身份模型上提供残差增益：

```text
E100: identity-preserving PRESENT topology residual attribution
source/data = E98-C原六折与全部候选不变
anchor      = coordinate_deepsets
candidate   = coordinate identity path + true PRESENT P-layer residual
control     = 同参数coordinate identity path + fixed wrong-P residual
seeds/folds = 0,1 / 6
epochs      = 40，最终epoch，不用test选权重
device      = local CPU
metrics     = Recall@1、Recall@5、MRR
```

只改变残差中的P映射；true-P和wrong-P使用配对初始化、相同参数量和batch顺序。由于E99的
Recall@5已接近天花板，主门改用仍有空间的Recall@1/MRR：true-P两seed均需相对coordinate anchor
`Recall@1 +0.02、MRR +0.01`，同时相对wrong-P `Recall@1 +0.01、MRR +0.005`，且Recall@5下降
不得超过0.005、最差折不得崩溃。通过才允许设计独立来源确认；不通过则保留“九轮通用坐标关系
识别”结果，停止当前PRESENT拓扑归因和远程放大。

禁止路线：直接远程扩大、继续训练当前丢失绝对身份的拓扑网、把unlabeled称为negative、用二分类
accuracy/AUC汇报、或把同公开语料六折结果称为九轮区分器/PRESENT-80攻击/SOTA。
