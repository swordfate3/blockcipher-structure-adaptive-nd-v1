# 创新2 E99：PRESENT九轮ATM支撑组件互斥PU神经排序本地门

日期：2026-07-20

状态：预注册后暂停 / E99训练前发现候选旋转轨道跨折 / 等待E98-C

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

## 2. 冻结来源与语义

```text
source run  = i2_present_r9_atm_support_component_pu_readiness_20260720
source gate = pass / innovation2_present_r9_atm_support_component_pu_ready
gate sha256 = 2f3f3d0cce46d3e786a39899ed87949eddb6c614deb52e16c8aaca623a5c0cb9
rounds      = 9
cipher      = PRESENT round function
key model   = independent 64-bit round keys
positive    = corrected canonical 468-relation independent basis
unlabeled   = synchronous 64-bit rotations after known-positive and support filters
```

它不是PRESENT-80密钥调度数据，也不是独立论文复现。每个pool只有一条公开已知正关系；其余候选的
真实密码学性质未知，因此训练目标是positive-unlabeled ranking，不是严格正负分类。

## 3. 六折协议

六个E98-B支撑组件组逐一作为test group，其余五组作为train positives。每个fold重新构造候选：

- train pool候选不得含test support坐标；
- test pool候选不得含train support坐标；
- 所有候选排除全部470个公开已知positive；
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
