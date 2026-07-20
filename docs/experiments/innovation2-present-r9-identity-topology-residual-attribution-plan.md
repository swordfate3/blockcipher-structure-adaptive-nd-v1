# 创新2 E100：PRESENT九轮坐标身份保持拓扑残差归因

日期：2026-07-20

状态：预注册 / 待实现与本地执行

## 1. 研究问题

E99在E98-C修复后的九轮PU六折上得到接近满分的`coordinate_deepsets`，但单独的
`present_topology_set`明显落后。后者在bit维度做共享消息和全局池化，丢失了前者最有效的可学习
绝对bit身份。因此E99不能回答真实P-layer在强坐标模型上是否还有残差价值。E100只回答：

```text
保留E99坐标身份主干后，加入真实PRESENT P-layer消息残差，
能否在两seed六折上稳定提高仍有空间的Recall@1和MRR，
且提升是否超过同参数、同循环结构的错误P-layer残差？
```

## 2. 冻结来源和数据

```text
source run     = i2_present_r9_atm_support_component_pu_neural_ranking_seed0_seed1_20260720
source status  = hold / innovation2_present_r9_pu_generic_neural_signal_only
source sha256  = a303939a7749452cbf92e4d17b12bec1677b74f20675093877a8907497adab6c
data gate      = E98-C pass / support + rotation-orbit double-disjoint
positive bank  = corrected 468-relation independent basis
folds          = 6 groups x 78 known positives
train/test all-relation overlap = 0
minimum train/test unlabeled    = 55 / 51
candidate semantics             = unlabeled, not negative
```

仍是公开ATM在独立64-bit轮密钥下的九轮关系语料，不是PRESENT-80密钥调度、独立论文来源、严格负类、
区分器或攻击。

## 3. 精简同协议矩阵

```text
coordinate_anchor
  = E99 CoordinateDeepSets原结构，重跑同一六折/seed/预算

identity_true_p_residual
  = 完整coordinate identity主干
  + 固定真实PRESENT P-layer与nibble消息残差

identity_wrong_p_residual
  = 与true-P完全相同的可训练参数和运算
  + 仅把P映射替换为固定共轭wrong-P
```

wrong-P定义为`Q o P o Q^-1`，其中`Q(i)=(i+1) mod 64`。它与真实P具有相同循环结构和边数，但
bit连接错误。true/wrong使用相同类、相同参数量、相同参数初始化、相同batch排列；固定P映射只存为
非训练buffer。

身份主干先编码每个`(u,v)`的输入/输出bit嵌入。拓扑路径从`self、P-neighbor、nibble mean`产生
coordinate residual，经零初始化可学习scale加入身份coordinate表示，再按关系集合池化。该设计保证
模型不会为了使用拓扑而先丢掉E99已确认的绝对bit身份。

## 4. 固定训练协议

```text
seeds        = 0, 1
folds/seed   = 6
epochs       = 40
batch pools  = 32
optimizer    = Adam
learning rate= 0.002
weight decay = 0.0001
loss         = listwise cross entropy
checkpoint   = final epoch, no test-selected checkpoint
device       = local CPU
```

所有行使用相同完整候选池、排序tie-break、参数初始化seed和batch顺序。报告参数量、训练时间、
Recall@1、Recall@5、MRR、Top-5 enrichment和每折指标。

## 5. 冻结锚点与裁决门

E99坐标锚点为：

```text
seed0 Recall@1=0.903846154 Recall@5=0.995726496 MRR=0.945690883
seed1 Recall@1=0.888888889 Recall@5=0.993589744 MRR=0.935543854
```

E100重跑的`coordinate_anchor`必须先复现：每seed Recall@1和MRR与E99绝对差均不超过`0.03`，
Recall@5绝对差不超过`0.01`。

`true_p_residual_attributed`必须同时满足：

```text
两seed分别相对本次coordinate_anchor：
  Recall@1 >= +0.020
  MRR      >= +0.010
  Recall@5 >= -0.005

两seed分别相对paired wrong-P：
  Recall@1 >= +0.010
  MRR      >= +0.005

每seed最差fold Recall@5 >= 0.95；
true-P两seed的Recall@1提升方向一致；
true-P和wrong-P参数量完全相等。
```

通过只允许设计独立来源/新关系集确认，不直接启动远程规模，不宣称九轮区分器或PRESENT-80攻击。

`wrong_p_or_capacity_only`：true-P提高坐标锚点，但未稳定超过wrong-P。保留容量/正则化观察，停止真实
拓扑归因，远程关闭。

`identity_anchor_remains_best`：true-P未稳定提高坐标锚点。保留E99“九轮通用坐标关系识别”结果，
停止当前PRESENT拓扑分支，不增加epoch、宽度或远程样本。

`protocol_invalid`：E99/E98-C来源、fold、候选、配对初始化、参数相等、wrong-P循环结构、最终权重或
指标协议漂移。只修协议。

## 6. 执行与计划产物

```text
run_id = i2_present_r9_identity_topology_residual_attribution_seed0_seed1_20260720
output = outputs/local_diagnostic/i2_present_r9_identity_topology_residual_attribution_seed0_seed1_20260720/
```

产物包括`results.jsonl`、`gate.json`、`summary.json`、`metadata.json`、`progress.jsonl`、
`fold_metrics.csv`、`history.csv`、`curves.svg`和`visual_qa_passed.marker`。完成后更新本文件正式结果、
刷新最近结果索引，并明确独立确认、停止拓扑分支或协议修复三者中的唯一下一动作。

## 7. 正式结果

待执行。
