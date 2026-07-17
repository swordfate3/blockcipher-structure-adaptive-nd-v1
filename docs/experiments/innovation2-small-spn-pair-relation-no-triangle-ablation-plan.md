# 创新2 E40：SPN-PRR no-triangle路径归因消融计划

日期：2026-07-18

状态：已完成 / hold / pair-state保留，triangle路径贡献未隔离

## 1. 研究问题

E39的SPN-PRR在扩展16-bit合成SPN benchmark上通过ID门和公平错误P-layer归因门，但当前
同时改变了pair表示与triangle路径处理器。E40只回答一个问题：

```text
E39相对GraphGPS/CETT的组外收益，是否需要显式组合 i->k->j 路径，
还是同样的16x16 pair表示配合逐pair局部MLP已经足够？
```

## 2. 冻结来源与锚点

```text
data/split/selected = 与E37、E38、E39完全相同
E39 source run      = i2_small_spn_pair_relation_reasoner_seed0_seed1_20260718
E39 topology gate   = i2_small_spn_pair_relation_fair_control_seed0_seed1_20260718
E39 seed0 dual      = 0.6958499108479165
E39 seed1 dual      = 0.7374530800936864
E39 mean dual       = 0.7166514954708014
ID dual baseline    = 0.6843931010265274
```

不得改变labels、selected mask、fit/validation cell split、heldout topology、checkpoint metric、
seed、训练epoch、batch或优化器。

## 3. 唯一模型变量

E40保留E39的全部外层结构：

```text
initial pair state       = 16 x 16 x hidden
pair输入                 = identity/P-edge/inverse-P/same-cell/same-lane/active/mask
S-box/round conditioning = 不变
hidden / rank            = full 64 / 8
shared update steps      = [2,3,4,5]
readout/head             = 不变
absolute IDs             = none
```

唯一变化：

```text
E39: message[i,j] = projection(sum_k left(R[i,k]) * right(R[k,j]))
E40: message[i,j] = projection(left(R[i,j]) * right(R[i,j]))
```

E40消息不读取任何其他pair位置，因此没有`i->k->j`路径组合。左右投影、projection、残差更新
MLP与normalization的形状全部不变，要求E39/E40 full参数量精确相等为`111825`。不得用增加
hidden、rank、block、epoch或新特征补偿移除的路径组合。

## 4. Readiness门

```text
run_id = i2_small_spn_pair_relation_no_triangle_smoke_seed0_20260718
rows   = no-triangle true seed0 / label-shuffle seed0
budget = hidden32 / rank4 / 8 epochs / batch128 / local CPU
```

必须全部通过：

```text
E39 Phase A与Phase B来源裁决匹配
initial pair shape = batch x 16 x 16 x hidden
triangle block count = 0
pair-local block count = 1
step schedule = [2,3,4,5]
E39/E40对应配置参数量精确相等
改变一个输入pair只允许改变同位置pair-local block输出
cell重标号最大logit误差 <= 1e-6
同权重true/corrupted topology最大logit差异 >= 1e-5
两行训练、checkpoint与全部AUC有限
```

smoke AUC不作科学裁决。readiness失败只修实现或协议，不改变冻结门；通过后自动进入full。

## 5. Full同预算消融

```text
run_id = i2_small_spn_pair_relation_no_triangle_seed0_seed1_20260718
rows   = no-triangle true seed0 / true seed1 / label-shuffle seed0
budget = hidden64 / rank8 / dropout0.10 / 40 epochs / batch128
optimizer = AdamW lr1e-3 / weight_decay1e-4
checkpoint = train-topology validation AUC
device = local CPU
```

协议门：

```text
source E39 Phase A = candidate_screened
source E39 Phase B = topology_confirmed
E39与E40逐seed数据/split/budget/parameter count匹配
label-shuffle dual AUC <= 0.60
```

路径归因通过必须同时满足：

```text
E39 seed0 dual > E40 seed0 dual
E39 seed1 dual > E40 seed1 dual
E39 mean dual - E40 mean dual >= 0.03
```

通过裁决：

```text
decision = innovation2_small_spn_pair_relation_triangle_attributed
next     = 设计真实密码标签迁移readiness，不直接启动远程训练
```

这表示当前合成benchmark的增量需要跨pair路径组合。E40不过上述门则裁决：

```text
decision = innovation2_small_spn_pair_relation_triangle_not_isolated
next     = 保留pair-state候选，停止triangle特异性声明，审计query-conditioned NBFNet或结构化P族
```

不得在结果揭示后放宽`0.03`、增加seed、epoch、hidden或改split重新裁决。

## 6. 产物与范围

```text
outputs/local_smoke/i2_small_spn_pair_relation_no_triangle_smoke_seed0_20260718/
outputs/local_diagnostic/i2_small_spn_pair_relation_no_triangle_seed0_seed1_20260718/

results.jsonl
history.csv
gate.json
metadata.json
summary.json
progress.jsonl
checkpoints/
curves.svg
visual_qa_passed.marker
```

E40仍只属于16-bit合成SPN结构条件积分标签预测，不是PRESENT/GIFT真实密码、高轮攻击或
SOTA证据。E40在本地CPU运行，不使用远程GPU。

## 7. 2026-07-18完成记录

Readiness smoke于`06:53:45--06:55:34`完成，共`1m49s`：

```text
status / decision       = pass / innovation2_small_spn_pair_relation_no_triangle_readiness_passed
triangle blocks         = 0
pair-local blocks       = 1
pair count              = 256
parameter count         = 33001，与triangle smoke精确一致
off-pair influence      = 0.0
cell relabel error      = 1.1920928955078125e-07
true/corrupted logit Δ  = 0.006246432662010193
```

Full固定三行矩阵于`06:56:17--07:41:05`完成，共`44m48s`。两颗真实标签seed：

```text
seed0 unseen-S / unseen-P / dual = 0.8883638431 / 0.7114663662 / 0.7097153611
seed1 unseen-S / unseen-P / dual = 0.9083361424 / 0.6889813237 / 0.6951612408
mean  unseen-S / unseen-P / dual = 0.8983499927 / 0.7002238449 / 0.7024383010
label-shuffle dual               = 0.5132262145
parameter count                  = 111825，与E39 triangle精确一致
```

冻结路径归因比较：

```text
triangle mean dual               = 0.7166514955
no-triangle mean dual            = 0.7024383010
triangle - no-triangle mean      = +0.0142131945
seed0 triangle - no-triangle     = -0.0138654502
seed1 triangle - no-triangle     = +0.0422918393
no-triangle mean dual - ID       = +0.0180451999
```

label-shuffle协议门正常，但triangle没有逐seed领先，均值差也未达到预注册`0.03`。最终裁决：

```text
status   = hold
decision = innovation2_small_spn_pair_relation_triangle_not_isolated
```

这不是SPN-PRR整体失败。E39的真实拓扑归因仍成立，E40说明同样的pair-state配合逐pair局部更新
已经得到相近组外表现；因此保留“pair表示是当前最强方向”，停止“收益主要来自triangle路径
组合”的特异性声明。不得用增加hidden、epoch、seed或放宽阈值重开裁决。

最终SVG按`visual-qa-redraw`渲染为`1824x966`像素检查，标题、图例、数值、基线、裁决文字和
导出边界均无重叠、裁切或歧义，已记录`visual_qa_passed.marker`。

下一步不直接迁移真实密码或启动远程GPU。先做E41候选排序：比较query-conditioned NBFNet
与结构化P-layer数据族是否能提出比当前pair-state更明确、单变量且可公平归因的增量；若
没有，则优先把pair-state方法整理为真实密码标签迁移readiness，而不是继续枚举网络名称。
