# 创新2 E38：扩展拓扑上的GraphGPS/CETT分阶段归因计划

日期：2026-07-18

状态：Phase A已完成 / hold / GraphGPS与CETT均未过ID门，Phase B不运行

## 1. 研究问题

E37把独立训练P-layer从3个扩到12个，并在不读取heldout标签选cell的前提下通过全部
标签宽度、S×P交互、组外ID边际与公平控制契约。E33-R的cell-equivariant GraphGPS和
E35b的CETT此前都在3-P训练benchmark失败；现在只重新回答一个问题：

```text
在12个独立训练P-layer下，GraphGPS或CETT能否稳定超过train-only ID边际？
如果可以，CETT的显式edge token是否优于GraphGPS的neighbor gather？
```

不同时引入新特征、标签、split、优化器或网络家族。

## 2. 冻结来源与数据

```text
source run    = i2_small_spn_expanded_topology_4s16p_256keys_20260718
source gate   = innovation2_small_spn_expanded_topology_benchmark_ready
label shape   = 64 x 4 x 14 x 64
selected      = 320 base cells，规则9 <= train positives <= 27

train         = S0..S2 x P0..P11  = 36 topologies
unseen-S      = S3 x P0..P11      = 12
unseen-P      = S0..S2 x P12..P15 = 12
dual          = S3 x P12..P15      = 4
```

320个base cell使用固定`CELL_SPLIT_SEED=38001`打乱，80%只用于train-domain fit，20%只用于
train-domain checkpoint validation。四个heldout topology split不参与训练、cell选择、
checkpoint、early selection或阈值修改，只在训练完成后评价。

E37冻结ID锚点：

```text
unseen-S strongest marginal AUC = 0.6881980369450377
unseen-P strongest marginal AUC = 0.6487534140097411
dual strongest marginal AUC     = 0.6843931010265274
```

## 3. 候选网络

### N1：cell-equivariant GraphGPS

沿用E33-R：16个bit node，仅使用cell内lane role，不使用绝对bit/nibble/variant ID；
P-layer通过incoming/outgoing neighbor gather进入局部消息，另含全局attention。

### N2：Cipher Edge-Token Transformer（CETT）

沿用E35b：16 node token + 16 directed P-edge token + 4 S-box relation token + 1 mask
query，共37个token。P-layer边是显式对象，可发生edge-edge与query-edge attention。

两者读取完全相同的S-box truth table、P-layer、round index、active-bit mask和output mask；
都没有cipher ID或variant embedding。

## 4. 冻结预算

```text
hidden dimension        = 64
GraphGPS blocks         = 3
CETT Transformer layers = 3
attention heads         = 4
FFN dimension           = 128
dropout                 = 0.10
optimizer               = AdamW
learning rate           = 1e-3
weight decay            = 1e-4
batch size              = 128
epochs                  = 40
checkpoint              = train-topology validation AUC
seeds                   = 0,1
device                  = local CPU
```

readiness smoke使用hidden32、2层、8 epochs，只验证数据/forward/checkpoint/metric流程，
不参与科学裁决。

## 5. 分阶段矩阵

### Phase A：候选筛选

| 行 | seed | 作用 |
|---|---:|---|
| GraphGPS cell-equivariant + true P | 0,1 | neighbor-gather候选 |
| CETT + true P | 0,1 | edge-token候选 |
| CETT + label shuffle | 0 | 流程负控制 |

共5行。smoke只运行三个seed0行。

Phase A每个候选独立过门：

```text
label-shuffle dual AUC <= 0.60
两颗seed的unseen-S AUC均 > 0.6881980369450377
两颗seed的unseen-P AUC均 > 0.6487534140097411
两颗seed的dual AUC均 > 0.6843931010265274
候选mean dual AUC >= 0.7143931010265274
```

若两个候选都不过门：

```text
decision = innovation2_small_spn_expanded_neural_screen_not_ready
```

停止，不运行错误拓扑控制，不加层、epoch、seed或远程规模。

若至少一个候选过门：

```text
decision = innovation2_small_spn_expanded_neural_candidate_screened
```

按mean dual AUC选择一个最强候选进入Phase B；同分时优先参数更少者。CETT只有在mean dual
比GraphGPS至少高`0.01`时，才可写成“显式edge-token有增量”，否则只视为并列候选。

### Phase B：真实拓扑归因

只对Phase A选中的一个候选运行：

| 行 | seed | 作用 |
|---|---:|---|
| selected candidate + fair-corrupted P | 0,1 | 拓扑归因控制 |

控制对每个variant自身P-layer组合固定destination-cell rotation，不跨variant roll，不改变
label、split、cell、S-box、mask或训练预算。heldout corrupted P不得等于任何true或
corrupted train P。

最终保留门：

```text
true mean dual >= fair-corrupted mean dual + 0.03
true两seed dual均 > 对应seed fair-corrupted dual
```

通过才允许把收益归因于真实P-layer，并决定下一种关系算子；否则停止当前候选。

## 6. 产物与执行路径

```text
smoke run = i2_small_spn_expanded_neural_screen_smoke_seed0_20260718
full run  = i2_small_spn_expanded_neural_screen_seed0_seed1_20260718

outputs/local_smoke/<smoke run>/
outputs/local_diagnostic/<full run>/

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

smoke通过后自动继续本地Phase A full。Phase B是否运行由Phase A gate自动决定，不需要用户
重复审批。本实验是16-bit合成SPN上的结构条件输出平衡标签预测，不是积分multiset二分类、
真实密码高轮结果、攻击成功或SOTA比较。任何结果都不开放远程GPU。

## 7. 实际运行

readiness smoke：

```text
run_id   = i2_small_spn_expanded_neural_screen_smoke_seed0_20260718
status   = pass
decision = innovation2_small_spn_expanded_neural_screen_readiness_passed

GraphGPS cell重标号最大logit误差 = 1.4901161193847656e-07
CETT cell重标号最大logit误差     = 7.450580596923828e-08
CETT token count                 = 37
```

来源、64-variant shape、320个selected cell、36/12/12/4 split、fit/validation互斥、
12个独立train P、两模型forward/checkpoint/metric、公平控制身份和label-shuffle流程全部通过。

Phase A full：

```text
run_id   = i2_small_spn_expanded_neural_screen_seed0_seed1_20260718
status   = hold
decision = innovation2_small_spn_expanded_neural_screen_not_ready
```

| 模型 | seed | train AUC | 未见S AUC | 未见P AUC | dual AUC | best epoch |
|---|---:|---:|---:|---:|---:|---:|
| cell-equivariant GraphGPS | 0 | 0.986210 | 0.885221 | 0.666798 | 0.645458 | 29 |
| cell-equivariant GraphGPS | 1 | 0.988194 | 0.908880 | 0.681349 | 0.637906 | 29 |
| CETT | 0 | 0.945925 | 0.902657 | 0.593098 | 0.579995 | 34 |
| CETT | 1 | 0.946287 | 0.901728 | 0.708995 | 0.679318 | 29 |
| CETT label shuffle | 0 | 0.483657 | 0.492296 | 0.486858 | 0.506077 | 2 |

两seed均值：

```text
GraphGPS unseen-S / unseen-P / dual = 0.897050 / 0.674073 / 0.641682
CETT     unseen-S / unseen-P / dual = 0.902193 / 0.651047 / 0.629657

dual ID baseline                   = 0.684393
CETT dual - GraphGPS dual           = -0.012025
```

GraphGPS两seed的dual都低于ID锚点。CETT seed1达到`0.679318`，但仍低于`0.684393`，
seed0仅`0.579995`，不稳定；CETT均值还比GraphGPS低`0.012025`。label-shuffle dual为
`0.506077`，流程负控制正常。两个候选都没有资格进入Phase B，因此预注册的fair-corrupted
P两seed训练不运行。

最终SVG按`visual-qa-redraw`渲染到1900像素宽检查：模型/split柱、seed柱、ID与advance门线、
数值标签、中文标题、图例和裁决均无重叠、裁切或歧义，`visual_qa_passed.marker`已记录。

## 8. 解释与下一步

扩展独立训练拓扑解决了benchmark欠定问题，却没有自动解决模型的P-layer组合外推。
两种模型在未见S-box上都达到约`0.90`，说明它们能学习S-box truth-table相关规律；但dual
明显下降，说明当前neighbor gather或无显式关系位置的edge token不能稳定组合“新S-box +
新P-layer”。这不是数据无拓扑信号：E37已经直接证明selected cell的P敏感和S×P交互宽。

下一步先做E39关系表达审计与最小实现，唯一结构变量改为有向bit-pair关系状态：使用
`16 x 16`关系张量显式表示P-edge、同S-box cell关系和query角色，通过共享triangle update
组合`i -> k -> j`路径，按实际轮数执行2--5步。这对应2-WL/PPGN/Neural Bellman-Ford式
路径组合，直接测试当前模型缺失的多边组合能力。E39仍使用E37数据、同预算ID锚点和
label-shuffle；先做forward/等变/参数与小smoke，不直接扩大epoch或使用远程GPU。
