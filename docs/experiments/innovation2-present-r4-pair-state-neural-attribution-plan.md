# 创新2 E44：PRESENT r4严格平衡标签的64-bit pair-state神经归因计划

日期：2026-07-18

状态：完成 / hold / seed1与远程规模关闭

## 1. 研究问题

E43得到首个语义严格、structure-disjoint且一元边际平衡的真实PRESENT标签benchmark。
E44回答两个单变量问题：

1. 64-bit有向pair-state能否从`active structure × output mask × P-layer`交互预测全称平衡
   证书/反例标签，而不是停留在AUC=0.5的一元边际；
2. 若能，收益是否依赖正确P-layer，还是同容量网络在公平错误拓扑上也能取得同样结果。

E44不是新的标签实验，不允许改变E43的positive/negative/unknown、checkerboard、split或
输出mask。

## 2. 冻结来源

```text
source run = i2_present_r4_universal_balance_atlas_20260718
source file= outputs/local_audits/i2_present_r4_universal_balance_atlas_20260718/
             matched_contrast.csv

train       = 800 rows, 400/400, 53 structures
validation  = 236 rows, 118/118, 18 structures
unary AUC   = 0.5
```

启动前必须重新核验source run id、decision、行数、正负类、structure互斥、无重复边和每个
structure/mask内部正负平衡。

## 3. 模型与预算

所有神经行使用同一`SmallSpnPairRelationReasoner`：

```text
state bits       = 64
pair tensor      = batch × 64 × 64 × 16
round categories = 1
round step offset= 4
hidden           = 16
path rank        = 2
dropout          = 0.10
epochs           = 30
batch            = 8
optimizer        = AdamW(lr=1e-3, weight_decay=1e-4)
checkpoint       = best validation AUC
seed             = 0
device           = local CPU
```

矩阵最多四行：

```text
1. train-only unary marginal baseline（冻结AUC=0.5）
2. pair-local true P-layer
3. triangle true P-layer
4. 第2/3行验证AUC较高者的fair-corrupted destination-cell P-layer
```

第4行只改变P-layer，保持processor、参数、seed、训练顺序与预算一致。若第2/3行并列，
优先pair-local以保留更简单模型。

## 4. 门控

流程门：

```text
source E43 decision = innovation2_present_universal_balance_atlas_ready
source train/validation rows与类别计数精确匹配
source structure groups互斥、edge唯一、structure/mask正负平衡
64-bit模型前向/反向、有限logit/gradient通过
pair-local与triangle参数量相同
fair-corrupted P-layer仍为64-bit permutation且与true不同
全部metric有限
```

候选开放门：

```text
best true validation AUC            >= 0.60
best true - unary baseline          >= 0.05
```

拓扑归因门：

```text
best true - same-processor corrupted >= 0.03
```

全部通过：

```text
decision = innovation2_present_pair_state_topology_attributed
next     = 同一四行矩阵seed1确认
```

候选未过AUC门：停止当前r4 pair-state训练，不加epoch/hidden；检查标签是否主要反映高阶ANF
关系而pair-local/triangle无法表达。候选过门但拓扑差不足：保留预测结果但撤回SPN拓扑归因，
下一步审计active/mask set interaction或certificate复杂度捷径，不尝试NBFNet。任何协议失败
只修协议，不解释网络。

## 5. 产物与边界

```text
outputs/local_diagnostic/i2_present_r4_pair_state_neural_attribution_seed0_20260718/

results.jsonl
history.csv
gate.json
summary.json
metadata.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

E44是本地小数据readiness/归因实验，不是高轮积分区分器、远程规模训练或SOTA性能。只有
seed0候选与正确P-layer归因同时过门才允许seed1；禁止直接远程GPU或迁移r5/r7。

## 6. 2026-07-18实际结果

权威run：

```text
i2_present_r4_pair_state_neural_attribution_seed0_20260718
```

source gate、行数、类别、96个structure、300个mask、structure互斥、edge唯一和每个
structure/mask正负平衡全部重新核验通过。64-bit模型contract结果：

```text
initial pair tensor               = 8 x 64 x 64 x 16
pair count                        = 4096
pair-local / triangle parameters  = 10725 / 10725
step schedule                     = [4]
logit / loss / gradient finite    = true / true / true
corrupted P-layer                 = distinct 64-bit permutation
```

30轮seed0结果：

| 模型 | 最佳epoch | train AUC | validation AUC |
|---|---:|---:|---:|
| unary marginal | 0 | `0.500000` | `0.500000` |
| pair-local true P | 18 | `0.590031` | `0.549914` |
| triangle true P | 27 | `0.611363` | `0.561979` |
| triangle fair-corrupted P | 23 | `0.579644` | `0.549698` |

triangle是最强真拓扑候选，但裁决差值为：

```text
best true - unary baseline    = +0.061979
best true - fair-corrupted P  = +0.012281
```

它满足`true-unary >= 0.05`，但验证AUC未达到`0.60`，正确P-layer优势也未达到`0.03`。
因此：

```text
status   = hold
decision = innovation2_present_pair_state_candidate_not_ready
seed1    = no
remote   = no
```

结果说明pair-state从严格matched标签中学到少量非一元信号，但当前证据不能将其归因到正确
P-layer，也不能称为可用的真实密码积分预测器。不得通过增加hidden、epoch或远程GPU绕过
冻结门。

## 7. 推荐下一步

下一审判改为E45“certificate-complexity与非拓扑set interaction审计”，不训练新网络。对
E43 resolved/matched标签计算以下train-only确定性特征并在相同validation结构上评估：

```text
1. active-to-mask前向P-layer可达计数（拓扑）
2. 各输出位ANF support size / full-cube candidate count（证书复杂度）
3. mask weight、active bit spread、cell/lane occupancy交互（非拓扑set统计）
4. 去掉或置换P-layer后保持不变的complexity control
```

E45要判断`0.56`信号主要属于哪一类。若certificate-complexity baseline已经显著超过
triangle，下一网络应改为“证书状态序列/集合编码器”，而不是继续关系图；若只有真实P-layer
可达特征有效，再设计query-conditioned Bellman-Ford式处理器。E45完成前NBFNet、seed1、
r5迁移和远程GPU继续关闭。

最终`curves.svg`按`visual-qa-redraw`渲染为`1385×729`像素检查；标题、说明、30轮曲线、
柱图、数值、图例、阈值线、裁决和导出边界无重叠、裁切或缺字，已记录
`visual_qa_passed.marker`。
