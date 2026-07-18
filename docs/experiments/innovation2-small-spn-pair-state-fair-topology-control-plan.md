# 创新2 E41：no-triangle pair-state公平拓扑控制计划

日期：2026-07-18

状态：已完成 / pass / 局部pair-state真实拓扑贡献确认

## 1. 为什么必须先做这个控制

E39证明triangle版SPN-PRR真实拓扑优于公平错误P-layer；E40又发现同参数no-triangle局部
pair更新与triangle的mean dual只差`0.014213`，没有隔离triangle贡献。但E40只训练了
no-triangle真实拓扑，没有训练它自己的公平错误拓扑控制，因此仍无法回答：

```text
接近E39的表现是否来自pair-state对真实P-layer的利用，
还是pair-state/readout中的拓扑无关捷径？
```

直接开发NBFNet会跳过这个归因缺口。E41只补缺失的两行控制。

## 2. 冻结来源

```text
data/split/selected = 与E37--E40完全相同
true source run     = i2_small_spn_pair_relation_no_triangle_seed0_seed1_20260718
source decision     = innovation2_small_spn_pair_relation_triangle_not_isolated
true seed0 dual     = 0.7097153610915079
true seed1 dual     = 0.6951612408333590
true mean dual      = 0.7024383009624334
label-shuffle dual  = 0.5132262144627522
```

公平错误拓扑沿用E35b、E39已经验证的控制：对每个variant自身P-layer组合固定
destination-cell rotation，不跨variant替换，不把heldout P-family换成train-seen family。

## 3. 固定训练矩阵

```text
run_id = i2_small_spn_pair_relation_no_triangle_fair_control_seed0_seed1_20260718
rows   = no-triangle fair-corrupted P seed0 / seed1
model  = 16x16 pair-state + shared pair-local update
budget = hidden64 / rank8 / dropout0.10 / 40 epochs / batch128
optimizer = AdamW lr1e-3 / weight_decay1e-4
checkpoint = train-topology validation AUC
device = local CPU
```

不得重训E40 true行，不增加label-shuffle重复行，不改变参数、数据、split、checkpoint或seed。

## 4. 协议门

必须全部通过：

```text
E40 source run/decision匹配且label-shuffle dual <= 0.60
E40 true rows与E41 control rows均为seed0/1
control topology_mode = corrupted
processor_mode = local
triangle block count = 0 / local block count = 1
E40 true与E41 control参数量精确一致为111825
off-pair influence = 0.0
cell重标号误差 <= 1e-6
fair-corrupted heldout与true/corrupted train拓扑均不相交
全部指标有限且训练/checkpoint完成
```

## 5. 裁决门

pair-state真实拓扑贡献通过必须同时满足：

```text
true seed0 dual > fair-corrupted seed0 dual
true seed1 dual > fair-corrupted seed1 dual
true mean dual - fair-corrupted mean dual >= 0.03
```

通过：

```text
decision = innovation2_small_spn_pair_state_topology_confirmed
next     = 设计真实密码输出性质标签迁移readiness，triangle/local作为同pair-state两种处理器
```

不过门：

```text
decision = innovation2_small_spn_pair_state_topology_not_attributed
next     = pair-local只作容量控制；真实密码迁移readiness保留E39 triangle与ID基线
```

两种结果都不直接启动远程GPU，也不通过放宽`0.03`或增加seed重新裁决。NBFNet继续暂缓：
E39已有通过公平控制的路径候选，E40没有证明新增路径结构是当前首要瓶颈。

## 6. 产物与证据范围

```text
outputs/local_diagnostic/i2_small_spn_pair_relation_no_triangle_fair_control_seed0_seed1_20260718/

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

E41仍只属于16-bit合成SPN拓扑归因，不是PRESENT/GIFT真实密码、高轮攻击或SOTA证据。

## 7. 2026-07-18完成记录

两行控制于`07:52:14--08:22:23`完成，共`30m09s`：

```text
fair-corrupted seed0 unseen-S / unseen-P / dual = 0.9084894477 / 0.6417158831 / 0.6529989534
fair-corrupted seed1 unseen-S / unseen-P / dual = 0.8889612053 / 0.5773125771 / 0.5145421879
fair-corrupted mean  unseen-S / unseen-P / dual = 0.8987253265 / 0.6095142301 / 0.5837705706
```

与E40冻结true行比较：

```text
true mean dual                  = 0.7024383010
fair-corrupted mean dual        = 0.5837705706
true-control mean               = +0.1186677304
seed0 true-control              = +0.0567164077
seed1 true-control              = +0.1806190530
```

两颗seed均领先且均值差超过`0.03`，全部协议门和裁决门通过：

```text
status   = pass
decision = innovation2_small_spn_pair_state_topology_confirmed
```

因此E39--E41的联合结论是：SPN专用有向pair-state表示确实利用正确P-layer；triangle版预测
均值略高，但E40没有证明triangle路径组合是必要增量。创新方法应表述为结构条件化的有向
pair-state表示和SPN拓扑联合编码，不得把通用pair network或triangle算子本身声称为首创。

最终SVG按`visual-qa-redraw`渲染为`1824x966`像素复核，标题、图例、数值、基线、裁决文字和
导出边界无重叠、裁切或歧义，已记录`visual_qa_passed.marker`。

下一步是E42真实密码迁移readiness审计：核对现有PRESENT输出性质标签的语义、可用结构数、
key采样、finite-key噪声校正和64-bit pair-state内存预算。先冻结同数据的ID边际、triangle、
local和fair-corrupted必要控制；readiness通过前不训练真实密码模型、不启动远程GPU。

E42现已完成：64-bit pair-local/triangle共24个前向/反向配置全部通过，但PRESENT/SKINNY
四个真实标签族均未满足训练门，裁决为`innovation2_real_spn_pair_state_label_bank_not_ready`。
因此E41方法证据保留，真实密码训练继续关闭；下一步转确定性标签atlas提供者契约，不再
枚举神经网络结构。
