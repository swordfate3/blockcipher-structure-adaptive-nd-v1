# 创新2 E39：小状态SPN有向bit-pair路径推理器计划

日期：2026-07-18

状态：已完成 / pass / 真实拓扑与triangle候选均过门，下一步E40 no-triangle消融

## 1. 研究问题

E37扩展benchmark已通过；E38 GraphGPS/CETT均能泛化未见S-box，却不能稳定泛化到双重
未见S-box/P-layer。E39只改变关系处理器：把node/edge token表示替换为`16×16`有向
bit-pair状态，用共享triangle update显式组合`i→k→j`路径。

研究问题：

```text
低秩2-FWL/PPGN式路径组合，能否在不增加参数和训练预算的条件下，
超过E37 train-only ID边际并稳定外推到新S-box + 新P-layer？
```

## 2. 冻结数据与锚点

```text
source       = i2_small_spn_expanded_topology_4s16p_256keys_20260718
labels       = 64 x 4 x 14 x 64
selected     = 320 train-only matched base cells
train        = 36 topologies / 12 independent P-layers
unseen-S     = 12 topologies
unseen-P     = 12 topologies
dual         = 4 topologies
cell split   = CELL_SPLIT_SEED 38001，沿用E38相同fit/validation
```

确定性ID锚点：

```text
unseen-S AUC = 0.6881980369450377
unseen-P AUC = 0.6487534140097411
dual AUC     = 0.6843931010265274
```

历史同预算神经锚点：

```text
GraphGPS dual mean = 0.6416819230939694
CETT dual mean     = 0.6296566195848206
```

不得改变labels、selected mask、split、checkpoint metric或heldout使用边界。

## 3. 冻结模型

名称：`SPN Pair-Relation Reasoner (SPN-PRR)`。

```text
pair states             = 16 x 16
hidden dimension        = smoke 32 / full 64
path rank               = smoke 4 / full 8
processor               = 1个共享triangle block
steps                    = round index对应2、3、4、5步
dropout                  = smoke 0.0 / full 0.10
absolute bit/cell ID     = none
cipher/variant embedding = none
```

输入和更新严格按文献设计说明中的pair feature、低秩矩阵乘法与五路readout实现。唯一结构
增量是pair-state triangle composition；不同时加入basis branch、LapPE、预训练或新损失。

## 4. Readiness门

readiness run：

```text
run_id = i2_small_spn_pair_relation_reasoner_smoke_seed0_20260718
mode   = smoke
rows   = true P seed0 / fair-corrupted P seed0 / label-shuffle seed0
budget = hidden32 / rank4 / 8 epochs / batch128 / local CPU
```

必须全部通过：

```text
E37 source、shape、selected、split与12 train P契约通过
initial pair tensor shape = batch x 16 x 16 x hidden
共享triangle block数量 = 1
step schedule = [2,3,4,5]
参数量 <= 297409
cell重标号最大logit误差 <= 1e-6
同权重true与fair-corrupted fixture最大logit差异 >= 1e-5
fair-corrupted heldout不等于true/corrupted train P
三行训练、checkpoint与全部AUC有限
```

smoke AUC只验证流程，不用于科学裁决。readiness失败只修协议；通过后自动进入Phase A。

## 5. Phase A：同预算筛选

```text
run_id = i2_small_spn_pair_relation_reasoner_seed0_seed1_20260718
rows   = true P seed0 / true P seed1 / label-shuffle seed0

hidden64 / rank8 / shared triangle block / 40 epochs
AdamW lr1e-3 / weight decay1e-4 / batch128
checkpoint = train-topology validation AUC
device = local CPU
```

全部通过才进入Phase B：

```text
label-shuffle dual AUC <= 0.60
true两seed unseen-S均 > 0.6881980369450377
true两seed unseen-P均 > 0.6487534140097411
true两seed dual均 > 0.6843931010265274
true mean dual >= 0.7143931010265274
```

不过门：

```text
decision = innovation2_small_spn_pair_relation_reasoner_not_ready
```

停止，不增加容量、epoch、seed或远程规模。

过门：

```text
decision = innovation2_small_spn_pair_relation_candidate_screened
```

才运行Phase B的fair-corrupted P seed0/1，要求true mean dual超过control至少`0.03`且逐seed
领先，才能写成真实P-layer贡献。

Phase B固定为：

```text
run_id = i2_small_spn_pair_relation_fair_control_seed0_seed1_20260718
source = i2_small_spn_pair_relation_reasoner_seed0_seed1_20260718
rows   = fair-corrupted P seed0 / seed1
budget = 与Phase A true行完全相同
```

通过：

```text
decision = innovation2_small_spn_pair_relation_topology_confirmed
```

只允许下一步做同pair-state、同参数预算的no-triangle消融，确认增量来自路径组合；仍不直接
迁移真实密码或启动远程GPU。

不过true-control门：

```text
decision = innovation2_small_spn_pair_relation_topology_not_attributed
```

停止拓扑收益声明，不增加容量或训练时间。

## 6. 产物与证据范围

```text
outputs/local_smoke/i2_small_spn_pair_relation_reasoner_smoke_seed0_20260718/
outputs/local_diagnostic/i2_small_spn_pair_relation_reasoner_seed0_seed1_20260718/

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

本实验只属于16-bit合成SPN结构条件输出积分标签预测，不是实际密码高轮积分证明、攻击、
PRESENT/GIFT结果或SOTA。任何结果都不开放远程GPU。

## 7. 2026-07-18完成记录

Readiness smoke：

```text
run_id                  = i2_small_spn_pair_relation_reasoner_smoke_seed0_20260718
status / decision       = pass / innovation2_small_spn_pair_relation_readiness_passed
elapsed                 = 2m48s local CPU
parameter count         = 33001
pair count              = 256
step schedule           = [2,3,4,5]
cell relabel error      = 1.1920928955078125e-07
true/corrupted logit Δ  = 0.006170153617858887
```

Phase A固定三行矩阵于`05:17:56--06:03:23`完成，共`45m27s`。两颗真实拓扑seed结果：

```text
seed0 unseen-S / unseen-P / dual = 0.9000399278 / 0.7328710485 / 0.6958499108
seed1 unseen-S / unseen-P / dual = 0.9011855200 / 0.7374720283 / 0.7374530801
mean  unseen-S / unseen-P / dual = 0.9006127239 / 0.7351715384 / 0.7166514955
label-shuffle dual               = 0.5082623355
parameter count                  = 111825
```

同一E37数据与split上的比较：

```text
SPN-PRR mean dual - ID baseline   = +0.0322583944
SPN-PRR mean dual - GraphGPS mean = +0.0749695724
SPN-PRR mean dual - CETT mean     = +0.0869948759
```

所有Phase A门通过，裁决为
`innovation2_small_spn_pair_relation_candidate_screened`。Phase B随后在
`06:07:24--06:37:48`完成，共`30m24s`：

```text
fair-corrupted seed0 dual = 0.6580582915
fair-corrupted seed1 dual = 0.6702225222
fair-corrupted mean dual  = 0.6641404069

seed0 true-control        = +0.0377916194
seed1 true-control        = +0.0672305579
mean  true-control        = +0.0525110886
```

逐seed和均值拓扑归因门全部通过，最终裁决为
`innovation2_small_spn_pair_relation_topology_confirmed`。这确认SPN-PRR在当前合成benchmark
上使用了正确P-layer信息；它仍不是实际密码、高轮区分器、攻击或SOTA证据。

两张最终SVG均按`visual-qa-redraw`渲染为`1824x966`像素复核，标题、图例、数值、坐标、
裁决文字和导出边界无重叠、裁切或歧义，结果目录已记录`visual_qa_passed.marker`。

下一步只能运行E40同预算no-triangle消融：保持数据、输入、`16x16` pair state、迭代步数、
readout、参数量、seed和训练协议不变，移除跨中间节点`i->k->j`组合。E40前不迁移真实密码、
不启动远程GPU，也不通过增加hidden、epoch或样本机械放大E39。
