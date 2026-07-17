# 创新2 E33-R：小状态SPN cell重标号等变表示审计

日期：2026-07-18

状态：已完成 / hold / GraphGPS位置表示路线停止

## 1. 研究问题

E33的GraphGPS真实P-layer在dual-unseen上只有`0.682672`，低于ID边际
`0.726528`和错误P-layer控制`0.752444`。源码审计发现模型给每个node加入：

```text
absolute bit embedding
absolute nibble embedding
within-cell lane embedding
```

16-bit合成SPN的四个并行S-box cell使用同一S-box。若同时重标号四个cell、P-layer、
输入结构和输出mask，积分标签不应变化。绝对bit/nibble embedding破坏这个对称性。
E33-R回答：

```text
只修复cell重标号等变性后，真实P-layer是否稳定超过ID边际和错误P-layer？
```

这是表示缺陷审计，不是第二轮架构搜索。

## 2. 唯一变量

E33绝对位置表示：

```text
node base = bit ID + nibble ID + lane role
```

E33-R等变表示：

```text
node base = lane role
```

保留cell内lane role，因为4-bit S-box truth table的输入/输出bit有明确角色。去掉绝对
bit和nibble身份，使四个并行cell的同时重标号不改变输出。S-box编码、P-layer消息、
active/mask特征、readout、hidden维度、层数和参数优化全部不变。

## 3. 冻结数据与拆分

```text
label source = i2_small_spn_exact_label_width_16ciphers_256keys_seed0_20260718
cell source  = i2_small_spn_matched_contrast_readjudication_20260718
selected base cells = 589
total rows          = 9424
train topology      = S0..S2 x P0..P2
unseen S-box        = S3 x P0..P2
unseen P-layer      = S0..S2 x P3
dual unseen         = S3 x P3
```

沿用E33的`CELL_SPLIT_SEED=33001`。heldout不参与训练、checkpoint或调参。

## 4. 同预算矩阵

| 行 | seed | 作用 |
|---|---|---|
| cell-equivariant GraphGPS + true P | 0,1 | 候选 |
| cell-equivariant GraphGPS + wrong P | 0,1 | 拓扑归因控制 |
| cell-equivariant GraphGPS + label shuffle | 0 | 流程控制 |

E33已有绝对位置GraphGPS两seed结果作为冻结锚点，不重跑。ID边际也不重算。

完整预算：

```text
hidden dimension = 64
blocks           = 3
heads            = 4
dropout          = 0.10
optimizer        = AdamW
learning rate    = 1e-3
weight decay     = 1e-4
batch size       = 128
epochs           = 40
seeds            = 0,1（label-shuffle仅seed0）
device           = local CPU
```

先运行seed0、hidden32、2 blocks、8 epochs的三行readiness smoke。smoke只检查数据、
forward、cell重标号数值不变性、checkpoint和控制路径；不用于研究裁决。

## 5. 确定性等变性契约

冻结一个非平凡cell置换，例如`[2,0,3,1]`。同时对以下对象应用同一重标号：

```text
P-layer使用置换共轭
active-bit向量重排
output-mask向量重排
basis列重排
```

相同可训练参数下，原图与重标号图logit的最大绝对误差必须`<=1e-6`。该检查必须在
训练前通过；否则属于协议无效，不解释AUC。

## 6. 裁决门

推进门：

```text
label-shuffle dual AUC <= 0.60
true P 两seed dual AUC均 > 0.726528
true P mean dual >= 0.756528（ID边际 + 0.03）
true P mean dual >= wrong P mean + 0.03
true P unseen-S mean >= 0.765693
true P unseen-P mean >= 0.732532
cell重标号最大logit误差 <= 1e-6
```

通过：`innovation2_small_spn_cell_equivariance_repair_confirmed`，只开放同表示的SCGT
basis增益审计；仍不直接开放真实密码远程训练。

如果候选超过边际但未超过wrong P：
`innovation2_small_spn_cell_equivariance_topology_not_attributed`，停止拓扑贡献声明。

其余有效失败：`innovation2_small_spn_cell_equivariance_repair_not_ready`，停止当前
GraphGPS表示路线，不增加层数、epoch、样本或seed。

## 7. 产物与下一步边界

```text
results.jsonl
history.csv
gate.json
metadata.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

若本门失败，下一网络候选不能凭名称直接启动。必须根据失败归因二选一：只有出现
edge-edge表达不足证据才预注册TokenGT；只有跨轮而非拓扑组外失败才预注册共享权重
looped graph processor。当前不测试Mamba、KAN、更深MLP或大型纯Transformer。

## 8. 实际执行与结果

readiness smoke：

```text
run_id   = i2_small_spn_cell_equivariance_smoke_seed0_20260718
decision = innovation2_small_spn_cell_equivariance_readiness_passed
cell relabeling max logit error = 8.940696716308594e-08
```

冻结完整矩阵：

```text
run_id = i2_small_spn_cell_equivariance_seed0_seed1_20260718
rows   = 5
epochs = 40
```

| 方法 | unseen-S | unseen-P | dual-unseen |
|---|---:|---:|---:|
| E32b ID边际 | 0.775693 | 0.742532 | 0.726528 |
| E33绝对位置GraphGPS真实P | 0.834885 | 0.726330 | 0.682672 |
| E33-R等变GraphGPS真实P | 0.822269 | 0.684278 | 0.711548 |
| E33-R等变GraphGPS错误P | 0.825681 | 0.712182 | 0.708831 |
| E33-R标签打乱 | 0.607809 | 0.576364 | 0.489362 |

逐seed dual-unseen：

```text
equivariant true P  = 0.699539 / 0.723557
equivariant wrong P = 0.614071 / 0.803590
```

等变修复让真实P均值相对E33绝对位置提高`+0.028876`，但仍比ID边际低
`-0.014981`，只比错误P均值高`+0.002717`。两颗seed都没有超过ID边际；unseen-P
也低于边际停止线。label-shuffle dual接近随机，确定性cell重标号误差远低于`1e-6`，
因此这是有效负结果，不是协议失败。

最终裁决：

```text
status   = hold
decision = innovation2_small_spn_cell_equivariance_repair_not_ready
remote_scale = false
```

## 9. 后验逐轮诊断与下一网络计划

为了选择下一种结构而做的只读后验诊断显示，matched-contrast在2轮没有cell，5轮
dual split只有15个负类，二者不能计算有意义的分轮AUC。可比较的结果是：

```text
3轮 dual AUC = 0.500207 / 0.413740  （201个cell，121正/80负）
4轮 dual AUC = 0.772653 / 0.851589  （373个cell，239正/134负）
```

这不是预注册主门，也不能单独证明因果；它只用于下一架构排序。当前模型把轮数当全局
embedding，却固定运行3个不共享GraphGPS block，传播深度与实际2--5轮没有对应关系。
3轮与4轮差异因此优先支持一个新假设：使用共享权重的round processor，按每条样本的
真实轮数循环，而不是改用TokenGT或继续增加静态层数。

下一实验必须冻结：

```text
question      = 按真实轮数循环的共享处理器能否恢复dual-unseen拓扑归因
anchor        = E33-R cell-equivariant GraphGPS
one variable  = 3个静态独立block -> 1个共享block按rounds重复2..5次
data/split    = 同一9424行matched-contrast与同一cell split
budget        = hidden64, heads4, 40 epochs, batch128, seed0/1
controls      = wrong P-layer seed0/1 + label-shuffle seed0
execution     = 本地CPU；先8-epoch readiness，再完整矩阵
advance gate  = 两seed dual均过0.726528，mean同时领先边际和wrong P各0.03
stop gate     = 未过门即停止合成GraphGPS/looped family，不加层、epoch、seed或远程规模
forbidden     = 不直接启动SCGT、TokenGT、Mamba、KAN或真实密码训练
```

