# 创新2 E34：小状态SPN round-shared neural algorithmic reasoner计划

日期：2026-07-18

状态：已完成 / hold / 合成GraphGPS与looped家族停止

## 1. 研究问题

E33-R修复了cell重标号等变性，真实P-layer dual-unseen AUC由`0.682672`改善到
`0.711548`，但仍低于ID边际`0.726528`，并与错误P-layer`0.708831`持平。当前
GraphGPS还有一个明确的语义错配：

```text
样本轮数 = 2..5
现有传播 = 固定3个互不共享的GraphGPS block
轮数作用 = 只加入一个round embedding
```

密码经过多少轮本应决定结构传播执行多少次。E34回答：

```text
把同一个图处理器按实际密码轮数循环2..5次，
能否在未见S-box/P-layer上恢复稳定的真实拓扑贡献？
```

该候选属于小型neural algorithmic reasoner：共享处理器执行次数由问题实例控制。它不是
更深GraphGPS，也不是把当前网络换名。

## 2. 唯一变量

E33-R锚点：

```text
processor = 3个独立SmallSpnGpsBlock
execution = 每条样本固定3步
```

E34候选：

```text
processor = 1个共享SmallSpnGpsBlock
execution = round_index + 2，即每条样本执行2、3、4或5步
```

以下全部保持不变：

```text
cell-equivariant lane-role输入
round embedding
S-box truth-table编码
真实/错误P-layer构造
active与output-mask node feature
mask/active/global readout和分类头
hidden64、heads4、dropout0.10
AdamW、lr1e-3、weight_decay1e-4
batch128、40 epochs、checkpoint规则
```

共享模型参数少于E33-R；本门不通过时不能把失败归因成参数不足后再扩宽。该设计检验的是
共享算法步骤归纳偏置能否在更少参数下改善组外泛化。

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
cell split seed     = 33001
```

heldout仍不参与训练、checkpoint、超参数选择或停止规则。

## 4. 同预算矩阵

| 行 | seed | 作用 |
|---|---|---|
| round-shared reasoner + true P | 0,1 | 主候选 |
| round-shared reasoner + wrong P | 0,1 | 拓扑归因控制 |
| round-shared reasoner + label shuffle | 0 | 流程控制 |

冻结锚点不重跑：

```text
ID边际 dual                  = 0.726528
E33-R static true P dual     = 0.711548
E33-R static wrong P dual    = 0.708831
```

先运行三行seed0 readiness smoke：hidden32、8 epochs、batch128。smoke只证明共享参数、
按样本可变步数、cell等变性、训练和控制路径有效，不参与研究裁决。

## 5. 实现契约

必须通过以下确定性测试：

1. `processor_mode=round_shared`只注册一个`SmallSpnGpsBlock`；
2. round index `0/1/2/3`分别执行`2/3/4/5`次同一对象；
3. 混合轮数batch使用逐样本mask停止更新，不能把所有样本执行到最大轮数；
4. 同时重标号cell、P-layer、active、basis和mask时最大logit误差`<=1e-6`；
5. E33/E33-R默认`stacked`行为和row id保持不变。

## 6. 裁决门

推进条件全部满足：

```text
label-shuffle dual AUC <= 0.60
true P 两seed dual AUC均 > 0.726528
true P mean dual >= 0.756528（ID边际 + 0.03）
true P mean dual >= wrong P mean + 0.03
true P unseen-S mean >= 0.765693
true P unseen-P mean >= 0.732532
cell重标号最大logit误差 <= 1e-6
```

通过：`innovation2_small_spn_round_shared_reasoner_confirmed`。只开放round-shared
SCGT basis增益审计；真实密码迁移仍需新的readiness门，不直接远程训练。

若候选过边际但未超过wrong P：
`innovation2_small_spn_round_shared_topology_not_attributed`，停止拓扑贡献声明。

其余有效失败：`innovation2_small_spn_round_shared_reasoner_not_ready`，停止当前
合成GraphGPS/looped family，不增加宽度、block、epoch、seed或数据规模。

协议失败：`innovation2_small_spn_round_shared_protocol_invalid`，只修实现。

## 7. 产物与边界

```text
readiness run = outputs/local_smoke/i2_small_spn_round_shared_reasoner_smoke_seed0_20260718/
full run      = outputs/local_diagnostic/i2_small_spn_round_shared_reasoner_seed0_seed1_20260718/
results.jsonl
history.csv
gate.json
metadata.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

禁止把合成SPN AUC写成PRESENT/GIFT高轮结果。E34失败后不自动转TokenGT、Mamba、KAN、
大型Transformer或远程真实密码训练；下一结构必须重新从失败归因与文献匹配排序。

## 8. 实际执行与结果

readiness smoke：

```text
run_id   = i2_small_spn_round_shared_reasoner_smoke_seed0_20260718
decision = innovation2_small_spn_round_shared_readiness_passed
cell relabeling max logit error = 7.450580596923828e-08
```

冻结完整矩阵：

```text
run_id = i2_small_spn_round_shared_reasoner_seed0_seed1_20260718
rows   = 5
epochs = 40
candidate parameters = 146881
E33-R static parameters = 297409
```

| 方法 | unseen-S | unseen-P | dual-unseen |
|---|---:|---:|---:|
| ID边际 | 0.775693 | 0.742532 | 0.726528 |
| E33-R静态真实P | 0.822269 | 0.684278 | 0.711548 |
| E34共享处理器真实P | 0.824466 | 0.688850 | 0.683643 |
| E34共享处理器错误P | 0.814390 | 0.683288 | 0.708540 |
| E34标签打乱 | 0.528721 | 0.532618 | 0.397999 |

逐seed dual-unseen：

```text
round-shared true P  = 0.647404 / 0.719881
round-shared wrong P = 0.734328 / 0.682751
```

候选相对ID边际、E33-R静态真实P和E34错误P分别为：

```text
-0.042886
-0.027905
-0.024897
```

共享处理器只注册一个block，参数量约为静态锚点的一半，确定性可变步数与cell等变契约
均通过；label-shuffle接近随机。因此结果是有效的架构负证据，而不是实现或流程失败。

最终裁决：

```text
status       = hold
decision     = innovation2_small_spn_round_shared_reasoner_not_ready
remote_scale = false
```

## 9. 下一架构排序

E33绝对位置GraphGPS、E33-R cell-equivariant GraphGPS和E34 round-shared reasoner均未
建立真实P-layer归因。继续改变层数、循环次数或basis分支不再合理，合成GraphGPS/looped
家族按门停止。

下一候选优先级调整为`Cipher Edge-Token Transformer`（CETT），其依据是TokenGT式
node/edge统一token思想，但项目实现保持小型且密码结构专用：

```text
node token = lane role + active bit + output-mask bit
P-edge token = source node feature + destination node feature + directed edge type
S-box token = 4-bit cell relation + shared truth-table embedding
query token = output mask property query
processor = 小型Transformer在node/edge/query token之间交互
```

它真正改变当前失败的关系算子：P-layer边不再只是一次neighbor gather，而成为可被query
直接读取、可与其他边交互的对象。下一门仍应使用同一matched数据、ID边际、true/wrong-P
两seed和label-shuffle；必须先证明cell重标号不变性。若CETT也不超过错误P-layer，说明
该合成benchmark的神经拓扑贡献路线应整体关闭，转回标签/任务设计而不是继续搜模型。

