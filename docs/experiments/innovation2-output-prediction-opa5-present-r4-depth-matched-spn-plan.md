# 创新2 OPA5：PRESENT四轮深度匹配SPN-aware结构改造计划

日期：2026-07-22

状态：条件式预注册 / 未实现未启动 / 等待OPA3与OPA4

## 1. 研究问题

OPA1已经在同一三轮八输出任务上比较MLP、六层LSTM、十块ResCNN、七层Transformer和
PRESENT-SPN-aware，不存在“只使用LSTM”的模型限制。结果中LSTM和Transformer接近随机，ResCNN
保留部分信号，而三个SPN block的PRESENT-SPN-aware在第三、第四固定密钥上均达到八位置平均
`AUC=1.0`。因此后续优先级不是继续枚举通用网络名称，而是检验可归因的结构改造。

当前PRESENT-SPN-aware在三轮任务上恰好使用三个SPN block。OPA4会先保持模型不变，只把数据从
三轮改成四轮，以测量未改造架构的轮数边界。若OPA4性能门未通过，OPA5只回答一个预先冻结的问题：

> 在总参数量基本不变时，把网络从三个SPN block改为四个SPN block，能否因为“一轮对应一个结构
> block”而恢复四轮八个真实密文输出bit的可预测性？

这不是完整密文恢复、真假样本分类、积分平衡判断或结构化XOR预测。

## 2. 启动权限

OPA5不得与OPA3或OPA4并行抢跑。只有下列条件全部满足才允许实现本地readiness：

```text
OPA3 = verified result branch / plan-aligned / pass
OPA3 decision = innovation2_selected8_present_topology_independently_attributed
OPA4 = verified result branch / plan-aligned / protocol and execution checks all true
OPA4 performance gate = hold or fail
```

若OPA4通过，下一步仍是第五固定密钥`seed4`的原始三block四轮独立确认，不运行OPA5。若OPA4因结果
不完整、来源不可信、缓存错误、标签回放失败或其他协议问题失败，必须先修复或裁决协议，OPA5不得把
协议失败包装成模型改造机会。

OPA5是对OPA4性能失败的唯一预注册结构救援实验。不得并行加入Mamba、KAN、GNN、更深MLP、更大
Transformer、后验输出位置、XOR目标、更多样本或更多epoch。

## 3. 冻结任务契约

OPA5完整复用OPA4的数据与输出预测协议：

```text
cipher                  = PRESENT-80
rounds                  = 4
seed / fixed key        = 3
train rows              = 131072 total plaintext-ciphertext pairs
test rows               = 65536 total disjoint plaintext-ciphertext pairs
input                    = 64 MSB-first plaintext bits
target                   = 8 preregistered true ciphertext output bits
selected MSB positions   = [0, 2, 8, 10, 32, 34, 40, 42]
epochs                   = 100 per newly trained model
batch size               = 250
optimizer                = RMSprop
loss                     = raw-output MSE
learning rate            = 0.001
selection                = final epoch
```

必须通过OPA4缓存元数据和内容哈希复用相同训练/测试明文及四轮密文标签，不得重新抽样。测试标签始终
是真实四轮密文bit；标签打乱只作用于训练行，并复用OPA4冻结的排列。

## 4. 单变量结构改造

OPA4原始锚点：

```text
blocks = 3
token_dim = 189
parameters = 3,879,415
```

OPA5候选：

```text
blocks = 4
token_dim = 164
parameters = 3,894,181
relative parameter gap = +0.3806%
```

`token_dim`缩小只用于补偿新增block的参数量，使改造保持在锚点`1%`以内；研究变量是固定参数预算下
的SPN block深度分配。每个block仍使用相同的4-bit nibble局部混合、残差channel MLP、位置表示、
输出头和无参数P-layer映射，不加入新的特征、损失或中间状态监督。

## 5. 三行受控矩阵

OPA5直接引用OPA4已经完成的三block exact-P true结果作为同数据锚点，不重复训练。新增训练仅三行：

```text
selected8_present_spn_4block_exact_p_true_output
selected8_present_spn_4block_exact_p_label_shuffle
selected8_present_spn_4block_wrong_p_true_output
```

exact-P与wrong-P继续复用OPA3冻结的映射定义。三行的模型参数、初始化、batch顺序和优化协议相同；
exact-P true与shuffle只改变训练标签对应关系。OPA4的MLP、三block shuffle和三block wrong-P结果保留
在联合报告中，不增加远程训练矩阵。

## 6. 正式门

首先要求OPA4 gate和缓存哈希绑定正确，新增`24`条逐bit结果、`300`条history、三个checkpoint及
全部协议/执行检查完整。四block候选只有同时满足下列条件才通过：

```text
four-block exact-P mean true AUC >= 0.510
four-block exact-P - OPA4 three-block exact-P mean AUC >= +0.010
four-block exact-P - four-block shuffle mean AUC >= +0.010
four-block exact-P - four-block wrong-P mean AUC >= +0.030
```

至少`4/8`个预注册输出bit还必须同时满足：

```text
four-block exact-P AUC >= 0.510
four-block exact-P accuracy - majority >= +0.005
four-block exact-P - OPA4 three-block exact-P AUC >= +0.005
four-block exact-P - four-block shuffle AUC >= +0.005
four-block exact-P - four-block wrong-P AUC >= +0.020
```

报告全部八个位置，不按结果重选bit或阈值。

## 7. 裁决和后续

若OPA5通过，只能说明在第四固定密钥和当前预算下，轮数匹配的四block结构优于未改造三block锚点，
且收益超过匹配shuffle与错误P控制。下一步必须在第五固定密钥`seed4`上确认相同四block矩阵；独立
密钥确认前不得进入五轮或声称通用结构改进。

若OPA5未通过，停止当前八输出任务上的四轮模型枚举、容量增加、epoch/数据扩展和五轮推进。论文
保留OPA2三轮整体架构结果与OPA3拓扑归因结果（若成立），把OPA4/OPA5写成未改造与深度匹配改造
均未越过的四轮经验边界。

ResCNN在OPA1的三轮`mean AUC=0.588387942`使其仍是有效通用基线，但远低于已独立确认的SPN-aware
三轮结果；纯LSTM和纯Transformer均接近随机。因此OPA5不重复这些架构。只有四block候选在独立
密钥上确认后，才允许另行预注册一个SPN-ResCNN混合残差候选，并继续遵守同预算、同数据和匹配控制。

## 8. 执行和产物

获授权后先运行`64 train / 64 test / 1 epoch / CPU`本地readiness，只检查三行模型、参数差、gate
绑定、缓存拒绝错配、结果/history/checkpoint数量和绘图闭环，不作性能判断。

正式训练必须从已推送提交的远程run-owned干净clone在A6000运行，使用`G:\lxy`下的参数匹配磁盘
缓存、progress、checkpoint恢复、`cmd.exe /c`、verified result branch和本地tmux watcher。结果至少
包含`results.jsonl`、`history.csv`、`progress.jsonl`、`metadata.json`、`opa4_gate.json`、
`summary.json`、`gate.json`、`checkpoint_manifest.json`和`curves.svg`。回收后执行完整哈希/数量验证、
`visual-qa-redraw`真实像素质检，并刷新`outputs/00_RECENT_RESULTS.md/json`。
