# 创新2 OP11：PRESENT三轮固定八输出bit专用头独立密钥确认

日期：2026-07-21

状态：已完成 / pass（8/8跨第二固定密钥确认，专用八输出头归因通过）

## 1. 唯一研究问题

OP10在seed0固定秘密密钥下发现并用全新明文确认的八个真实密文输出位置，能否在另一把独立固定秘密
密钥上继续预测；把输出头从64位缩到冻结八位是否减少多任务干扰？

```text
selected_msb_indices = [0, 2, 8, 10, 32, 34, 40, 42]
input                = 未见明文P的64个MSB-first bit
target               = PRESENT_K^3(P)在上述八个位置的真实0/1输出值
key                  = seed1生成的独立固定未知PRESENT-80密钥
```

八个位置来自seed0 OP10，OP11不得在seed1上重新选位置。标签仍是真实密码输出，不是真假样本、平衡
性质、kernel或关系类别。

更精确地说，OP10完整64位排名有`27`个位置通过最低资格门，这八位是预注册候选上限下按发现综合
分数冻结的top-8；OP11只确认这一固定集合，不回答其余19个合格位置能否跨密钥，也不允许利用seed1
结果扩展或替换位置。

## 2. 同预算三行矩阵

```text
full64_mlp_true_output       现有两隐藏层MLP，输出完整64-bit，作为同密钥anchor
selected8_mlp_true_output    同一两隐藏层骨干，只输出冻结八个真实bit
selected8_mlp_label_shuffle  与selected8完全同构，只打乱训练标签行
```

三行共享相同明文、秘密密钥、训练/测试拆分、RMSprop、MSE、学习率、batch、epoch和随机初始化规则。
专用八输出头与其标签打乱控制参数完全匹配；完整64输出anchor只差最后输出层，参数差异必须报告。

## 3. 数据和训练规模

远程正式确认：

```text
train rows = 131072 total plaintext/ciphertext pairs
test rows  = 65536 total disjoint plaintext/ciphertext pairs
epochs     = 100
batch      = 250
device     = lxy-a6000 GPU0
```

该任务没有正负类别，不使用`/class`。数据按块落盘为`plaintexts.npy`、`features.npy`、
`full_targets.npy`和metadata，参数匹配时复用；每个模型逐epoch覆盖latest checkpoint并支持恢复。

本地实现门只用`64 train / 64 test / 1 epoch`和完整隐藏宽度，数字不作性能结论。

## 4. 逐bit指标和冻结门

每个模型在八个冻结位置分别报告：

```text
threshold_accuracy（raw output >= 0.5）
majority_accuracy
accuracy_minus_majority
AUC
MSE
invalid_numpy_rint_rate
```

一个位置通过独立密钥确认，必须同时满足：

```text
selected8 true AUC >= 0.510
selected8 true accuracy - majority >= +0.005
selected8 true AUC - selected8 shuffled AUC >= +0.005
```

主裁决：

```text
cross-key pass = 至少4/8个固定位置通过上述三门
dedicated-head attribution = 八位置平均AUC(selected8 true - full64 true) >= +0.002
hold = 少于4个位置通过，seed0结果暂按密钥条件信号处理
fail = 数据、位置顺序、checkpoint、标签打乱或拆分协议无效
```

`cross-key pass`不依赖专用头必须胜过full64 anchor；二者分别回答“位置能否跨密钥”与“专用输出是否
减少多任务干扰”。不得把两种结论混写。

## 5. 下一步与停止项

若跨密钥通过且专用头归因通过，OP12使用相同八位置与三行控制进入PRESENT四轮本地/远程阶梯；不在
r4重新选择位置。若跨密钥通过但专用头没有优势，保留易预测位置结论，使用表现更强的冻结anchor进入
r4。若跨密钥未通过，不增加seed、epoch、层数或数据，先判定seed0信号是否由固定密钥条件造成。

任何结果都不是完整密文恢复、跨大量密钥统计、攻击高轮突破或SOTA；但通过可支持毕业论文中的
“输出位置选择比完整密文同时预测更有效”方法证据。

## 6. 本地实现门结果

本地使用`64 train / 64 test / 1 epoch / CPU`完成三行端到端实现门，生成24行逐bit结果、三份最终
checkpoint及其SHA-256、逐epoch进度、磁盘数据缓存、门控和中文图。全部协议检查与执行检查通过：

```text
run_id   = i2_output_prediction_op11_present_r3_selected8_key1_smoke_20260721
status   = pass
decision = innovation2_selected8_independent_key_local_smoke_passed
output   = outputs/local_smoke/i2_output_prediction_op11_present_r3_selected8_key1_smoke_20260721/
```

`0/8`位置通过性能门只反映64条测试明文的随机波动，不作性能裁决。本地门唯一解锁动作是按本计划
固定的第二把密钥、八个位置、三行模型、数据规模和100 epochs启动A6000正式确认；不得根据smoke
数值重新选择位置或改阈值。

最终`curves.svg`经`visual-qa-redraw`渲染为像素检查。四个面板的标题、图例、柱体、曲线、门槛线、
坐标标签和底部裁决无文字重叠、遮挡、裁切或语义歧义，记录`visual_qa_passed.marker`。

## 7. 远程执行与回收契约

```text
formal run = i2_output_prediction_op11_present_r3_selected8_key1_gpu0_20260721
host       = lxy-a6000
GPU        = physical GPU0
source     = pushed commit, run-owned clean clone
remote     = G:\lxy\blockcipher-structure-adaptive-nd-runs\<run_id>
results    = 24 JSONL rows + cache/checkpoint/gate/progress evidence
retrieval  = verified result branch + local tmux watcher
```

远程不安装matplotlib、不生成图；验证结果分支回收并校验SHA-256后在本地绘图，再执行像素验收。正式
结果回来后必须在本文件记录逐bit三门、确认位置、三模型平均AUC、专用头归因、claim scope和唯一下一
动作。只有跨密钥门通过才允许预注册OP12；不允许把计划中尚未执行的四轮或结构化XOR写成结果。

## 8. 远程启动记录

```text
source commit = be4b93bfbb0cc472aafd453958cc28a54e6b97bc
remote state  = running
started gate  = readiness.txt + started.marker + torch/GPU logs present
local watcher = tmux:i2_op11_selected8_monitor
```

远程源代码位于run-owned干净克隆，启动提交在启动时与`origin/main`一致；其后的纯文档启动记录提交
不改变运行代码。本地watcher负责稀疏同步、失败标记识别、验证结果分支SHA-256回收、本地绘图和索引
刷新。主线程不进行SSH轮询。正式结果尚未回收，因此当前不得填写跨密钥通过数、专用头增益或OP12
裁决。

## 9. 条件式OP12掩码几何纠正

本节只冻结bit映射，既不是OP11结果，也不开放OP12。对MSB-first位置`m`，先换算整数bit `63-m`，
再通过PRESENT精确逆P-layer追踪末轮S-box输出来源：

| MSB位置 | 整数bit | 逆P来源bit | 来源S-box / 输出角色 |
|---:|---:|---:|---:|
| 0 | 63 | 63 | 15 / 3 |
| 2 | 61 | 55 | 13 / 3 |
| 8 | 55 | 31 | 7 / 3 |
| 10 | 53 | 23 | 5 / 3 |
| 32 | 31 | 61 | 15 / 1 |
| 34 | 29 | 53 | 13 / 1 |
| 40 | 23 | 29 | 7 / 1 |
| 42 | 21 | 21 | 5 / 1 |

因此，若OP11跨密钥门通过，OP12的主要双bit结构候选必须是同一末轮S-box的：

```text
(0,32), (2,34), (8,40), (10,42)
```

旧草案中的`(0,2)、(8,10)、(32,34)、(40,42)`只是密文显示布局中的同输出nibble邻近对；逆P后来自
不同S-box，只能作为同重量几何控制，不能标作同S-box主候选。四bit的`(0,2,8,10)`与
`(32,34,40,42)`仍分别对应来源S-box上的相同输出角色`3`与`1`，可保留为另一类结构候选；八bit全集
可作为联合mask。所有八个位置均通过`P(1 << inverse_source) == 1 << integer_bit`的确定性往返检查。

该逆P几何是在OP10统计选位完成后才用于解释和设计控制，并非OP10的输入特征或选位规则。因此
OP11的`8/8`复现支持固定位置跨第二密钥稳定，但不能单凭位置图案把预测增益因果归于精确P-layer；
后续拓扑实验必须继续保留错误P、无P和标签打乱控制。

## 10. OP11远程正式结果

验证结果分支已由本地watcher回收。SHA-256清单、固定源提交、GPU/torch记录、`196608`行完整磁盘
缓存、`24`行逐bit结果、`300`行训练历史、三份最终checkpoint哈希、真实输出标签重放、MSB-first
输入、明文唯一性与训练/测试零重合全部通过：

```text
run_id   = i2_output_prediction_op11_present_r3_selected8_key1_gpu0_20260721
source   = be4b93bfbb0cc472aafd453958cc28a54e6b97bc
status   = pass
decision = innovation2_selected8_cross_key_and_dedicated_head_supported
train    = 131072 total plaintext/ciphertext pairs
test     = 65536 total disjoint plaintext/ciphertext pairs
epochs   = 100 per model
results  = 24 / 24 rows
```

八个预注册位置全部同时通过`AUC >= 0.510`、`accuracy-majority >= +0.005`及
`true AUC - matched-shuffle AUC >= +0.005`：

| MSB位置 | 专用八输出AUC | 准确率 | 超多数类 | 匹配打乱AUC | true-shuffle | 完整64输出AUC | 专用-完整 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.527628 | 0.519730 | +0.015381 | 0.499809 | +0.027819 | 0.522927 | +0.004701 |
| 2 | 0.529898 | 0.520401 | +0.019608 | 0.501564 | +0.028333 | 0.521003 | +0.008894 |
| 8 | 0.534412 | 0.525146 | +0.023483 | 0.502269 | +0.032143 | 0.522549 | +0.011863 |
| 10 | 0.528777 | 0.521301 | +0.020920 | 0.501493 | +0.027284 | 0.518808 | +0.009970 |
| 32 | 0.532778 | 0.522415 | +0.020874 | 0.503602 | +0.029176 | 0.524600 | +0.008178 |
| 34 | 0.534711 | 0.523056 | +0.023056 | 0.501428 | +0.033283 | 0.522988 | +0.011722 |
| 40 | 0.528734 | 0.521423 | +0.019501 | 0.498359 | +0.030375 | 0.518608 | +0.010127 |
| 42 | 0.530262 | 0.521408 | +0.020355 | 0.497779 | +0.032483 | 0.523769 | +0.006493 |

联合指标：

```text
confirmed positions                         = 8 / 8
mean selected8 true AUC                    = 0.530900037
mean full64 AUC on the same positions      = 0.521906462
mean architecture-matched shuffle AUC      = 0.500787873
mean selected8 - full64 AUC                = +0.008993575
mean selected8 - matched-shuffle AUC       = +0.030112164
```

因此OP10的八个位置不是seed0单密钥偶然：它们在第二把独立固定秘密密钥上全部复现。专用八输出头也
明确超过完整64输出anchor，支持“聚焦易预测位置可以减少完整输出多任务干扰”的毕业论文方法证据。
该结论仍只覆盖两把固定密钥、PRESENT三轮和逐bit真实输出值，不是完整密文恢复、广泛跨密钥统计、
四轮结果、主流攻击轮数或SOTA。

最终`curves.svg`由验证结果分支数据在本地重绘，并经`visual-qa-redraw`渲染像素检查。标题、任务解释、
四个面板、三模型图例、近邻曲线、门槛线、数值标签和底部裁决均无重叠、遮挡、裁切、缺字或语义
歧义；`visual_qa_pending.marker`已替换为`visual_qa_passed.marker`。最近结果索引刷新通过，本结果为
`001`。

## 11. 证据支持的下一步

OP11同时通过跨密钥门和专用头归因门，因此唯一下一动作是OP12：在相同seed1固定秘密密钥、同一
`131072/65536`总行数、100 epochs和MLP骨干下进入PRESENT四轮，不重新选择八个位置。OP12只改变
轮数并比较四行同预算矩阵：专用八单bit anchor、结构化XOR直接预测、同重量几何/随机XOR控制、结构化
XOR标签打乱控制。

主要双bit mask使用第9节已验证的同末轮S-box配对；显示相邻旧配对作为几何控制。必须同时报告从
单bit概率派生的parity基线。只有结构化XOR在预注册主mask上超过匹配打乱、同重量控制、派生parity和
对应单bit anchor，才支持“多输出bit XOR提高预测轮数”。不得枚举后挑最佳mask，不增加epoch、数据、
seed或进入五轮机械扩展。若四轮门未通过，停止该轮数路线并保留三轮位置选择为论文结论。
