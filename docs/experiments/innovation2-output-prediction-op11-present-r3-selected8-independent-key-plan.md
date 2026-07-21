# 创新2 OP11：PRESENT三轮固定八输出bit专用头独立密钥确认

日期：2026-07-21

状态：本地实现门通过 / 远程正式确认运行中

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
