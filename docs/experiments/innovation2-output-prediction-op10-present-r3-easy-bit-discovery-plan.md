# 创新2 OP10：PRESENT三轮易预测真实输出bit发现与独立确认

日期：2026-07-21

状态：本地实现门通过 / 自动等待OP9模型完成

## 1. 唯一研究问题

在固定未知PRESENT-80密钥和未见明文上，64个真实密文输出位置中是否存在少数bit能被稳定预测，且该
优势能在未参与候选选择的全新明文上复现？

```text
C = PRESENT_K^3(P)
input  = P的64个MSB-first bit
target = 某个固定位置j的真实密文bit C[j]
```

`0/1`是该位置的真实输出值，不是正负样本类别。完整64-bit同时命中、真假区分、积分平衡、kernel和
ATM关系都不是OP10标签。

## 2. bit编号

所有产物同时记录三种无歧义位置：

```text
msb_index        = 0..63，0表示密文最高位
integer_bit      = 63 - msb_index，0表示密文最低位
nibble_msb_index = msb_index // 4
bit_in_nibble    = msb_index % 4，0表示该nibble最高位
```

论文、图和用户报告默认使用`msb_index`，不得只写“bit 7”而不说明方向。

## 3. 冻结模型与必要控制

不重新训练或根据OP10结果调整OP9模型。直接加载OP9完成的三个最终checkpoint：

```text
kimura_lstm_true_output       逐bit候选发现模型
matched_mlp_true_output       同参数量非序列基线
kimura_lstm_label_shuffle     架构匹配的标签打乱控制
```

每个bit分别计算真实输出LSTM和真实输出MLP的冻结发现分数，并选择两者中分数更强的模型作为该bit的
`selector_model`；最多8个候选bit，不会把同一位置按两个模型重复计数。标签打乱LSTM是LSTM候选的
架构匹配控制，对MLP候选只能作为跨架构负控制。若MLP候选fresh确认，OP11必须补同预算
`matched_mlp_label_shuffle`后才能作架构归因或跨密钥声明。模型、阈值和候选数不得在fresh确认结果
揭盲后修改。

## 4. 两个严格分离的数据阶段

### 发现集

复用OP9中未参与训练的`65536`条测试明文/真实密文。对64个位置和三种模型分别报告：

```text
threshold_accuracy（raw output >= 0.5）
majority_accuracy
accuracy_minus_majority
AUC
MSE
invalid_numpy_rint_rate
```

### fresh确认集

在候选列表及SHA-256写盘后，使用相同固定秘密密钥生成另外`65536`条唯一随机明文。它们必须与OP9的
`196608`条训练/测试明文完全不重合，使用独立冻结RNG seed，分块落盘并记录进度。冻结模型只推理，
不得继续训练、校准阈值或更改候选。

## 5. 候选选择与确认门

发现阶段每个bit必须同时满足才进入候选池：

```text
LSTM discovery AUC >= 0.510
LSTM discovery accuracy - majority >= +0.005
LSTM discovery AUC - shuffled-LSTM AUC >= +0.005
```

对LSTM和MLP分别计算`min(AUC-0.5, accuracy-majority, AUC-shuffled_AUC)`，每个bit冻结分数更强的
模型，再按该分数降序冻结最多8个候选；并始终输出64位完整排名，即使候选池为空。fresh阶段对冻结
的同一`(bit, selector_model)`使用同样三个阈值，且AUC方向必须仍大于0.5。

主裁决：

```text
pass = 至少1个候选在fresh 65536条明文上通过全部确认门
hold = 候选池为空，或所有候选在fresh确认失效
fail = checkpoint、bit顺序、明文互斥、真实密文重放或候选冻结协议无效
```

`pass`只表示“该固定密钥PRESENT三轮存在确认的易预测输出bit”，不是跨密钥结论、四轮结论或SOTA。

## 6. 执行与产物

OP10在远程A6000上读取OP9磁盘缓存与checkpoint；不会复制大模型进Git，也不会在本地CPU重训。

```text
results.jsonl                 64位 x 3模型 x discovery/fresh逐bit指标
ranking.csv                   64位发现排名与fresh确认结果
candidates.json               揭盲前冻结候选、阈值与SHA-256
fresh_data/cache_metadata.json
summary.json
gate.json
progress.jsonl
curves.svg
```

结果完成后必须校验checkpoint hash、候选文件hash、fresh数据互斥、PRESENT输出重放、结果行数和门控；
随后运行`visual-qa-redraw`检查逐bit图的标签、缩放、候选标注、控制曲线与中文标题。

## 7. 下一步和停止项

若至少一个bit确认，通过OP11只对确认bit训练专用小输出头，并在第二把独立固定秘密密钥上复验同一
位置；专用头必须与共享64输出头、MLP和各自架构匹配的标签打乱控制使用相同数据预算。第二密钥前
不扩r4。

若没有bit确认，不把完整模型机械扩到更多数据、epoch、层数、seed或更高轮；先审计专用单bit头是否
属于合理的多任务干扰诊断。不得退回真假样本分类、平衡性质分类或用完整exact-match替代逐bit裁决。

## 8. 本地实现门结果

使用OP9本地完整架构smoke的三个checkpoint完成OP10端到端实现门：

```text
run_id          = i2_output_prediction_op10_present_r3_easy_bit_modelselect_smoke_20260721
discovery       = 64条OP9未见明文
fresh           = 64条新生成且与OP9全部明文不重合的明文
result rows     = 64 bits x 3 models x 2 splits = 384
protocol checks = 20/20 true
status          = pass
decision        = innovation2_output_bit_discovery_local_smoke_passed
```

候选文件及SHA-256在第一条fresh数据生成前写盘；三个checkpoint hash、MSB-first映射、真实PRESENT
输出重放、发现/fresh拆分、逐bit LSTM/MLP模型冻结选择和结果行数全部通过。`64+64`小样本没有候选，
不作性能结论。

`curves.svg`经过两轮`visual-qa-redraw`像素检查：修正内部英文decision id和缺失的smoke边界后，最终
标题、四面板、图例、bit方向、曲线缩放和中文裁决没有重叠、裁切或语义歧义。结果索引已刷新为
`outputs/00_RECENT_RESULTS.md`的`001`。

## 9. 远程依赖式执行

OP10实现与远程包固定在已推送提交：

```text
source commit = d23d16a6e474e002be4a4f0d497d76404d1c4bec
watcher       = tmux:i2-op10-bit-monitor
watcher state = waiting_for_verified_op9_retrieval
```

watcher只有看到OP9从验证结果分支回收成功的本地marker后，才会调度OP10。远程OP10读取OP9保留在
`G:\lxy`的三个最终checkpoint和数据缓存，生成`2^16`条fresh确认明文；完成后自动验证384行结果、
候选hash、fresh缓存和结果分支，再回收并刷新索引。主线程不重复SSH轮询。
