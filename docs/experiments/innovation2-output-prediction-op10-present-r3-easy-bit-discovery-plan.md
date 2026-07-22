# 创新2 OP10：PRESENT三轮易预测真实输出bit发现与独立确认

日期：2026-07-21

状态：远程fresh确认通过 / 等待架构匹配控制与独立密钥复验

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

发现阶段对每个bit的LSTM和MLP模型选项分别应用同一资格门；一个模型选项必须同时满足：

```text
true-output model discovery AUC >= 0.510
true-output model discovery accuracy - majority >= +0.005
true-output model discovery AUC - shuffled-LSTM AUC >= +0.005
```

其中LSTM选项使用架构匹配的shuffled-LSTM；MLP选项在OP10只能使用该shuffled-LSTM作为跨架构负
控制，所以MLP候选即使通过本门，也必须在OP11补做匹配的shuffled-MLP后才能作架构或跨密钥归因。

对LSTM和MLP分别计算`min(AUC-0.5, accuracy-majority, AUC-shuffled_AUC)`，每个bit冻结分数更强的
模型，再按该分数降序冻结最多8个候选；并始终输出64位完整排名，即使候选池为空。fresh阶段对冻结
的同一`(bit, selector_model)`使用同样三个阈值，且AUC方向必须仍大于0.5。

`candidate_limit=8`是查看fresh确认标签前冻结的候选容量，不是结果揭盲后按图形或P-layer几何调整
出来的数量。它让后续专用八输出head、独立密钥确认和逐bit联合门使用固定宽度；最低资格门负责排除
弱候选，综合分数只在合格位置中决定前八名。fresh确认集不得参与候选数量、位置或排序选择。

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
source commit = 06f76382fed86fc0b231e48c5a358cf0c116f54b
watcher       = tmux:i2-op10-bit-monitor
watcher state = verified_results_retrieved_and_indexed
```

watcher只有看到OP9从验证结果分支回收成功的本地marker后，才会调度OP10。远程OP10读取OP9保留在
`G:\lxy`的三个最终checkpoint和数据缓存，生成`2^16`条fresh确认明文；完成后自动验证384行结果、
候选hash、fresh缓存和结果分支，再回收并刷新索引。主线程不重复SSH轮询。

## 10. OP10远程正式结果

远程发现与fresh确认已完成并从验证结果分支回收。候选SHA-256、三个源checkpoint、384行逐bit结果、
`65536`条fresh缓存、明文唯一性/互斥性、MSB-first映射和真实PRESENT输出重放全部通过。

```text
status                = pass
decision              = innovation2_true_output_bits_fresh_confirmed
discovery candidates  = 8
fresh confirmed       = 8 / 8
confirmed MSB indices = 0, 2, 8, 10, 32, 34, 40, 42
selector model        = matched_mlp_true_output（8个位置全部相同）
```

完整64位发现排名中实际有`27`个位置通过最低资格门；由于预注册`candidate_limit=8`，这里只冻结综合
分数最高的八个，发现顺序为`32,34,2,42,40,8,0,10`，报告时再按MSB位置排序为
`0,2,8,10,32,34,40,42`。因此这八位是“合格池中的固定top-8”，不能表述成只有这八位有信号，也
不能在后续密钥或模型结果上改选其余19个合格位置。

| MSB位置 | 整数bit | nibble / 内部bit | fresh准确率 | 超多数类 | fresh AUC | AUC-打乱LSTM |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 31 | 8 / 0 | 0.518173 | +0.017197 | 0.524560 | +0.024620 |
| 34 | 29 | 8 / 2 | 0.518600 | +0.017441 | 0.524725 | +0.024723 |
| 2 | 61 | 0 / 2 | 0.517242 | +0.015396 | 0.523485 | +0.023422 |
| 42 | 21 | 10 / 2 | 0.514175 | +0.013428 | 0.520558 | +0.020404 |
| 40 | 23 | 10 / 0 | 0.518478 | +0.016815 | 0.525110 | +0.025295 |
| 8 | 55 | 2 / 0 | 0.512665 | +0.011734 | 0.519002 | +0.019002 |
| 0 | 63 | 0 / 0 | 0.519394 | +0.018600 | 0.526567 | +0.026507 |
| 10 | 53 | 2 / 2 | 0.515320 | +0.014221 | 0.522252 | +0.022344 |

八个位置在发现集冻结后全部于全新明文复现，说明单固定密钥PRESENT三轮存在可预测的真实输出bit；
完整64-bit exact-match为0不影响该逐bit结论。位置还呈现规则子集：nibble只为`0/2/8/10`，且
`bit_in_nibble_msb`只为`0/2`，值得在独立密钥上检验是否为稳定SPN结构而非当前密钥偶然。

上述规则子集是top-8冻结后的结构观察，不是OP10候选选择特征。OP10只按发现集预测指标选择位置；
逆P来源、同S-box配对和输出角色不得反向写成初始选位原因。它们只能作为后续预注册结构对照的假设
来源，并且必须与同重量几何控制比较。

证据边界必须保留：八个候选全部由MLP选择，当前标签打乱行是LSTM，只能作为跨架构负控制。OP10
通过的是fresh输出信号门，不是MLP架构匹配归因门，也不是跨密钥结论。下一步OP11固定这八个位置，
在独立秘密密钥上比较完整64输出MLP、八输出专用MLP和八输出标签打乱MLP；第二密钥结果揭盲前不得
更改位置或阈值。

最终`curves.svg`经过`visual-qa-redraw`重绘：候选圆点和准确率曲线改为每个bit实际冻结的模型，确认
条形图明确标出八个MLP位置；像素检查无重叠、裁切、曲线指代错误或中文语义歧义。
