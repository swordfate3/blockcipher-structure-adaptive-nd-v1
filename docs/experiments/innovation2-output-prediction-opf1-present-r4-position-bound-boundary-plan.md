# 创新2 OPF1：PRESENT四轮位置绑定网络同协议输出预测边界计划

日期：2026-07-22

状态：正式远程完成并验证回收 / 四轮输出预测门未通过 / 三至四轮经验边界成立

本地readiness记录：

```text
run_id      = i2_output_prediction_opf1_present_r4_position_bound_spn_rescnn_smoke_seed7_20260722
scale       = 64 train / 64 test / 1 epoch / CPU
status      = pass
decision    = innovation2_position_bound_r4_local_readiness_passed
results     = 40 rows
history     = 5 rows
checkpoints = 5
visual QA   = pass（2264x1340像素渲染检查，无重叠、裁切、缺字或标题歧义）
next        = 从推送提交启动A6000正式131072/65536、100 epochs五模型矩阵
```

readiness中的随机小样本AUC不作性能判断；它只授权执行已经冻结的正式远程矩阵。

正式远程启动记录：

```text
source commit = 3ad333ecd599e450159d464c2f474ccf49830113
remote root   = G:\lxy\blockcipher-structure-adaptive-nd-runs\i2_opf1_r4_poshead_k7_20260722
GPU           = physical GPU0
started       = 2026-07-22，started.marker已确认
readiness     = status=pass
progress      = progress.jsonl已开始持久化
monitor       = local tmux i2_opf1_r4_poshead_monitor
result state  = running；尚未完成或回收，不作四轮性能裁决
```

tmux watcher已经完成等待并自动回收验证分支产物；本地回收、哈希验证、40/500/5产物门、结果索引
刷新和最终SVG像素检查均已完成，因此本记录现在允许写入四轮边界裁决。

正式结果记录：

```text
completed remotely = 2026-07-22 20:39:53 +08:00
retrieval           = verified result branch archive
source commit       = 3ad333ecd599e450159d464c2f474ccf49830113
status              = hold
decision            = innovation2_position_bound_r4_boundary_observed
results/history     = 40 / 500 rows
checkpoints         = 5
SHA256 archive      = pass
visual QA           = pass（2264x1340像素渲染检查）
```

五模型正式平均AUC：

```text
全局头ResCNN             = 0.511421832
无P位置头                = 0.508510437
真实P位置头              = 0.513755358
错误P位置头              = 0.511067172
真实P位置头标签打乱       = 0.500934143
```

主候选控制差值与三轮对照：

```text
真实P - 标签打乱         = +0.012821215  < +0.030主门
真实P - 全局头           = +0.002333526
真实P - 无P位置头        = +0.005244921
真实P - 错误P位置头      = +0.002688186
平均accuracy-majority    = +0.010000229
通过逐bit门              = 0 / 8
OPD1三轮真实P平均AUC      = 0.999996158
OPF1四轮 - OPD1三轮      = -0.486240800
```

八个预注册bit的真实P位置头AUC范围为`0.508588--0.517537`，全部低于`0.55`逐bit门。虽然平均准确率
相对多数类高约一个百分点，但AUC、标签打乱差值和`4/8`逐bit要求没有同时满足，所以不能把四轮称为
输出预测成功。

证据支持的推荐动作是保留OPD1三轮正结果，并把本实验写成同密钥、同明文划分、同八输出、同五模型、
同训练预算下仅增加一轮后出现的经验边界。当前不启动五轮，不机械增加样本或epoch，也不继续用同一
位置绑定路线搜索模型；后续若提出新方法，必须作为新的假设，以本OPF1为同预算四轮锚点重新预注册。

## 1. 为什么必须补这个实验

OPD1在PRESENT三轮、seed7固定未知密钥、`131072/65536`训练/测试明文对和100 epochs下，使用位置
绑定SPN-ResCNN预测八个真实密文输出bit：

```text
exact-P位置头平均AUC = 0.999996158
wrong-P位置头平均AUC = 0.999974160
普通ResCNN平均AUC    = 0.570093368
标签打乱平均AUC      = 0.500234930
```

exact-P没有超过wrong-P，所以OPD1不能把性能归因于精确PRESENT P-layer；但这不否定位置绑定模型的
输出预测能力。此前四轮OP12改用了MLP并预测2-bit/4-bit XOR，不能回答同一个位置绑定网络是否在四轮
失效。OPF1补齐这个缺口。

## 2. 唯一改变的变量

```text
OPD1 anchor rounds = 3
OPF1 candidate rounds = 4
```

其余协议全部保持：

```text
cipher / key        = PRESENT-80 / OPD1同一seed7固定未知密钥
train/test          = 131072 / 65536 total明密文对
plaintext sequence  = 与OPD1完全相同的生成seed、顺序和划分
input               = 64个MSB-first明文bit
target              = [0,2,8,10,32,34,40,42]八个独立真实密文bit
models              = OPD1原五行矩阵
epochs / batch      = 100 / 250
optimizer / loss    = RMSprop / raw-output MSE
learning rate       = 0.001
selection           = final epoch
sample class        = false
```

远程OPD1原始明文文件已只读核验：

```text
plaintexts.npy SHA256 = 0f08d171c5b833ee1223da07bfc80e10d7ea99bbc0bef1b068547d3a7e8120e1
```

OPF1正式缓存生成完成后必须得到同一SHA256；四轮只允许改变密文目标，不允许改变明文或划分。

## 3. 五模型同预算矩阵

```text
selected8_global_head_rescnn_anchor_true_output
selected8_position_head_rescnn_no_p_true_output
selected8_position_head_spn_rescnn_exact_p_true_output
selected8_position_head_spn_rescnn_wrong_p_true_output
selected8_position_head_spn_rescnn_exact_p_label_shuffle
```

普通ResCNN为`3,955,904`参数，三个位置头模型均为`3,956,928`参数，差约`0.026%`。exact-P、wrong-P
和shuffle行保持相同初始化、容量、batch顺序和训练预算；shuffle只打乱训练标签，测试标签始终为真实
四轮密文输出bit。

## 4. 为什么性能门不要求超过wrong-P

OPF1首先回答“同一个OPD1候选进入四轮后是否仍能预测真实输出”。因此输出预测主门只要求exact-P：

```text
平均AUC >= 0.55
平均AUC - 标签打乱平均AUC >= 0.03
平均accuracy-majority >= 0.005
至少4/8 bit通过逐bit门
```

逐bit门为：

```text
AUC >= 0.55
AUC - 同bit标签打乱AUC >= 0.015
accuracy-majority >= 0.005
```

exact-P相对global、no-P和wrong-P的差值继续完整报告，但它们回答架构/拓扑归因，不阻断“输出仍可
预测”的性能结论。尤其wrong-P如果也成功，表示精确P仍不可归因，不表示四轮输出预测失败。

## 5. 裁决与下一步

如果输出预测主门通过：

```text
decision = innovation2_position_bound_r4_output_supported
next = 使用全新固定未知密钥原样确认四轮五行矩阵
r5 = 未授权
```

只有新密钥确认也通过，才允许另行预注册五轮实验。不得在确认前换输出位置、增加epoch、扩大模型或
根据四轮结果选择bit。

如果输出预测主门未通过：

```text
decision = innovation2_position_bound_r4_boundary_observed
next = 保留OPD1三轮正结果，报告同网络/同目标/同预算的三至四轮经验边界
r5 / more data / more epochs / model search = no
```

这仍是单固定密钥和Kimura单密钥样本预算下的经验边界，不是所有网络、所有密钥或所有四轮输出函数
的数学不可能性证明，也不是完整密文恢复、真假样本分类或SOTA攻击轮数结果。

## 6. 执行路径与产物门

先运行本地`64/64`、1 epoch CPU readiness，只验证四轮真实密文回放、同seed明文序列、五模型参数、
训练/测试零重合、磁盘缓存、断点、40条结果、5条history、5个checkpoint与SVG，不用随机小样本数值
作性能判断。

readiness通过后，从推送提交在A6000运行正式矩阵。正式远程必须：

```text
run root   = G:\lxy\blockcipher-structure-adaptive-nd-runs\i2_opf1_r4_poshead_k7_20260722
cache      = plaintexts.npy / features.npy / full_targets.npy / cache_metadata.json
progress   = progress.jsonl逐chunk和逐epoch落盘
resume     = 参数匹配缓存与latest checkpoint恢复
results    = 40 rows
history    = 500 rows
checkpoints= 5 hashes
```

远程启动后只做一次只读启动确认，随后由本地tmux watcher等待完成、回收验证分支产物、校验SHA256、
绘制中文SVG和刷新最近结果索引。最终SVG必须执行`visual-qa-redraw`像素检查后才能完成裁决。

## 7. 证据支持的推荐动作

当前唯一推荐动作是执行OPF1本身。不要先做新的XOR mask、四轮MLP、五轮模型、更多seed或更大样本；
这些都会绕开“同一OPD1网络只增加一轮”的核心问题。OPF1结果将直接决定创新2能否把PRESENT输出
预测正结果从三轮推进到四轮，或者把四轮写成严格得多的同协议经验边界。
