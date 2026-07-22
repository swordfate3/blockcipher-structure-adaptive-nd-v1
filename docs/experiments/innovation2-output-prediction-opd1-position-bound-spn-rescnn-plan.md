# 创新2 OPD1：PRESENT三轮位置绑定 SPN-ResCNN 输出预测计划

日期：2026-07-22

状态：实现、本地readiness与远程正式包验证通过 / 待提交推送启动

## 1. 研究问题与来源门

OPC1在第七把固定秘密密钥上的正式平均AUC为：

```text
普通ResCNN       = 0.573571593
全局头真实P混合  = 0.546633861
全局头错误P混合  = 0.545181187
全局头标签打乱   = 0.500163496
```

真实P既低于普通ResCNN，又只比错误P高`0.001453`，因此OPC1全局头混合路线已经停止。OPN1进一步
证明该模型的最后一次P重排可被`Linear(252 * 64, 8)`的列重排完全吸收，最大数值误差仅
`1.99e-13`。

OPD1只测试一个新假设：

> 保持ResCNN骨干、P路由、数据、目标和训练预算不变，将全局扁平head换成参数匹配的位置绑定多输出
> head，使每个输出只读取对应最终位置，能否让真实P同时超过普通ResCNN、无路由位置头、错误P和
> 标签打乱？

OPD1是由OPN1确定性非可识别性直接支持的新机制，不是给OPC1增加深度、数据、epoch或搜索更多P。

正式模式必须同时验证：

```text
OPC1 gate SHA256 = ebb86a9feab6d2d9993937f5c0a7f4afe1bfe3597c8c1dff083956381e0310b4
OPC1 status/decision = hold / innovation2_spn_rescnn_hybrid_not_supported
OPN1 gate SHA256 = 887a7db3643e73bdda67958bcaae470881a09db25ab0ba5ff6c3d6bb0a2503d7
OPN1 status/decision = pass / innovation2_spn_rescnn_final_routing_absorbable_by_global_head
```

## 2. 唯一模型变化

OPC1全局head：

```text
全部64位置 x 252 channels -> Flatten -> Linear(16128, 8)
```

OPD1位置绑定head：

```text
八个冻结输出位置各自取252维向量
-> 八个独立 MLP(252 -> 64 -> 1)
-> 拼成八个真实密文输出bit
```

八个独立局部head约`130056`参数，原全局head为`129032`参数；整体参数差约`0.03%`。候选不读取
密钥、轮密钥或中间状态。最后P重排会改变每个局部head实际读取的位置，不能再通过一个读取全部位置的
全局线性层吸收。

## 3. 五行同预算归因矩阵

```text
selected8_global_head_rescnn_anchor_true_output
selected8_position_head_rescnn_no_p_true_output
selected8_position_head_spn_rescnn_exact_p_true_output
selected8_position_head_spn_rescnn_wrong_p_true_output
selected8_position_head_spn_rescnn_exact_p_label_shuffle
```

五行共享相同明文、密文、八个冻结位置、252 channels、10个残差块、optimizer、loss、epoch和batch。
第二至第五行使用完全相同的位置绑定head；第三与第四行可训练参数初始化逐项相同，唯一差异是无参数
位置映射；第五行只打乱训练标签，测试标签始终是真实密文输出值。

## 4. 正式协议

```text
cipher / rounds         = PRESENT-80 / 3
fixed key               = seed7，第八把独立秘密密钥
train/test              = 131072 / 65536 total plaintext-ciphertext pairs
selected MSB positions  = [0,2,8,10,32,34,40,42]
epochs / batch          = 100 / 250
optimizer / lr          = RMSprop / 0.001
loss / selection        = raw-output MSE / final epoch
device                  = lxy-a6000 physical GPU0
result/history/checkpoint rows = 40 / 500 / 5
disk cache rows         = 196608
```

这仍是Kimura单固定密钥样本预算，不是多密钥paper-scale复现、样本分类、完整密文恢复、四轮结果或
主流SOTA攻击比较。

## 5. 正式性能门

```text
候选平均AUC >= 0.550
候选 - 全局头ResCNN平均AUC >= +0.010
候选 - 无P位置头平均AUC >= +0.010
候选 - 错误P平均AUC >= +0.020
候选 - 标签打乱平均AUC >= +0.030
至少4/8个bit同时通过：
  candidate AUC >= 0.550
  candidate-global >= +0.005
  candidate-no-P >= +0.005
  candidate-wrong >= +0.015
  candidate-shuffle >= +0.015
  accuracy-majority >= +0.005
```

正式通过只开放全新固定密钥原样确认，不开放四轮。任一平均门或`4/8`联合门失败，则保留全局头
ResCNN锚点并停止位置绑定路线，不增加head宽度、深度、数据、epoch、输出位或错误P。

## 6. 本地readiness与执行路径

本地只运行seed7、`64/64` total rows、1 epoch、CPU的五行实现门。readiness必须验证：

1. OPC1/OPN1来源gate及SHA-256所有权；
2. seed7与seed0--6秘密密钥不同；
3. 明文零重合、真实密文目标逐条回放、标签不是样本类别；
4. 全局head与三个位置head模型参数差不超过`0.1%`，三个位置head参数严格相等；
5. exact/wrong可训练状态初始化相同，最终selected位置来源不同；
6. 标签打乱只作用训练标签；
7. cache、40条results、5条history、5个checkpoint及SVG闭环。

readiness AUC不作性能判断。若实现门通过，按仓库工作流范围提交并推送，从推送提交生成Windows
`.cmd`和本地tmux watcher，在A6000 GPU0启动正式矩阵；若实现门失败，只修复协议，不远程启动。

## 7. 证据支持的下一动作

立即实现上述最小位置绑定head和五行readiness，不同时加入GNN、LSTM、Transformer、扩张卷积或
四轮目标。这样既响应多模型/改造要求，也保证新实验只改变OPN1指出的不可识别输出头机制。

## 8. 本地readiness结果

本地冻结实现门已经完成：

```text
run_id = i2_output_prediction_opd1_present_r3_position_bound_spn_rescnn_smoke_seed7_20260722
train/test = 64/64 total plaintext-ciphertext pairs
models/epochs = 5/1
results/history/checkpoints = 40/5/5
status = pass
decision = innovation2_position_bound_spn_rescnn_local_readiness_passed
```

全部protocol和execution checks为真，包括：OPC1/OPN1来源gate及SHA-256匹配；seed7秘密密钥与
seed0--6不同；训练/测试明文零重合；真实密文目标回放；全局头模型`3,955,904`参数，三个位置头
变体均为`3,956,928`参数，差距约`0.026%`；exact/wrong可训练状态初始化相同；八个selected最终
来源在exact/wrong下全部不同；五行训练、40条结果、5条history和5个checkpoint hash完整。

本地平均AUC为：

```text
全局头ResCNN = 0.474708
无P位置头    = 0.492786
真实P位置头  = 0.537688
错误P位置头  = 0.505384
标签打乱     = 0.552127
```

这些值来自64条随机测试样本，只证明结果可回放，不能应用正式性能门或用于模型选择。

readiness `curves.svg`已按`visual-qa-redraw`在`1920x1137`和`1280x758`像素下检查；中文标题、五行
热图、四个独立差值轴、阈值线、数值标签、裁决和证据边界均无重叠、裁切、遮挡、缺字或尺度歧义。

readiness通过后的唯一下一动作是生成并测试seed7远程正式包，范围提交并推送，从推送提交在A6000
GPU0启动`131072/65536`、`100 epochs x 5`模型矩阵并交由本地tmux watcher自动回收。不得把本地
AUC写成候选正结果，也不得在正式结果前修改head、门槛或输出位置。

## 9. 远程正式包验证

seed7正式包已经生成并通过冻结审计：

```text
run_id = i2_output_prediction_opd1_present_r3_position_bound_spn_rescnn_key7_gpu0_20260722
remote root = G:\lxy\blockcipher-structure-adaptive-nd-runs\i2_opd1_poshead_k7_retry1_20260722
expected results/history/checkpoints/cache = 40 / 500 / 5 / 196608
source = exact pushed commit in a run-owned clean clone
launch = Windows Task Scheduler, SYSTEM, cmd.exe /c, physical GPU0
retrieval = local tmux watcher plus verified result branch
```

远程运行脚本在训练前同时验证OPC1/OPN1 gate身份、SHA-256、状态、裁决和全部来源协议；训练数据按
`plaintexts.npy`、`features.npy`、`full_targets.npy`与`cache_metadata.json`持续落盘，进度写入
`progress.jsonl`，参数匹配时允许复用。启动脚本拒绝脏源码和错误提交，不使用`cmd.exe /k`或延迟
变量展开，所有项目文件均位于`G:\lxy`。

本地验证结果：

```text
focused pytest = 200 passed
Python compileall = pass
CLI help/import = pass
monitor bash syntax = pass
git diff --check = pass
remote package regression = pass
```

当前环境未声明或安装`ruff`，因此`uv run ruff`无法启动；这不是测试失败，也没有为此修改项目依赖。
下一动作保持唯一：范围提交并推送本计划、模型、runner、测试、配置和生成脚本，从推送后的精确HEAD
启动A6000正式任务，然后只做一次只读启动确认并交给本地tmux watcher。

## 10. 首次远程启动门失败与修复

首次从推送提交`f04f7fe815f4a6f8899847711b8888f7db545a4d`启动后，Windows计划任务实际执行，GPU与
PyTorch检查通过，但在训练和数据生成之前由readiness gate拒绝：

```text
started marker = absent
failed marker = present
readiness stderr = AssertionError
cache artifacts = absent
training result = not produced
```

根因是Windows Git checkout将提交中的OPN1 authority JSON从LF转换为CRLF。JSON内容没有改变，但
冻结的字节SHA-256从`887a7db3643e73bdda67958bcaae470881a09db25ab0ba5ff6c3d6bb0a2503d7`
变为`7948b1440d0a5b1a74ae168c0a84fe4072c961607bcd0c314d5dc99cd3efa8c9`，所以来源所有权门按设计
失败。OPC1远程gate仍保持期望SHA-256，CUDA可用且只暴露一张A6000。

修复不放宽哈希门：仓库新增`.gitattributes`，对
`configs/experiment/innovation2/authorities/*.json`设置`-text`，确保Windows克隆保留冻结字节；
同时让远程脚本把readiness失败写入明确的`failure_reason.txt`，并加入回归测试。修复提交推送后，
不得把首次失败记作模型性能结果或写入最近结果索引。

为避免复用首次checkout中已经发生换行转换的工作树，实际重试改用新的run-owned干净目录：

```text
G:\lxy\blockcipher-structure-adaptive-nd-runs\i2_opd1_poshead_k7_retry1_20260722
```

逻辑`run_id`、seed7、五模型矩阵、数据、epoch、门槛和结果分支名称不变；只改变远程物理目录与归档
目录。首次失败目录保留为启动门诊断证据，不删除、不覆盖。
