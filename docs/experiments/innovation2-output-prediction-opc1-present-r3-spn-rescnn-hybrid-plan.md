# 创新2 OPC1：PRESENT三轮 SPN-ResCNN 混合输出预测计划

日期：2026-07-22

状态：本地readiness通过 / OPB1负归因正式授权 / seed6远程正式矩阵运行中

## 1. 研究问题

OPA1并非只测试MLP或LSTM，而是在相同八输出任务上比较了MLP、六层LSTM、位置保持ResCNN、
Transformer和PRESENT-SPN-aware。第三固定密钥上的平均AUC分别为`0.531657`、`0.500000`、
`0.588388`、`0.499477`和`1.000000`。ResCNN是当前最强的非异常饱和锚点；原SPN-aware虽强，
但OPA3中真实P与错误P都为`1.0`，不能归因于精确PRESENT连线。

OPC1只测试一个新假设：

> 保留已证实有信号的位置保持ResCNN，在三个残差阶段之间加入固定P-layer位置重排，能否在不增加
> 参数和训练预算的条件下，同时超过普通ResCNN、错误P-layer控制和标签打乱控制？

它不是重新搜索模型名单，不测试四轮、XOR、完整密文恢复、积分性质或真假样本分类。

## 2. 模型与唯一变量

普通ResCNN锚点：

```text
64个明文bit -> Conv1D stem -> 10个残差块 -> 保留64位置的扁平输出头 -> 8个真实密文bit
```

SPN-ResCNN候选：

```text
64个明文bit -> Conv1D stem
              -> 3个残差块 -> 固定P重排
              -> 3个残差块 -> 固定P重排
              -> 4个残差块 -> 固定P重排
              -> 同一扁平输出头 -> 8个真实密文bit
```

候选与锚点使用完全相同的252 channels、10个残差块和输出头，参数量必须严格相等。真实P与错误P
候选使用相同初始化，唯一差异是无参数位置映射。

## 3. 四行同预算矩阵

```text
selected8_rescnn_anchor_true_output
selected8_spn_rescnn_exact_p_true_output
selected8_spn_rescnn_wrong_p_true_output
selected8_spn_rescnn_exact_p_label_shuffle
```

四行共享相同明文、真实密文、八个冻结位置、optimizer、loss、epoch和batch。标签打乱只打乱训练
标签对应关系，测试标签始终是真实输出值。

## 4. 正式协议与启动硬门

```text
cipher / rounds         = PRESENT-80 / 3
fixed key               = seed6，第七把独立秘密密钥
train/test              = 131072 / 65536 total plaintext-ciphertext pairs
selected MSB positions  = [0,2,8,10,32,34,40,42]
epochs / batch          = 100 / 250
optimizer / lr          = RMSprop / 0.001
loss / selection        = raw-output MSE / final epoch
device                  = lxy-a6000 physical GPU0
```

正式模式只有在已回收、全部协议与执行检查通过的OPB1 gate同时满足下列条件时才可运行：

```text
status = hold
decision = innovation2_topology_bottleneck_not_attributed
metrics.attribution_passed = false
```

如果OPB1通过，则优先执行其seed5不变确认，OPC1保持关闭；如果OPB1仅有归因但性能下降，则按OPB1
计划只允许一次低秩条件修复，OPC1仍不抢占正式实验槽。

## 5. 正式性能门

```text
候选平均AUC >= 0.550
候选 - ResCNN平均AUC >= +0.010
候选 - 错误P平均AUC >= +0.020
候选 - 标签打乱平均AUC >= +0.030
至少4/8个bit同时通过逐bit AUC、锚点增益、控制增益和accuracy-majority门
```

正式通过只产生新密钥确认候选，不开放四轮。正式失败则保留ResCNN为当前非泄漏锚点，停止该混合
路线，不通过加深网络、增加数据、换输出位或搜索更多错误P绕过门。

## 6. 本地readiness

本地只运行：

```text
seed6
64 train / 64 test total pairs
1 epoch
CPU
4 models / 32 result rows / 4 history rows / 4 checkpoints
```

readiness只校验真实输出回放、独立密钥、模型参数相等、真实/错误P初始权重相同、标签打乱隔离、
缓存、checkpoint、JSONL/CSV/gate/SVG闭环。其AUC不作性能裁决。

完成产物位于：

```text
outputs/local_readiness/i2_output_prediction_opc1_present_r3_spn_rescnn_hybrid_smoke_seed6_20260722/
```

## 7. 推荐下一步

先完成本地readiness并保留为待命方法。OPB1正式结果回收后严格分支：OPB1完整通过则做seed5原样
确认；OPB1有归因但性能不足则只做一次低秩修复；只有OPB1未归因时，才从推送提交构造OPC1远程
包并启动上述seed6四行矩阵。这样既扩大模型探索，也避免在同一问题上并行堆模型造成后验选择偏差。

## 8. 本地readiness结果

实际运行完成：

```text
run_id = i2_output_prediction_opc1_present_r3_spn_rescnn_hybrid_smoke_seed6_20260722
train/test = 64/64 total plaintext-ciphertext pairs
models = 4
epochs = 1
result/history/checkpoint = 32/4/4
status = pass
decision = innovation2_spn_rescnn_hybrid_local_smoke_passed
```

全部协议与执行检查通过，包括：seed6秘密密钥与seed0--5均不同；训练/测试明文互异且零重合；目标
逐条回放为真实PRESENT三轮密文bit；普通ResCNN、真实P混合模型和错误P混合模型参数量严格相等；
真实P与错误P可训练状态在初始化时逐参数相同；三阶段拓扑重排存在；标签打乱只作用训练标签；磁盘
缓存、32条结果、4条history和4个checkpoint hash完整。

本地64条测试集上的平均AUC为：

```text
ResCNN anchor     = 0.506087
hybrid exact-P    = 0.493491
hybrid wrong-P    = 0.513269
hybrid shuffle    = 0.466814
```

这些值只记录产物回放，不应用正式性能门，不能用于保留或删除候选。`curves.svg`已通过
`visual-qa-redraw`像素检查：中文标题与裁决、热图数字、色条、图例、坐标和证据边界无重叠、裁切或
低对比度问题。

正式结果回收前又用候选/错误P接近`1.0`、shuffle接近`0.5`的极端合成值预检绘图路径。原共享差值
纵轴会让关键`真实P - 错误P`小差值几乎不可见，且图例遮挡顶部柱形；现已改为ResCNN、错误P和
shuffle三个独立纵轴面板，分别标出逐bit门槛和三位小数差值。重新生成的readiness `curves.svg`在
1920px与1280px渲染下通过`visual-qa-redraw`，无文字重叠、裁切、尺度歧义或不可读关键比较。

OPB1正式结果已经回收并满足唯一授权分支：

```text
run_id = i2_output_prediction_opb1_present_r3_topology_bottleneck_key4_gpu0_20260722
status = hold
decision = innovation2_topology_bottleneck_not_attributed
candidate exact-P mean AUC = 1.0
candidate wrong-P mean AUC = 1.0
candidate - wrong-P = 0.0
attributed bits = 0/8
gate SHA256 = 776a43a7e0b13e9db17d825ec20f83fc6ce54ca8a36408849d7007a8ec46a549
```

因此OPC1正式seed6四行矩阵现已授权。下一动作是验证冻结远程包，范围提交并推送，从推送提交在A6000
GPU0启动`131072/65536`、`100 epochs × 4`模型实验，并交由独立本地tmux watcher回收。该实验仍只
是PRESENT三轮模型归因，不开放四轮或五轮。

## 9. 正式远程启动状态

正式矩阵已经从推送提交启动：

```text
run_id = i2_output_prediction_opc1_present_r3_spn_rescnn_hybrid_key6_gpu0_20260722
source commit = 286cd0cd44238c3ae7095461570c80146066659c
remote device = lxy-a6000 physical GPU0
train/test = 131072/65536 total plaintext-ciphertext pairs
models/epochs = 4 models x 100 epochs
remote root = G:\lxy\blockcipher-structure-adaptive-nd-runs\i2_opc1_hybrid_k6_20260722
status = running
```

启动后的单次只读确认已经看到`readiness=status=pass`、started marker、`progress.jsonl`以及参数匹配的
磁盘缓存`features.npy`、`full_targets.npy`、`plaintexts.npy`和`cache_metadata.json`。本地tmux
会话`i2_opc1_hybrid_k6_watch_20260722`已接管稀疏监控、验证分支回收、hash与协议验证、绘图和结果
索引刷新；主线程不再SSH轮询。

当前只能称为“远程运行中”，不能提前填写正式AUC或裁决。结果回收后严格执行冻结分支：通过则仅做
全新固定密钥原样确认；失败或hold则保留ResCNN发现锚点并停止SPN-ResCNN混合路线，不后验增加
网络深度、数据、epoch、错误P或输出位置，也不直接开放四轮。
