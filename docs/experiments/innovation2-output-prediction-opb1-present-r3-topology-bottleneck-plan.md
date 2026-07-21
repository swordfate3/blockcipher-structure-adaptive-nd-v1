# 创新2 OPB1：PRESENT三轮拓扑瓶颈输出预测计划

日期：2026-07-22

状态：已预注册 / 等待实现门

## 1. 研究问题

OPA1已经在相同八输出任务上比较MLP、LSTM、位置保持ResCNN、Transformer和
PRESENT-SPN-aware。OPA2在第四把固定秘密密钥上确认，原SPN-aware网络的八位置平均
`AUC=1.0`，显著超过MLP和匹配标签打乱。OPA3随后发现：原网络的真实P-layer与固定错误P-layer
都达到`AUC=1.0`，而identity仅为`0.531989557`。因此已有证据支持“分层跨nibble扩散有用”，但不
支持“精确PRESENT连线是增益来源”。

源码审计给出一个可检验机制：原网络为64个输入位置分别学习完整的189维位置向量；三个SPN block
之后，真实P与错误P都形成`1 -> 4 -> 16 -> 64`全局感受野。高维绝对位置自由度可能让网络只需任意
快速混合，而不必依赖真实PRESENT传播路径。

OPB1只回答：

> 在第五把独立固定秘密密钥上，将完整位置向量替换成每轮低秩固定密钥条件，同时保持局部4-bit混合、
> 三层传播、数据、输出位置和训练预算不变，能否保留真实输出预测能力，并使真实P-layer显著超过
> 固定错误P-layer？

这是新的结构瓶颈方法假设，不重开OPA4/OPA5，不测试四轮、XOR、完整密文恢复、样本分类或积分性质。

## 2. OPA3来源边界

OPB1必须读取下列已验证OPA3 gate，并确认它确实是`hold`而不是伪造的pass：

```text
run_id = i2_output_prediction_opa3_present_r3_selected8_topology_attribution_key3_gpu0_20260722
status = hold
decision = innovation2_selected8_present_topology_not_attributed
gate SHA256 = def55214d46acf0e199f465fda66e6ca394f094ceec78d419354357df1c50943
exact-P mean AUC = 1.0
wrong-P mean AUC = 1.0
attributed bits = 0/8
protocol/execution checks = all true
```

该来源只证明旧架构存在“错误拓扑同样饱和”的缺口。它不授权修改seed3结果、搜索更多错误排列、
改变输出位置，或直接扩展四轮。

## 3. 冻结任务契约

```text
cipher                  = PRESENT-80
rounds                  = 3
seed / fixed key        = 4，第五把独立固定秘密密钥
train rows              = 131072 total plaintext/ciphertext pairs
test rows               = 65536 total disjoint plaintext/ciphertext pairs
input                    = 64 MSB-first plaintext bits
target                   = 8 preregistered true ciphertext output bits
selected MSB positions   = [0, 2, 8, 10, 32, 34, 40, 42]
sample classification    = false
epochs                   = 100 per model
batch size               = 250
optimizer                = RMSprop
learning rate            = 0.001
loss                     = raw-output MSE
selection                = final epoch
```

明文、完整64-bit真实密文标签和固定秘密密钥必须按块持久化；参数匹配时复用，参数不匹配时拒绝。
训练/测试明文必须唯一且零重合。八个输出位置继续冻结自OP10，不在seed4重新选择。

## 4. 单一模型改造

原SPN-aware锚点：

```text
bit scalar -> token embedding
           + 每位置独立189维可训练位置向量
           -> [共享4-bit局部MLP -> 固定P重排 -> 共享channel MLP] x 3
           -> 八个冻结位置的共享输出头
```

新拓扑瓶颈候选只替换位置条件：

```text
删除：每位置独立189维位置向量
加入：每个block的64个标量条件 x 该block共享的189维方向
保留：bit embedding、共享4-bit局部MLP、固定P重排、共享channel MLP、三block和共享输出头
```

每个位置因此只能沿一个共享方向表达该轮固定密钥/位置条件，而不能携带一整套自由189维特征。
真实P与错误P候选使用相同参数量和相同初始化；唯一差异仍是无参数P映射。标签打乱行与真实P候选
使用同一模型和初始化，只改变训练标签对应关系。候选与原锚点参数差必须不超过3%。

冻结参数量为：

```text
原SPN-aware锚点          = 3,879,415
拓扑瓶颈exact-P候选      = 3,868,078
拓扑瓶颈wrong-P控制      = 3,868,078
候选相对锚点参数差        = -0.2922%
```

## 5. 四行受控矩阵

```text
selected8_present_spn_anchor_exact_p_true_output
  原SPN-aware，真实P，当前同协议最强锚点

selected8_topology_bottleneck_exact_p_true_output
  低秩位置/固定密钥条件，真实P，候选

selected8_topology_bottleneck_wrong_p_true_output
  同一候选，固定错误P，拓扑归因控制

selected8_topology_bottleneck_exact_p_label_shuffle
  同一候选，真实P，训练标签匹配打乱
```

不重复训练LSTM、Transformer、ResCNN或MLP；它们已在OPA1相同任务和预算下完成发现比较。OPB1也不
加入GNN、SPN-ResCNN、不同低秩宽度或更多wrong-P，以保持一个方法变量和四行必要控制。

## 6. 正式门

首先要求全部协议检查、`32`条逐bit结果、`400`条history、四个checkpoint hash和完整磁盘缓存通过。

候选的平均门为：

```text
anchor exact-P mean AUC >= 0.900
bottleneck exact-P mean AUC >= 0.900
bottleneck exact-P - anchor exact-P mean AUC >= -0.050
bottleneck exact-P - bottleneck wrong-P mean AUC >= +0.030
bottleneck exact-P - bottleneck shuffle mean AUC >= +0.030
```

至少`4/8`个冻结输出bit还必须同时满足：

```text
bottleneck exact-P AUC >= 0.550
bottleneck exact-P - wrong-P AUC >= +0.020
bottleneck exact-P - shuffle AUC >= +0.020
bottleneck exact-P - anchor AUC >= -0.100
accuracy - majority >= +0.005
```

本地`64 train / 64 test / 1 epoch / CPU`只检查实现和产物，不应用性能门。

## 7. 裁决与下一步

若全部正式门通过：保留拓扑瓶颈候选，先在第六把独立固定密钥`seed5`复现同一四行矩阵；只有
seed5也通过，才允许另行预注册四轮八输出实验。OPB1本身不构成四轮或高轮结果。

若候选超过wrong-P和shuffle，但未保持相对锚点的性能：裁决为“拓扑可归因但存在性能代价”，只允许
基于训练内验证预注册一次低秩条件表达改造；不得直接扩轮。

若候选未超过wrong-P或shuffle：停止该拓扑瓶颈路线，不增加数据、epoch、错误排列、输出位置或模型
名单。下一候选按现有证据优先考虑SPN-ResCNN混合，并建立新的独立计划。

## 8. 执行和产物

本地readiness通过后，范围提交并推送，从推送提交在远程A6000的run-owned干净clone启动。正式产物
全部位于`G:\lxy`，使用参数匹配磁盘缓存、逐epoch checkpoint、progress、`cmd.exe /c`、verified
result branch和本地tmux watcher。

每次完成的结果至少包含：

```text
results.jsonl
history.csv
progress.jsonl
metadata.json
opa3_gate.json
summary.json
gate.json
checkpoint_manifest.json
data/cache_metadata.json
curves.svg
```

结果回收后校验OPA3 gate哈希、来源提交、数据缓存、结果/history/checkpoint数量；可视化必须通过
`visual-qa-redraw`真实像素检查，并刷新`outputs/00_RECENT_RESULTS.md/json`。每个正式裁决必须在本
文追加实际指标、证据范围和下一推荐动作。

## 9. 本地readiness结果与远程交接

本地实现门已经完成：

```text
run_id = i2_output_prediction_opb1_present_r3_topology_bottleneck_smoke_20260722
train/test = 64/64 total plaintext-ciphertext pairs
models = 4
epochs = 1 each
result/history/checkpoint = 32/4/4
cache = complete, 128/128 rows
protocol checks = all true
execution checks = all true
status = pass
decision = innovation2_topology_bottleneck_local_smoke_passed
```

该门验证了第五固定密钥、真实密文输出回放、冻结八位置、OPA3 hold所有权、真实/错误P初始参数一致、
候选无完整位置向量、每block仅64个标量位置条件、参数匹配、标签打乱只作用训练行、测试标签保持真实，
以及磁盘缓存、progress、checkpoint和结果闭环。64条测试数据上的AUC不作性能解释。

用户可查看产物：

```text
outputs/local_readiness/
  i2_output_prediction_opb1_present_r3_topology_bottleneck_smoke_20260722/
```

`curves.svg`已按`visual-qa-redraw`渲染为1920×1260像素检查；标题、协议说明、热图、色条、模型标签、
图例、差值轴和底部裁决均无重叠、裁切或含混范围。最近结果索引已刷新，本次readiness为完成时的
`001`。

冻结的正式远程运行是：

```text
run_id = i2_output_prediction_opb1_present_r3_topology_bottleneck_key4_gpu0_20260722
remote directory = G:\lxy\blockcipher-structure-adaptive-nd-runs\i2_opb1_tbneck_k4_20260722
physical GPU = 0
train/test = 131072/65536 total pairs
epochs = 100 x 4 models
expected result/history/checkpoint/cache = 32/400/4/196608
OPA3 gate SHA256 = def55214d46acf0e199f465fda66e6ca394f094ceec78d419354357df1c50943
```

推荐下一步是：范围提交并推送本次计划、模型、CLI、测试和远程包；随后从推送提交在A6000启动该
四行正式实验，并把等待/回收交给本地tmux watcher。理由是readiness已经排除实现与协议阻塞，而
低秩瓶颈是否能在固定预算下拉开真实P和错误P只能由正式训练回答。不得把readiness的随机小样本
AUC用于取消或宣传候选。
