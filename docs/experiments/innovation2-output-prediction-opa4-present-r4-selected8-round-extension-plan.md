# 创新2 OPA4：PRESENT四轮固定八输出SPN-aware轮数扩展门

日期：2026-07-22

状态：条件式预注册 / 等待OPA3正式门 / 未实现未启动

## 1. 研究问题

OPA1在第三固定密钥发现PRESENT-SPN-aware明显优于MLP，OPA2在第四固定密钥及两种架构匹配标签
打乱下独立确认了整体架构优势。OPA3正在判断该优势能否归因于精确PRESENT P-layer。

若OPA3通过，OPA4只回答一个问题：保持第四固定密钥、明文split、八个预注册输出位置、模型、参数
预算、损失、优化器和训练预算不变，只把PRESENT从三轮改为四轮后，精确P-layer SPN-aware网络是否
仍能预测同八个真实密文输出bit，并同时超过MLP、匹配标签打乱和固定错误P控制？

```text
input  = 未见明文P的64个MSB-first bit
C      = PRESENT_K^4(P)
target = [C[0], C[2], C[8], C[10], C[32], C[34], C[40], C[42]]
```

目标是八个真实密文输出值，不是真假样本、积分平衡类别、完整64-bit恢复或结构化XOR。

## 2. 与OP12的边界

OP12已经在第二固定密钥上否定“MLP直接预测六个结构化XOR可把三轮信号推进到四轮”：六个mask均
接近随机且`0/6`过门。OPA4不重新开启该停止路线，也不更换mask或增加OP12数据。

OPA4的新增依据是OP12完成后才得到的OPA1/OPA2 SPN-aware架构证据。它保留原始八个逐bit输出
契约，测试结构网络能否延长轮数；因此不得把OPA4结果写成OP12 XOR路线的成功或失败复现。

## 3. 唯一启动权限

OPA4必须同时绑定OPA2与OPA3正式gate。OPA3必须满足：

```text
status = pass
decision = innovation2_selected8_present_topology_independently_attributed
metrics.priority_passed = true
protocol_checks = all true
execution_checks = all true
```

OPA3为`hold`、`fail`、结果不完整、来源未验证或正式图未完成质检时，OPA4不得实现或启动。不得通过
删除identity/wrong-P控制、放宽OPA3门、增加seed/epoch/数据或手工覆盖gate来获得授权。

## 4. 冻结任务契约

OPA4与OPA3相比只改变轮数：

```text
cipher                  = PRESENT-80
rounds                  = 4
seed / fixed key        = 3，与OPA2/OPA3相同的第四固定未知密钥
train rows              = 131072 total plaintext-ciphertext pairs
test rows               = 65536 total disjoint plaintext-ciphertext pairs
plaintext split         = 与OPA2/OPA3相同的确定性唯一明文split
input                    = 64 MSB-first plaintext bits
target                   = 八个预注册MSB-first真实密文bit
selected MSB positions   = [0, 2, 8, 10, 32, 34, 40, 42]
sample classification    = false
epochs                   = 100 per model
batch size               = 250
optimizer                = RMSprop
loss                     = raw-output MSE
learning rate            = 0.001
selection                = final epoch
```

四轮密文目标必须重新按块生成并写入OPA4独立磁盘缓存；缓存元数据必须包含`rounds=4`并拒绝复用
三轮目标。训练/测试明文唯一、零重合，PRESENT官方向量和逐bit标签重放继续作为协议门。

## 5. 五行受控矩阵

OPA4是轮数扩展phase gate，五行均为必要锚点或控制：

```text
selected8_mlp_true_output
selected8_mlp_label_shuffle
selected8_present_spn_exact_p_true_output
selected8_present_spn_exact_p_label_shuffle
selected8_present_spn_wrong_p_true_output
```

MLP与精确P模型沿用OPA2参数量和超参数；错误P模型与精确P模型参数、初始化、局部nibble混合、训练
顺序和输出头完全相同，唯一差异是OPA3冻结的destination轴循环移动1位的无参数双射。两种shuffle
使用同一个预注册训练行排列，测试标签始终是真实四轮密文输出bit。

OPA4不加入identity-P、ResCNN、LSTM、Transformer、XOR头、更多输出位置或改造网络。identity-P已在
OPA3用于三轮机制归因；OPA4保留固定wrong-P作为非平凡同容量拓扑控制，以限制远程矩阵规模。

## 6. 正式门

首先要求全部protocol/execution checks、40条逐bit结果、500条history、5个checkpoint hash和
196608条缓存完整。性能门必须同时满足：

```text
exact-P mean true AUC >= 0.510
exact-P mean true AUC - MLP mean true AUC >= +0.003
[(exact true - exact shuffle) - (MLP true - MLP shuffle)] mean AUC >= +0.003
exact-P mean true AUC - wrong-P mean true AUC >= +0.030
```

至少`4/8`个输出bit还必须逐项满足：

```text
exact-P true AUC >= 0.510
exact-P accuracy - majority >= +0.005
exact-P true AUC - exact-P shuffle AUC >= +0.005
exact-P true AUC - MLP true AUC >= +0.002
exact-P true AUC - wrong-P true AUC >= +0.020
```

所有阈值在OPA3揭盲前冻结，不按四轮结果修改。报告全部八位置，不得只展示最高bit。

## 7. 裁决和后续

若OPA4通过，允许的结论仅为：在第四固定未知密钥、冻结的八个真实输出bit和当前预算下，经过三轮
独立确认与拓扑归因的PRESENT-SPN-aware网络把受控输出预测信号延伸到四轮。下一步不是直接五轮，
而是使用第五固定未知密钥`seed4`、同一四轮五行矩阵做一次独立密钥确认。

只有seed4复现同一主门，才允许预注册五轮readiness；届时仍不得称为完整密文恢复、普遍跨密钥
统计、主流最高轮次或SOTA。OPA4已经通过时不运行结构救援实验。

若OPA4协议与执行门完整、但性能门未通过，只开放揭盲前已冻结的OPA5单次结构救援：在参数差小于
`1%`时把三个SPN block改成四个并缩窄token维度，比较同数据三block锚点、四block exact-P、匹配
shuffle和wrong-P。OPA5以
`docs/experiments/innovation2-output-prediction-opa5-present-r4-depth-matched-spn-plan.md`为唯一协议。

除此之外，停止当前SPN-aware路线的四轮/五轮、seed、数据、epoch和模型扩展。不得用后验bit筛选、
结构化XOR、更多参数、其他任务或临时网络枚举补写为同一轮数轨迹。OPA4若因协议/来源/完整性失败，
OPA5保持关闭，必须先解决原实验有效性。

## 8. 执行与产物要求

OPA3通过后才允许实现本地`64 train / 64 test / 1 epoch / CPU` readiness。smoke只验证gate绑定、
四轮缓存隔离、共享shuffle、五行训练、结果/历史/checkpoint数量和中文绘图，不应用性能门。

正式训练属于远程实验，必须从已推送提交的run-owned干净短路径clone在A6000运行，使用磁盘缓存、
进度日志、checkpoint恢复、`cmd.exe /c`、verified result branch和本地tmux watcher。至少生成：

```text
results.jsonl
progress.jsonl
history.csv
metadata.json
opa2_gate.json
opa3_gate.json
summary.json
gate.json
checkpoint_manifest.json
data/cache_metadata.json
curves.svg
```

回收后必须验证gate哈希、来源提交、缓存、40/500/5数量和全部协议门；正式图通过
`visual-qa-redraw`真实像素检查，并刷新`outputs/00_RECENT_RESULTS.md/json`，之后才能报告裁决。
