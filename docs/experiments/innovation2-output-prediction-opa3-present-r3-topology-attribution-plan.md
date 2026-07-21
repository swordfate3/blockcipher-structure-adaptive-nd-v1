# 创新2 OPA3：PRESENT三轮输出预测P-layer拓扑归因计划

日期：2026-07-22

状态：条件式预注册 / 等待OPA2正式门

## 1. 研究问题

OPA1在第三把固定未知密钥上发现`PRESENT-SPN-aware`对八个预注册真实密文输出bit的平均
AUC为`1.0`，OPA2正在第四把密钥上用同架构标签打乱和MLP匹配控制确认该发现。即使OPA2通过，
仍只能证明整个SPN-aware网络有效，不能单独证明收益来自真实PRESENT拓扑。

OPA3只回答一个归因问题：保持网络参数、局部nibble混合、训练数据和预算全部不变时，精确
PRESENT P-layer是否显著优于identity和固定错误排列？

## 2. 启动权限

OPA3不得手工选择候选或绕过OPA2。启动前必须读取OPA2正式`gate.json`并同时满足：

```text
run_id = i2_output_prediction_opa2_present_r3_selected8_present_spn_key3_gpu0_20260722
status = pass
decision = innovation2_selected8_architecture_priority_independently_confirmed
candidate_architecture = present_spn
protocol_checks = all true
execution_checks = all true
metrics.priority_passed = true
```

若OPA2为`hold`、`fail`、结果不完整或未通过来源哈希验证，OPA3保持关闭，不得通过增加seed、
数据、epoch或删除控制重新开启。

## 3. 冻结任务契约

OPA3复用OPA2第四固定密钥及相同输出预测任务：

```text
cipher                  = PRESENT-80
rounds                  = 3
seed / fixed key        = 3
train rows              = 131072 total plaintext/ciphertext pairs
test rows               = 65536 total disjoint plaintext/ciphertext pairs
input                    = 64 MSB-first plaintext bits
target                   = 8 preregistered true ciphertext output bits
selected MSB positions   = [0, 2, 8, 10, 32, 34, 40, 42]
sample classification    = false
epochs                   = 100 per model
batch size               = 250
optimizer                = RMSprop
loss                     = raw-output MSE
learning rate            = 0.001
selection                = final epoch
```

明文、密钥、64-bit目标和特征必须继续使用参数匹配的磁盘缓存；训练/测试明文唯一且零重合。
不得改变输出位置、目标语义、损失、指标、训练样本或测试样本。

## 4. 单变量三行矩阵

三行都使用`token_dim=189`、三个SPN block、相同位置嵌入、局部4-bit混合、channel MLP、输出头、
参数初始化和batch顺序。唯一变量是每个block中的无参数bit排列：

```text
present_spn_exact_p_true_output
  精确PRESENT MSB-first P-layer source-for-destination映射

present_spn_identity_p_true_output
  identity映射，64个位置不发生跨nibble扩散

present_spn_wrong_p_true_output
  对精确映射的destination轴固定循环移动1位；仍是64位置双射，
  与精确映射64/64位置不同，但不是PRESENT P-layer
```

identity与wrong-P不是随机多次搜索；映射在揭盲前固定且只运行一次。三行参数量必须完全相同。
OPA2已经提供同一seed、同一任务下的exact-P匹配标签打乱和MLP控制，OPA3不重复扩大矩阵。

## 5. 正式门

首先要求exact-P复跑与OPA2 exact-P平均AUC之差绝对值不超过`0.005`，排除代码变化或训练协议
漂移。拓扑归因通过还必须同时满足：

```text
exact-P mean AUC >= 0.510
exact-P mean AUC - max(identity-P, wrong-P) mean AUC >= +0.030
至少4/8个bit满足：
  exact-P AUC >= 0.510
  exact-P AUC - identity-P AUC >= +0.020
  exact-P AUC - wrong-P AUC >= +0.020
```

全部protocol/execution checks、24条逐bit结果、300条history和三个checkpoint hash必须完整。
smoke只验证实现，不应用性能门。

## 6. 裁决和下一步

若通过，允许的结论仅为：在第四固定未知密钥、PRESENT三轮、八个预注册真实输出bit和当前预算
下，精确P-layer拓扑对SPN-aware输出预测有可重复的因果归因证据。下一步才允许预注册OPA4：
以精确SPN-aware、错误P控制、MLP和标签打乱构成的PRESENT四轮单bit输出预测门。OPA3通过本身
不是四轮、高轮、完整密文恢复、跨密钥总体统计或SOTA证据。

若未通过，保留OPA2的整体架构结果（若OPA2通过），但不得声称收益来自精确PRESENT拓扑；停止
错误排列搜索、seed扩展和四轮推进，回到架构机制解释或论文边界收束。

## 7. 产物要求

正式运行至少生成：

```text
results.jsonl
progress.jsonl
history.csv
metadata.json
opa2_gate.json
summary.json
gate.json
checkpoint_manifest.json
data/cache_metadata.json
curves.svg
```

结果回收后必须校验OPA2 gate哈希、源提交、缓存、结果行、history和checkpoint；图像经过
`visual-qa-redraw`真实像素检查后才能标记完成，并在同一回合刷新`outputs/00_RECENT_RESULTS.md/json`。
