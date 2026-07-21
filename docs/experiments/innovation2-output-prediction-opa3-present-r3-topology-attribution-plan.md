# 创新2 OPA3：PRESENT三轮输出预测P-layer拓扑归因计划

日期：2026-07-22

状态：正式seed3完成 / verified result branch回收 / hold / OPA4与OPA5关闭

## 1. 研究问题

OPA1在第三把固定未知密钥上发现`PRESENT-SPN-aware`对八个预注册真实密文输出bit的平均
AUC为`1.0`；OPA2随后在第四把密钥上用同架构标签打乱和MLP匹配控制独立确认了该发现。OPA2
只能证明整个SPN-aware网络有效，不能单独证明收益来自真实PRESENT拓扑。

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
下，精确P-layer拓扑对SPN-aware输出预测有可重复的因果归因证据。下一步才允许执行预注册OPA4：
以精确SPN-aware、错误P控制、MLP和标签打乱构成的PRESENT四轮八输出预测门。OPA3通过本身
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

## 8. 本地实现门结果

```text
run_id = i2_output_prediction_opa3_present_r3_selected8_topology_attribution_smoke_20260722
train/test = 64/64 total pairs
epochs = 1
result rows = 24/24
history rows = 3/3
checkpoint hashes = 3/3
cache rows = 128/128
protocol/execution checks = all true
status = pass
decision = innovation2_selected8_topology_attribution_local_smoke_passed
```

smoke使用synthetic OPA2 pass gate只验证gate ownership接口、三种映射、数据、训练、指标和产物闭环，
不构成正式OPA2授权，也不解释64条测试上的AUC。真实、identity和固定错误映射均为64位置双射；
错误映射与真实P在`64/64`个destination位置不同；三个模型参数量完全相同。

`curves.svg`由`visual-qa-redraw`转为1920×1080等效像素图检查。标题、三段协议说明、热图数字、
色条、坐标标签、四个面板和底部裁决无重叠、裁切、缺字或模糊范围，已写入
`visual_qa_passed.marker`。最近结果索引已刷新，当前为`001`。

本地smoke不得替代OPA2正式授权或OPA3正式结果。

## 9. OPA2正式授权与远程执行包

OPA2已经从verified result branch完整回收并满足本计划第2节的全部授权条件：

```text
run_id = i2_output_prediction_opa2_present_r3_selected8_present_spn_key3_gpu0_20260722
source commit = 3ddd346e70fb68f5d143a3253639acf10f33f4b3
status = pass
decision = innovation2_selected8_architecture_priority_independently_confirmed
candidate_architecture = present_spn
candidate mean true AUC = 1.000000000
candidate mean shuffle AUC = 0.500839804
candidate - MLP mean true AUC = +0.467737769
adjusted matched-control gain = +0.465751518
result/history/checkpoint/cache = 32/400/4/196608
protocol/execution checks = all true
gate SHA256 = 97943c59a8d88f8bbc1b6845aa6372a8b91d5693e0abd3a23e63b41259601284
```

因此OPA3正式门已解锁。冻结远程运行：

```text
run_id = i2_output_prediction_opa3_present_r3_selected8_topology_attribution_key3_gpu0_20260722
remote run root = G:\lxy\blockcipher-structure-adaptive-nd-runs\i2_opa3_ptopo_k3_20260722
physical GPU = 0
train/test = 131072/65536 total plaintext-ciphertext pairs
models = exact-P / identity-P / fixed-wrong-P
epochs = 100 x 3
expected result/history/checkpoint/cache = 24/300/3/196608
```

正式experiment JSON、remote JSON、短路径Windows run/launch脚本和本地tmux watcher均绑定上述
OPA2 gate哈希。运行只允许从已推送提交的run-owned干净clone启动，使用`cmd.exe /c`，所有数据、
缓存、checkpoint、日志、归档和结果均位于`G:\lxy`。完成后优先从verified result branch回收，重新
绘图、执行`visual-qa-redraw`和刷新最近结果索引。

## 10. 正式远程启动

OPA3于`2026-07-22`从已推送提交启动：

```text
source commit = 1dc74e9d5cb703f22018f7e5fffff30e39e348db
run_id = i2_output_prediction_opa3_present_r3_selected8_topology_attribution_key3_gpu0_20260722
remote run root = G:\lxy\blockcipher-structure-adaptive-nd-runs\i2_opa3_ptopo_k3_20260722
physical GPU = 0
status = completed remotely / verified result branch retrieved
```

启动使用run-owned短路径干净clone，没有修改或重置旧远程主克隆。一次性有界确认已经验证：远程
`source_expected_commit.txt`精确匹配上述提交、readiness=`pass`、started marker与`progress.jsonl`
存在、`data/cache_metadata.json`已经持久化，且磁盘缓存完成`196608/196608`条。`data_ready`事件中的
所有数据、密钥、标签和三种P-layer协议检查均为真。

本地tmux会话`i2_opa3_ptopo_k3_watch_20260722`随后完成了结果分支等待、自动回收、校验、绘图和
最近结果索引刷新；主线程没有使用重复SSH轮询替代watcher。

## 11. 正式结果与裁决

OPA3已经从verified result branch完整回收。来源与产物检查为：

```text
source commit        = 1dc74e9d5cb703f22018f7e5fffff30e39e348db
gate SHA256          = def55214d46acf0e199f465fda66e6ca394f094ceec78d419354357df1c50943
results/history      = 24 / 300
checkpoints          = 3，全部具有SHA256
cache                = complete，196608 / 196608
protocol checks      = all true
execution checks     = all true
```

Windows生成的`SHA256SUMS`含CRLF；Linux端只对清单输入流去除行尾CR后执行`sha256sum -c`，20个归档
文件全部通过，没有改写清单或载荷。正式平均AUC为：

| 模型 | 八位置平均AUC | 平均accuracy-majority |
|---|---:|---:|
| exact-P | 1.000000000 | +0.499097824 |
| identity-P | 0.531989557 | +0.030904770 |
| fixed-wrong-P | 1.000000000 | +0.499097824 |

exact-P在八个位置分别比identity高`0.463337294`至`0.473109524`，说明跨nibble扩散相对不扩散控制有
明显作用；但exact-P与wrong-P在八个位置的AUC差值全部精确为`0.0`。因此：

```text
exact mean - best control mean AUC = +0.000000000
attributed bits                      = 0 / 8
exact reproduction delta vs OPA2    = 0.000000000
status                               = hold
decision = innovation2_selected8_present_topology_not_attributed
```

确定性感受野审计解释了控制结果：exact-P与wrong-P都按`1 -> 4 -> 16 -> 64`扩展，三个block后每个
预注册输出位置都可接触全部64个明文bit；identity-P则保持`1 -> 4 -> 4 -> 4`。正式结果支持“分层
跨nibble扩散骨架有效”，不支持“只有精确PRESENT连线才有效”。这不是协议失败，也不推翻OPA2的整体
架构结果，但精确P-layer因果主张不成立。

正式`curves.svg`经`visual-qa-redraw`渲染为1920×1279像素检查。标题、说明、热图、色条、坐标、
数值、四个面板和底部裁决均无重叠、裁切、缺字、模糊尺度或误导性放大；pending标记已替换为
`visual_qa_passed.marker`。`scripts/index-results`返回`status=pass`，OPA3正式结果为最近结果`001`。

按第6节预注册裁决，OPA4与OPA5均保持关闭且从未实现或启动；不继续wrong-P搜索、seed扩展、四轮、
五轮、加样本、加epoch或临时模型枚举。下一步是把OPA2整体架构正结果和OPA3“精确连线不唯一”的
机制边界写入论文，而不是把OPA3改写成精确拓扑成功。
