# 创新2 OPF3：PRESENT四轮共享逐轮SPN递推输出预测条件计划

日期：2026-07-23

状态：盲预注册 / OPF2运行中仅开放实现准备 / readiness、训练和启动仍由OPF2正式hold授权

## 1. 条件授权

OPF3不是与OPF2并行的模型搜索。OPF2正式指标仍不可见时，只允许按照本计划已冻结的网络结构编写模型代码和
结构单元测试；不得读取OPF2性能、生成OPF3数据、运行readiness、训练、调宽度或启动远程任务。只有本地回收
并验证OPF2正式产物后，同时满足以下条件，才开放readiness和后续实验执行：

```text
OPF2 run_id   = i2_output_prediction_opf2_present_r4_position_bound_spn_rescnn_2p20_key7_gpu0_20260722
OPF2 status   = hold
OPF2 decision = innovation2_position_bound_r4_scale_not_supported
results/history/checkpoints = 40 / 500 / 5
cache rows    = 1114112 complete
source commit、SHA256、切分hash、protocol和execution checks全部通过
```

若OPF2主门通过，OPF3保持关闭；下一步必须是OPF2原样更换全新固定未知密钥确认，确认前不进入五轮。

## 2. 为什么候选只能是逐轮SPN递推器

现有同任务证据已经排除缺少机制依据的模型枚举：

```text
OPA1三轮：LSTM和Transformer接近随机；不再重复。
OPA3三轮：普通SPN exact-P和wrong-P都饱和到AUC=1；不再搜索错误P。
OPB1三轮：低秩拓扑瓶颈仍不能区分exact-P和wrong-P；路线停止。
OPC1三轮：在通用ResCNN阶段之间插P层低于普通ResCNN；混合路线停止。
OPD1三轮：位置绑定头恢复预测，但exact-P不优于wrong-P。
OPF1四轮：位置绑定ResCNN在2^17训练行下平均AUC仅0.513755358。
```

这些结果支持的缺口不是“网络还不够大”，而是当前四轮候选仍用通用残差块近似完整密码函数；公开的
`S-box局部非线性 -> P层跨cell路由`只作为少数无参数重排插入，且各阶段没有共享同一个轮函数归纳偏置。

OPF3只测试一个新机制：使用同一组可训练局部算子显式递推四次，每次严格先在16个4-bit S-box cell内
混合，再按公开P层路由；每轮位置上下文只吸收固定未知轮密钥造成的轮次差异。它不是LSTM、Transformer、
GNN、更多ResCNN block、更多epoch或更多训练样本。

## 3. 研究问题与同预算锚点

研究问题：

> 在OPF2完全相同的PRESENT-80四轮固定密钥、明文、八个真实输出bit、`2^20/2^16`训练测试规模和
> 100 epochs下，共享权重的逐轮SPN递推器能否超过OPF2位置绑定ResCNN，并通过真实输出预测门？

同预算锚点是OPF2的`selected8_position_head_spn_rescnn_exact_p_true_output`正式结果。OPF3不重新训练
OPF2五模型，也不更改数据、目标、指标或门槛。

唯一模型变量：

```text
OPF2：10个通用ResCNN block，按3+3+4分段，在阶段后插入P路由
OPF3：1个共享SPN round block递推4次，每次执行4-bit局部混合后立即执行P路由
```

## 4. 冻结数据与训练协议

```text
cipher / rounds        = PRESENT-80 / 4
fixed unknown key      = OPF2同一seed7密钥
input                  = 64个MSB-first明文bit
target positions       = [0, 2, 8, 10, 32, 34, 40, 42]
target semantics       = 同一输入真实四轮密文的八个独立输出bit
sample classification  = false
train rows             = 1048576 = 2^20 total
test rows              = 65536 = 2^16 total
train indices          = [0,131072) U [196608,1114112)
test indices           = [131072,196608)
epochs / batch         = 100 / 250
optimizer / lr         = RMSprop / 0.001
loss                   = raw-output MSE
checkpoint selection   = final epoch
device                 = remote A6000 GPU0 after OPF2 releases it
```

必须保留OPF1训练前缀和测试段hash：

```text
OPF1 train plaintext raw SHA256 = eca0f5705c2d9a6b4f0475bfb90e55d2bfa2d5e4d7b8c380b10ab55778a4555a
OPF1 test plaintext raw SHA256  = 5c5410d4c0761f729f5f705d43a7392bf90f6ae0bee65a57321760d515b82fec
```

`plaintexts.npy/features.npy/full_targets.npy/cache_metadata.json`必须在新run root中按chunk落盘，记录
非连续切分和参数匹配复用。不得从OPF2结果中后验重选bit、明文、密钥或训练段。

## 5. 冻结网络结构

候选名预留为`SelectedOutputRoundRecurrentSpn`，结构冻结为：

```text
64个输入bit
  -> 每bit Linear(1,316) + 64位置嵌入
  -> 对round t=0..3重复同一个共享block：
       加第t轮可训练的64位置上下文
       reshape为16个cell x 4个bit x 316维
       共享 local MLP：1264 -> 1264 -> 1264，GELU，残差
       reshape回64位置
       exact-P无参数路由
       共享 channel MLP：316 -> 632 -> 316，GELU，残差
  -> 加最终白化层64位置上下文
  -> 八个冻结位置各自使用 MLP(316 -> 64 -> 1)
  -> 八个真实密文输出值
```

局部MLP、channel MLP和归一化参数在四轮间严格共享；仅轮次位置上下文独立。模型不读取秘密密钥、
轮密钥、中间状态或密文标签以外的信息。`token_dim=316`用于把总参数量保持在OPF2约396万参数的
`3%`范围内；readiness必须计算真实参数量，超出范围则视为实现失败，不得根据AUC调宽度。

## 6. 三行受控矩阵

```text
selected8_round_recurrent_spn_exact_p_true_output
  主候选：共享逐轮block + exact PRESENT P路由 + 真实输出标签

selected8_round_recurrent_spn_identity_p_true_output
  等参数控制：共享逐轮block + identity路由 + 真实输出标签

selected8_round_recurrent_spn_exact_p_label_shuffle
  匹配控制：与主候选相同初始化和架构，只打乱训练标签
```

不再加入wrong-P搜索。identity行只判断跨cell逐轮路由是否有用；即使exact通过，也不据此声称精确
PRESENT连线相对所有错误双射具有唯一性。OPF2正式exact-P结果作为外部同预算锚点加入裁决，不重复训练。

## 7. 裁决门

输出预测主门沿用OPF2，并增加同预算方法增益：

```text
exact-P平均AUC >= 0.55
exact-P - label-shuffle平均AUC >= 0.03
exact-P平均accuracy-majority >= 0.005
exact-P - OPF2 exact-P平均AUC >= 0.010
至少4/8 bit同时满足：
  AUC >= 0.55
  candidate - shuffle >= 0.015
  accuracy-majority >= 0.005
```

机制归因单独报告，不阻断输出预测结论：

```text
exact-P - identity-P平均AUC
逐bit exact-P - identity-P
```

若主门与同预算增益全部通过，下一步原样换全新固定未知密钥确认OPF3；仍不直接进入五轮。若未通过，
停止该模型、`2^22`、300 epochs、宽度/深度调参和事后bit选择；保留PRESENT三轮输出预测正结果与四轮
受控边界，重新做方法级文献审查，不把失败写成所有四轮输出函数不可预测。

## 8. 实施与远程路径

模型实现和结构单元测试可以在OPF2指标未知时盲态准备。OPF2正式hold并通过来源门后才执行第2步及以后步骤：

1. 实现共享逐轮block和参数匹配、共享权重、exact/identity路由的结构单元测试；三行runner仍保持关闭。
2. 本地只跑`64/64`、1 epoch CPU readiness，验证真实输出回放、切分hash、初始化公平性、参数量、
   磁盘缓存、24条逐bit结果、3条history、3个checkpoint和中文SVG；小样本AUC不解释。
3. readiness通过后范围提交并推送。
4. 从精确推送提交在run-owned干净clone启动：

```text
run_id      = i2_output_prediction_opf3_present_r4_round_recurrent_spn_2p20_key7_gpu0_20260723
remote root = G:\lxy\blockcipher-structure-adaptive-nd-runs\i2_opf3_r4_rrspn_2p20_k7_20260723
GPU         = physical GPU0
```

5. 远程使用磁盘缓存、逐chunk/逐epoch progress和参数匹配resume；启动后只做一次有界确认，后续由本地
   tmux watcher等待verified result branch并自动回收。
6. 回收后验证来源OPF2 gate SHA256、源提交、缓存、`24/300/3`产物门，生成SVG并执行
   `visual-qa-redraw`像素检查，刷新`outputs/00_RECENT_RESULTS.md/json`，更新本记录并给出证据支持的下一步。

## 9. 当前停止状态

本计划是在OPF2正式指标未知时盲预注册。当前允许OPF2继续运行并盲态准备冻结模型实现；OPF3仍没有readiness、
数据、训练、远程任务或性能结果，不得把实现代码本身写成高轮进展或候选成功。若OPF2通过，OPF3实现保持休眠，
不得据此绕过新固定密钥确认分支。
