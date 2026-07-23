# 创新2 AES1：AES-128真实输出预测与SPN字节结构归因条件计划

日期：2026-07-23

状态：条件计划冻结 / 密码实现与公开协议已审计 / 未实现、未训练、未启动

## 1. 条件授权与任务边界

AES1只在PRESENT与GIFT-64的当前冻结分支完成或形成有效hold、且GIFT已给出可解释的跨SPN裁决后开放
readiness。前序结果不是AES性能成立的前提；顺序用于避免同时改变密码、表示、模型和计算预算。

条件开放前只允许协议审计、数据适配器和模型单元测试，不得生成正式数据、运行性能screen或启动远程训练。

任务严格定义为：

```text
固定但不提供给网络的128-bit AES秘密密钥 K
输入 = 未见128-bit明文 P
标签 = 同一输入的真实轮减AES-128密文 E_K^r(P)
```

完整128-bit密文是训练目标。发现集冻结恰好16个真实密文bit，并在未读取标签的fresh集上确认；该selected-bit
结果仍是同一样本真实密文值预测，不是真假样本、差分分类、积分平衡、内部状态或关系成立分类。

## 2. 外部边界与研究问题

Jeong等2024的Encryption Emulation协议使用`2^22`训练、`2^15`测试、300 epochs、batch 128、BCE、
AdamW和`lr=0.001`。Table 3中AES-128一至三轮的FCNN/BiLSTM `BAPavg`均约为`0.499--0.500`；正文称
即使一轮和`2^22`训练数据也没有恢复信号。该论文没有给出本项目要求的独立`key_seed`、匹配标签打乱、
discovery/fresh拆分和AES结构错误控制，因此这些数值只作外部通用网络参考，不称固定未知密钥精确复现。

这形成一个可证伪问题：公开通用FCNN/BiLSTM在AES低轮近随机，究竟是AES输出本身不可学习，还是因为bit数组
网络没有利用一轮AES的强字节局部性和公开状态拓扑？AES1只测试一个结构假设：

> 保持`4x4 byte state`，使用共享字节局部变换、正确ShiftRows路由和正确MixColumns列扩散顺序的网络，能否
> 在同预算下超过通用网络、错误路由和标签打乱，并在第二把固定未知密钥上复现？

## 3. 冻结数据与密钥协议

AES1-A以一轮作为低轮正控制：

```text
cipher                  = AES-128
rounds                  = 1
data/model/shuffle seed = 51
key_seed                = 51
secret key derivation   = Random(1_510_000 + key_seed).getrandbits(128)
plaintext RNG           = numpy.default_rng(1_520_000 + seed)
input                    = 16个byte，按AES column-major 4x4 state组织；另保存128个MSB-first bit
target                   = 128个MSB-first真实密文bit
train                    = 2^17 total
discovery                = 2^15 total
fresh                    = 2^15 total
total evaluation         = 2^16 total
epochs / batch           = 100 / 128
loss / optimizer / lr    = BCE / AdamW / 0.001
selection                = final epoch
data chunk rows          = 4096
sample classification    = false
```

AES1-B保持全部字段，只将`key_seed=52`。明文、features、初始化、batch顺序、shuffle置换和冻结输出位置必须
逐值相同；只改变秘密密钥及由它生成的真实密文targets。每把固定密钥独立训练，不复用checkpoint。

三个split的明文必须全局唯一且两两零重合。正式数据按chunk写入：

```text
plaintexts.npy
features.npy
full_targets.npy
cache_metadata.json
progress.jsonl
```

metadata冻结cipher、rounds、两个seed、128-bit key hex、AES状态序列化、split边界、SHA256、completed rows、
RNG state和参数匹配恢复。fresh targets在候选位置及顺序写入冻结manifest前不得被runner读取。

## 4. AES轮减语义

仓库`Aes128`的轮减语义是：

```text
initial AddRoundKey
for round 1 .. r-1:
    SubBytes -> ShiftRows -> MixColumns -> AddRoundKey
final round r:
    SubBytes -> ShiftRows -> AddRoundKey
```

因此一轮没有MixColumns；二轮包含一次完整列扩散和一次末轮。计划和模型必须使用这一确切语义，不得把
`rounds=1`误写成带MixColumns的一轮，也不得把AES-like或SAES数值当作AES-128结果。

readiness必须验证FIPS-197向量：

```text
key        = 000102030405060708090a0b0c0d0e0f
plaintext  = 00112233445566778899aabbccddeeff
ciphertext = 69c4e0d86a7b0430d8cdb78070b4c55a
```

并对一轮、二轮缓存抽样执行标量真实密文回放。

## 5. 网络家族与唯一结构变量

### 5.1 FCNN论文族锚点

使用Jeong论文族的`128 -> 512 -> 1024 -> 512 -> 128`全连接结构，隐藏层BatchNorm/ReLU，输出128个
logit。实现差异全部记录为`paper-family approximation`，不能仅凭层宽称精确复现。

### 5.2 通用byte-ResCNN锚点

把16个byte作为长度16的token序列，使用共享byte embedding和普通位置保持残差卷积，输出128个bit。它保留
byte边界，但不硬编码ShiftRows或MixColumns，是结构候选的最强同预算通用锚点之一。

### 5.3 AES-state共享递推候选

候选将token严格放置为AES column-major `4x4`状态，每轮依次执行：

```text
共享byte-local非线性块
正确ShiftRows固定路由
非末轮：共享列混合块，只连接同一MixColumns列的4个byte
round context
```

round context只能表示轮编号及固定未知子密钥造成的位置偏置，不能读取密钥、真实中间状态或测试标签。
输出头按最终状态坐标绑定，每个bit只读取其路由后的byte token；禁止使用可吸收末端固定置换的全局
`Flatten + Linear`头。

错误结构控制只把ShiftRows行偏移从正确的`[0,1,2,3]`替换为`[0,2,3,1]`。该映射仍为逐行循环置换，
保持token数、算子数、参数量和列混合强度。二轮及以上继续在错误路由后的列上执行同一列混合块，因此唯一
改变的是公开扩散路由。正确和错误候选总参数量必须相同，并冻结在最强通用锚点的`±5%`内。

## 6. 分阶段精简矩阵

AES1-A1三行true screen：

```text
aes128_full128_fcnn_true_output
aes128_full128_byte_rescnn_true_output
aes128_full128_state_recurrent_true_output
```

AES1-A2只在A1来源、checkpoint和hash完整时新增两行，正确候选不重复训练：

```text
aes128_full128_wrong_shiftrows_recurrent_true_output
aes128_full128_state_recurrent_label_shuffle
```

shuffle只打乱训练标签行，discovery/fresh标签始终是真实密文。模型、明文、初始化和batch顺序匹配。

AES1-B在第二密钥上只运行：

```text
strongest_generic_true
state_recurrent_true
wrong_shiftrows_recurrent_true
state_recurrent_label_shuffle
```

最强generic按A1的discovery完整输出`BAPavg`、macro AUC、再优先byte-ResCNN确定；A2或fresh揭盲后不得
更换。最强generic、16个输出位置及其顺序必须写入同一个冻结manifest，随后才允许生成或读取fresh targets。

## 7. 输出位置冻结与指标

A1训练完成后，仅使用discovery集按候选逐bit AUC降序冻结恰好16个位置；相同位置、顺序和模型用于所有控制、
fresh集及第二密钥。不得按fresh结果减小位置数或改选模型。

每行必须报告：

```text
128个逐bit threshold accuracy / majority / margin / AUC / BCE
完整128-bit BAPavg、macro AUC、exact-match count/rate
冻结16-bit的平均AUC、BAPavg、selected-vector exact-match
参数量、训练时间、最终epoch、checkpoint和无效输出率
```

输出可预测门在fresh冻结16-bit上判定：

```text
候选平均AUC >= 0.55
候选BAPavg >= 0.55
候选 - matched shuffle平均AUC >= 0.03
候选 - matched shuffle BAPavg >= 0.03
至少8/16 bit同时满足：
  AUC >= 0.55
  accuracy-majority >= 0.005
  candidate-shuffle AUC >= 0.015
```

结构增益门要求：

```text
候选在key51和key52都优于各自最强generic
两密钥合计candidate-generic平均AUC增益 >= 0.005
至少8/16 bit的增益方向一致
正确ShiftRows在两密钥都优于wrong ShiftRows
两密钥合计correct-wrong平均AUC >= 0.003
```

正确/错误持平时可保留真实输出可预测结论，但不能宣称精确AES路由贡献。完整128-bit exact match很低不否定
预注册选定位结果；反之，单个事后高AUC bit不能通过本门。

## 8. 临界轮推进

```text
r1两密钥通过
  -> 只将rounds=1改为2，继承同预算、模型和冻结规则

r2两密钥通过
  -> 逐次测试r3；每次只增加一轮

某轮单密钥通过、第二密钥失败
  -> 保留单密钥条件结果，不增加轮数

首个完整失败轮
  -> 只允许一次预注册的2^20训练规模审判，模型/epoch/测试集不变

2^20仍失败
  -> 冻结最后通过轮/首个失败轮，不追加模型、epoch或事后bit
```

若r1所有true模型都近随机，先审计固定密钥、AES轮减语义、byte序列化和训练实现。Jeong的`2^22/300`近随机
参考不能替代本项目协议有效性检查，也不能把`2^17/100`失败写成AES输出不可预测。

## 9. readiness、远程与产物

条件开放后，本地CPU只运行`64/32/32`、1 epoch五行readiness，验证官方向量、column-major状态、真实输出
回放、split零重合、独立`key_seed`、fresh盲态、matched shuffle、错误路由唯一变化、参数量、缓存恢复、
checkpoint和结果行数；随机小样本指标不解释。

正式`2^17`及`2^20`使用远程A6000、推送提交、run-owned干净clone和`G:\\lxy`磁盘缓存。启动后一次有界
确认，随后由本地tmux watcher等待、验证和回收；主线程不SSH轮询。

每个完成阶段必须生成`results.jsonl`、`history.csv`、`gate.json`、缓存/来源/checkpoint manifest和中文SVG；
图像必须通过`visual-qa-redraw`像素检查，随后刷新`outputs/00_RECENT_RESULTS.md/json`，更新本记录并给出
证据支持的下一动作。

明确禁止：

```text
不得把AES-like、SAES或PRESENT输出位置直接迁移到AES-128
不得使用真假差分、积分、kernel或内部关系标签
不得用全局head吸收末端ShiftRows后宣称拓扑归因
不得从fresh集事后选择bit、模型、轮数或门
不得同时改变轮数、密钥、数据规模、epoch和模型
不得在GIFT分支闭环前运行AES性能实验
```
