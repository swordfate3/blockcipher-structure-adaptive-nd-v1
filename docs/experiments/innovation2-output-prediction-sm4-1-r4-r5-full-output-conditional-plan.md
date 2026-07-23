# 创新2 SM4-1：SM4四字递推真实输出预测条件计划

日期：2026-07-23

状态：条件计划冻结 / 密码语义与直通基线已审计 / 未实现、未训练、未启动

## 1. 条件授权与任务边界

SM4-1是创新2对Feistel-like四字递推结构的最终覆盖，只在DES FEISTEL1完成两密钥机制裁决后开放readiness。
DES结果只授权迁移“共享递推、正确依赖、错误结构和独立密钥”的方法，不能迁移左右两半表示或性能结论。

条件开放前只允许协议审计、数据适配器和模型单元测试，不得生成正式数据、运行性能screen或远程训练。

任务为：

```text
固定但不提供给网络的128-bit SM4秘密密钥 K
输入 = 未见128-bit明文 P
标签 = 同一输入的真实轮减SM4密文 E_K^r(P)
```

完整128-bit密文是训练目标；发现后冻结的真实密文bit是主边界指标。不得使用Innovation 1的SM4真假差分
数据、pair特征、AUC或checkpoint作为本任务输出预测证据。

## 2. 为什么从四轮开始

仓库`Sm4Reduced`按标准四字递推：

```text
X_(i+4) = X_i xor T(X_(i+1) xor X_(i+2) xor X_(i+3) xor rk_i)
output  = reverse(last four chronological words)
```

因此低于四轮时，序列化密文仍直接包含未变换的原始明文字：

```text
r1：输出中有3个原始明文字 = 96个直通bit
r2：输出中有2个原始明文字 = 64个直通bit
r3：输出中有1个原始明文字 = 32个直通bit
r4：四个输出字均至少经过一次轮递推
```

一至三轮的完整输出`BAPavg`会被确定性复制捷径抬高，不能作为神经模型学到SM4轮函数的正证据。SM4-1
从r4开始，并强制报告直通/多数/逐位置确定性基线。readiness仍可用r1--r3验证序列化，但不产生性能结论。

本地已验证32轮官方向量：

```text
key/plaintext = 0123456789abcdeffedcba9876543210
ciphertext    = 681edf34d206965e86b3e94f536e4246
```

## 3. 文献边界与研究问题

当前本地核验的输出预测论文链没有SM4固定未知密钥真实密文Encryption Emulation结果。Jeong 2024只在相关
工作中提到SM4密钥恢复；本项目持有的SM4 Conv-ResNet论文属于差分/分类路线，任务标签不同。因此SM4-1是
跨结构方法检验，不称论文复现，也不预设能达到某个公开轮数。

唯一结构假设为：

> 显式保持四个32-bit字的时间角色、正确三字轮函数依赖、固定输出反转和共享轮递推，能否在真实SM4输出
> 预测上超过同预算通用网络、错误字依赖和标签打乱，并在第二把固定未知密钥上复现？

## 4. 冻结数据与密钥协议

SM4-1-A：

```text
cipher                  = SM4
rounds                  = 4
data/model/shuffle seed = 61
key_seed                = 61
secret key derivation   = Random(1_610_000 + key_seed).getrandbits(128)
plaintext RNG           = numpy.default_rng(1_620_000 + seed)
input                    = (X0,X1,X2,X3)四个32-bit字；另保存128个MSB-first bit
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

SM4-1-B保持全部字段，只将`key_seed=62`；明文、features、初始化、batch、shuffle置换和冻结bit必须逐值相同，
只改变秘密密钥和真实targets。每把固定密钥独立训练。

train/discovery/fresh明文全局唯一且两两零重合。正式缓存必须按chunk写入`plaintexts.npy`、`features.npy`、
`full_targets.npy`、`cache_metadata.json`和`progress.jsonl`，支持参数匹配恢复。metadata冻结轮数、两个seed、
key hex、四字顺序、最终反转、split、SHA256、completed rows和RNG state。冻结manifest写入前禁止读取fresh标签。

## 5. 网络家族与错误结构

### 5.1 通用word-ResCNN锚点

将128-bit明文按四个32-bit word token编码，使用位置保持残差卷积和128-bit输出头，不硬编码轮递推、三字依赖
或SM4旋转。这是主通用锚点。

### 5.2 四word BiLSTM锚点

把四个32-bit字作为长度4序列，使用三层BiLSTM及128-bit输出头。它提供与DES/SPECK论文族一致的通用序列
模型比较，但不能使用DES的两半输入布局。

### 5.3 SM4四字共享递推候选

候选保持有序状态窗口`(W0,W1,W2,W3)`，每一步执行：

```text
u   = shared_three_word_mixer(W1,W2,W3, round_context)
new = xor_skip(W0, shared_T_block(u))
next state = (W1,W2,W3,new)
```

`shared_T_block`保持SM4字节S-box局部性和线性层旋转通道`ROL(2,10,18,24)`；round context只吸收轮编号及
固定未知轮密钥造成的位置偏置，不能读取密钥或真实中间状态。完成r步后使用固定逆序把最后四个时间字映射到
真实密文坐标。输出头按字和bit位置绑定，不允许全局head吸收最终字序。

错误依赖控制保持相同状态窗口、共享块、旋转、shift次数和参数量，只将：

```text
correct u = mixer(W1,W2,W3)
wrong   u = mixer(W0,W2,W3)
```

即把正确的第一个后继依赖字替换为当前skip字。其余运算完全相同。正确和错误候选参数量必须相等，并冻结在
最强通用锚点的`±5%`内。

## 6. 分阶段精简矩阵

SM4-1-A1三行true screen：

```text
sm4_full128_word_rescnn_true_output
sm4_full128_four_word_bilstm_true_output
sm4_full128_word_recurrent_true_output
```

SM4-1-A2只在A1来源完整时新增，正确候选不重复训练：

```text
sm4_full128_wrong_word_dependency_true_output
sm4_full128_word_recurrent_label_shuffle
```

shuffle只打乱训练标签行，discovery/fresh保持真实密文。错误依赖只改变`W1 -> W0`这一项。

SM4-1-B第二密钥四行：

```text
strongest_generic_true
word_recurrent_true
wrong_word_dependency_true
word_recurrent_label_shuffle
```

最强generic按A1 discovery完整输出`BAPavg`、macro AUC、再优先word-ResCNN确定；A2或fresh揭盲后不得更换。
最强generic、16个输出位置及其顺序必须写入同一个冻结manifest，随后才允许生成或读取fresh targets。

## 7. 输出冻结、指标与门

A1完成后仅使用discovery集，按正确候选逐bit AUC降序冻结恰好16个MSB-first输出位置。相同位置、顺序和模型
用于所有控制、fresh及第二密钥，不得按fresh结果改变。

每行报告：

```text
128个逐bit threshold accuracy / majority / margin / AUC / BCE
完整128-bit BAPavg、macro AUC、exact-match count/rate
冻结16-bit平均AUC、BAPavg、selected-vector exact-match
r1--r3直通位审计与r4零直通证明
参数量、训练时间、最终epoch、checkpoint和无效输出率
```

fresh输出可预测门：

```text
候选冻结16-bit平均AUC >= 0.55
候选冻结16-bit BAPavg >= 0.55
候选 - matched shuffle平均AUC >= 0.03
候选 - matched shuffle BAPavg >= 0.03
至少8/16 bit同时满足：
  AUC >= 0.55
  accuracy-majority >= 0.005
  candidate-shuffle AUC >= 0.015
```

结构增益门：

```text
候选在key61和key62都优于各自最强generic
两密钥合计candidate-generic平均AUC增益 >= 0.005
至少8/16 bit增益方向一致
correct dependency在两密钥都优于wrong dependency
两密钥合计correct-wrong平均AUC >= 0.003
```

正确/错误持平时只能保留真实输出可预测结论，不能宣称精确四字依赖贡献。r1--r3复制位即使达到AUC 1.0也
只记为确定性基线，不计入通过轮。

## 8. 临界轮与规模推进

```text
r4两密钥通过
  -> 只将rounds=4改为5，继承同预算、模型和冻结规则

r5两密钥通过
  -> 每次只增加一轮

单密钥通过、第二密钥失败
  -> 保留条件结果，不增加轮数

首个完整失败轮且存在数据稀缺证据
  -> 只允许一次2^20训练规模审判，epoch和测试集不变

2^20仍失败
  -> 冻结最后通过轮/首个失败轮，不追加模型、epoch或事后bit
```

若r4所有true模型近随机，先审计四字序列化、固定密钥、T函数、最终反转、缓存和训练实现。由于没有本地核验
的SM4同任务论文正锚点，不允许用机械`2^22/300`替代协议诊断，也不能将单预算失败写成SM4不可预测。

## 9. readiness、远程与产物

条件开放后，本地CPU只运行`64/32/32`、1 epoch五行readiness，并额外运行r1--r4确定性直通审计。readiness
验证官方向量、四字顺序、真实输出回放、r4零直通、split零重合、独立`key_seed`、fresh盲态、matched
shuffle、错误依赖唯一变化、参数量、缓存恢复、checkpoint和结果行数；随机小样本指标不解释。

正式`2^17`及`2^20`使用远程A6000、推送提交、run-owned干净clone和`G:\\lxy`磁盘缓存。启动后一次有界
确认，随后由本地tmux watcher等待、验证和回收；主线程不SSH轮询。

每个完成阶段必须生成`results.jsonl`、`history.csv`、`gate.json`、缓存/来源/checkpoint manifest和中文SVG；
图像必须通过`visual-qa-redraw`像素检查，随后刷新`outputs/00_RECENT_RESULTS.md/json`，更新本记录并给出
证据支持的下一动作。

明确禁止：

```text
不得把r1--r3直通位准确率写成SM4轮函数神经预测成功
不得复用DES左右两半网络、PRESENT bit-P网络或Innovation 1真假结果
不得使用真假差分、积分、kernel、密钥恢复或内部关系标签
不得从fresh集事后选择bit、模型、轮数或门
不得同时改变轮数、密钥、数据规模、epoch和模型
不得在DES FEISTEL1闭环前运行SM4性能实验
```
