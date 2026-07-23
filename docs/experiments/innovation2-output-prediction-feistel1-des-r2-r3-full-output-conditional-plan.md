# 创新2 FEISTEL1：DES二至三轮完整输出临界点条件计划

日期：2026-07-23

状态：条件计划冻结 / 数据适配器已实现并通过单元测试 / 网络与CLI未实现 / 未训练、未启动

## 1. 条件授权与任务定义

FEISTEL1是创新2的Feistel真实输出预测机制锚点，只在PRESENT、GIFT和SPECK当前冻结分支依次完成或形成
有效hold后开放readiness。前序路线成功不是DES的性能前提；顺序用于保持远程算力和研究变量单一。

条件开放前只允许协议、数据适配器和网络结构单元测试，不得生成正式数据、运行性能screen或远程训练。

截至2026-07-23，已在`tasks/innovation2/feistel/`子包中分离协议、NumPy轮减DES和磁盘数据缓存，完成
固定密钥真实输出数据合同，但没有生成
F1-A/B/C/R正式数据。定向测试验证了官方16轮向量、1/2/3/16轮NumPy与标量DES逐项一致、MSB-first
序列化和32-bit两半、训练/测试唯一且零重合、key31到key32只改变密钥和真实targets、训练扩容前缀、
参数错配/部分缓存拒绝，以及首个测试chunk中断后逐值恢复。网络、matched shuffle、错误F分支、训练CLI、
结果门和远程包仍未实现；本状态只证明数据层实现就绪，不是readiness结果或DES可预测性证据。

任务为：

```text
固定但不提供给网络的64-bit DES秘密密钥 K
输入 = 未见64-bit明文 P
标签 = 同一输入的真实64-bit轮减DES密文 E_K^r(P)
```

标签是完整真实密文值，不是真假样本、差分分类、积分平衡、active S-box或内部关系。Innovation 1已有的
DES真假区分、canonical layout和shuffled mapping结果只说明代码/表示准备度，不能计入本计划输出预测证据。

## 2. 论文边界与研究问题

Jeong等2024论文的完整输出Encryption Emulation任务使用`2^22`训练、`2^15`测试、300 epochs、batch 128、
BCE/AdamW和FCNN/BiLSTM。Table 3报告：

```text
DES r1 EE：FCNN 1.000 / BiLSTM 1.000 BAPavg
DES r2 EE：FCNN 0.812 / BiLSTM 0.875 BAPavg
DES r3 EE：FCNN 0.507 / BiLSTM 0.510 BAPavg
```

这给出公开的“二轮明显可预测、三轮接近随机”经验边界。论文没有显式Feistel递推网络、错误分支控制或
标签打乱；2024单篇的固定密钥描述也不够完整，本计划按2026同作者后续论文族明确的固定未知密钥KPA解释，
只称论文族校准，不称逐细节精确复现。

研究问题分两层：

1. 在明确固定未知密钥和匹配shuffle下，能否恢复DES r2完整输出正信号并在第二密钥复现？
2. 显式Feistel左右分支、正确F输入分支和逐轮交换的共享递推器，是否优于同预算FCNN/BiLSTM及错误分支？

三轮只在二轮两密钥闭环后运行，用来测量最后通过轮与首个失败轮；不得与二轮架构开发同时调参。

## 3. 分阶段预算

```text
F1-A：DES r2，2^20/2^15，100 epochs，单密钥架构与归因screen
F1-B：DES r2，相同预算，只更换第二固定未知密钥
F1-C：条件论文规模，DES r2，2^22/2^15，300 epochs
F1-R：DES r3，继承F1-A/B预算和两密钥确认矩阵
```

若F1-A的FCNN和BiLSTM都没有达到输出信号门，不能关闭DES r2；先审计实现，协议完整时运行F1-C论文规模
通用锚点校准。若论文规模通用锚点仍远低于公开`0.812/0.875`，结论是复现差距，不是DES不可预测。

## 4. 冻结数据协议

F1-A：

```text
cipher                  = DES
rounds                  = 2
block / key             = 64 / 64 bits（含DES parity位的序列化主密钥）
data/model/shuffle seed = 31
key_seed                = 31
secret key derivation   = Random(1_310_000 + key_seed).getrandbits(64)
plaintext RNG           = numpy.default_rng(1_320_000 + seed)
input                    = 64个MSB-first明文bit
target                   = 64个MSB-first真实轮减DES密文bit
train / test             = 2^20 / 2^15 total
epochs / batch           = 100 / 128
loss / optimizer / lr    = BCE / AdamW / 0.001
selection                = final epoch
data chunk rows          = 4096
sample classification    = false
```

F1-B保持全部字段，只将`key_seed=32`；明文、features、初始化、batch和shuffle置换必须逐值相同，仅秘密密钥
和真实targets改变。

F1-C保持seed31、key_seed31和原测试集，唯一预算变化为`train=2^22, epochs=300`。`2^20`训练明文必须是
扩展训练集前缀，测试明文必须被非连续切分保护，不能进入扩展训练段。

F1-R只将`rounds=2 -> 3`；先key31再key32。不得同时改变数据量、epoch、输出位置或模型。

所有正式数据逐chunk落盘，缓存metadata冻结cipher、rounds、两个seed、64-bit key hex、IP/FP语义、MSB顺序、
split、completed rows、RNG state和参数匹配恢复。训练/测试明文唯一且零重合。

## 5. 公共DES语义和真实输出

仓库`Des`实现对轮减版本仍执行标准公开初始置换IP、指定数量Feistel轮、最终交换和FP。readiness必须验证：

```text
16轮官方向量：
  key       = 0x133457799BBCDFF1
  plaintext = 0x0123456789ABCDEF
  ciphertext= 0x85E813540F0AB405

轮减标量回放：
  full_targets[i] == bits(Des(rounds=r,key=K).encrypt(P_i))
```

网络可在输入端应用公开IP并在输出端应用公开FP，但训练标签始终保存和评估真实序列化密文。内部IP状态只是一种
公开可逆表示，不能改写成预测真实中间状态或泄露轮密钥。

## 6. 三个网络家族

### 6.1 FCNN论文族锚点

按Jeong正文/图3实现四个全连接层、最后sigmoid，并记录正文未完全明确的激活、归一化与节点数差异。任何
近似项写入metadata，不能称精确模型复现。

### 6.2 两分支BiLSTM论文族锚点

```text
raw plaintext -> 两个32-bit半块组成长度2序列
three-layer bidirectional LSTM / hidden 256
-> 64个真实密文输出bit
```

这是最强公开同任务通用锚点，不与Innovation 1的DES pair-set LSTM混用。

### 6.3 共享Feistel递推候选

候选显式执行：

```text
公开IP -> (L0,R0)
for t in 0..r-1，共享主体：
  learned F token = local/global branch block(R_t, round_context_t)
  L_(t+1) = R_t
  R_(t+1) = L_t xor learned F token
最终公开swap -> FP -> 64个位置绑定输出logit
```

每轮位置上下文只吸收固定未知子密钥造成的轮次差异；模型不读取秘密密钥、中间真实状态或密文以外的标签。
共享F主体与branch mixer在各轮复用。总参数量冻结在BiLSTM锚点`±5%`，并在揭盲前记录。

错误结构控制保持全部参数、交换次数和算子次序，只把F输入从正确`R_t`改为错误`L_t`：

```text
wrong: L_(t+1)=R_t, R_(t+1)=L_t xor learned_F(L_t)
```

它比简单identity/no-swap更接近等强度控制，也避免只测试最终可被灵活head吸收的固定置换。正确和错误模型都
使用位置绑定输出与固定FP，不允许全局线性head重新参数化最后分支映射。

## 7. 精简矩阵

F1-A1三行true screen：

```text
des_full64_fcnn_true_output
des_full64_bilstm_true_output
des_full64_feistel_recurrent_true_output
```

F1-A2在完整A1来源上只新增：

```text
des_full64_wrong_f_branch_recurrent_true_output
des_full64_feistel_recurrent_label_shuffle
```

正确候选结果从A1复用，不重复训练。shuffle只打乱训练标签行，测试标签保持真实密文。

F1-B第二密钥四行：

```text
strongest_generic_true
feistel_recurrent_true
wrong_f_branch_recurrent_true
feistel_recurrent_label_shuffle
```

最强通用锚点按A1的BAPavg、macro AUC、再优先BiLSTM确定，A2揭盲后不得更换。

F1-C只运行FCNN、BiLSTM以及架构匹配BiLSTM shuffle，先校准公开论文规模；结构候选是否进入`2^22`由
F1-A/B增益另行决定，不能自动把全部五行扩大。

F1-R r3使用F1-B同样四行和两把密钥，不重新搜索模型。

## 8. 指标与门

每行报告：

```text
64个逐bit threshold accuracy / majority / margin / AUC / BCE
BAPavg、macro AUC
完整64-bit exact-match count/rate
参数量、训练时间、最终epoch、checkpoint和无效输出率
```

F1-A/B输出可预测门：

```text
候选BAPavg >= 0.55
候选macro AUC >= 0.55
候选 - matched shuffle BAPavg >= 0.03
候选 - matched shuffle macro AUC >= 0.03
至少32/64 bit同时满足：
  AUC >= 0.55
  accuracy-majority >= 0.005
  candidate-shuffle AUC >= 0.015
```

结构增益门：

```text
候选在key31和key32都优于各自最强通用锚点
两密钥合计candidate-generic BAPavg >= 0.005
至少32/64 bit增益方向一致
correct F-input在两密钥都优于wrong F-input
两密钥合计correct-wrong BAPavg >= 0.003
```

正确/错误持平时可保留输出可预测结论，但精确Feistel分支归因失败。

F1-C必须报告FCNN/BiLSTM相对公开r2 `0.812/0.875`的差值；公开数值不是协议有效性的等式门，单密钥结果
也不是论文复现。F1-R r3沿用输出与结构门，并报告相对公开`0.507/0.510`，不得为了制造边界而放宽门。

## 9. 临界轮裁决

```text
r2两密钥通过、r3两密钥hold
  -> last confirmed predictable round = 2
  -> first complete failed round = 3

r3两密钥通过
  -> 才允许同预算测试r4

r2未通过且2^22论文规模通用锚点也未校准
  -> reproduction gap；不进入r3，不称DES输出不可预测
```

完整输出逐bit分析可以用于解释，但selected-bit新实验必须另设discovery/fresh split并在测试前冻结位置；不能
从F1测试集挑容易bit再写成确认结果。

## 10. SM4迁移边界

DES闭环后进入SM4，但只迁移“共享轮递推、正确依赖和错误结构控制”的方法，不迁移DES左右半块几何：

```text
SM4 state = (X_i, X_(i+1), X_(i+2), X_(i+3)) 四个32-bit字
X_(i+4) = X_i xor F(X_(i+1),X_(i+2),X_(i+3),rk_i)
```

SM4候选必须是四字递推，错误控制改变字依赖或顺序；Innovation 1现有SM4真假区分模型和结果不能直接作为
真实输出模型。SM4正式轮数、规模和目标必须在DES结果后单独预注册，不在本计划中猜测成功轮数。

## 11. readiness、远程与停止边界

条件开放后本地CPU只运行`64/64、1 epoch`五行readiness，验证官方向量、IP/FP、MSB顺序、两半角色、真实
输出回放、split、key_seed、matched shuffle、wrong F唯一变化、参数门、缓存恢复和结果产物；随机指标不解释。

所有`2^20/2^22`运行使用推送提交、远程A6000、run-owned干净clone和`G:\\lxy`磁盘缓存，交由本地tmux
watcher自动回收。正式图必须用中文解释二轮正锚点、三轮边界、完整输出和控制差值，并通过
`visual-qa-redraw`像素检查。

明确禁止：

```text
不得使用真假差分、积分、kernel或内部关系标签
不得把2^20失败写成2^22/300论文规模失败
不得根据A1结果增加网络或调整候选宽度
不得同时改变轮数、数据、epoch和模型
不得混合变化密钥或从测试集事后挑bit
不得在r2两密钥闭环前进入r3
不得把DES结果外推SM4或所有Feistel密码
```

每个完成阶段必须生成JSONL、history CSV、gate、缓存/来源/checkpoint manifest和SVG；更新本记录，刷新
`outputs/00_RECENT_RESULTS.md/json`，完成可视化检查、范围提交和推送，并给出门与控制支持的下一动作。
