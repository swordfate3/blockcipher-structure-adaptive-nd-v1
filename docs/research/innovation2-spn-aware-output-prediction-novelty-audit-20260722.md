# 创新2 SPN-aware真实输出预测：核心论文创新边界审计

日期：2026-07-22

状态：核心全文与联网核验完成 / OPA3与OPB1正式hold / OPC1远程运行中 / 精确P-layer创新主张删除

## 1. 审计问题与声明范围

本审计只回答：当前核心输出预测论文是否已经在固定未知密钥PRESENT任务中，把S-box分组和精确
P-layer显式写入网络，并用identity/错误P同参数控制证明拓扑贡献。

审计语料是项目已验证全文的四篇核心论文，不是穷尽性系统综述。因此本文可以限定已有核心基线与
项目主张，不能据此使用“首次”“无人研究”或“国际领先”等语言。

```text
Kimura et al. 2022  Output Prediction Attacks on Block Ciphers Using Deep Learning
Kimura et al. 2023  A Deeper Look into Deep Learning-based Output Prediction Attacks Using Weak SPN Block Ciphers
Watanabe et al.     On the Effects of Neural Network-based Output Prediction Attacks on the Design of Symmetric-key Ciphers
Singh 2025          PRESENT Full Round Emulation: Structural Flaws and Predictable Outputs
```

全文与已核验书目信息位于`papers/innovation_two/`和
`docs/research/innovation2-output-prediction-literature-index.md`。

## 2. 核心论文实际做法

### 2.1 Kimura等，2022

Kimura等定义了已知明文、单固定秘密密钥下的密文预测与明文恢复：网络输入未见明文，输出对应真实
密文；主要成功指标是完整输出精确命中概率。其攻击刻意采用black-box设定，攻击者只知道接口，不知道
目标密码内部结构，甚至不假设知道密码是置换。

主网络是通用stacked LSTM。PRESENT三轮密文预测使用六层、hidden 300、MSE、RMSprop、100 epochs、
`2^17`训练与`2^16`测试。论文附录另做普通Keras `Conv1D` CNN对比，并报告在三个16-bit玩具密码上
LSTM整体优于CNN。

论文研究了交换S-layer/P-layer或替换S-box后攻击成功率如何变化，但这是修改目标密码后重新训练通用
网络的白盒分析，不是把精确P-layer作为网络内无参数传播算子，也没有exact/identity/wrong-P同参数
模型归因。

### 2.2 Kimura等，2023

该工作延续相同black-box固定密钥输出预测协议和LSTM模型，重点把small PRESENT-[4]的原S-box替换为
两个已知弱S-box，再比较深度学习攻击与差分/线性攻击轮数。它说明目标SPN组件强弱会影响通用网络的
输出预测能力，但没有提出PRESENT拓扑感知网络，也没有对网络内部拓扑做错误连线控制。

### 2.3 Watanabe等，2024

Watanabe等把输出预测用于SIMON32变体设计分析。其冻结网络仍是通用LSTM：一层、hidden 300、完整
16/32-bit输出、MSE、Adam、100 epochs。工作重点是目标密码的扩散与经典攻击性质，而不是比较MLP、
ResCNN、Transformer或密码拓扑网络。

因此它占据“用输出预测反馈对称密码设计”和“跨变体比较”的位置，但没有占据PRESENT显式P-layer
消息传播或同参数拓扑因果归因。

### 2.4 Singh，2025

Singh最接近逐轮PRESENT建模：为31轮分别训练64-512-512-512-64 MLP，并把上一轮预测传给下一轮。
但其数据生成明确删除每轮key addition并使用全零key，目标函数只有公开的S-box与P-layer确定性变换。
论文自己把“加入key material”列为future work。

这占据“去密钥PRESENT公开轮函数的逐轮MLP仿真”，却不是固定未知秘密密钥下的输出预测攻击。其网络
仍是通用MLP，没有在模型内部显式执行nibble局部混合和P-layer重排，也没有真实/错误拓扑同参数控制。

### 2.5 Jeong等，2024/2026补充近邻

联网检索新增了Jeong、Ahmadzadeh和Moon的2024论文`Comprehensive Neural Cryptanalysis on Block
Ciphers Using Different Encryption Methods`，Crossref DOI为`10.3390/math12131936`。其正式摘要覆盖EE、
PR、key recovery和ciphertext classification，比较全连接、RNN与Transformer，目标密码为DES、SDES、
AES、SAES和SPECK。

Jeong、Park和Moon的2026后续论文`Scalable Neural Cryptanalysis of Block Ciphers in Federated Attack
Environments`，DOI `10.3390/math14020373`，把EE/PR扩展到多服务器联邦环境并比较全连接与BiLSTM。
两篇论文说明“输出仿真使用非LSTM架构”和“多架构比较”本身不是空白，但其Crossref摘要均不包含
PRESENT、显式S-box/P-layer网络或错误拓扑归因。MDPI落地页在当前环境拒绝访问，因此这里是正式
书目与摘要级核验，不冒充全文协议审计。

另一新增候选`Neural Cryptanalysis of Lightweight Block Ciphers Using Residual MLPs`，DOI
`10.1109/CSR64739.2025.11130149`，经IEEE搜索记录和摘要核对属于SIMON/SPECK all-in-one差分分类，
不是明文到真实密文输出值预测。

## 3. 与当前项目的逐项对照

| 维度 | Kimura 2022/2023 | Watanabe 2024 | Singh 2025 | 当前OPA1--OPB1与OPC1 |
|---|---|---|---|---|
| 密钥协议 | 单固定未知密钥，逐密钥训练 | 单固定未知密钥，逐密钥训练 | 删除key addition，全零key | 单固定未知密钥，seed0--4已分别训练，seed6运行中 |
| 输入/目标 | 明文到完整密文，或反向 | 完整输出预测，主要SIMON | 前一轮状态到下一轮公开变换 | 明文到八个预注册真实密文bit |
| 主要网络 | stacked LSTM | 一层LSTM | 每轮独立MLP | MLP/LSTM/ResCNN/Transformer/SPN-aware/SPN-ResCNN |
| 密码结构进入网络 | 不进入，black-box | 不进入，black-box | 不进入MLP，只体现在标签 | 4-bit局部混合 + 精确P重排进入每个block |
| 输出位置保持 | 序列LSTM，完整头 | 序列LSTM，完整头 | 64维完整头 | bit token保持到选定输出头 |
| 同参数错误拓扑 | 无 | 无 | 无 | OPA3 exact/identity/wrong-P；OPB1低秩exact/wrong-P |
| 匹配标签打乱 | 核心论文未作为主门 | 未作为主门 | 无 | OPA2、OPB1含架构匹配shuffle；OPC1同样冻结 |
| 当前证据轮数 | PRESENT三轮完整输出论文锚点 | 非PRESENT主结果 | 去密钥31轮链式仿真 | PRESENT三轮；四轮同目标尚未运行 |

## 4. 当前网络的准确技术定义

当前`PRESENT-SPN-aware`不是把密文排成图像的普通CNN，也不是完整实现已知PRESENT加密。输入仍只有
64个明文bit，网络不能读取秘密密钥、轮密钥或真实中间状态。其一个block为：

```text
64个bit token
  -> 每4个相邻bit合并为一个nibble
  -> 共享nibble MLP进行局部4-bit非线性混合
  -> 按精确PRESENT P-layer执行无参数bit token重排
  -> 每个位置共享channel MLP与残差
```

三个block依次堆叠，最后只在八个预注册输出位置应用共享线性头。网络学习局部非线性和固定密钥作用，
精确P-layer本身不学习。约388万参数与MLP、LSTM、ResCNN和Transformer控制在3%以内。

OPB1把每个位置的完整189维自由嵌入压缩为每轮64个标量乘共享方向，但exact-P和wrong-P仍同时达到
`1.0`，所以低秩条件没有恢复精确拓扑归因。当前运行中的OPC1采用另一种机制：以OPA1中非饱和的
位置保持ResCNN为锚点，在10个残差块的`3+3+4`阶段之间插入三次无参数P-layer重排，并比较普通
ResCNN、exact-P、wrong-P和匹配shuffle。前三个网络均为`3,955,904`个参数；该矩阵仍只预测同八个
三轮真实密文bit，尚无正式结果。

该设计使用已知密码算法结构，威胁模型比Kimura的black-box攻击者更强，应该表述为
`algorithm-known, fixed-secret-key known-plaintext output prediction`，不能声称复现或直接优于Kimura
black-box协议。

## 5. 暂定可主张的创新

OPA2已经支持：

> 在PRESENT三轮固定秘密密钥的八个预注册真实输出bit任务中，显式保持bit位置并按SPN轮结构组织
> 局部混合与扩散的网络，在第三密钥发现屏和第四密钥匹配控制确认中均明显优于参数匹配MLP；第四
> 密钥上网络同时显著超过自身标签打乱控制。

OPA3正式结果不支持更强的精确拓扑主张：exact-P和fixed-wrong-P八位置平均AUC均为`1.0`，identity-P
为`0.531989557`，exact相对wrong的逐bit差值全部为`0.0`。因此可以写“跨nibble分层扩散明显优于
不扩散”，不能写“精确PRESENT连线优于同容量错误拓扑”。

OPB1进一步证明，删除高维位置嵌入并加入低秩条件瓶颈仍不能让exact-P超过wrong-P。因此不能把原
OPA3负归因简单归咎于自由位置容量。OPC1若同时超过普通ResCNN、wrong-P和shuffle，才会开放全新
固定密钥确认；在结果回收前，它不能进入创新主张。

适合毕业论文的方法名暂定为：

```text
中文：PRESENT结构感知选定位输出预测网络
英文：PRESENT-Structure-Aware Selected-Output Predictor
缩写：PSA-SOP
```

名称只是写作占位；`structure-aware`只能指位置保持、4-bit局部混合和跨nibble扩散的整体结构归纳
偏置，不能解释为已经验证的精确P-layer因果贡献。更保守的结果描述可用“SPN式分层扩散选定位输出
预测网络”，避免方法名暗示精确连线已经归因。

## 6. 明确不能主张

- 不能声称首次提出神经网络输出预测攻击；Kimura等已经占据该任务。
- 不能声称首次使用CNN预测密码输出；Kimura 2022附录已有普通Conv1D。
- 不能声称首次逐轮神经仿真PRESENT；Singh已有去密钥逐轮MLP。
- 不能把ResCNN或Transformer本身作为创新；它们在OPA1只是同预算候选。
- 不能声称OPA3证明精确P-layer导致`AUC=1.0`；错误P控制同样为`1.0`。
- 不能声称OPB1低秩瓶颈已经隔离精确拓扑；低秩exact-P与wrong-P仍同为`1.0`。
- 不能把正在运行的OPC1写成SPN-ResCNN已获得正式正结果。
- 不能把固定密钥、八位置、三轮AUC写成完整密文恢复、未知密钥泛化或主流七至九轮攻击。
- 不能把算法已知结构网络与Kimura black-box LSTM直接排名为同协议SOTA。
- 核心四篇全文审计不能替代穷尽性系统检索，不能使用“文献中没有任何人做过”的绝对表述。

## 7. OPA3与OPB1对创新边界的裁决

OPA3是方法创新的必要因果门：

```text
exact-P mean AUC >= 0.510
exact-P - best(identity, wrong-P) mean AUC >= +0.030
至少4/8 bit同时满足exact-identity和exact-wrong >= +0.020
exact-P复跑与OPA2 mean AUC差 <= 0.005
```

正式结果为：exact-P `1.000000000`、identity-P `0.531989557`、fixed-wrong-P `1.000000000`，
exact-best-control=`0.000000000`，attributed bits=`0/8`，来源和全部协议/执行门完整，裁决为`hold`。
因此只保留同预算架构发现、独立密钥匹配控制和“分层扩散相对identity有效”的结果；删除精确PRESENT
拓扑因果措辞，不继续OPA4/OPA5四轮扩展。

OPB1在第五固定密钥上复核低秩条件假设：原SPN锚点、低秩exact-P和低秩wrong-P平均AUC均为
`1.000000000`，shuffle为`0.499476582`，exact-wrong=`0`，attributed bits=`0/8`。因此低秩路线
停止，不调整rank、数据、epoch、错误P或输出位置；该结果只授权单独预注册的OPC1三轮混合网络，
不重新开放OPA4/OPA5或四轮。

## 8. 下一步文献与实验动作

1. 等待OPC1 verified result branch自动回收，验证冻结协议、OPB1 gate hash和正式视觉门；不从训练
   中间损失推断最终AUC。
2. OPC1通过只开放全新固定密钥原样确认；失败或hold则保留ResCNN发现锚点并停止混合路线。
3. 继续关闭OPA4/OPA5，不通过错误排列搜索、seed、数据、epoch或网络枚举绕过归因门。
4. 在最终使用“新方法”措辞前，核验联网检索新增候选的实际落地页和全文，并把任务不同的BiLSTM、
   neural distinguisher、active-S-box预测与积分分类明确排除或纳入对照。

## 9. 2026-07-22联网交叉检索

使用Tavily advanced search在IACR ePrint、arXiv、Springer、IEEE、ACM和Semantic Scholar范围执行
三组关键词：

```text
structure-aware neural output prediction block cipher SPN PRESENT P-layer cipher topology
PRESENT neural network P-layer output prediction ciphertext prediction structure aware
cipher topology neural network output prediction block cipher wrong topology shuffled topology ablation
```

返回的密码学近邻为Kimura/Watanabe输出预测、Jeong等通用EE/PR多架构实验、Singh去密钥逐轮MLP、
Wu/Guo积分神经区分器、active-S-box预测、相关密钥/Residual-MLP差分神经区分器和早期通用PRESENT
神经分析。没有返回同时满足“固定未知密钥真实输出值 + 网络内显式P-layer +
exact/identity/wrong-P同参数消融 + 匹配shuffle”的记录。

这个未命中只提高暂定创新边界的可信度，不构成首次性证明。2024/2026 Jeong等论文已经通过Crossref
核验标题、作者、年份、DOI与摘要任务，但尚未完成全文协议审计；因此它们只用于限制“非LSTM/多架构”
措辞，不用于同协议数值比较。

可复核搜索日志：

```text
sources/research_innovation2_spn_aware_output_prediction_web_20260722.md
```

## 10. 核心来源

- Kimura et al., *Output Prediction Attacks on Block Ciphers Using Deep Learning*, ACNS Workshops 2022,
  DOI `10.1007/978-3-031-16815-4_15`, IACR ePrint `2021/401`。
- Kimura et al., *A Deeper Look into Deep Learning-based Output Prediction Attacks Using Weak SPN Block
  Ciphers*, Journal of Information Processing 31 (2023), DOI `10.2197/ipsjjip.31.550`。
- Watanabe, Ito, and Ohigashi, *On the Effects of Neural Network-based Output Prediction Attacks on the
  Design of Symmetric-key Ciphers*, IACR ePrint `2024/1310`, CSCML 2024 DOI
  `10.1007/978-3-031-76934-4_13`。
- Singh, *PRESENT Full Round Emulation: Structural Flaws and Predictable Outputs*, IACR ePrint
  `2025/1069`。
- Jeong, Ahmadzadeh, and Moon, *Comprehensive Neural Cryptanalysis on Block Ciphers Using Different
  Encryption Methods*, Mathematics 12(13), 1936 (2024), DOI `10.3390/math12131936`；元数据/摘要级。
- Jeong, Park, and Moon, *Scalable Neural Cryptanalysis of Block Ciphers in Federated Attack
  Environments*, Mathematics 14(2), 373 (2026), DOI `10.3390/math14020373`；元数据/摘要级。
