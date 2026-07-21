# 创新2 SPN-aware真实输出预测：核心论文创新边界审计

日期：2026-07-22

状态：核心全文语料审计完成 / 暂定创新边界 / 等待OPA3拓扑归因

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

## 3. 与当前项目的逐项对照

| 维度 | Kimura 2022/2023 | Watanabe 2024 | Singh 2025 | 当前OPA1--OPA3 |
|---|---|---|---|---|
| 密钥协议 | 单固定未知密钥，逐密钥训练 | 单固定未知密钥，逐密钥训练 | 删除key addition，全零key | 单固定未知密钥，seed0--3分别训练 |
| 输入/目标 | 明文到完整密文，或反向 | 完整输出预测，主要SIMON | 前一轮状态到下一轮公开变换 | 明文到八个预注册真实密文bit |
| 主要网络 | stacked LSTM | 一层LSTM | 每轮独立MLP | MLP/LSTM/ResCNN/Transformer/SPN-aware |
| 密码结构进入网络 | 不进入，black-box | 不进入，black-box | 不进入MLP，只体现在标签 | 4-bit局部混合 + 精确P重排进入每个block |
| 输出位置保持 | 序列LSTM，完整头 | 序列LSTM，完整头 | 64维完整头 | bit token保持到选定输出头 |
| 同参数错误拓扑 | 无 | 无 | 无 | OPA3 exact/identity/wrong-P |
| 匹配标签打乱 | 核心论文未作为主门 | 未作为主门 | 无 | OPA2含MLP/SPN各自shuffle |
| 当前证据轮数 | PRESENT三轮完整输出论文锚点 | 非PRESENT主结果 | 去密钥31轮链式仿真 | PRESENT三轮；四轮尚未运行 |

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

该设计使用已知密码算法结构，威胁模型比Kimura的black-box攻击者更强，应该表述为
`algorithm-known, fixed-secret-key known-plaintext output prediction`，不能声称复现或直接优于Kimura
black-box协议。

## 5. 暂定可主张的创新

OPA2已经支持：

> 在PRESENT三轮固定秘密密钥的八个预注册真实输出bit任务中，显式保持bit位置并按SPN轮结构组织
> 局部混合与扩散的网络，在第三密钥发现屏和第四密钥匹配控制确认中均明显优于参数匹配MLP；第四
> 密钥上网络同时显著超过自身标签打乱控制。

只有OPA3通过后，才能进一步主张：

> 在网络参数、初始化、数据、局部nibble混合和训练预算不变时，精确PRESENT P-layer显著优于
> identity和固定错误双射，说明收益可归因于正确扩散拓扑，而不只是容量或局部4-bit处理。

适合毕业论文的方法名暂定为：

```text
中文：PRESENT结构感知选定位输出预测网络
英文：PRESENT-Structure-Aware Selected-Output Predictor
缩写：PSA-SOP
```

名称只是写作占位；OPA3未通过时不得把`structure-aware`解释为已经验证的精确拓扑贡献。

## 6. 明确不能主张

- 不能声称首次提出神经网络输出预测攻击；Kimura等已经占据该任务。
- 不能声称首次使用CNN预测密码输出；Kimura 2022附录已有普通Conv1D。
- 不能声称首次逐轮神经仿真PRESENT；Singh已有去密钥逐轮MLP。
- 不能把ResCNN或Transformer本身作为创新；它们在OPA1只是同预算候选。
- OPA3完成前不能声称精确P-layer导致`AUC=1.0`。
- 不能把固定密钥、八位置、三轮AUC写成完整密文恢复、未知密钥泛化或主流七至九轮攻击。
- 不能把算法已知结构网络与Kimura black-box LSTM直接排名为同协议SOTA。
- 核心四篇全文审计不能替代穷尽性系统检索，不能使用“文献中没有任何人做过”的绝对表述。

## 7. OPA3对创新边界的裁决

OPA3是方法创新的必要因果门：

```text
exact-P mean AUC >= 0.510
exact-P - best(identity, wrong-P) mean AUC >= +0.030
至少4/8 bit同时满足exact-identity和exact-wrong >= +0.020
exact-P复跑与OPA2 mean AUC差 <= 0.005
```

若通过，保留PSA-SOP方法主张并进入保持八输出契约的四轮OPA4。若不通过，只能写“同预算架构发现”
和整体模型结果，不能把性能归因于精确PRESENT拓扑，也不继续四轮扩展。

## 8. 下一步文献与实验动作

1. 等待OPA3 verified result branch回收并完成正式图像质检，不从训练MSE预判。
2. OPA3通过后，把exact/identity/wrong-P结果加入论文核心表，并实现条件式OPA4。
3. OPA3不通过时，删除论文中的精确拓扑因果措辞，保留OPA2整体架构边界。
4. 在最终使用“新方法”措辞前，再做一次venue-native补充检索，关键词至少覆盖
   `structure-aware output prediction block cipher`、`SPN-aware neural output prediction`、
   `PRESENT P-layer neural network`和`cipher-topology output prediction`。

## 9. 核心来源

- Kimura et al., *Output Prediction Attacks on Block Ciphers Using Deep Learning*, ACNS Workshops 2022,
  DOI `10.1007/978-3-031-16815-4_15`, IACR ePrint `2021/401`。
- Kimura et al., *A Deeper Look into Deep Learning-based Output Prediction Attacks Using Weak SPN Block
  Ciphers*, Journal of Information Processing 31 (2023), DOI `10.2197/ipsjjip.31.550`。
- Watanabe, Ito, and Ohigashi, *On the Effects of Neural Network-based Output Prediction Attacks on the
  Design of Symmetric-key Ciphers*, IACR ePrint `2024/1310`, CSCML 2024 DOI
  `10.1007/978-3-031-76934-4_13`。
- Singh, *PRESENT Full Round Emulation: Structural Flaws and Predictable Outputs*, IACR ePrint
  `2025/1069`。
