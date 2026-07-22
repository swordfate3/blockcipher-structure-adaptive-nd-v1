# 创新2真实输出预测论文协议审计

日期：2026-07-21

状态：完成 / 已纳入OPD1位置绑定结构归因路线

## 1. 审计问题

本审计只回答：Kimura 2022、Kimura 2023与Watanabe 2024实际训练的是否为真实输出预测，以及
当前OP1--OP8和论文协议之间有哪些足以改变实验结论的差异。主任务保持：

```text
固定未知秘密密钥 K
输入 = 未见明文 P
真实密文 = C = PRESENT_K^r(P)
目标 = C的真实bit、完整C，或C的预注册输出parity
```

不存在正负样本类别。`0/1`若出现，是待预测的密文输出值。

## 2. 来源与精确协议

| 项目 | Kimura 2022 | Kimura 2023 | Watanabe 2024 | 当前OP1--OP8 |
|---|---|---|---|---|
| 攻击目标 | 明文到完整密文；密文到完整明文 | 同一完整输出协议，研究弱SPN S-box | 主要为SIMON完整明文恢复 | 明文到64-bit密文或16个密文parity |
| 样本标签 | 同一输入的完整真实输出bit串 | 同左 | 同左 | 同一输入的真实密文bit/parity |
| 任务语义 | 输出回归，不是区分 | 输出回归，不是区分 | 输出回归，不是区分 | 输出预测，不是样本分类 |
| 单模型密钥 | 训练/测试同一固定密钥 | 同左 | 明确确认同一固定密钥最有效 | 同左；seed0/seed1分别训练 |
| 网络 | stacked LSTM + Dense(blocksize) | stacked LSTM | stacked LSTM | 两层MLP、SPN-local或bit-role网络 |
| PRESENT r3 CP | 6层、hidden 300 | 沿用基础方法 | 非PRESENT主实验 | hidden 128 MLP与结构网络 |
| 损失/优化器 | MSE；r3 CP为RMSprop、lr 0.001 | MSE与论文族优化 | MSE、Adam用于SIMON固定配置 | BCE、AdamW、lr 0.001 |
| epoch/batch | 100 / 250 | 100 / 250 | 100 / 250 | 5 / 128 |
| PRESENT r3数据 | `2^17`训练、`2^16`测试 | 16-bit弱SPN为`2^15/2^15` | SIMON为`2^15/2^15` | 4096/1024/2048训练/验证/测试 |
| 主指标 | 完整输出exact match成功概率 | 完整输出exact match | 平均逐bit match；32-bit exact近似 | parity macro AUC；完整bit AUC/accuracy为支持项 |
| PRESENT r3论文结果 | CP exact-match `2^-1.30 ≈ 0.4061` | 沿用基础方法 | 非PRESENT主实验 | 尚无同协议结果 |
| 跨密钥报告 | 独立固定密钥模型后对100把密钥取平均 | 100把密钥 | 10把密钥；附录审计多密钥退化 | 两把密钥，仅小规模诊断 |

原文锚点：

```text
Kimura 2022 lines 422--449: single-key known-plaintext ciphertext prediction
Kimura 2022 lines 477--516: LSTM regression and complete-output exact match
Kimura 2022 lines 552--580: 100 epochs, batch 250, independent fixed-key models
Kimura 2022 Table 7: PRESENT r3 CP = hidden300, 6 layers, RMSprop, lr 0.001
Kimura 2022 Table 12/C.2: PRESENT r3 = 2^17 train, 2^16 test
Kimura 2022 Table 12: PRESENT r3 CP exact-match success probability = 2^-1.30
Watanabe 2024 Sect. 3.2/Table 1: LSTM, MSE, 2^15/2^15, 100 epochs
Watanabe 2024 Appendix A.2: one fixed key across train/test is the effective setting
```

本地PDF与文本均已存在于`papers/innovation_two/`，三篇核心论文也已通过
`paper-research-workflow`导入`/home/fate/paper-workspace`。

## 3. 对当前证据的纠正

OP1--OP8已经证明以下事实：

```text
固定密钥与明文互斥协议有效
真实密文bit/parity标签可重放
aligned parity在r1/r2可预测，在r3只剩弱残差
当前MLP与两种SPN结构网络在4096训练行、5 epochs下未恢复r3 parity
```

它们没有证明：

```text
Kimura LSTM在PRESENT r3失败
完整64-bit输出预测在论文数据规模失败
100 epochs或2^17训练无效
完整输出exact-match已回到随机基线
PRESENT三轮是输出预测路线的最终上限
```

特别地，论文的`2^17`是每个固定密钥模型的总训练明密文对，不是`2^17/class`；该任务没有
正负类别。论文跨100把密钥的含义是分别训练100个固定密钥模型再汇总，不是把100把密钥混进
同一网络形成矛盾标签。

## 4. 下一实验裁决

审计裁决为：

```text
paper protocol alignment = compatible at threat-model level
current implementation   = not an exact or scaled Kimura reproduction
next experiment          = OP9 PRESENT r3 full-output Kimura-LSTM calibration
parity mechanical scale  = closed
sample classification    = forbidden
```

OP9先用极小本地实现门验证MSB-first完整输出、LSTM/MSE/RMSprop、exact-match与落盘缓存，再从
推送提交启动远程单固定密钥`2^17/2^16/100 epochs`校准。远程矩阵只含论文LSTM、参数量匹配MLP、
论文LSTM标签打乱三个必要行。若论文LSTM没有超过标签打乱控制，停止该论文族扩展；若通过，下一步
只做第二独立固定密钥确认，不先扩四轮或宣称复现论文100密钥结果。

结果裁决必须分成两层。OP9原冻结门中的“至少一次完整命中”和bit/AUC控制差值只回答当前实现是否
恢复了非零真实输出信号；论文水平对照必须另外报告观察到的exact-match rate/count、Table 12的
`2^-1.30 ≈ 0.4061`参考值，以及相对差距。对`2^16`测试集，论文均值对应约`26616`次完整命中。
OP9只有一把密钥，因此即使接近该数值也只是论文族单密钥校准，不是100密钥复现；若仅有少量命中，
不得因通过弱信号门而称为论文水平。

用户随后冻结了创新2更窄且更实用的主目标：不要求完整64-bit同时命中，而是发现并预测容易的真实密文
输出bit。Kimura完整输出协议继续作为共享64位置扫描器和外部文献锚点；创新2主裁决改为逐bit预测，
且候选发现与fresh明文确认必须分离。完整exact-match为零不能推出所有单bit均不可预测。

## 5. 证据边界

Kimura的成功定义与有限测试集可观察性存在方法学风险：`2^16`测试样本无法稳定估计接近`2^-64`
的随机完整输出命中率。因此OP9同时报告原始64-bit exact match、逐bit match、macro AUC、无效舍入率
和相对标签打乱差值，但不会用AUC替换论文主指标，也不会把单密钥校准写成论文复现或高轮突破。

## 6. Jeong 2024/2026通用输出仿真协议补充审计

两篇开放全文已经通过标题、作者、DOI、PDF元数据和正文交叉核验：

```text
Jeong et al., Mathematics 2024, DOI 10.3390/math12131936
Jeong et al., Mathematics 2026, DOI 10.3390/math14020373
```

它们与创新2共享的核心任务是：输入明文bit，输出同一输入对应的完整真实密文bit串。2026论文明确
冻结同一未知秘密密钥，属于固定密钥KPA场景；2024正文的数据生成段没有单独把密钥冻结规则写得足够
明确，因此只能结合其2026后续论文确认论文族设定，不能从2024单篇过度推断。

| 项目 | Jeong 2024 | Jeong 2026 | 当前创新2 OPD1 |
|---|---|---|---|
| 密码 | SDES、SAES、DES、AES-128、SPECK32/64 | 同左 | PRESENT-80 |
| 目标 | 完整密文/明文逐bit输出 | 完整密文/明文逐bit输出 | 八个冻结真实密文bit |
| 网络 | FCNN；3层BiLSTM-256 | FCNN；3层BiLSTM-256 | 普通ResCNN；位置绑定SPN-ResCNN及控制 |
| 结构先验 | DES/SPECK两半作为长度2序列 | BiLSTM逐bit序列 | 精确P、无P、错误P、固定位置head |
| 训练规模 | 架构比较`2^22`训练、`2^15`测试 | `2^20`训练、`2^15`测试 | `2^17`训练、`2^16`测试 |
| 训练协议 | 300 epochs、batch 128、BCE/AdamW | 10全局轮 x 10本地epoch、batch 4096、BCE/Adam | 100 epochs、batch 250、MSE/RMSprop |
| 指标 | 完整输出平均逐bit准确率BAPavg | BAPavg与二项检验 | 逐bitAUC、准确率超多数类及控制差值 |
| 拓扑归因 | 无 | 无 | exact-P/no-P/wrong-P/标签打乱 |

Jeong 2024在`2^22`训练对下报告SPECK32/64三轮EE的FCNN/BiLSTM BAPavg分别为`0.587/0.883`。
这是ARX密码、完整输出和更大预算结果，不能与PRESENT三轮八位置AUC直接比较，也不能作为PRESENT
高轮证据。2026论文的主要新增量是联邦训练扩展，不是密码结构网络。

## 7. 对当前路线的约束

这次核验排除了三个不严谨表述：

```text
不能把“使用非LSTM网络”本身写成创新
不能把“逐bit输出准确率”本身写成首次提出
不能把联邦训练扩展写成PRESENT结构编码
```

当前可检验的新意保持为：先从全部64个真实密文位置冻结易预测bit，再用位置绑定输出头和
`exact-P / no-P / wrong-P / label-shuffle`同参数控制，判断PRESENT拓扑是否带来可识别增益。

OPD1完成前不插入新的Jeong复现或BiLSTM训练。若OPD1通过，按冻结计划先做全新固定密钥原样确认；
若OPD1未通过，只能停止当前`2^17/100 epochs`同预算位置绑定路线，不能据此声称输出预测或所有网络
架构达到上限。Jeong 2024的`2^22/300 epochs`只保留为后续论文规模边界与可选外部架构校准，不用于
绕过OPD1的预注册停止门。
