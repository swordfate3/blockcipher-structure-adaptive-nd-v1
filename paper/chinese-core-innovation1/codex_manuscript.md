# 面向异构 SPN 的运行时结构先验神经区分方法：Dialga 拓扑归因与 uKNIT 非线性语义验证

> 稿件状态说明：本文为中文核心期刊投稿风格初稿，实验数据截至 2026 年 8 月 3 日。正文严格区分本地完整闭环、原始回收并重裁决、远程 gate 完成但归档待闭环、协议无效和用户终止五类状态。本文把 `262144/class` 设为当前论文的最高训练规模；百万规模 DFC1 已按作者决定终止，六行矩阵仅完成一行，所有不完整指标均排除出正文结论。投稿定稿时仍需逐条核验参考文献和作者信息，并按目标期刊体例调整篇幅。

## 摘要

针对异构代换—置换网络（substitution-permutation network，SPN）中非连续状态单元、逐阶段非线性语义及多线性层拓扑难以被通用神经区分器显式利用的问题，提出一种运行时结构先验神经区分方法。该方法将公开密码结构编码为可执行说明书，包括原生单元映射、单元内比特角色、S 盒真值语义、逐轮 GF(2) 连接关系和观察窗口；在统一消费接口下，分别构造 uKNIT-BC 非线性语义专家和 Dialga-128 Runtime-E4 拓扑专家。uKNIT-BC 5 轮 K1-U 在每类 65536 个训练样本、两个随机种子下得到 0.974540/0.967867 的正确 S 盒 AUC，错误 S 盒控制为 0.503901/0.505827；位置不变控制为 0.977201/0.974682，说明正确非线性语义稳定有效，但精确位置编码的必要性未获支持。Dialga-128 4 轮 D2 在同一冻结检查点下仅替换说明书，正确、扰动和无拓扑 AUC 分别为 0.958417/0.958679、0.925256/0.918380 和 0.517403/0.526351。最高规模 DMC2 在每类 262144 个训练样本下，正确拓扑 AUC 为 0.984964/0.984212，扰动拓扑为 0.967531/0.967088，AutoND/DBitNet 为 0.502370/0.501358。结果支持把机器可读结构说明书和反事实控制用于特定密码、轮数与差分下的结构归因；不支持共享权重网络适配任意 SPN，不代表对既有论文大规模协议的复现，也不构成攻击轮数或公开最佳结果的突破。

**关键词：** 神经区分器；代换—置换网络；结构先验；uKNIT-BC；Dialga-128；跨密钥评估

## 1 引言

神经区分器将约化轮密码分析转化为统计学习问题：模型根据一个或多个密文对，判断样本来自给定输入差分下的真实加密分布，还是来自协议规定的负样本分布。自神经差分密码分析在 ARX 密码上显示出有效性以来，卷积网络、残差网络和自动化网络搜索逐渐被用于多种公开分组密码的离线安全评估[1-4]。现有方法常把密文比特直接排列为向量或规则张量，再由网络自行发现有效统计关系。该范式便于迁移，但也弱化了一个基本事实：分组密码的 S 盒、状态单元划分和线性扩散拓扑均为公开且具有明确语义的结构信息。

对结构规则、轮函数对齐的传统 SPN，固定卷积感受野或按 nibble 分组的输入表示有时能够隐式吸收部分结构。异构 SPN 则提出了更严格的要求。例如，状态中的逻辑 cell 可能由非连续比特组成，不同阶段可能使用不同 S 盒，线性层也可能随轮次变化。若仅依赖密码名称绑定一个固定网络，模型无法清楚表达“某一比特在当前轮中属于哪个 cell、承担什么角色、由哪些源比特经 GF(2) 关系生成”。若模型在正确结构和错误结构下均取得相近结果，则性能也不能归因于密码结构本身。

本文选择 uKNIT-BC 与 Dialga-128 作为两个互补案例。uKNIT 的非轮对齐设计使阶段身份、原生 cell 位置和非线性语义成为主要建模对象；Dialga 使用多个线性层，其非连续 cell 与 GF(2) 源—目标连接更适合检验拓扑信息。本文并不假设二者共享同一种充分统计量，而是统一结构说明书、运行时消费接口和控制实验原则，再针对不同结构机制配置差异化专家。

围绕上述目标，本文研究三个问题：

1. 在数据、优化器、训练轮数和负样本定义保持一致的条件下，结构专家能否优于通用 AutoND/DBitNet 及其他公开网络架构适配器？
2. 观察到的优势是否依赖正确的 S 盒或 GF(2) 拓扑，而不是参数量、训练随机性或简单的多密文对聚合？
3. 结构信号在增加一轮后能否保持，其可重复边界在哪里？

本文的主要工作如下。

（1）构建面向异构 SPN 的运行时结构说明书，把 `cell_membership`、`bit_role`、S 盒真值表、逐轮线性源关系和观察窗口从固定模型代码中分离，作为可校验的结构输入。

（2）针对两类结构机制设计差异化专家。uKNIT 专家由两轮算子组合窗口生成五阶段视图，在原生 cell 上统计 4 bit 取值分布；Dialga Runtime-E4 专家按运行时说明书重组非连续 cell，并沿多轮 GF(2) 拓扑融合密文对信息。

（3）形成正确结构、错误结构、无结构、通用模型、公开论文架构适配器及同检查点反事实替换的控制链，并同时报告成功窗口和相邻轮数失败边界。当前结论仅限于给定差分、模型与实验预算下的离线约化轮区分，不涉及密钥恢复，也不声称统一网络适配全部 SPN。

## 2 相关工作

### 2.1 神经差分区分器

Gohr 将深度残差网络用于 SPECK 的约化轮差分区分，推动了神经网络与传统差分分析的结合[1]。后续研究从可解释性、数据复杂度、训练稳定性和泛化等角度重新评估神经区分器，指出高精度结果必须与输入差分、负样本构造、密钥抽样和测试协议共同解释[2-3]。因此，仅比较来自不同数据协议的单个 accuracy 或 AUC 并不足以支持方法优越性。本文采用同协议控制，并把 train、validation、seed、pairs/sample 和 samples/class 分开记录。

### 2.2 通用模型与自动化建模

AutoND/DBitNet 一类方法试图通过统一输入编码和自动化网络构造降低手工设计成本[4]。该类模型适合作为“没有显式消费目标密码运行时结构”的通用基线，但通用性不等于在每个密码上均达到最强效果。本文不把 AutoND 的同协议诊断等同于其公开论文的大规模训练协议；即使本文最高的 `262144/class` 训练也只有每模型、每种子 524288 个总训练样本，仍明显小于公开代码每轮 (10^7) 个总训练样本的量级。因而本文的比较只回答同数据与同优化预算下的架构和结构归因问题，不用于判定 AutoND 的论文规模能力上限。

### 2.3 SPN 专用架构、表示与多密文对聚合

SPN 神经区分研究主要从输入组织和网络几何两方面引入结构。Zhang 和 Wang 面向 DES、Chaskey 与 PRESENT 构造多密文对网络，以核宽为 1、2、4 的多分支一维卷积提取不同尺度特征，再通过递增奇数核残差块和全局平均池化完成预测[6]。Liu 等面向 SKINNY 与 MIDORI 将状态写成三通道二维矩阵，并采用二维卷积残差网络；其 Case-3 表示还包含逆轮处理后的状态与差分[7]。这些工作说明，多尺度卷积、状态矩阵和逆轮表示均是重要的公开先例，但其密码、差分、负样本、训练规模和评价指标与本文 uKNIT 协议并不一致。

为避免只与 AutoND/DBitNet 比较，本文把上述两类骨干迁移到冻结的 uKNIT K1-BS 协议：MCND 适配器保留多分支一维卷积、递增核残差块与全局平均池化；Case-3 Conv2D 适配器使用 \(C\)、\(C'\) 和原始 \(C\oplus C'\) 三通道状态矩阵。后者有意不使用 Liu 等的逆轮表示，以单独考察 Conv2D 骨干，而不把架构收益与特征工程混为一谈。因此，这两行是统一协议下的架构适配，不是原论文复现。

多密文对输入可以通过均值、注意力或集合网络降低单对预测方差，但 pair 数增加本身不保证产生新的可归因信号[8]。如果正确结构与错误结构随 pair 数同步改善，则不能把增益解释为语义先验。本文在 uKNIT K1-BV 中同时设置 exact 4-pair、exact 16-pair 和 wrong-S-box 16-pair，以区分数量放大与结构语义。

### 2.4 uKNIT 与 Dialga 的公开设计

uKNIT 通过打破传统轮对齐组织轻量级密码结构，其公开设计强调不同非线性原语与线性变换的组合[9]。Dialga 是使用多个线性层的低时延可调分组密码族，具有适合硬件时延约束的结构安排[10]。本文仅在二者公开算法的约化轮版本上进行离线区分实验，目标是研究神经模型对结构说明书的消费能力，而不是重新评价其完整轮安全性。

## 3 问题定义与证据分级

### 3.1 区分任务

设公开分组密码的约化轮映射为 \(E_r(\cdot,K)\)，输入差分为 \(\Delta P\)。每个样本包含 \(q\) 个密文对。正样本中，

\[
P_i'=P_i\oplus \Delta P,\qquad
(C_i,C_i')=(E_r(P_i,K),E_r(P_i',K)),
\]

其中 \(i=1,\ldots,q\)。严格负样本同样经过公开加密算法生成，但使用独立随机明文构造，使标签差异不退化为“密文是否经过加密”的简单判别。uKNIT 协议使用 encrypted random plaintexts，Dialga 协议使用 encrypted random plaintext pairs。二分类标签只表示正、负样本来源，不表示密钥比特或明文属性。

模型输出分数 \(s(x)\)，主要指标为受试者工作特征曲线下面积（AUC）；accuracy 依赖阈值，仅作为补充。结构归因主要比较

\[
\Delta_{\mathrm{sem}}=\mathrm{AUC}_{\mathrm{correct}}-\mathrm{AUC}_{\mathrm{wrong}},
\]

以及正确结构相对无结构或通用基线的差值。本文对不同 seed 分别报告结果，不以两点均值掩盖方向不一致。

### 3.2 证据状态

为防止远程训练完成被误写为论文证据闭环，本文使用表 1 的证据状态。这里的“gate”指依据预先记录的阈值、必要控制和计划一致性作出的研究裁决，不等同于训练脚本正常退出。

**表 1  实验结果的证据状态**

| 状态 | 判定条件 | 本文实验 |
|---|---|---|
| 本地完整闭环 | 本地具有规范结果、验证、gate 及可追溯产物 | uKNIT K1-BS、K1-BZ；Dialga D1、D2、D3 |
| 原始回收并重裁决 | 从远程运行根回收原始归档，校验清单、验证和本地 gate 可追溯，但不是完整的验证结果分支归档 | uKNIT K1-U、Dialga DMC2 |
| 远程 gate 已完成，归档待闭环 | 远程验证和研究 gate 已完成，但最终归档或本地规范重裁决不完整 | uKNIT K1-BV、Dialga DMC1 |
| 协议无效 | 产物来源、缓存、配置或 gate 的必要条件发生冲突；指标不得用于结论 | uKNIT K1-BT |
| 用户终止 | 计划矩阵未完成，不能进行组间裁决 | Dialga DFC1（1/6 行完成，排除指标） |

该分级直接限制可写结论：本地闭环结果可用于当前协议下的机制与边界判断；原始回收结果必须保留归档来源限定；协议无效或用户终止的矩阵即使出现有利数值也必须撤出性能表，不能从不完整对照中择取趋势。

## 4 运行时结构先验方法

### 4.1 结构说明书与统一接口

运行时结构说明书是机器可读、可校验的公开结构描述，而非密码名称或自然语言标签。设状态宽度为 \(n\)，4 bit cell 数为 \(m=n/4\)。说明书至少包含：

\[
\mathcal S=\{n,\pi_{\mathrm{cell}},\pi_{\mathrm{role}},T_S^{(t)},A^{(t)},W\},
\]

其中，\(\pi_{\mathrm{cell}}:\{0,\ldots,n-1\}\rightarrow\{0,\ldots,m-1\}\) 表示比特所属原生 cell，\(\pi_{\mathrm{role}}\) 表示比特在 cell 中的 0—3 号角色，\(T_S^{(t)}\) 为第 \(t\) 个阶段或轮次的 S 盒真值语义，\(A^{(t)}\) 为 GF(2) 上目标比特与源比特的连接关系，\(W\) 为模型消费的结构窗口。

统一接口完成三项工作：校验状态宽度与 cell 完整性；把任意物理比特位置重组为有序原生 cell；向结构专家提供逐轮算子而不在网络中硬编码密码名称。本文所谓“统一”限于该说明书和消费协议。uKNIT 与 Dialga 的特征提取器及已训练权重并不共享。

### 4.2 uKNIT 五阶段原生单元位置直方图专家

uKNIT 专家消费两轮运行时结构窗口。对每个密文对，程序按公开算子精确组合生成 `COMPOSITION_STAGE_NAMES` 对应的五个阶段视图。每个阶段中的比特依据 \(\pi_{\mathrm{cell}}\) 和 \(\pi_{\mathrm{role}}\) 重组为有序 4 bit 原生单元，得到取值 \(v_{t,c,i}\in\{0,\ldots,15\}\)。对一个样本的 \(q\) 个密文对，计算

\[
h_{t,c,a}=\frac{1}{q}\sum_{i=1}^{q}\mathbf 1[v_{t,c,i}=a],
\quad a\in\{0,\ldots,15\}.
\]

由此形成“阶段 × 原生 cell × 16 类取值”的 pair-mean one-hot 直方图。16 维取值分布先经共享编码器映射，再保留阶段和位置身份进行投影；所得结构残差通过有界门控与基础密文对表示融合。该表示不是简单的“逆 S 盒若干次”，而是精确算子组合视图在原生位置上的统计读出。

wrong-S-box 控制只替换 S 盒真值语义，线性矩阵、cell 布局、样本、pair 数和训练预算保持不变。若正确 S 盒未稳定优于错误 S 盒，则不能把结果归因于正确非线性语义。

### 4.3 Dialga Runtime-E4 拓扑专家

Dialga 说明书给出非连续 `cell_membership`、cell 内 `bit_role`、S 盒真值及多轮 GF(2) `target_sources`。Runtime-E4 首先从每个密文对构造左状态、右状态及其异或差分，并按说明书把物理位置重组为 cell token；随后利用真实源—目标连接形成图上下文或精确逆线性上下文，在若干结构处理步中融合 cell、拓扑和 S 盒信息。pair 内部和多个 pair 之间分别通过等变混合、均值/最大值与注意力聚合，最后输出区分分数。

正确拓扑控制使用公开说明书；扰动拓扑在保持张量尺寸和训练流程的条件下确定性改变连接关系；无拓扑控制关闭真实关系；AutoND/DBitNet 不消费该运行时拓扑。D2 是最强的机制控制：固定 D1 正确拓扑模型的同一检查点，不再训练，仅在推理时换入正确、扰动或无拓扑说明书。这样可以排除不同初始化和不同优化轨迹，把预测变化直接关联到运行时说明书。

### 4.4 方法边界

本方法不是“一个网络通吃所有 SPN”。其框架可表示多类公开结构，但结构专家仍依赖问题机制：uKNIT 侧重阶段—位置—取值分布，Dialga 侧重非连续 cell 和 GF(2) 拓扑。若目标密码需要新的结构算子，仍需新增与必要控制一同验证的专家。结构说明书也不自动产生可区分性；当输入差分在更高轮数扩散后不再保留可学习统计，正确结构模型仍可能接近随机。

## 5 实验设置

### 5.1 公共设置

全部实验均针对公开算法的离线约化轮区分。优化器为 Adam，损失函数为均方误差，训练 10 个 epoch；每项比较冻结设备、数据协议、负样本定义与优化预算，不在同预算比较中途切换设备。表中 `samples/class` 表示每类样本数，总训练行数为其两倍。K1-BS/K1-BZ 的 2048/class 是本地小样本架构诊断；K1-U 和 DMC1 的 65536/class 是中等规模结果；DMC2 的 262144/class 是本文最高规模的双种子确认。本文不把上述规模称为 AutoND 原论文级复现或通用正式规模。除报告 accuracy 外，裁决优先使用不依赖固定阈值的 AUC。

### 5.2 uKNIT 协议

K1-BS 使用 uKNIT-BC r5、16 pairs/sample、2048/class 训练和 1024/class 跨密钥验证，seed 为 3、4；比较结构专家、AutoND/DBitNet 与两个项目内部通用 SPN 网络。K1-BZ 逐项复用 K1-BS 的差分、密钥、数据缓存、负样本、优化器和验证集，只新增 Zhang/Wang MCND 与 Liu raw Case-3 Conv2D 两个适配器。MCND 适配器含 650177 个参数；Conv2D 适配器含 130945 个参数。由于本机 CUDA 不可用，该极小诊断按预注册例外在 CPU 上执行，未把中等规模训练转移到本地 CPU。

K1-BZ 的晋级门要求同一适配器在两个 seed 上均满足 AUC 不低于 0.550，且相对对应 AutoND/DBitNet 至少提高 0.010。该门只决定是否值得进入远程中等规模验证，不把 2048/class 结果升级为正式证据。历史 K1-BT 原计划使用 65536/class，但同一 run id 出现两套不同的 progress/results，当前远程产物又由预期的四次缓存创建与四次复用漂移为八次创建、零复用；其研究 gate 为 `invalid`，故本文不报告其 AUC。

K1-U 使用 uKNIT-BC r5、4 pairs/sample、65536/class 训练和 32768/class 跨密钥验证，seed 为 3、4。它比较正确 S 盒加原生位置、错误 S 盒加原生位置和正确 S 盒加位置不变聚合，分别检验非线性语义与精确位置身份。K1-BV 使用 uKNIT-BC r6，训练 2048/class，跨密钥验证 1024/class，seed 为 3、4；比较 exact 4-pair、exact 16-pair 与 wrong-S-box 16-pair。K1-BV 是远程小规模边界诊断，不用于估计 r6 的方法上限。

### 5.3 Dialga 协议

D1 使用 Dialga-128 prefix-r4，4 pairs/sample，训练 2048/class，验证 1024/class，seed 为 0、1，分别训练正确拓扑、扰动拓扑和无拓扑模型。D2 复用 D1 正确拓扑检查点及相同验证协议，只替换推理说明书。DMC1 和 DMC2 保持 prefix-r4、4 pairs/sample、两个 seed 及三模型矩阵，训练规模分别为 65536/class 和 262144/class；DMC2 的验证规模为 65536/class。Runtime-E4 有 442466 个参数，AutoND/DBitNet 有 797633 个参数；两者数据和优化预算一致，但并非参数量匹配。D3 使用 prefix-r5、2048/class、4 pairs/sample、seed 0/1，检查相邻轮数窗口能否复制 r4 结果。

**表 2  实验矩阵与当前证据状态**

| 实验 | 算法/轮数 | pairs | 训练 samples/class | 验证 samples/class | seeds | 主要控制 | 状态 |
|---|---|---:|---:|---:|---|---|---|
| K1-BS | uKNIT-BC r5 | 16 | 2048 | 1024 | 3、4 | AutoND、两个内部通用 SPN 网络 | 本地完整闭环，gate=`pass` |
| K1-BZ | uKNIT-BC r5 | 16 | 2048 | 1024 | 3、4 | MCND、raw Case-3 Conv2D | 本地完整闭环，gate=`hold` |
| K1-BT | uKNIT-BC r5 | 16 | 65536 | 16384 | 3、4 | AutoND/DBitNet | 协议无效，不报告指标 |
| K1-U | uKNIT-BC r5 | 4 | 65536 | 32768 | 3、4 | wrong-S-box、位置不变 | 原始回收，协议有效，研究 gate=`hold` |
| K1-BV | uKNIT-BC r6 | 4/16 | 2048 | 1024 | 3、4 | wrong-S-box | 远程 gate=`hold`，归档待闭环 |
| D1 | Dialga-128 prefix-r4 | 4 | 2048 | 1024 | 0、1 | 扰动/无拓扑 | 本地完整闭环，gate=`pass` |
| D2 | Dialga-128 prefix-r4 | 4 | 不再训练 | 复用 D1 协议 | 0、1 | 同检查点替换说明书 | 本地完整闭环 |
| DMC1 | Dialga-128 prefix-r4 | 4 | 65536 | 16384 | 0、1 | 扰动拓扑、AutoND | 远程 gate=`pass`，归档待闭环 |
| DMC2 | Dialga-128 prefix-r4 | 4 | 262144 | 65536 | 0、1 | 扰动拓扑、AutoND | 原始回收并本地重裁决，gate=`pass` |
| D3 | Dialga-128 prefix-r5 | 4 | 2048 | 1024 | 0、1 | 扰动/无拓扑 | 本地完整闭环，gate=`hold` |

百万规模 DFC1 不属于本文有效矩阵。该任务按作者决定于 2026 年 8 月 3 日终止；终止时只完成正确拓扑 seed 0 一行，扰动拓扑 seed 0 尚处第 6 个 epoch，其余四行未运行。因此本文不报告 DFC1 的任何性能数字，也不以其局部结果增强 DMC2 结论。

## 6 实验结果与分析

### 6.1 uKNIT K1-BS/K1-BZ：公开架构补充对比

表 3 汇总冻结的 K1-BS 锚点与 K1-BZ 新增架构。结构专家在 seed 3/4 的 AUC 为 0.902802/0.932539；AutoND/DBitNet 为 0.511321/0.526423。新增的 Zhang/Wang MCND 适配器为 0.493508/0.493233，Liu raw Case-3 Conv2D 适配器为 0.526742/0.505930。图 1 左侧给出统一协议下的四模型 AUC，右侧给出新增适配器相对 AutoND/DBitNet 的逐 seed 差值。

**表 3  uKNIT-BC r5 统一小样本协议下的公开架构补充对比**

| 模型 | 参数量 | seed 3 AUC | seed 4 AUC | 相对 AutoND（seed 3/4） |
|---|---:|---:|---:|---:|
| 五阶段位置直方图结构专家（K1-BS） | 214316 | 0.902801514 | 0.932538986 | +0.391480446/+0.406115532 |
| AutoND/DBitNet（K1-BS） | 985985 | 0.511321068 | 0.526423454 | 0/0 |
| Zhang/Wang MCND 适配（K1-BZ） | 650177 | 0.493507862 | 0.493232727 | −0.017813206/−0.033190727 |
| Liu raw Case-3 Conv2D 适配（K1-BZ） | 130945 | 0.526742458 | 0.505930424 | +0.015421390/−0.020493030 |

![图 1 uKNIT K1-BZ 公开论文架构补充对比](figures/fig_uknit_k1bz_published_architecture_comparison.svg)

**图 1  uKNIT K1-BZ 公开论文架构补充对比。固定 r5、cell11 差分、16 pairs/sample、2048/class、跨密钥验证、seed 3/4 和 10 个 epoch；图中结果是架构适配诊断，不是三篇基线论文的原协议复现。**

MCND 在两个 seed 上均低于 AutoND。Liu Conv2D 在 seed 3 高 0.015421，但 seed 4 低 0.020493，且两颗 seed 的 AUC 都未达到 0.550。预注册门要求同一适配器在两颗 seed 上同时达到 AUC 0.550，并相对 AutoND 至少提高 0.010，因此 gate 状态为 `hold`，决策为 `innovation1_uknit_k1bz_no_published_adapter_local_promotion`，不进入远程扩样。

本结果只说明两个具体适配器在当前 uKNIT 小样本协议中没有稳定恢复信号。MCND 行改变了原论文的密码、负样本和训练规模；Liu 行还去掉了原框架的逆轮表示，只保留三通道状态矩阵与 Conv2D-ResNet 骨干。因而不能由表 3 推导 Zhang/Wang 或 Liu 原方法失效，也不能认为这两类架构已经完成充分超参数搜索。另一方面，K1-BS 结构专家的优势同样不是容量匹配的因果证明；它只支持在该冻结协议下保留当前结构专家作为后续研究对象。

### 6.2 uKNIT K1-U：正确 S 盒语义有效，位置必要性未获支持

K1-U 把 uKNIT r5 结构语义验证扩展到 65536/class。表 4 显示，正确 S 盒加原生位置在两个 seed 上的 AUC 为 0.974540/0.967867，错误 S 盒控制为 0.503901/0.505827，语义差值达到 +0.470640/+0.462040。该方向在两个 seed 上一致，支持模型信号依赖正确 S 盒语义，而不是只依赖相同形状的结构输入。

**表 4  uKNIT-BC r5 K1-U 中等规模语义与位置控制**

| 条件 | seed 3 AUC | seed 4 AUC |
|---|---:|---:|
| 正确 S 盒 + 原生位置 | 0.974540495 | 0.967867357 |
| 错误 S 盒 + 原生位置 | 0.503900695 | 0.505827348 |
| 正确 S 盒 + 位置不变 | 0.977200513 | 0.974682369 |
| 正确位置分支 − 错误 S 盒 | +0.470639800 | +0.462040009 |
| 正确位置分支 − 位置不变 | −0.002660018 | −0.006815012 |

![图 2 uKNIT K1-U 的中等规模 S 盒语义与位置控制](figures/fig_uknit_k1u_semantic_position.svg)

**图 2  uKNIT K1-U 的跨密钥 AUC 与归因差值。正确 S 盒相对错误 S 盒的优势在两个 seed 上稳定，而原生位置分支没有优于位置不变控制。**

然而，位置不变控制在两个 seed 上分别高 0.002660 和 0.006815。K1-U 的研究 gate 因而为 `hold`，决策为 `innovation1_uknit_family_ctspn_k1u_medium_signal_without_position_necessity`。准确结论是：正确 S 盒语义具有稳定归因证据，但固定原生位置展开不是必要条件，较简单的位置不变单元计数更适合作为后续表示。该结果来自原始回收归档；校验清单、缓存事件、六行结果和检查点重算均通过，但它不是验证结果分支的完整归档，也不是论文级大规模基准。

### 6.3 uKNIT K1-BV：6 轮密文对放大未获支持

K1-BV 的主要裁决依据是 AUC，而非阈值相关的 accuracy。由表 5 可见，seed 3 从 exact 4-pair 增至 exact 16-pair 的 AUC 增量为 0.028973，但 wrong-S-box 16-pair 反而比 exact 16-pair 高 0.001780；seed 4 的 pair 增量仅为 0.001619，exact 16-pair 比 wrong-S-box 低 0.006320。两个 seed 均未形成“增加 pair 后正确语义稳定优于错误语义”的必要组合。

**表 5  uKNIT-BC r6 K1-BV 的 AUC 与 gate 派生量**

| 条件 | seed 3 AUC | seed 4 AUC |
|---|---:|---:|
| exact 4-pair | 0.492288589 | 0.496891975 |
| exact 16-pair | 0.521261692 | 0.498510838 |
| wrong-S-box 16-pair | 0.523041725 | 0.504830837 |
| pair gain：exact16 − exact4 | +0.028973103 | +0.001618862 |
| semantic gap：exact16 − wrong16 | −0.001780033 | −0.006320000 |

作为补充，三种条件的 accuracy 分别为：seed 3 下 0.498535、0.508789、0.500000；seed 4 下 0.500977、0.500977、0.501953。图 3 直接展示 AUC、pair 增益和正确 S 盒相对错误 S 盒的差值，与表 5 的 gate 量一致。

![图 3 uKNIT K1-BV 的 AUC、pair 增益与 S 盒语义差值](figures/fig_uknit_k1bv_boundary.svg)

**图 3  uKNIT K1-BV 的 AUC、pair 增益与 S 盒语义差值。两个 seed 均未形成“增加 pair 且正确 S 盒稳定占优”的组合。**

远程 gate 状态为 `hold`，决策为 `innovation1_uknit_k1bv_pair_amplification_not_supported`，等级为 `unsupported`；最终结果打包失败。因此准确结论是：在该固定差分、固定结构专家和 2048/class 诊断预算下，没有证据支持通过 4-pair 到 16-pair 的机械放大获得 r6 结构优势。该结果不能外推为 uKNIT r6 不存在神经区分信号，更不能视为方法在所有更大规模上的上限。

### 6.4 Dialga D1：4 轮拓扑控制

D1 已完成本地结果、验证和 gate 闭环。表 6 中，正确拓扑在两个 seed 上均约为 0.9585，分别高于扰动拓扑 0.022107 和 0.020863；无拓扑结果接近随机，正确拓扑优势超过 0.455。扰动拓扑仍保留较高 AUC，说明 D1 信号并非全部来自精确接线，基础输入统计与部分结构可能已经提供区分信息；但正确拓扑在两个 seed 上方向一致的增量表明精确连接仍有贡献。

**表 6  Dialga-128 prefix-r4 D1 本地闭环结果**

| seed | 正确拓扑 AUC | 扰动拓扑 AUC | 无拓扑 AUC | 正确−扰动 | 正确−无拓扑 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.958416939 | 0.936309814 | 0.497210026 | +0.022107124 | +0.461206913 |
| 1 | 0.958679199 | 0.937816620 | 0.503405571 | +0.020862579 | +0.455273628 |

### 6.5 Dialga D2：同一冻结检查点的反事实说明书替换

D2 不重新训练模型，只在 D1 正确拓扑检查点上更换推理时结构说明书。表 7 显示，正确说明书的 AUC 保持为 0.958417/0.958679，扰动说明书下降至 0.925256/0.918380，无拓扑下降至 0.517403/0.526351。相对于扰动拓扑，正确拓扑优势为 0.033161 和 0.040299；相对于无拓扑，优势为 0.441014 和 0.432328。

**表 7  Dialga D2 同检查点结构替换结果**

| seed | 正确拓扑 AUC | 扰动拓扑 AUC | 无拓扑 AUC | 正确−扰动 | 正确−无拓扑 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.958416939 | 0.925255775 | 0.517402649 | +0.033161163 | +0.441014290 |
| 1 | 0.958679199 | 0.918379784 | 0.526350975 | +0.040299416 | +0.432328224 |

![图 4 Dialga D2 同一检查点下的结构说明书替换结果](figures/fig_dialga_d2_same_checkpoint.svg)

**图 4  Dialga D2 同一检查点下的结构说明书替换结果**

更换说明书还引起明显的逐样本预测变化：seed 0 的最大预测概率变化在扰动和无拓扑条件下分别为 0.912204、0.918676，seed 1 分别为 0.902221、0.869230。D2 因此提供了比“分别训练三个模型”更直接的功能依赖证据：在权重和检查点不变时，模型输出确实随运行时拓扑改变。不过，该实验仍只能证明模型依赖当前说明书，不能推导为模型已经学习到 Dialga 的完整密码学语义。

### 6.6 Dialga DMC1/DMC2：4 轮拓扑效应的规模确认

DMC1 将训练规模提高到 65536/class，DMC2 进一步提高到 262144/class。两级实验均保持正确拓扑、扰动拓扑和 AutoND/DBitNet 三模型矩阵。表 8 显示，DMC2 正确拓扑 AUC 为 0.984964/0.984212，扰动拓扑为 0.967531/0.967088，正确拓扑增量为 +0.017434/+0.017124；AutoND/DBitNet 为 0.502370/0.501358。与 DMC1 相比，正确拓扑 AUC 提高，但关键归因不是绝对 AUC，而是两个 seed 上均保持正向的正确—扰动拓扑差值。

**表 8  Dialga-128 prefix-r4 DMC1/DMC2 规模梯度结果（AUC）**

| 规模与条件 | seed 0 | seed 1 |
|---|---:|---:|
| DMC1 正确运行时拓扑 | 0.978120916 | 0.978711151 |
| DMC1 扰动拓扑 | 0.964931685 | 0.964382136 |
| DMC1 AutoND/DBitNet | 0.507460726 | 0.500493946 |
| DMC1 正确−扰动 | +0.013189230 | +0.014329014 |
| DMC2 正确运行时拓扑 | 0.984964147 | 0.984212034 |
| DMC2 扰动拓扑 | 0.967530599 | 0.967087931 |
| DMC2 AutoND/DBitNet | 0.502369641 | 0.501357568 |
| DMC2 正确−扰动 | +0.017433548 | +0.017124103 |

![图 5 Dialga DMC2 两个随机种子下的 AUC 与拓扑归因结果](figures/fig_dialga_dmc2_auc.svg)

**图 5  Dialga DMC2 两个随机种子下的 AUC 与正确—扰动拓扑差值。训练规模为 262144/class；该图已经过渲染像素检查。**

DMC2 的六行结果、四次缓存创建、八次参数匹配复用和六个检查点均通过本地重裁决，研究 gate 为 `pass`，决策为 `innovation1_dialga_dmc2_scale_topology_supported`。其归档来自远程运行根的原始回收，而不是完整验证结果分支；同时 `final_test_repeats=0`，没有独立最终测试。因而 DMC2 可作为本文最高规模的项目内拓扑确认，但不能称为 AutoND 论文规模复现或通用正式基准。Runtime-E4 参数少于 AutoND/DBitNet，二者又未严格匹配参数量，表 8 不能单独分解结构、容量和优化难度的独立效应。

### 6.7 Dialga D3：相邻 5 轮窗口未复制

D3 已在本地完成 gate，状态为 `hold`，决策为 `innovation1_dialga_runtime_e4_d3_adjacent_window_not_replicated`。表 9 中，正确拓扑在 seed 0 上略高于扰动拓扑 0.002948，却低于无拓扑 0.036830；seed 1 上同时低于扰动和无拓扑。方向不一致且整体接近随机，不能支持 prefix-r5 的拓扑效应。

**表 9  Dialga-128 prefix-r5 D3 本地边界结果**

| seed | 正确拓扑 AUC | 扰动拓扑 AUC | 无拓扑 AUC | 正确−扰动 | 正确−无拓扑 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.507329941 | 0.504382133 | 0.544159889 | +0.002947807 | −0.036829948 |
| 1 | 0.493329525 | 0.528279781 | 0.519388676 | −0.034950256 | −0.026059151 |

D3 只表明 D1/D2 的 prefix-r4 机制在当前 D3 配置下没有复制到相邻 prefix-r5 窗口。它不抹除 r4 的本地机制证据，也不能据此宣称 Dialga 神经区分的总体上限。

## 7 讨论

### 7.1 两个案例的共同主线

uKNIT 与 Dialga 不是为了堆叠两个高 AUC 案例。前者检验“阶段 × 原生位置 × 非线性取值分布”，后者检验“非连续 cell × GF(2) 源—目标拓扑”。二者共同支持的最小主张是：公开密码结构能够通过机器可读说明书进入神经区分流程，并可使用错误语义和反事实说明书检查模型是否真正消费该结构。论文层面的统一对象是方法接口和实验逻辑，而不是共享模型参数。

### 7.2 结构基线与通用基线的不同作用

AutoND/DBitNet 回答“同数据和同优化预算下，不显式使用目标运行时结构的通用网络表现如何”；MCND 与 Case-3 Conv2D 适配器回答“其他公开网络几何在同一 uKNIT 原始可观测量上是否恢复信号”；wrong-S-box、位置不变、扰动拓扑和无拓扑则回答“结构专家的增益究竟来自哪类正确语义”。三类基线作用不同，不能相互替代。K1-BZ 排除了“只与 AutoND 比较”的单一基线问题，却没有完成参数量匹配或原论文超参数搜索；K1-U 将可归因部分收窄为正确 S 盒语义，并否定了精确位置编码的必要性；Dialga DMC2 包含扰动拓扑，D2 又在同一检查点上替换说明书，其拓扑机制链更完整。

### 7.3 成功窗口与失败边界应同时报告

K1-BS 在 r5 uKNIT 的小样本协议中显示可学习信号，K1-U 则在 65536/class 下确认正确 S 盒语义并收窄位置主张；D1/D2/DMC1/DMC2 在 prefix-r4 Dialga 中形成从本地机制到 262144/class 的拓扑证据链。K1-BZ 表明把另外两种公开网络骨干直接迁移到同一 raw-observable 协议并不能稳定复制信号。K1-BV 和 D3 进一步说明，增加 pair 或沿用相同网络并不能机械拓展一轮。失败边界有助于防止选择性报告，也提示下一步应优先审计差分传播、表示与观察窗口，而不是继续机械扩样。本文据此把实验上限停在 262144/class，并把论文定位为结构机制与反事实归因研究。

### 7.4 与传统密码分析结论的关系

本文结果属于离线、约化轮、固定协议的神经区分评估。AUC 高不直接给出密钥恢复复杂度，也不能与使用不同数据、查询模型和成功准则的传统攻击数字直接比较。特别是 Dialga 已有确定性线性分析背景，本文的意义是检验神经模型对异构拓扑的消费和归因，而不是声称超过确定性方法。多密文对聚合也属于应用层样本组织方式，不能把 16-pair 结果写成单密文对的原始区分能力。

## 8 局限性

本文仍存在以下限制。

第一，证据闭环尚不一致。K1-BS、K1-BZ、D1、D2、D3 已在本地完成规范验证与 gate；K1-U 和 DMC2 为原始回收后重裁决，仍需保留归档来源限定；K1-BV 和 DMC1 的规范归档不完整。K1-BT 已被来源冲突和缓存契约破坏判为协议无效。百万规模 DFC1 由作者主动终止，矩阵不完整，不能通过报告已完成单行恢复为有效比较。

第二，本文最高训练规模为 DMC2 的 262144/class，uKNIT 主语义结果为 65536/class，均采用两个 seed。它们足以支撑本文限定的机制归因与规模确认叙事，但不足以支持“正式规模复现”“通用模型失效”或“达到性能上限”等更强主张。DMC2 也没有独立最终测试，统计重复性仍弱于多 seed、多最终测试的基准研究。本文选择收窄主张，而不是继续用百万规模换取与方法问题不相称的等待时间。

第三，比较并非完全容量匹配。K1-BS/K1-BZ 中结构专家、AutoND/DBitNet、MCND 和 Liu Conv2D 适配器的参数量分别为 214316、985985、650177 和 130945；Dialga Runtime-E4 与 AutoND/DBitNet 分别为 442466 和 797633。相同训练预算不能消除容量、归纳偏置与优化难度差异。更严格的实验应加入参数量近似的无结构控制或等容量通用基线。

第四，当前成功结论受密码、轮数、差分与专家共同约束。uKNIT r5 与 Dialga prefix-r4 的结果不能外推到完整轮、任意差分、其他 SPN 或跨密码零样本场景。本文也未训练一组共享权重在 uKNIT 与 Dialga 之间直接迁移。

第五，K1-BZ 只进行了冻结超参数下的架构适配，没有复现原论文的数据构造、规模和超参数搜索。尤其 Liu 行不含逆轮表示，不能代表其完整 Case-3 框架；MCND 行也不代表其 PRESENT 原协议。K1-U 已提供协议有效的中等规模 wrong-S-box 控制，但仍缺少与 Dialga D2 同等级的冻结检查点反事实替换。

第六，本文尚未系统报告训练时间、推理吞吐、显存占用、置信区间和校准误差。两个 seed 只能说明初步重复性，不能提供充分的方差估计。

## 9 结论

本文提出一种面向异构 SPN 的运行时结构先验神经区分方法：把公开算法中的 cell、比特角色、S 盒语义和 GF(2) 拓扑编码为可执行说明书，再由与结构机制相匹配的专家消费。uKNIT K1-U 在 65536/class 下表明正确 S 盒语义稳定优于错误语义，但位置不变控制略优于精确位置分支，因此支持非线性语义而不支持位置必要性。Dialga D1、D2 的本地闭环结果表明，正确拓扑在同一冻结检查点下直接影响预测；DMC2 在 262144/class、两个 seed 下保持正确拓扑相对扰动拓扑和 AutoND/DBitNet 的优势。K1-BV 与 D3 则分别给出 pair 机械放大和相邻轮直接延伸的失败边界。

因此，现有证据支持“统一结构说明书与控制协议、针对机制采用差异化专家”这一有限结论，但不支持共享权重网络自动适配任意 SPN，也不支持攻击轮数突破。论文不再以百万规模实验为完成条件；后续工作应优先补齐原始回收归档、独立最终测试、参数量匹配控制和 uKNIT 同检查点反事实，而不是继续机械增加样本量。

## 数据与材料可得性声明（占位）

本文实验配置、模型实现、结果文件、验证记录和 gate 产物拟在论文录用或审稿政策允许的阶段公开。投稿前需根据匿名审稿要求替换为匿名仓库或补充材料链接。

## 伦理与利益冲突声明（占位）

本文仅研究公开分组密码的离线约化轮安全评估，不涉及真实网络系统、用户数据或未授权密钥材料。作者声明不存在需要披露的利益冲突（待全体作者确认）。

## 作者贡献与经费声明（占位）

作者贡献、基金项目名称及编号待作者团队确认后按目标期刊格式填写。

## 人工智能辅助写作声明（占位）

本文初稿使用生成式人工智能辅助整理结构、润色中文和核对实验表格；算法判断、实验设计、数据真实性、引用准确性及最终文本由作者负责。正式投稿时按目标期刊政策决定是否保留及如何表述本声明。

## 参考文献（占位，投稿前逐条核验）

[1] GOHR A. Improving attacks on round-reduced Speck32/64 using deep learning[C]//Advances in Cryptology—CRYPTO 2019. 2019.（卷号、页码待核验）

[2] BENAMIRA A, GERARD B, PEYRIN T, et al. A deeper look at machine learning-based cryptanalysis[C]. 2021.（正式题名、会议、页码待核验）

[3] GOHR A, LEANDER G, NEUMANN P. An assessment of differential-neural distinguishers[C]. 2022.（正式出版信息待核验）

[4] AutoND/DBitNet cipher-agnostic neural distinguisher pipeline[EB/OL]. 2023.（作者、正式题名、版本与公开代码地址待核验）

[5] 神经差分密码分析系统化综述（SoK）[C]. 2024.（作者、题名与出版信息待核验）

[6] ZHANG L, WANG Z. Improving Differential-Neural Distinguisher Model for DES, Chaskey and PRESENT[EB/OL]. arXiv:2204.06341, 2022.（正式出版版本与页码待核验）

[7] LIU J S, LI M M, REN J J, CHEN S Z. A Highly Efficient Neural Distinguisher Framework for IoT-Friendly Lightweight SPN Block Ciphers[J]. IEICE Transactions on Information and Systems, 2026. DOI: 10.1587/transinf.2025EDP7070.（卷期页码待核验）

[8] 多密文对神经区分与聚合方法相关工作[C/J].（作者、题名与出版信息待核验）

[9] HU K, KHAIRALLAH M, PEYRIN T, TAN Q Q. uKNIT: Breaking Round-Alignment for Cipher Design[J]. IACR Transactions on Symmetric Cryptology, 2026(2). DOI: 10.46586/tosc.a0zo-4njsuvm.（页码待核验）

[10] BANIK S, et al. Dialga: A Family of Low-Latency Tweakable Block Ciphers Using Multiple Linear Layers[J]. IACR Transactions on Symmetric Cryptology, 2025(4): 70-124. DOI: 10.46586/tosc.v2025.i4.70-124.（完整作者列表待核验）

[11] GPD 等通用密码区分模型相关工作[C/J]. 2025.（作者、题名与出版信息待核验）

## 附录 A 证据路径与复核边界

**表 A1  本文关键结果的本地证据入口**

| 实验 | 证据入口 | 当前边界 |
|---|---|---|
| uKNIT K1-BS | `docs/experiments/innovation1-uknit-r5-neural-architecture-ablation-k1bs-plan.md`；`outputs/local_diagnostic/i1_uknit_r5_neural_architecture_ablation_k1bs_16pair_2048_seed3_seed4_20260731/` | 本地完整闭环；小样本架构诊断 |
| uKNIT K1-BZ | `docs/experiments/innovation1-uknit-r5-published-architecture-baselines-k1bz-plan.md`；`outputs/local_diagnostic/i1_uknit_r5_published_architecture_baselines_k1bz_16pair_2048_seed3_seed4_20260802/` | 本地完整闭环，gate=`hold`；架构适配而非原论文复现 |
| uKNIT K1-BT | `docs/experiments/innovation1-uknit-r5-neural-architecture-medium-k1bt-plan.md`；`outputs/remote_results_incomplete/i1_uknit_r5_neural_architecture_medium_k1bt_16pair_65536_seed3_seed4_20260731_monitor/` | run-id 冲突与缓存复用失败；协议无效，不报告指标 |
| uKNIT K1-U | `docs/experiments/innovation1-uknit-family-ctspn-position-residual-k1u-medium-plan.md`；`outputs/remote_results_incomplete/i1_uknit_family_ctspn_position_residual_k1u_medium_65536_seed3_seed4_20260728/` | 原始回收、校验与协议检查通过；研究 gate=`hold`，支持 S 盒语义但不支持位置必要性 |
| uKNIT K1-BV | `docs/experiments/innovation1-uknit-r6-pair-amplification-k1bv-plan.md`；`outputs/remote_results_incomplete/i1_uknit_r6_pair_amplification_k1bv_2048_seed3_seed4_20260731_monitor/` | 远程 gate=`hold`；最终归档失败 |
| Dialga D1 | `docs/experiments/innovation1-dialga128-runtime-e4-d1-r4-2048-plan.md`；`outputs/local_diagnostic/i1_dialga128_runtime_e4_d1_r4_2048_seed0_seed1_20260725/results.jsonl` | 本地完整闭环 |
| Dialga D2 | `docs/experiments/innovation1-dialga128-runtime-e4-d2-same-checkpoint-plan.md`；`outputs/local_audits/i1_dialga128_runtime_e4_d2_same_checkpoint_20260725/results.jsonl` | 本地完整闭环 |
| Dialga DMC1 | `docs/experiments/innovation1-dialga128-runtime-e4-dmc1-r4-medium-plan.md`；`outputs/remote_results_incomplete/i1_dialga128_runtime_e4_dmc1_r4_65536_seed0_seed1_20260731_monitor/` | 远程 gate=`pass`；最终归档失败 |
| Dialga DMC2 | `docs/experiments/innovation1-dialga128-runtime-e4-dmc2-r4-262144-plan.md`；`outputs/remote_results_incomplete/i1_dialga128_runtime_e4_dmc2_r4_262144_seed0_seed1_20260801/` | 原始回收并本地重裁决，gate=`pass`；本文最高规模结果，不含独立最终测试 |
| Dialga D3 | `docs/experiments/innovation1-dialga128-runtime-e4-d3-r5-2048-plan.md`；`outputs/local_diagnostic/i1_dialga128_runtime_e4_d3_r5_2048_seed0_seed1_20260725/results.jsonl` | 本地完整闭环，gate=`hold` |
| Dialga DFC1 | `docs/experiments/innovation1-dialga128-runtime-e4-dfc1-r4-formal-plan.md`；`outputs/remote_results_incomplete/i1_dialga128_runtime_e4_dfc1_r4_1000000_seed0_seed1_20260802_monitor/` | 2026-08-03 用户终止；仅 1/6 行完成，全部指标排除出论文结论 |
