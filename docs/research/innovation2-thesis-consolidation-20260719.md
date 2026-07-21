# 创新2最终论文综合：结构条件积分输出平衡谱预测

> 历史路线说明：本文收束的是2026-07-19时的结构条件积分输出平衡谱路线。2026-07-21以后，用户
> 冻结的当前创新2主任务改为固定未知秘密密钥下的真实密码输出值预测；OP9--OP12的最新权威边界见
> `docs/research/innovation2-output-prediction-thesis-boundary-20260721.md`。两种任务不得混为同一结果。

**冻结日期：** 2026-07-19

**最终状态：** PRESENT/GIFT双密码方法证据确认；高轮严格标签未就绪；停止继续枚举网络

**主要证据：** E65--E80、E96、E97

**论文定位：** 方法与严格评测贡献，不是高轮区分器、新攻击或SOTA结果

## 1. 最终创新点

建议在毕业论文中将创新2表述为：

> 提出一种面向SPN密码积分输出性质的前缀引导逐节点平衡谱预测方法。该方法把一个输入
> 活动结构对应的64个单位输出掩码建模为图节点，以目标轮之前的ANF支撑与次数谱作为节点
> 特征，并沿密码真实P-layer及S-box分组关系进行共享消息传递，一次输出完整的64维严格平衡
> 谱。通过全称正证书、具体负见证、unknown保留、structure-disjoint checkerboard和同参数
> 错误拓扑控制，在PRESENT-80与GIFT-64两套独立四轮数据上分别训练后，真实密码拓扑在两颗
> seed上均稳定优于独立节点和错误拓扑。

推荐方法名：

```text
中文：前缀引导逐节点平衡谱算子
英文：Prefix-Guided Nodewise Balance-Profile Operator
缩写：PG-NBPO
```

代码和历史实验中使用的`r3-only Profile Operator`指的就是该方法的冻结实现。论文首次定义
缩写后，可以简称`平衡谱算子`。

## 2. 它预测的不是具体密文

创新2的基本查询为：

```text
输入：密码算法、轮数r、8个活动输入bit组成的cube、输出bit位置j
输出：该cube经过r轮后，第j个输出bit对完整cube求XOR是否恒为0
```

对活动结构`S`和单位输出掩码`e_j`，目标性质写为：

```text
B_r(S,j) = XOR_{x in S} <e_j, E_K^r(x)>
```

网络不是预测某个明文的具体输出bit，也不是判断两个密文差分是否匹配。它对同一结构一次输出：

```text
(B_r(S,0), B_r(S,1), ..., B_r(S,63))
```

也就是64个输出bit的平衡谱。训练标签只在能够严格判定时使用：

```text
positive = 活动变量ANF支撑上界中不存在完整cube单项式，
           因而对所有主密钥和所有inactive plaintext offset都能证明XOR为0；

negative = 存在一个真实key schedule生成的主密钥和一个inactive offset，
           对完整2^8=256个cube明文计算后指定输出bit XOR为1；

unknown  = 当前证书没有证明平衡，witness bank也没有找到反例；
           不强行改成negative，也不进入训练。
```

因此，这里的`positive`是全称性质，`negative`是存在性反例；有限密钥上没有观察到失败不能代替
positive证书。

## 3. 一个训练样本到底是什么

神经网络的一个batch元素不是一对密文，而是一个8-bit活动cube结构。该结构经过确定性前缀
分析后形成`64 x 13`节点特征，网络一次给出64个logit。标签矩阵使用masked形式：

```text
profile_targets  : structures x 64，取值{-1, 0, 1}
profile_observed : structures x 64，只有严格positive/negative匹配边为true
```

损失只计算`profile_observed=true`的边，unknown和未被checkerboard选中的边不参与训练。

冻结数据集：

| 密码 | 轮数 | 原始结构 | 输出节点 | 训练匹配边 | 验证匹配边 | 总观测边 |
|---|---:|---:|---:|---:|---:|---:|
| PRESENT-80 | 4 | 96 | 64 | 178正 + 178负 | 60正 + 60负 | 476 |
| GIFT-64 | 4 | 192 | 64 | 248正 + 248负 | 62正 + 62负 | 620 |

PRESENT和GIFT的结构库、标签分布和观测边数量不同，所以只能在各自密码内部比较真实拓扑和
控制，不能用两个密码的绝对AUC高低判断哪个密码“更容易”或哪个模型“更强”。

## 4. 13维r3前缀特征

四轮输出性质不能把第4轮最终证书、负见证、key/offset parity或标签派生量输入网络，否则会
发生目标泄漏。PG-NBPO只读取目标轮前一轮，也就是第3轮的13维ANF前缀统计。

对每个输出bit节点，第3轮特征为：

```text
支撑集合大小的 mean / max / normalized sum     3维
多个选择支撑的 union size                        1维
ANF单项式次数0,1,...,8的归一化直方图             9维
合计                                             13维
```

历史完整prefix特征包含r1、r2、r3三段，共`39=3x13`维；E73/E79确认只保留r3的13维后，
PRESENT和GIFT仍能保持质量与拓扑归因，同时参数从完整39维版本的5,679降为4,795。

## 5. 神经网络结构

PG-NBPO的冻结结构为：

```text
输入                         batch x 64 nodes x 13 features
输入归一化                   LayerNorm(13)
节点投影                     Linear(13, 32)
共享消息传递                 2 steps
每步输入                     当前节点状态、所在nibble的4-bit均值、P-layer前驱状态
共享更新                     Linear(96,32) -> GELU -> Dropout(0.10)
                             -> Linear(32,32) -> residual -> LayerNorm(32)
输出头                       LayerNorm(32) -> shared Linear(32,1)
输出                         batch x 64 logits
总参数                       4,795
```

同预算三行控制为：

| 行 | 唯一区别 | 要回答的问题 |
|---|---|---|
| `independent` | 不使用真实nibble/P关系 | 单节点局部特征是否已经足够 |
| `true` | 使用密码真实P-layer与nibble分组 | 正确SPN拓扑是否提供独立增益 |
| `corrupted` | 使用同family、同参数的确定性错误P-layer | 收益是否来自任意消息混合而非真实拓扑 |

三行参数均为4,795，hidden、message step、dropout、optimizer、epoch和seed保持一致。错误P控制
保持同一密码family身份，避免把错误拓扑和分布漂移混在一起。

## 6. 训练与评测协议

```text
device          = local CPU
epochs          = 30
batch size      = 8 structures
hidden          = 32
message steps   = 2
optimizer       = AdamW
learning rate   = 1e-3
weight decay    = 1e-4
dropout         = 0.10
seeds           = 0, 1
loss            = masked binary cross entropy
checkpoint      = best structure-disjoint validation AUC
primary metric  = validation AUC on observed strict edges
```

拆分和控制遵循：

1. train/validation的structure不重叠；
2. checkerboard在每个被选structure和output bit内部配平正负边；
3. duplicate edge、structure class delta和output class delta均为0；
4. global、output-bit、active-bit一元边际AUC为0.5；
5. 每颗seed分别要求`true-independent >= 0.03`且`true-corrupted >= 0.03`；
6. PRESENT和GIFT分别从各自严格标签重新训练，不迁移checkpoint。

## 7. 正式双密码结果

| 密码 | seed | independent AUC | true-P AUC | wrong-P AUC | true-ind | true-wrong |
|---|---:|---:|---:|---:|---:|---:|
| PRESENT-80 r4 | 0 | 0.657500 | 0.945556 | 0.820000 | +0.288056 | +0.125556 |
| PRESENT-80 r4 | 1 | 0.671944 | 0.947778 | 0.879444 | +0.275833 | +0.068333 |
| GIFT-64 r4 | 0 | 0.571280 | 0.913111 | 0.774714 | +0.341831 | +0.138398 |
| GIFT-64 r4 | 1 | 0.569719 | 0.911030 | 0.784599 | +0.341311 | +0.126431 |

两颗seed均通过冻结`+0.03`归因门。方法级均值：

```text
PRESENT mean true AUC          = 0.946667
PRESENT mean true-independent = +0.281944
PRESENT mean true-wrong        = +0.096944

GIFT mean true AUC             = 0.912071
GIFT mean true-independent     = +0.341571
GIFT mean true-wrong           = +0.132414
```

E80的最终裁决为：

```text
status   = pass
decision = innovation2_cross_spn_r3_profile_method_confirmed_skinny_labels_not_ready
```

科学含义是：相同13维r3前缀、64-node共享算子和真实P-layer归纳偏置，在两个真实64-bit SPN
的独立严格标签上分别训练后，均获得双seed可归因增益。这是方法家族证据，不是PRESENT模型向
GIFT零样本迁移或checkpoint迁移。

## 8. 必须保留的负结果

### 8.1 r5位置先验先导实验

早期PRESENT r5结构条件候选排序在弱线性/随机对照下表现较好，但E6加入训练输出位置先验和
位置匹配线性后，MLP只领先位置先验`+0.014816`、领先位置匹配线性`+0.005951`，没有通过
预注册门。该结果应作为“为什么必须做强边际控制”的方法学反例，而不是主模型成绩。

### 8.2 第三密码与共享模型

```text
SKINNY：论文kernel可以复现，但同语义严格profile标签未就绪；后续稀疏残差没有机制增益。
RECTANGLE：unit-profile候选没有稳定通过真实拓扑归因；nested cube标签虽然sound，
           但真实nesting相对错误关系的margin不足。
共享PRESENT/GIFT模型：GIFT相对独立模型质量回退，不能替代分别训练。
```

这些结果说明当前贡献应写成“一个方法在两个密码上分别确认”，不能升级成通用SPN模型。

### 8.3 高轮严格标签边界

E97重放E52--E55、E61、E64和E69，并审计12个PRESENT五轮未决多bit查询：

```text
providers audited                      = 6
target-semantics matching providers    = 2
fully eligible providers               = 0
strict nontrivial PRESENT positives    = 0
resolved frozen queries                = 0 / 12
```

P0支撑缺失提供器语义正确且可执行，但不识别GF(2)相消并且正类为0；full-superpoly sparse ANF
能够表达正确证书，但在三轮第4个query已经达到`5,000,001`项，越过冻结500万项硬cap。因而：

```text
PRESENT r5非平凡多bit严格标签未就绪；
PRESENT r7--r9输出性质预测网络没有训练资格；
不得用有限密钥投票、提高旧cap或小型SPN证书替代。
```

这不是数学上否定PRESENT存在高轮积分性质，而是当前开放、可执行标签提供器的工程与语义边界。

## 9. 与创新1和主流高轮结果的关系

创新1的PRESENT r7/r8/r9是`real-vs-random`神经区分任务，创新2是
`structure x output-mask -> universal balance property`预测任务。两者可以在论文总实验章并列，
但不能把AUC或最高轮数直接混为同一指标。

当前可报告：

| 路线 | 任务 | 轮数 | 当前证据 |
|---|---|---:|---|
| 创新1 | real-vs-random神经区分 | PRESENT r7 | 1M/class双seed AUC约0.797，独立高轮辅助证据 |
| 创新1 | real-vs-random神经区分 | PRESENT r8 | 1M/class单seed baseline AUC 0.554963；本项目pair候选未胜 |
| 创新1 | real-vs-random神经区分 | PRESENT r9 | 262144/class诊断接近随机，不能称正式失败或突破 |
| 创新2 PG-NBPO | 严格输出平衡谱预测 | PRESENT/GIFT r4 | 双密码、双seed、正确拓扑归因通过 |
| 创新2 provider | 高轮严格标签生成 | PRESENT r5及以上 | E97未通过，r7--r9未训练 |

论文的高轮主流比较必须另外列出任务定义、训练/验证/评估总量、每类样本、密钥采样、negative
定义和epoch。创新2不得用四轮AUC冒充八轮区分准确率，也不得用E97的停止门声称高轮不存在。

## 10. 论文可以主张什么

可以主张：

1. 定义并实现了结构条件的64维积分输出平衡谱预测任务，区别于具体输出恢复和
   structured-vs-random二分类；
2. 构造了全称正证书、具体负见证和unknown保留的三态严格标签流程；
3. 提出了4,795参数的PG-NBPO，将目标轮前ANF支撑/次数特征与真实SPN拓扑结合；
4. 在PRESENT-80和GIFT-64上分别取得双seed、同参数错误拓扑控制下的稳定归因增益；
5. 通过位置先验、第三密码、共享参数、错误关系和provider硬cap给出完整适用边界。

不能主张：

- 达到PRESENT 7--9轮积分输出预测；
- 发现新的确定性积分区分器或密钥恢复攻击；
- PRESENT checkpoint可以零样本迁移到GIFT；
- 一个共享模型已经覆盖不同SPN；
- 绝对AUC高于某文献就代表同协议SOTA；
- finite-key零失败等价于全密钥证明；
- Transformer、GraphGPS或更大网络能够绕过标签缺口。

## 11. 建议论文章节结构

```text
第4章 面向SPN密码的结构条件积分输出平衡谱预测

4.1 问题定义
    4.1.1 具体输出预测、神经区分与积分输出性质预测的区别
    4.1.2 structure x output-mask查询与64维平衡谱
    4.1.3 positive / negative / unknown严格标签

4.2 数据集构造与无捷径拆分
    4.2.1 8-bit活动cube与单位输出mask
    4.2.2 ANF支撑正证书与scheduled-key负见证
    4.2.3 structure-disjoint checkerboard及边际AUC控制

4.3 前缀引导逐节点平衡谱算子
    4.3.1 13维r3 ANF前缀特征
    4.3.2 nibble上下文与P-layer消息传递
    4.3.3 independent / true / wrong-P同参数归因矩阵

4.4 PRESENT-80实验
    4.4.1 标签宽度和476条观测边
    4.4.2 seed0与seed1结果
    4.4.3 正确拓扑归因

4.5 GIFT-64独立验证
    4.5.1 192结构标签扩展与620条观测边
    4.5.2 双seed独立重训
    4.5.3 跨密码方法一致性与不可直接比较边界

4.6 消融、失败路线与高轮边界
    4.6.1 r5输出位置先验
    4.6.2 SKINNY/RECTANGLE/共享模型负结果
    4.6.3 E97严格标签提供器复杂度边界

4.7 本章小结
```

## 12. 建议表格和图

论文正文建议保留：

1. `表4-1`：任务输入、输出与三态标签定义；
2. `图4-1`：8-bit cube到64维输出平衡谱的数据流；
3. `图4-2`：PG-NBPO的13维节点输入、nibble聚合、P-layer消息和64输出结构；
4. `表4-2`：PRESENT/GIFT数据规模、拆分与边际控制；
5. `表4-3`：四行双密码双seed AUC及true-control margin；
6. `图4-3`：E80双密码方法综合图；
7. `表4-4`：SKINNY、RECTANGLE、共享模型和E97边界；
8. `图4-4`：E97 provider资格漏斗。

可直接复用的已有图：

```text
outputs/local_audits/i2_cross_spn_r3_profile_operator_method_synthesis_20260719/curves.svg
outputs/local_audits/i2_post_e95_architecture_portfolio_boundary_20260719/curves.svg
outputs/local_audits/i2_present_r5_cancellation_provider_feasibility_20260719/curves.svg
```

这些图均已通过`visual-qa-redraw`。论文排版时应从SVG导出统一字号的PDF/PNG，并保留中文图注；
若重新生成，仍需重新执行像素级视觉检查。

## 13. 可直接写入论文的本章结论

> 本章针对SPN密码的积分输出性质，提出了前缀引导逐节点平衡谱算子PG-NBPO。与预测具体
> 密文比特或判断结构样本真伪不同，该方法输入一个活动cube结构，利用目标轮之前的ANF支撑
> 与次数分布构造64个输出节点特征，并沿S-box分组和真实P-layer进行共享消息传递，一次预测
> 完整单位输出平衡谱。实验采用全称正证书、具体负见证、unknown保留和结构互斥checkerboard，
> 从而避免把有限密钥经验现象或输出位置边际当作确定性积分规律。在PRESENT-80与GIFT-64
> 四轮数据上，真实拓扑模型在两颗随机种子中均稳定优于独立节点和同参数错误拓扑，平均
> true-wrong AUC margin分别为0.096944和0.132414，说明正确SPN置换关系对输出平衡谱预测具有
> 可归因作用。进一步实验表明，该结论不能直接推广到SKINNY、RECTANGLE、共享跨密码模型或
> PRESENT高轮任务；五轮非平凡多bit严格标签仍受现有provider语义与复杂度限制。因此，本章
> 的贡献是一个经双密码验证、带严格控制的结构条件输出性质预测方法，而不是新的高轮积分攻击。

## 14. 答辩时的一句话

```text
我的创新不是让网络猜某个密文bit，而是让它读取输入cube在前三轮形成的代数支撑谱，
沿密码真实P-layer一次预测64个输出bit是否保持严格XOR平衡；这个拓扑增益在PRESENT和
GIFT两种真实SPN、两颗seed和错误拓扑控制下都成立，但我没有把四轮方法证据包装成
七到九轮攻击，高轮标签提供器的失败边界也被完整报告。
```

## 15. 后续执行计划

创新2当前不再分配训练或远程GPU预算。下一步直接进入论文写作：

```text
1. 以本文件第11节生成第4章初稿；
2. 从E65/E75整理表4-2，从E80整理表4-3；
3. 复用E80和E97图，统一论文图注与编号；
4. 将创新1 r7/r8/r9放入独立实验章，明确任务不可直接比较；
5. 最终检查摘要、创新点、实验结论和答辩PPT使用完全一致的声明范围。
```

只有出现新的sound、可执行且能在冻结cap内生成真实PRESENT非平凡严格正证书的provider，
才重新打开高轮输出预测研究；新的网络名称、更多seed、更多epoch或机械扩大现有标签池都不能
解除E97停止门。

## 16. 权威证据入口

```text
方法综合：
docs/experiments/innovation2-cross-spn-r3-profile-operator-method-synthesis-plan.md
outputs/local_audits/i2_cross_spn_r3_profile_operator_method_synthesis_20260719/

架构边界：
docs/experiments/innovation2-post-e95-architecture-portfolio-boundary-plan.md
outputs/local_audits/i2_post_e95_architecture_portfolio_boundary_20260719/

高轮标签边界：
docs/experiments/innovation2-present-r5-cancellation-provider-feasibility-plan.md
outputs/local_audits/i2_present_r5_cancellation_provider_feasibility_20260719/

历史r5先导：
docs/research/innovation2-thesis-ready-conclusion-20260716.md
```
