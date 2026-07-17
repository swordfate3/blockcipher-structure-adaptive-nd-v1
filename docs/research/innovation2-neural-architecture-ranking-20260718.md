# 创新2神经网络结构排名与开放条件

日期：2026-07-18

状态：架构调研完成 / E32b合成SPN标签门通过 / E33小规模比较可启动 / 真实密码训练未开放

## 1. 结论先行

创新2不是把密文排成图像后做普通二分类。目标输入和输出是：

```text
input  = cipher topology + rounds + input affine/subspace structure + output mask
target = balance/kernel property, fresh-key stability, or candidate ranking
```

当前最匹配的神经候选不是单独的CNN、LSTM或通用Transformer，而是一个小型
双编码器：

```text
canonical subspace basis --AllSet/Set encoder--+
                                                +-- mask-query decoder --> property score
cipher S-box/P-layer graph --small GraphGPS-----+
```

暂定名为 `Subspace-Cipher Graph Transformer`（SCGT）。这个名称只描述待验证的
项目候选，不能在实验过门前宣称为已完成创新或有效模型。

E30已经证明：网络架构不是当前第一瓶颈。四个坐标16维子空间有joint kernel，
32个随机orientation却是 `0/32`。没有宽标签族时，更强模型只会记忆位置、mask或
确定性候选生成器的输出。因此本文件冻结未来网络矩阵，但不开放训练。

## 2. 输入结构要求

一条样本至少包含：

| 输入对象 | 建议表示 | 原因 |
|---|---|---|
| 输入子空间 | RREF规范basis；每个basis row是一个set token或hyperedge | 不同basis可表示同一子空间，必须先规范化 |
| 64个状态bit | bit node | 承载位置、cell、round和活动关联 |
| basis-bit incidence | 16个basis hyperedge到64个bit node的关联 | 一个basis向量同时连接多个bit，不是普通相邻边 |
| S-box结构 | 同一4-bit cell内的typed relation | 保留SPN局部非线性单元 |
| P-layer结构 | source-bit到target-bit的有向typed edge | 保留真实扩散拓扑 |
| 轮数和密码族 | graph/global token | 允许跨轮或跨密码条件化 |
| 输出mask | 64个mask-bit query或一个mask token | 任务是查询指定输出性质，不是一次输出固定类别 |

必须满足的表示不变量：

1. basis row顺序变化不得改变预测；
2. 等价basis必须先通过RREF得到同一规范输入；
3. 同时重标号bit node、cipher edge和mask时，模型行为应保持等变；
4. 只打乱真实P-layer edge时，性能应下降，否则不能归因于密码拓扑；
5. 输出mask必须作为query参与交互，不能仅在最后拼接一个mask ID。

AllSet只能直接保证多重集合排列不变，不能自动保证任意
`GL(d,2)` basis变换不变；因此RREF规范化仍是数据契约的一部分，不能把该责任交给
网络自己学习。

## 3. 候选架构排名

### 第一名：AllSet/Set encoder + small GraphGPS + mask query

匹配度：最高；密码学直接证据：尚无；实现风险：中等。

AllSet把hypergraph层写成两个可学习的multiset函数，正好对应
`basis rows -> bit nodes -> basis rows` 的高阶关联。GraphGPS同时使用真实边上的局部
message passing和全局attention，适合短程S-box/P-layer关系与跨轮远程依赖并存的
64-bit状态图。mask query只读取与候选输出mask相关的图表示。

项目候选应保持小型：2--4层、固定64个bit node、16个以内basis token，不引入大型
图预训练框架。真正待验证的创新点是“子空间incidence + cipher graph + mask query”
的组合及其组外泛化，不是重新命名GraphGPS或AllSet。

### 第二名：small GraphGPS

匹配度：高；密码学直接证据：尚无；实现风险：低于组合模型。

将每个basis row对bit node的关联压成节点特征，再运行PRESENT真实图上的GraphGPS。
它是SCGT的必要消融：若不显式建hyperedge也达到同等结果，复杂的AllSet分支没有保留
价值。由于图只有64个节点，GraphGPS的超大图内存问题不构成本项目主要风险。

### 第三名：AllSetTransformer

匹配度：高；密码学直接证据：尚无；实现风险：中等。

它最忠实地编码basis-bit incidence和basis row无序性，但单独使用时缺少S-box/P-layer
有向拓扑。它适合作为“结构集合编码”消融，不应作为唯一候选。

### 第四名：TokenGT

匹配度：中高；密码学直接证据：尚无；实现风险：中高。

TokenGT把node和edge统一成token，理论表达力强，可以统一编码basis incidence、
cipher edge和output mask。但本任务图很小且edge type明确；完整TokenGT比小型
GraphGPS更重，也更容易在有限标签上过拟合。只有SCGT无法表示必要的edge-edge交互时
才升级到该路线。

### 密码领域直接基线：DenseNet + MBConv

匹配度：中；密码学直接证据：有。

Wu/Guo 2024在PRESENT积分神经区分器中用MBConv改造DenseNet。它处理的是积分multiset
与随机multiset的密文型输入，不天然满足“结构描述 + 输出mask query”任务，但它是
当前最接近的PRESENT积分神经架构基线。未来比较时必须明确：复用网络模块不等于复现
论文任务，也不能把其round-reach准确率直接与结构性质预测AUC混为一谈。

### 探索候选：Neural Algorithmic Reasoner / Looped Hypergraph Transformer

匹配度：概念上高；密码学直接证据：无；实现风险：高。

TransNAR证明GNN式algorithmic reasoner可以通过cross-attention向Transformer提供
结构计算表示；Looped Hypergraph Transformer用共享权重循环模拟hypergraph算法。
它们启发“每层对应一轮传播”或“共享round processor”，但原始任务是CLRS文本/图算法
和理论hypergraph算法，不是密码分析。只有SCGT先证明标签可学、且跨未见轮数明显失败
时，才值得测试共享权重的round-loop版本。

## 4. 当前不推荐路线

| 架构 | 暂不优先的原因 |
|---|---|
| Mamba/普通SSM | 输入没有天然长序列；人为bit顺序会弱化密码图归纳偏置 |
| BiLSTM | 把64个bit强行线性化，容易把位置顺序当成性质 |
| 更深MLP | 项目已有位置/mask捷径证据；增加容量不会修复标签语义 |
| KAN | 没有针对GF(2)子空间、basis等价或cipher拓扑的直接优势证据 |
| 大型纯Transformer | 标签量不足，参数预算和归因控制都不利 |
| DRSN | 自适应阈值适合连续噪声型差分密文信号，不匹配离散结构描述主任务 |

这些判断是当前任务匹配排名，不是对模型家族的一般性否定。

## 5. 标签门通过后的最小网络矩阵

每次固定相同训练行、参数预算、optimizer、epoch和seed，只比较三行：

| 行 | 作用 |
|---|---|
| deterministic marginal/direct-kernel baseline | 判断是否只是在重放现有算法或边际 |
| small GraphGPS | 测试真实cipher topology是否足够 |
| SCGT：AllSet + GraphGPS + mask query | 测试高阶basis incidence是否提供额外泛化 |

必要控制不额外膨胀成六个主模型；以消融或重跑形式执行：

```text
label shuffle
basis-row permutation
equivalent-basis -> same RREF
topology shuffle
position/mask marginal baseline
structure-disjoint split
mask-disjoint split
dual-disjoint split
fresh-key labels
```

推进门至少要求：

1. GraphGPS和SCGT都显著优于标签打乱；
2. SCGT在同参数预算下优于GraphGPS，才保留hypergraph分支；
3. true topology显著优于shuffled topology，才声称密码结构贡献；
4. 至少一个structure/mask dual-disjoint结果优于确定性边际基线；
5. fresh-key标签上的排序或membership结果保持；
6. 失败时先检查标签宽度，不机械增加层数、样本或GPU预算。

## 6. 确定性候选提供者审计对架构的影响

两条公开工具路线已做源代码级初审：

### CLAASP-MP

公开仓库commit `f2239d639ae5c4a013947ce9121c6f4464584758` 同时包含
`present_block_cipher.py` 与 `Gurobi/monomial_prediction.py`。后者明确要求Gurobi
license，仓库相关测试全部标记为skip；当前项目环境有Sage但没有`gurobipy`。
所以它是强理论/工具基线，却还不是当前可执行标签流水线。

### Algebraic Transition Matrices

公开仓库commit `b2ffbb2bf0ef8f2ffabe3203896006874aa1c40b` 使用
`ortools + python-sat + galois`，含PRESENT 9轮notebook和8份预计算basis。静态解析得到：

```text
result files                         = 8
unique serialized basis elements    = 470
GF(2) union rank                     = 468
coordinates appearing in supports   = 673
singleton standard-basis elements   = 305
```

305个singleton按输入指数重量分为 `60:1, 61:40, 62:66, 63:198`，按输出单项式
次数分为 `1:198, 2:66, 3:40, 4:1`。它们描述的是key-independent constant
generalized integral。论文的Remark 3明确指出该搜索本身不告诉常数是0还是1；多项
basis还表示多个 `(input exponent, output monomial)` 的线性组合。

因此这些结果不能直接冒充当前二分类标签：

```text
XOR over affine input set of a specified linear output mask == 0
```

它们可以成为候选提供者、确定性强基线或未来的“广义积分关系预测”扩展任务，但改变
target前必须单独裁决，不能与当前线性mask平衡任务混写。

## 7. 核验过的主要来源

- Lee et al., *Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks*, ICML 2019: https://proceedings.mlr.press/v97/lee19d.html
- Chien et al., *You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks*, ICLR 2022, arXiv:2106.13264: https://arxiv.org/abs/2106.13264
- Rampasek et al., *Recipe for a General, Powerful, Scalable Graph Transformer*, NeurIPS 2022: https://proceedings.neurips.cc/paper_files/paper/2022/hash/5d4834a159f1547b267a05a4e2b7cf5e-Abstract-Conference.html
- Kim et al., *Pure Transformers are Powerful Graph Learners*, NeurIPS 2022, arXiv:2207.02505: https://arxiv.org/abs/2207.02505
- Bounsi et al., *Transformers meet Neural Algorithmic Reasoners*, arXiv:2406.09308: https://arxiv.org/abs/2406.09308
- Huang et al., *Neural Algorithmic Reasoning for Hypergraphs with Looped Transformers*, arXiv:2501.10688: https://arxiv.org/abs/2501.10688
- Wu and Guo, *Improved integral neural distinguisher model for lightweight cipher PRESENT*, Cybersecurity 7, 65 (2024), DOI `10.1186/s42400-024-00258-0`: https://link.springer.com/article/10.1186/s42400-024-00258-0
- Beyne and Verbauwhede, *Integral Cryptanalysis Using Algebraic Transition Matrices*, ToSC 2023(3), project full text: `papers/innovation_two/text/2023_beyne_verbauwhede_algebraic_transition_matrices.txt`
- Bellini et al., *CLAASP-MP: An Automated MILP Framework for Monomial Prediction*, ePrint 2026/735, project full text: `papers/innovation_two/text/2026_bellini_claasp_mp.txt`

原始网络检索结果保存在：

```text
sources/research_innovation2_neural_architectures_20260718.json
sources/research_innovation2_set_graph_transformers_20260718.json
sources/research_innovation2_neural_cryptanalysis_architectures_20260718.json
```

## 8. 推荐下一步

E31已确认provider语义不完整；E32原始合成标签又被ID边际解释。E32b使用train-only
matched contrast后通过宽度与组外捷径门，因此下一步执行E33三行同预算小规模比较：
deterministic marginal、small GraphGPS和SCGT。训练只使用9个train topology，三个
heldout split不参与选择或优化；同时运行label-shuffle与P-layer-shuffle归因控制。
只有真实topology模型超过 `dual-unseen AUC=0.726528`边际并优于shuffled topology，
才进入真实密码迁移审计。该开放不等于真实PRESENT高轮训练已获批准。
