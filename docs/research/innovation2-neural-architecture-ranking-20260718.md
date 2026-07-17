# 创新2神经网络结构排名与开放条件

日期：2026-07-18

状态：E37扩展拓扑benchmark通过 / E38最小GraphGPS-CETT归因矩阵待跑 / 真实密码训练未开放

## 1. 结论先行

创新2不是把密文排成图像后做普通二分类。目标输入和输出是：

```text
input  = cipher topology + rounds + input affine/subspace structure + output mask
target = balance/kernel property, fresh-key stability, or candidate ranking
```

在实验前，最匹配的神经候选不是单独的CNN、LSTM或通用Transformer，而是一个小型
双编码器：

```text
canonical subspace basis --AllSet/Set encoder--+
                                                +-- mask-query decoder --> property score
cipher S-box/P-layer graph --small GraphGPS-----+
```

暂定名为 `Subspace-Cipher Graph Transformer`（SCGT）。这个名称只描述待验证的
项目候选，不能在实验过门前宣称为已完成创新或有效模型。

E33现已进一步证明：网络名称和容量不是有效性的证据。四个坐标16维子空间有joint kernel，
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

## 8. E33结果与架构重排

E33在E32b的9424行matched-contrast数据上完成两seed同预算比较。dual-unseen AUC为：

```text
ID marginal baseline = 0.726528
GraphGPS true         = 0.682672
SCGT true             = 0.726947
GraphGPS wrong P      = 0.752444
label shuffle         = 0.465781
```

因此原排名不能直接解释为“SCGT第一名且可以继续加规模”。GraphGPS真实拓扑低于边际，
错误P-layer反而更高；SCGT仅与边际持平。源码审计发现当前模型同时加入绝对bit、nibble、
lane embedding，这破坏了并行S-box cell同时重标号时应有的对称性，并可能让网络优先记住
坐标编号。

当前探索顺序调整为：

1. `cell-equivariant GraphGPS`：去掉绝对bit/nibble身份，只保留cell内lane role；先修复
   已识别的表示缺陷，不增加容量。
2. `equivariant SCGT`：仅当第一步证明真实P-layer贡献后，再测试basis hyperedge增益。
3. `round-shared / looped graph processor`：仅当等变GraphGPS在未见拓扑通过、但跨轮失败时开放。
4. `TokenGT edge-token`：仅当消融证明当前node-message无法表达必要edge-edge交互时开放。

Mamba、KAN、更深MLP和大型纯Transformer仍不优先；它们不会自动修复cell重标号对称性，
也缺少当前失败模式对应的归纳偏置。

## 9. 推荐下一步

E33已经失败，且错误P-layer优于真实P-layer。下一步不是直接尝试更多新模型，而是执行
E33-R单变量cell-equivariance审计：冻结E33数据、split、hidden64、3 blocks、40 epochs、
seed0/1和优化器，只替换绝对位置表示。候选同时与E33绝对位置锚点、ID边际、错误P-layer
和label-shuffle比较。只有真实拓扑在两seed上超过dual边际、均值超过边际与错误拓扑各
`0.03`，才开放equivariant SCGT；否则停止当前GraphGPS家族，不进入真实密码迁移。

E33-R现已完成。cell重标号最大logit误差为`8.94e-08`，证明实现满足冻结等变契约；
真实P-layer dual均值由`0.682672`改善到`0.711548`，但仍低于ID边际`0.726528`，且只比
错误P-layer`0.708831`高`0.002717`。所以绝对位置确实是部分缺陷，却不是主要瓶颈；
equivariant SCGT不开放。

后验逐轮诊断中，3轮dual为`0.500/0.414`，4轮为`0.773/0.852`；2轮无matched cell，
5轮只有负类，不能解释。这使下一候选从TokenGT转为小型round-shared neural algorithmic
reasoner：保留cell-equivariant node表示和同一GraphGPS operator，只把三个静态独立block
替换成一个共享block，并按样本真实轮数执行2--5次。该候选直接对应当前“轮数只是embedding、
传播深度固定”的源码错配，仍需同预算true/wrong-P和label-shuffle裁决；失败即停止该家族。

E34现已完成。round-shared真实P-layer dual均值为`0.683643`，低于ID边际`0.726528`、
E33-R静态锚点`0.711548`和错误P-layer`0.708540`；两seed分别为`0.647404/0.719881`。
共享处理器只有`146881`参数，约为静态锚点的一半，cell等变和标签打乱控制均正常，因而
不能用实现错误解释失败。GraphGPS/looped家族停止，SCGT不再开放。

下一候选改为小型`Cipher Edge-Token Transformer`。它吸收TokenGT把node与edge统一为
token的思想，但不采用大型通用框架：16个bit node、16个有向P-layer edge、4个S-box
relation和1个mask query组成固定小token集。候选的关键增量是让P-layer边成为显式对象，
支持edge-edge和query-edge交互；这与GraphGPS的neighbor gather是不同关系算子。仍必须
通过cell重标号不变性、错误P-layer和label-shuffle门，失败后关闭该合成神经拓扑路线。

E35首次运行后发现控制协议错误：跨variant `np.roll`把所有heldout P3替换成train-seen
P2，使rolled-P dual虚高到`0.822889`。该run已重分类为protocol-invalid，不能用
`true-wrong`差值做拓扑结论。

E35b改用每个variant自身P-layer的固定destination-cell rotation，明确保持heldout
P-family身份。公平重跑中，CETT true/fair-control/label-shuffle的dual均值分别为
`0.671767/0.664944/0.484934`。true只领先公平控制`0.006823`，且低于ID边际
`0.054761`；两seed均未过基线。CETT有效失败，当前合成benchmark上的GraphGPS、SCGT、
looped reasoner和edge-token架构搜索关闭。

下一步不再排名新网络名称，而是E36拓扑标签可识别性审计：量化固定S-box下P0--P3的
标签翻转、P-layer interaction宽度、train-only matched cell中的P敏感比例，以及这些
信号在unseen-P/dual拆分是否仍有正负支持。只有标签本身先过因果敏感性与公平控制门，
才允许设计新的benchmark或恢复神经模型比较。

E36现已通过全部冻结门。589个matched cell中，train P-sensitive any-S为`585`，
train S×P interaction为`521`，full interaction为`579`；在S3下有`391`个dual P-effect
cell，目标正负为`234/157`。因此标签确实包含宽P-layer条件信号，不能把E33--E35b失败
解释为标签完全不依赖拓扑。

新的主瓶颈是独立拓扑样本量。5301条训练行只来自`3 S-box x 3 P-layer=9`张图，独立
P-layer仅3个；round/structure/mask行不会增加图分布样本量。对随机生成的P permutation，
从3个训练图外推第4个图严重欠定。下一步E37应先生成`4 S-box x 16 P-layer`的精确标签，
形成12个train P、4个heldout P，再重新做train-only matched、interaction、边际和公平控制
门。扩展标签门通过前，不再比较网络结构。

E37现已完成并通过。扩展后的64个拓扑中，36个train topology包含12个独立P-layer；
冻结`9 <= train positive <= 27`规则选出320个base cell。train P-sensitive any-S、
train interaction、dual P-effect和full interaction比例分别为
`1.000000/0.981250/0.984375/1.000000`。unseen-S、unseen-P和dual的最强train-only ID
边际AUC分别为`0.688198/0.648753/0.684393`，全部低于预注册上限；四个split均具有
足够正负类，公平destination-cell corruption不把heldout topology换成train-seen topology。

因此最小神经矩阵重新开放，但E37本身不是神经收益。E38冻结hidden64、3层、40 epochs、
seed0/1，只比较cell-equivariant GraphGPS与CETT，并给两者各自的fair-corrupted P控制，
另保留一个label-shuffle流程控制。主问题是显式edge token在更多独立训练拓扑下是否比
neighbor gather获得可归因的组外增量；不同时重开SCGT、round-shared、Mamba、KAN或
大型Transformer。E38不过门则停止这两种结构，转向有向边对关系或结构化P-layer数据族。

E38 Phase A现已完成且未过门。GraphGPS两seed dual为`0.645458/0.637906`，CETT为
`0.579995/0.679318`，均未逐seed超过E37 ID边际`0.684393`；均值分别为
`0.641682/0.629657`。两模型未见S-box均值约`0.90`，但双重未见拓扑明显下降；
label-shuffle dual为`0.506077`，流程正常。因为没有候选先过ID门，Phase B公平错误拓扑
控制按预注册停止，不用增加两行训练。

当前失败模式把下一候选限定为关系路径组合，而不是更大Transformer。E39优先审计一个
cell-equivariant有向bit-pair网络：为`16 x 16`关系状态编码P-edge、同S-box cell、active与
output-query角色，使用共享triangle update聚合`R[i,k]`与`R[k,j]`，按实际轮数执行2--5步。
这类2-WL/PPGN/Neural Bellman-Ford式算子能显式表达多跳边组合，正对应GraphGPS单节点消息
和CETT无关系位置attention的共同缺口。先做等变性、真实/错误P forward差异和小smoke；
不过门则转向结构化P-layer族，不继续枚举通用网络名称。
