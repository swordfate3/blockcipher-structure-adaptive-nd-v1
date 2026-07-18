# 创新2神经网络结构排名与开放条件

日期：2026-07-18

状态：E68性能第一 / E73简洁性第一 / E74 GIFT严格标签matching容量未就绪

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

E39现已完成并成为当前最强候选。SPN-PRR两seed的unseen-S、unseen-P和dual均值分别为
`0.900613/0.735172/0.716651`；dual超过同split ID边际`0.032258`，超过GraphGPS与CETT均值
`0.074970/0.086995`，label-shuffle dual为`0.508262`。公平错误P-layer控制的dual均值为
`0.664140`，真实拓扑逐seed领先`0.037792/0.067231`，平均领先`0.052511`。因此当前证据
支持“有向pair表示加路径处理器能利用真实SPN拓扑”，但尚未隔离收益来自pair表示还是
`i->k->j`路径组合。

E40只做参数完全匹配的no-triangle消融：保留左右低秩投影、更新MLP、`16 x 16`状态、
readout和2--5步共享迭代，把跨pair矩阵乘法替换为每个pair独立的局部门控消息。若E39在
两seed上均领先且mean dual至少领先`0.03`，才把新增收益归因给路径组合；否则保留pair-state
结果但撤回triangle特异性解释。E40完成前不迁移真实密码或开放远程训练。

E40现已完成。no-triangle两seed的unseen-S、unseen-P和dual均值为
`0.898350/0.700224/0.702438`，label-shuffle dual为`0.513226`；模型参数`111825`与E39
triangle精确一致，局部block的off-pair influence为`0.0`。E39 triangle相对no-triangle
的dual均值只领先`0.014213`，逐seed差为`-0.013865/+0.042292`，没有通过冻结的逐seed和
`0.03`均值门。

因此排名结论需要收窄：当前最强方法级候选是`structure-conditioned directed pair state`，
不是已证明必要的triangle/2-FWL更新。triangle仍可作为SPN-PRR实现的一部分，但论文论证
不能把E39收益主要归因给`i->k->j`组合。下一步先排序query-conditioned NBFNet与结构化
P-layer数据路线前，先补一个更近的归因缺口：E40没有给no-triangle自身运行公平错误P控制。
E41只训练pair-local fair-corrupted seed0/1；若true稳定领先，说明pair-state本身已足够使用
拓扑；否则local只作容量控制，E39 triangle仍是唯一通过拓扑归因的候选。这个裁决完成前
不开发NBFNet，不迁移真实密码，远程GPU继续关闭。

E41公平控制现已通过。pair-local真实拓扑与公平错误拓扑的dual均值为
`0.702438/0.583771`，平均差`+0.118668`；逐seed差为`+0.056716/+0.180619`。这确认
E40的局部pair-state不是拓扑无关容量捷径。结合E39 triangle也通过公平控制而E40未隔离
triangle增量，当前排名更新为：

```text
1. SPN专用结构条件化有向pair-state表示（已确认真实拓扑贡献）
2. triangle共享路径处理器（可保留为候选，但必要性未确认）
3. pair-local处理器（已确认拓扑贡献，作为更简单同表示锚点）
4. query-conditioned NBFNet（暂缓；当前没有证据表明路径算子仍是首要瓶颈）
```

下一步不再继续合成网络枚举。E42先审计现有PRESENT输出性质标签和64-bit pair-state迁移条件，
冻结同数据ID、triangle、local与fair-corrupted控制的最小矩阵；只有标签宽度、finite-key
噪声校正、内存与split门通过，才开放真实密码本地readiness训练。远程GPU继续关闭。

E42已从机器产物重新核验真实迁移条件。64-bit pair-local与triangle在hidden16/32/64、
batch1/2/4/8的24个前向/反向配置全部通过；4096 pair、7/8步、cell等变性、错误P-layer
敏感性和local off-pair契约均成立。因此模型扩展不是当前瓶颈。

标签侧0/4族就绪。PRESENT r7 context虽然过`6/8`项，但fresh-key签名复现只有`0.4375`，
mask-disjoint简单基线AUC为`0.950623`；SKINNY r7/r8族最多只有4个非平凡结构，均未达到
`8/32`宽度门。当前架构排名不再开放新候选：pair-state仍是最强方法，triangle/local作为
同表示处理器；NBFNet、FloydNet和更深Transformer均暂缓。下一研究优先级转为满足完整
0/1 balance语义与negative契约的真实SPN标签atlas提供者，标签门通过后才恢复网络比较。

E43已解除这个标签阻塞，但只在真实PRESENT-80四轮锚点上。新的全称标签定义为：给定
8-bit coordinate cube与线性output mask，性质是否对所有80-bit key和所有inactive offset
保持XOR=0。正类使用sound活动变量ANF支撑超集中的完整cube单项式缺失证书，负类使用
具体`(key, offset)`上masked XOR=1的反例，无法裁决的候选保留为unknown。

96个structure与300个mask产生`3572/18590/6638`个positive/negative/unknown；72个
structure同时含正负类，三态签名65种，32个抽样negative均通过标量PRESENT复验。原始
mask-only AUC高达`0.967092`，不能训练；2x2 checkerboard匹配后得到800条train和236条
validation，覆盖71个互斥structure，global/mask/family/active-bit一元AUC均为`0.5`。

因此真实密码网络比较重新开放，但范围严格限定为E44本地seed0 readiness。排名保持
`pair-local`为简单主锚点、`triangle`为未证明必要的候选增量；首个四行矩阵只允许
unary baseline、pair-local true、triangle true和最强候选fair-corrupted P-layer。
NBFNet、FloydNet、更深Transformer和远程GPU仍不开放。只有最强真拓扑AUC至少`0.60`、
超过一元基线`0.05`且领先公平错误拓扑`0.03`，才运行seed1或讨论更强关系推理器。

E44已在E43严格matched benchmark上完成本地seed0。pair-local、triangle true和triangle
fair-corrupted的validation AUC分别为`0.549914/0.561979/0.549698`；triangle相对一元
基线为`+0.061979`，但相对同processor错误P-layer只有`+0.012281`。它没有通过冻结的
`0.60`候选门或`+0.03`拓扑归因门，因此seed1与远程规模关闭。

这与E39--E41合成拓扑结果不矛盾：pair-state在拓扑族标签上确实能使用P-layer，但E43
真实PRESENT证书/反例标签还包含由ANF支撑饱和、mask组合和active-set几何产生的复杂决策
边界。当前不能证明P-layer是E44弱信号的主要来源。架构排名暂时冻结，不把NBFNet排到
triangle之前，也不增加Transformer容量。

下一优先级是E45确定性归因审计：比较真实P-layer可达特征、ANF support/certificate
complexity、active/mask set统计与P-layer置换不变控制。若证书复杂度基线解释E44，下一
网络应编码单项式支撑状态或集合序列；若只有真实拓扑特征有效，才开放query-conditioned
Bellman-Ford关系推理器。这个审计比盲目枚举新网络更直接对应当前失败模式。

E45确定性归因已经完成。static-set、错误拓扑、正确拓扑、ANF 1--3轮前缀和最终证书
oracle的validation AUC分别为`0.504309/0.459063/0.648161/0.686082/1.000000`。正确
拓扑相对错误拓扑为`+0.189098`，说明标签确有P-layer路径信息；ANF前缀又比正确拓扑高
`+0.037920`且比静态set高`+0.181772`。按冻结门，下一网络明确选择MSPN，而不是NBFNet。

E46实现了`Monomial Support Propagation Network`：64个bit维护32维压缩支撑状态，以
PRESENT S-box真实ANF项组合、共享step执行4次、经true/fair-corrupted P-layer搬运，并由
output-mask query池化。模型参数`17788`，是E44的`1.66x`；cell重标号误差`5.22e-08`，
true/corrupted初始logit差`0.054705`，没有预计算prefix/oracle buffer。

两轮true/corrupted/label-shuffle AUC为`0.506500/0.507325/0.504668`，只作为readiness；
全部有限性、等变性、参数、source和shuffle门通过。当前架构排序更新为：

```text
1. MSPN（E45证据直接支持；E46 readiness通过，待E47正式seed0）
2. directed pair-state triangle（E44真实标签AUC 0.561979，拓扑归因不足）
3. directed pair-state local（更简单锚点，E44 AUC 0.549914）
4. query-conditioned NBFNet（正确拓扑特征有效，但弱于ANF前缀，继续暂缓）
5. static Set Transformer/DeepSets（static AUC约随机，不优先）
```

下一步只开放E47 MSPN 30轮seed0正式归因。它必须同时比较E45 prefix ridge、E44 triangle、
MSPN fair-corrupted和label-shuffle；不过门则停止当前MSPN，不通过加容量或远程GPU机械补救。

E47正式30轮结果否定了当前MSPN实现。true/corrupted/shuffle validation AUC为
`0.518673/0.560830/0.527291`；true低于E44 triangle `0.043307`、低于E45 prefix ridge
`0.167409`，且低于错误P-layer `0.042157`。true train AUC达到`0.794375`，说明失败不是
完全无法拟合，而是structure-disjoint泛化和正确transport归因失败。

因此“ANF前缀特征有效”不能直接推出“degree-compressed可微传播有效”。当前MSPN只保留
匿名degree/support强度，term mean/product会把不同活动变量组合映射到同类状态；E45精确
support集合包含的变量身份可能才是关键。MSPN排名撤回到未就绪，不运行seed1。

下一架构前必须执行support-state collision审计。只有固定维变量身份sketch在不读取最终
full-cube oracle的前提下显著降低跨标签碰撞并保持组外AUC，才开放`Identity-Sketch
Monomial Propagator`。否则神经证书近似路线停止，E45确定性ANF-prefix归因作为当前最强
解释，E44 triangle保留为最强真实神经锚点。

E48已经完成，变量身份碰撞假设没有获得支持。degree-only、exact identity、sketch64、
local-ID-permuted sketch64和fair-corrupted-P sketch64的validation AUC分别为
`0.689170/0.599109/0.670712/0.407785/0.599325`。identity sketch对变量对应和P-layer敏感，
但相对degree-only低`0.018457`；精确身份也低`0.090060`。degree-only跨标签冲突行率仅
`2.6062%`，精确身份降到`0.1931%`并没有转化为更高组外AUC。

因此关闭`Identity-Sketch Monomial Propagator`与`Monomial Token Set Transformer`，不再把
E47失败归因于身份碰撞。当前证据排序更新为：

```text
1. E45/E48确定性ANF-prefix degree spectrum（解释锚点，AUC约0.69，不是神经结果）
2. directed pair-state triangle（当前最强真实PRESENT神经锚点，AUC 0.561979）
3. directed pair-state local（更简单神经锚点，AUC 0.549914）
4. MSPN（组外过拟合且错误P-layer更高，当前实现关闭）
5. identity-sketch/token网络（E48不支持，关闭）
6. query-conditioned NBFNet（没有证据表明路径算子是当前首要瓶颈，继续暂缓）
```

唯一仍直接对应证据的神经问题是“网络能否学会中间degree spectrum”。E49只允许在E47 MSPN
上加入训练期1--3轮辅助蒸馏，不把teacher feature喂给最终head，并用target-shuffle和
self-consistent fair-corrupted-P控制排除普通正则化与错误transport。E49是两轮本地readiness，
不是新候选结论；不过门则停止证书传播神经路线，不再枚举更宽MSPN、identity token、
Transformer或远程规模。

E49两轮degree-spectrum蒸馏readiness没有通过。真谱、target-shuffle和自洽错误P的validation
teacher normalized MSE为`0.797746/0.850731/0.735733`；真谱只比打乱低`0.052985`，没有
达到冻结的`0.10`优势。对应balance AUC为`0.475151/0.465384/0.453246`，真谱还低于
`0.48`防退化线和E47 label-only `0.518673`。

这关闭了“让MSPN从活动位和拓扑自己近似证书传播”的分支：不再增加epoch、hidden、
auxiliary scale、identity token或teacher head。当前排名收敛为两个不同证据层：E45/E48
确定性ANF/degree约`0.69`是最强解释锚点；E44 triangle `0.561979`是最强真实神经锚点。

下一候选改为`Certificate-Guided Pair-State Residual`：把可公开计算、非oracle的E45 1--3轮
ANF-prefix作为冻结确定性base，只让E44 pair-state处理器学习base未解释的残差。它不属于
被E49关闭的“神经传播近似”，但也尚未获得性能证据。E50只做两轮实现readiness，并用
prefix-only residual与fair-corrupted-P residual控制区分非线性容量和真实拓扑增量；正式
性能门必须另建计划。若后续正式true residual不能超过E45 ridge与两个控制，则网络探索
应回到E44方法学结果，不再为E43四轮标签枚举新架构。

E50已完成`Certificate-Guided Pair-State Residual`两轮实现readiness。39维ANF-prefix ridge
精确复现`0.686082`；三种残差初始化与ridge逐样本误差为`0`，训练后ridge权重变化为`0`。
prefix-only/pair参数为`10659/10725`，差`0.6154%`；正确/错误P同权重pair embedding最大
差`0.027220`，说明残差分支内部确实感知拓扑。

两轮validation AUC为prefix-only `0.703174`、true pair `0.685938`、corrupted pair
`0.685938`。这些值只证明训练流程稳定；正确/错误P完全相同，不能声称pair残差已有贡献，
也不能因prefix-only两轮较高而后验改路线。

E51是这个E43四轮架构循环的正式终局门：固定30轮seed0，比较ridge、prefix-only、true pair
和fair-corrupted pair。true pair必须达到`0.70`并分别超过ridge/prefix-only/错误P
`0.02/0.02/0.03`。全部通过才允许seed1；否则停止新网络枚举，回到E45确定性约`0.69`
与E44最强真实神经`0.561979`两项可写结论。

E51正式30轮seed0已经关闭CGPR。ridge、prefix-only、true pair和corrupted pair的validation
AUC为`0.686082/0.703174/0.685938/0.685938`；true pair相对ridge、prefix-only和错误P的
差值为`-0.000144/-0.017236/+0.000000`。所有协议门通过，但候选、pair残差和拓扑归因门
全部失败。

因此E43四轮架构排名不再增加新模型：确定性ANF/degree约`0.69`是最强可解释方法，E44
triangle `0.561979`是最强纯神经锚点；E47--E51证明MSPN、identity、degree蒸馏和CGPR均
没有在强控制下增加组外价值。prefix-only `0.703174`只是seed0容量控制，相对ridge优势
`0.017093`低于预告实质margin，不升格主创新。

下一研究轴从“再换网络”转为“提高严格标签轮数”。E52只审计PRESENT-80五轮全密钥/全offset
证书与反例覆盖，以及能否构造structure-disjoint、边际匹配的checkerboard。标签门未通过前，
NBFNet、Transformer、r4调参、seed1和远程GPU全部关闭；标签门通过后才重新排序网络。

E52正式全池确认五轮瓶颈在标签提供者。P0对`96 x 300 = 28800`候选得到
`positive/negative/unknown = 0/27446/1354`；32个抽样negative全部通过标量PRESENT复验，
但6144个结构-输出位支撑全部饱和为`256/256`，无法证明任何正类，mixed structure为0。
因此五轮训练集未就绪，不能用经验平衡率或有限密钥零失败补正类。

CLAASP-MP frozen source具备PRESENT-80模型与完整superpoly接口，但当前Sage环境缺`bitstring`、
`gurobipy`和已验证Gurobi license，也没有非Gurobi monomial backend。语义上必须使用保留
inactive plaintext变量的`find_superpoly_of_specific_output_bit`；`find_keycoeff...`把非cube
明文固定为0，只证明零offset，不满足当前全offset目标。多bit mask还需异或对应bit的完整
superpoly并证明结果恒零。

网络排名继续冻结。下一优先级为E53开放3SDP provider门：利用本机已验证的Sage/GLPK MILP
后端，先在PRESENT 1--2轮exact-ANF fixture验证3SDP trail奇偶消去与bit order，再尝试冻结的
`16 structures x 64 masks`五轮子集。若只能复现普通2SDP可达性而不能处理消去，则停止该实现；
只有子集新增严格正类、正负均非零且证书复验全过，才扩大完整池并重新开放神经结构搜索。

E53-A现已完成上述exact oracle。完整PRESENT-80一、二轮输出ANF分别含`1907`与`4352830`
个单项式，随机向量与标量加密对拍`8/8`和`4/4`。每轮均生成8个全key/全offset严格正类、
8个具体反例负类及4个multi-bit mask fixture，所有证书边界和反例复验通过。

PRESENT S-box的256个exponent pair中，166个至少存在一条raw trail，但只有90个奇数parity、
与exact ANF非零系数一致；其余76个是existence-only误报，单个pair最多228条路径偶数抵消。
这给后续GLPK实现提供了可执行的强控制：它必须复现全部transition parity、正负fixture、bit
order与multi-mask XOR，不能只报告SAT/UNSAT可达性。网络排名和五轮训练门保持关闭。

E53-B进一步验证了Sage/GLPK逐解blocking。代表output exponent `v=1/3/7`分别完整枚举
`4/28/224`个term-choice解，raw count与GF(2) parity逐项匹配E53-A，说明约束语义正确；对应
独立进程墙钟约`0.58/0.55/2.11`秒。最重`v=15`需要1792个解，在冻结10秒内未完成，parity
保持unknown。当前也没有可替代的PySAT、CryptoMiniSat、Z3、BDD或model-counter后端。

因此关闭per-solution GLPK扩到PRESENT电路的路线，不通过加timeout或解释部分枚举补救。下一
provider门E54改审计exact local transition tensor的GF(2)变量消元宽度：只有真实PRESENT-80
五轮因子图在确定性min-fill/query-aware顺序下最大factor变量数不超过26、估计峰值不超过4GiB，
并复现E53-A一、二轮fixture，才实现实际收缩。网络排名、五轮训练和远程GPU继续冻结。

E54首先执行了比内部treewidth更早的full-superpoly语义边界门。8-bit cube之外必须保留56个
inactive plaintext变量，再加80个master-key变量，唯一语义匹配的最终输出边界为136变量、
`2^136`项；即使bit-packed也需约`1.0141e31 GiB`，远超冻结的26变量/4GiB门。固定key、
zero-offset和固定赋值虽然维度更小，却都不能证明all-key/all-offset正类。因此内部factor graph
与min-fill没有构造，dense tensor路线在语义边界处关闭。

E53-A的稀疏完整输出ANF从一轮1907项增长到二轮4352830项，fixture最大superpoly从13项增长
到53392项，尚无可靠五轮稀疏上界。最后一个不改变标签语义的开放provider门是E55：本地CPU
沿12个冻结output query的反向依赖锥计算三轮exact sparse ANF，每query硬限500万项、60秒、
4GiB，并先重放E53-A一、二轮fixture。任何越界即关闭当前全变量provider家族；全部通过才以
相同cap进入四轮，再通过才允许五轮固定子集。E55结束前，网络排名、四轮重新枚举、五轮训练
和远程GPU继续冻结。

E55已经完成并关闭当前exact full-variable provider家族。query-cone实现先对E53-A完成
`32/32`行一、二轮exact重放，superpoly、unit输出hash、随机赋值标量PRESENT、错误P-layer和
zero-offset语义控制全部通过。三轮Q00--Q02完成，query内最大项数为
`2149131/1417246/1775929`，三者都从二轮strict positive转为三轮strict negative，且反例
标量复验通过。Q03在`3.2048s`达到`5000001`项并触发冻结term cap，Q04--Q11按停止线跳过。

因此不进入四轮/五轮query-cone，不提高cap、不转远程、不使用部分多项式标签。连同E52--E54，
当前可执行的support、GLPK逐解、dense tensor与exact sparse-ANF四条五轮严格标签provider路线
均已关闭。这个结论关闭的是当前提供器家族，不是证明五轮积分正类不存在，也不是神经网络
结构上限。

神经结构排名继续冻结：E45/E48确定性ANF/degree约`0.69`仍是四轮解释锚点，E44 triangle
`0.561979`仍是最强纯神经锚点；MSPN、identity、degree蒸馏和CGPR没有获得新增组外价值。
下一步不能靠换NBFNet、Transformer或更大模型绕过标签。

新的E56只审计“广义积分关系输出预测”契约：检查Algebraic Transition Matrices现有预计算
basis能否形成真实PRESENT key schedule下可验证的`输入指数/子空间 + 输出单项式关系`正负标签，
以及这种关系与当前linear-mask XOR平衡任务的映射边界。只有标签宽度、group-disjoint拆分、
强边际控制和确定性复验全部通过，才重新开放一个三行最小神经矩阵；否则不改变target来制造
可训练结果，创新2保留四轮方法学与五轮provider边界结论。

E56正式审计确认ATM广义relation正类有宽度，但神经标签仍未就绪。8份公开basis包含470个
去重relation、union rank 468；relation size为`1:305, 2:136, 4:29`。其中316个relation在
全部8文件共同出现，只有24个仅出现于单文件，因此按搜索split文件划分数据会发生严重relation
泄漏。公开结果没有任何已证明key-dependent negative或negative witness。

源码进一步确认九轮PRESENT round function使用独立64-bit局部轮密钥，没有80-bit master-key
schedule。独立轮密钥下constant relation作为正类原则上比actual schedule更强，但常数0/1仍需
直接求值，actual schedule下严格负类也必须有具体key witness；二者不能从basis缺失推断。

E57最初把ATM坐标误读为普通输入monomial `x^u`，并错误估算每坐标只需
`2^(64-wt(u))=2--16`个明文。论文Definition 7和Algorithm 1表明输入实际采用precursor basis：
`pi_u=1_{x<=u}`，其支持大小为`2^wt(u)`。因此当前`wt(u)=60--63`对应`2^60--2^63`个明文。
错误`x^u` evaluator的`0/470`跨key稳定只能标记为wrong-basis诊断，不能当密码学结果。

修正后的E57只审计precursor数据复杂度与标量cap，不执行大规模加密。若最小relation-key成本
已经超过`2^24`本地门，就关闭直接标量求常数/negative witness；只有获得可执行的algebraic/SAT
constant与key-dependence provider，才重新开放`deterministic marginal / coordinate-set
DeepSets / Relation-Cipher Cross-Attention`矩阵。其他新网络继续关闭。

E57正式结果已关闭直接标量路线。470个relation共693个坐标，`wt(u)`直方图为
`60:3, 61:40, 62:98, 63:552`；单relation、单key最小/中位/最大成本为
`2^60/2^63/2^65`，最小双key witness为`2^61`，而冻结本地cap只有`2^24`。来源和语义控制
全部通过，但成本门全部失败；没有执行大规模加密、神经训练或远程GPU。

下一阶段只改变provider表示：先用E53-A一、二轮exact fixture校准，再对一个冻结九轮relation
做硬cap可执行性审计。候选必须输出可复验constant或actual PRESENT-80 master-key witness；
只输出可达性、只支持独立轮密钥或超cap均停止。通过前网络排名继续冻结。

E58优先审计ATM作者代码已有的PySAT投影模型，而不是立即安装另一个通用求解框架。作者
`is_key_dependent/get_sum`可在理论上导出独立轮密钥key-monomial，但`get_sum`存在tuple/int
成员检查错误，`*_limited`还会把cap exhaustion返回为dependent/1。E58将通过项目侧adapter修正
前者、拒绝后者作为证书，并用S-box/toy全真值校准后再做一个九轮60秒硬cap探针。即使找到
witness，也只开放“独立轮密钥广义relation”标签宽度审计，不自动声称PRESENT-80负类。

E58-A机制复现已经通过。作者原生Glucose4模型对PRESENT S-box全部256个`(u,v)`代数转移
系数与直接真值完全一致，其中90个非零；一位`F_k(x)=x XOR k`返回key exponent `0x1`且独立
重放为odd，常数`x`系数无key witness，强制低cap返回unknown。Python 3.13的QMC并行
`CpModelProto`不可pickle，因此adapter只将等价约束生成改为单进程，并由256项真值门约束。

这只开放E58-B单个九轮mutation的60秒硬cap探针。未得到relation级odd witness前，公开470
正类仍没有可配对的严格负类，DeepSets/Cross-Attention训练继续关闭。

E58-B已经按冻结stop rule关闭九轮exact witness。公开singleton正类
`(u=0xFFFFFFFFFFFFFFF0,v=0x1)`只把输出bit旋转为`v=0x2`，relation size、输入重量60和输出
重量1均保持，candidate也不在公开正类span；但worker在60秒内未完成，没有输出key exponent、
odd parity或relation replay。candidate因此保持unknown，不是strict negative。

ATM九轮监督标签路线现已关闭：不换mutation、不提高cap、不转远程、不训练预注册的
DeepSets/Cross-Attention。下一架构研究只能建立在另一套可执行严格正负标签上；优先审计低轮
PRESENT独立轮密钥relation是否能在同一exact provider下形成足够宽且边际受控的benchmark，
而不是再次改变九轮provider预算。

E59据此冻结两轮16-query readiness：固定`u=0xFFFFFFFFFFFFFFF0`与unit output `e0..e15`，
在60秒总墙钟、`2^12` projected-key和`2^16` trail cap下要求至少12条完成，并同时得到至少4条
exact constant和4条odd-witness key-dependent标签。通过后只扩大标签宽度和捷径审计；只有
严格正负各256且边际匹配，才实现`deterministic / DeepSets / Relation-Cipher Cross-Attention`
三行矩阵。这个顺序避免在没有benchmark时再次枚举网络。

E59的16条两轮query全部在2秒内完成，模型含2080条CNF和192个独立key变量，逐query中位
时间`0.000609s`；但`e0..e15`全部是exact constant，非零projected key mask数量均为0，strict
key-dependent为0。provider执行性通过，标签宽度失败，因此RCCA仍不实现。

下一门只能预注册依赖锥内/外平衡的输出位置并先比较exact reachability基线。若singleton标签
可被reachability完全解释，则新网络没有研究价值，必须转multi-coordinate GF(2) cancellation
relation；不得靠后验扫64位挑出有负类的位置后直接训练。

E60预注册两轮`v=e0`的真实反向依赖锥`bits 0..15`。对输入重量1--8分别配对全锥内prefix与
同重量、仅最后一位换成bit16的锥外control。这样标签前就冻结16条query，并能直接检验
degree-only与cone-membership强基线。即使得到严格正负类，若cone AUC超过`0.80`也停止
singleton任务；只有残差信号存在，才继续RCCA标签宽度审计。

E60已完成并关闭singleton任务。16条查询全部完成且无unknown，结果为`16 constant / 0
key-dependent`；16条constant均通过key exponent 0精确重放以及三组独立轮密钥的完整标量系数
对拍。第一条查询恒为1，其余15条恒为0。由于只有一个类别，degree-only和cone-membership AUC
均不可定义，这不是反捷径通过。RCCA、DeepSets、远程扩展和1024-query singleton扫描全部关闭。

下一优先级不再是继续更换神经网络名称，而是E61多坐标GF(2) cancellation relation：将2--4个
坐标的key-polynomial support做对称差，以零支撑作为严格positive，以保留具体odd key-monomial
并精确重放作为strict negative。只有同relation size和结构边际匹配后能得到各至少256条正负类，
才比较deterministic cancellation、coordinate-set DeepSets和RCCA；否则创新2保留现有四轮
方法学结果，不用模型容量制造表面可训练性。

E61-A进一步否决了“先完整导出240个坐标key-polynomial，再离线找nullspace”的实现。60秒内只
落盘8条，其中7条exact且全部key-dependent；unit output只有41个非零odd项、约0.11秒，但二次
output monomial已经有1763项、约7.7秒，三次`v=0x7`在首个projected mask即超过`2^16` trail
cap并耗时31.45秒。最终为`7 exact / 233 unknown / 0 positive relation / 0 matched negative`。

因此PRESENT两轮ATM完整支撑矩阵关闭，不通过提高cap、远程GPU或事后删掉高次数输出继续。
RCCA仍未实现。下一架构证据门应迁移到项目已有、可全枚举且无unknown的小型SPN严格标签源：
先构造同尺寸多坐标GF(2) relation正负类，再以`ID/重量边际 -> DeepSets -> RCCA true/wrong-P`
最小矩阵验证cross-attention增益。通过只能证明small-SPN方法可行，之后仍需独立解决PRESENT
严格标签provider；失败则直接关闭RCCA，不再用新网络名称延长路线。

E62已证明small-SPN多坐标任务满足训练前提。65,536个标签盲坐标pair经全256主密钥审计后，
train-only选出2048个relation模板，覆盖3轮与4轮各1024条；训练/dual正负分别为
`41531/32197`和`6158/2034`，共有1370种64-topology标签pattern。全部positive逐key为零，
全部negative都有具体odd master-key witness。

最强dual拓扑无关边际AUC为`0.685895`，relation-ID、coordinate和结构group均未越过0.75；
dual P-effect与train SxP interaction比例为`0.985352/0.989258`。因此RCCA现在首次有一个严格、
宽且反捷径通过的多坐标benchmark。下一排名只保留两项：同预算DeepSets基线与RCCA候选；
wrong-P和label-shuffle是必要控制。其他新网络继续冻结，E63失败即关闭RCCA。

E63 readiness已通过全部实现契约。DeepSets/RCCA参数为`70465/79073`，relation token交换与cell
重标号误差为`2.98e-08/7.45e-08`，true/wrong-P同权重logit差为`9.19e-04`。四行8-epoch
流程完整，label-shuffle dual为`0.516807`。

readiness AUC不作科学裁决，但风险信号明确：DeepSets/RCCA true/wrong-P dual分别为
`0.701076/0.599471/0.678291`，RCCA true暂时低于DeepSets且wrong-P更高。不能据此后验改模型；
下一步执行冻结40-epoch双seed Phase A。若RCCA不逐seed超过DeepSets与0.685895边际，直接关闭，
不再增加容量或改换attention变体。

E63正式Phase A已经否决RCCA。DeepSets dual为`0.603005/0.678269`，RCCA为
`0.526025/0.532732`；paired差为`-0.076981/-0.145537`，RCCA mean仅`0.529378`，低于
E62边际`0.685895`。label-shuffle为`0.515766`，不变量、参数预算和训练流程均有效。

因此不运行wrong-P Phase B，RCCA关闭。下一步不是继续试Transformer/attention变体，而是E64
精确分解E62标签：区分both-singleton-balanced产生的简单positive与两个nonzero parity vector真正
相等产生的cancellation positive，并测singleton-status强基线。这个审计决定多坐标任务是否仍有
独立学习价值；不过门则停止该路线，避免把单坐标AND组合包装成新神经结构创新。

E64已经确认后一种情况。dual的6158个positive中，6152个是both-coordinate-zero，只有6个是
两个nonzero parity vector真正相等；train/unseen-S/unseen-P的nontrivial positive也只有
`158/30/50`。both-balanced基线在unseen-S/unseen-P/dual的AUC为
`0.999084/0.998178/0.999513`。

因此E62严格标签本身无误，但原任务被重分类为singleton balance组合主导；E63的复杂RCCA没有
独立的非平凡消去目标。多坐标网络搜索停止，不再实现pair-path relation变体。创新2当前最强可写
神经结构证据仍是E39 SPN-PRR在16-bit合成单坐标benchmark上的拓扑归因，不是PRESENT/GIFT突破。

E65没有重启上述多坐标relation路线，而是把E43真实PRESENT四轮的64个unit-mask严格标签重排
为每个活动结构一次输出64维masked balance profile。checkerboard得到`356/120`条train/
validation观察坐标、`50/18`个互斥structure和`32/23`个输出bit；行列边际AUC均为`0.5`。

确定性路由中，static/错误P可达/正确P可达/ANF前缀validation AUC为
`0.514722/0.695694/0.704306/0.793611`。正确P只领先错误P`0.008611`，因此纯拓扑profile
operator不开放；ANF前缀领先正确拓扑`0.089306`，开放一次prefix-guided nodewise profile
operator readiness。该方向的单变量不是增加Transformer容量，而是把逐query分类改为同一
structure下共享前缀表示、一次产生64个输出logit，并用独立node MLP与fair-corrupted-P消息
处理器控制跨输出交互。它仍限于PRESENT-80四轮严格标签，不构成高轮或攻击结果。

E66两轮readiness已经通过。三种关系模式参数均为`5679`，cell重标号误差`1.94e-7`、masked
loss误差`0`；independent/true-P/corrupted-P validation AUC分别为
`0.717778/0.799167/0.692222`。两轮AUC不作为正式收益，但正确P相对同容量独立node和错误P
已分别显示`+0.081389/+0.106944`的防退化信号，因此只开放E67固定30轮seed0正式归因；
E67必须重新满足绝对AUC、独立容量和错误拓扑三重门，不能直接把readiness写成创新结论。

E67 seed0正式归因进一步通过。independent/true-P/corrupted-P validation AUC为
`0.762500/0.953056/0.800833`，E65 ANF-prefix ridge为`0.793611`；正确P分别领先
`+0.190556/+0.152222/+0.159444`，train-validation gap仅`0.031006`。这是当前首个在真实
PRESENT-80严格输出性质标签上同时超过同容量、错误拓扑和安全确定性前缀的神经结构结果，
但仍限四轮且只有seed0。下一步只运行完全相同的seed1矩阵；在复核前不能升级为稳定结论，
更不能写成高轮积分区分器或攻击突破。

E68 seed1已独立复现：independent/true-P/corrupted-P AUC为
`0.765000/0.961389/0.819444`，正确P相对三类锚点的增益为
`+0.196389/+0.141944/+0.167778`。双seed mean true AUC为`0.957222`，mean增益为
`+0.193472/+0.147083/+0.163611`，两seed所有绝对、关系、ridge和过拟合门均通过。

因此当前架构排名第一更新为`Prefix-Guided Nodewise Profile Operator`，但限定为真实
PRESENT-80四轮unit-output universal-balance profile方法证据；E39 SPN-PRR仍是合成跨拓扑
归因锚点，两者不能合并成高轮结论。下一候选不是更大网络，而是先审计E43非unit linear-mask
族能否形成同级严格profile；只有宽度和反捷径门通过，才给已确认的64-node operator增加一个
轻量mask-query decoder。

E69已经否决该mask-query扩展。E43四个非unit family共有`2254`个raw positive，但
nontrivial positive为`0`；当前证书定义使multi-bit positive严格等价于所有component unit
positive，componentwise validation AUC为`1.0`。匹配宽度也只有`180/40`条train/validation
边，nibble与adjacent-nibble validation缺失。即使combined正确P可达AUC达到`0.735`，也不能
越过语义强基线。

所以架构排名不增加mask-query decoder。已确认的unit operator继续保留；下一有价值的结构
问题是其64-node/shared-round归纳偏置能否跨活动维度或跨SPN迁移。任何迁移训练前必须先生成
与E43同级的严格unit正负标签、structure-disjoint split和边际控制，不能把componentwise组合
或有限key投票重新包装成输出预测创新。

E70已经完成跨活动维度标签与零样本门。4-bit provider完整运行16个结构，但`1024/1024`
候选全为unknown；sound support未证明positive，`8 keys x 4 offsets`也未找到negative。
12-bit的16个结构都在第四轮达到`4,741,632`个候选组合，超过冻结硬上限`2,000,000`，因此
provider完成`0/16`。source/hash/checkpoint和8-bit前缀重放通过，但两个维度均无matched rows，
空集`0.5` AUC不解释。

因此E70裁决为`active_dimension_transfer_labels_not_ready`。它不降低E68在8-bit域内的排名，
也不支持跨维度泛化。当前禁止提高cap、把unknown当negative、做dimension-conditioned微调或
远程规模。下一候选回到同一E65/E68严格benchmark，只改变轮次建模：测试一个小型、共享权重的
`Round-Recurrent Prefix-Guided Profile Operator`，按r1->r2->r3依次更新64个node状态。E68
true-P双seed作为只读同预算锚点，并训练相同参数预算的错误轮序和fair-corrupted-P控制；只有
显式轮序逐seed维持E68质量并稳定超过两项控制，才把round recurrence加入方法。

E71 Phase A已经否决原正向RR-PGPO。三行均为`5461`参数，协议、masked loss、cell等变和
E65重放全部通过；正确轮序/反向轮序/错误P的两轮validation AUC分别为
`0.716667/0.867222/0.697222`。正确轮序相对反向轮序为`-0.150556`，相对错误P仅
`+0.019444`，未过两项`+0.02`门，因此不进入30轮。

反向轮序领先不是39维切片错误：E65特征确实按r1/r2/r3连续排列。不过该结果来自预注册控制，
不能事后改名直接升级。E68仍排名第一。反向序列实际把r1放在最靠近输出头的位置，所以E72
不预设传播方向，只做双seed checkpoint切片中和与single-round ridge归因。若r1稳定主导，
候选应是保留正确轮序的early-round skip/gated residual，而不是反向GRU；若证据不一致或r3
主导，则停止轮递归分支。

E72已经确认r3稳定主导。single-round ridge的r1/r2/r3 AUC为
`0.636111/0.672500/0.799444`；中和r3使seed0/seed1从`0.953056/0.961389`降至
`0.597500/0.690000`，drop为`0.355556/0.271389`。中和r1只下降
`0.025556/0.009722`，r2近零。三组argmax和全部效应门一致通过。

因为r3本来就位于正向序列末端，这个稳定机制与E71反向两轮高分矛盾；轮递归、backward
reasoning和early-round skip全部关闭，E68保持第一。下一优先级不是新网络名称，而是E73
r3-only同结构压缩门：只删除r1/r2输入，仍比较true/independent/fair-corrupted P。若保持
E68质量，得到更简单且更可解释的方法；失败则保留完整39维E68。

E73已经双seed通过。r3-only只输入13维、参数`4795`，比E68的39维/`5679`减少66.7%输入与
15.57%参数。true-P validation AUC为`0.945556/0.947778`，平均`0.946667`；相对完整E68
逐seed为`-0.007500/-0.013611`，平均`-0.010556`，均在预注册`-0.02`容差内。
它逐seed领先independent `+0.288056/+0.275833`，领先fair-corrupted P
`+0.125556/+0.068333`，并保持小于`0.02`的train-validation gap。

因此架构排名分成两个维度：完整39维E68仍是性能第一（mean `0.957222`）；r3-only E73是
简洁性与可解释性第一（mean `0.946667`，输入和参数显著减少），两者都只限PRESENT-80四轮
8-bit活动cube严格unit profile。停止同benchmark继续枚举容量。下一结构搜索必须先在第二真实
SPN上通过同级strict profile标签门，优先审计GIFT-64或SKINNY-64；标签未就绪前不训练。

E74已经完成第二真实SPN的第一道门，但裁决为hold。GIFT-64四轮`96 x 64`原始atlas得到
`3022 positive / 1381 negative / 1741 unknown`，72个结构同时含正负类、共有91种三态签名。
官方向量、S-box ANF、四轮向量化/标量对拍、24个negative witness复验和全部split协议均通过。

问题只出在冻结checkerboard容量：train为`92/92`、validation为`28/28`，低于
`150/150`与`50/50`门；行列class delta均为0，一元边际AUC也全部为0.5。现有标签矩阵的
逐输出配平理论上界只有train 318边和validation 100边，而目标总边为300和100，说明96结构
几乎没有packing余量。因此不降低门槛、不扩大有限key投票，也不启动GIFT神经网络。

下一实验只把确定性结构库从96扩大到192，其他标签语义、4轮、8-bit cube、16 keys、8 offsets、
split、matching与所有门保持不变。若宽度通过，才开放GIFT r3-only三行本地readiness；若仍失败，
关闭当前GIFT unit-profile迁移并转SKINNY或新的sound标签表示。E68/E73排名不受E74 hold影响。

E75已经在唯一改变结构数的条件下通过。前96结构定义、三态标签和39维前缀逐项重放E74；
192结构得到`6087/3286/2915`个positive/negative/unknown、168个mixed结构和187种签名。
checkerboard扩展到train `248/248`、validation `62/62`，覆盖143个结构与39个validation输出bit，
行列class delta仍为0，一元边际AUC仍全部0.5。

因此GIFT-64四轮严格profile现在只开放E76本地两轮readiness：同参数比较independent node、
true GIFT P-layer与same-family fair-corrupted P，并报告r3-only/full39 train-only ridge。
这不是跨密码神经成功；只有正确P逐项超过独立与错误P且r3 deterministic信息没有明显落后完整
前缀，才进入30轮seed0。E68/E73仍分别是当前性能与简洁性第一。

E76正式裁决为hold，但暴露了一个重要基线错位。单节点full39/r3 ridge validation AUC只有
`0.477888/0.507804`，所以预注册`r3 ridge >= 0.60`失败；三行神经两轮AUC却为
independent/corrupted/true `0.560874/0.704475/0.760666`。真实GIFT P领先独立与same-family错误P
`+0.199792/+0.056191`，所有神经readiness和协议门均通过。

因此不直接进入30轮，也不把两轮结果写成跨SPN成功。下一步E77必须先对齐信息范围：比较local、
`local+cell+true-P predecessor`和对应错误P的train-only ridge，并用同一个E76 true checkpoint
在多个错误P上做推理反事实。只有确定性拓扑展开与同权重反事实都归因到真实GIFT P，才能判定
E76失败的是原单节点ridge门而不是网络路线；否则关闭GIFT r3-only。E68/E73排名暂不变化。

E77已经用无新训练证据修复上述信息范围错位。给ridge加入与消息算子相同的local/cell/P前驱后，
local/最强错误P/真实P AUC为`0.507804/0.702133/0.743496`，真实P增益为
`+0.235692/+0.041363`。冻结E76 true checkpoint全部可学习参数只替换P-layer，真实P AUC
`0.760666`，三个错误P最高`0.705515`，margin `+0.055151`。

所以E76的hold不被抹除，但“r3信息不足”被更公平的E77证据限定为单节点ridge不可见，而不是
拓扑交互不可学。E78只允许同一4795参数三行、30轮seed0正式归因，并以E77 true-P ridge
`0.743496`作为确定性安全锚点；通过后才运行seed1。PRESENT E68/E73仍保持现有排名，GIFT尚未
获得正式双seed神经结论。

E78 seed0正式归因已经通过。independent/corrupted/true validation AUC为
`0.571280/0.774714/0.913111`，真实GIFT P相对两类同参数控制和E77公平ridge的增益为
`+0.341831/+0.138398/+0.169615`；train-validation gap为`-0.004748`，全部质量、拓扑与协议门
通过。

这把GIFT路线从标签/两轮诊断推进到真实密码四轮的正式seed0神经归因，但尚未成为稳定双seed或
跨SPN泛化结论。下一步只运行完全相同的30轮seed1；不得在复现前改变模型、输入、错误P或门槛。
PRESENT E68仍是完整前缀性能第一，E73是PRESENT简洁性第一；GIFT E78暂列第二真实SPN的单seed
候选证据。
