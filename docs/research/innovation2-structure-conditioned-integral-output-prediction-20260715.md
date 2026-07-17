# Innovation 2：结构条件化积分输出性质预测

**日期：** 2026-07-15

**状态：** 真实PRESENT/SKINNY/SPECK标签族仍暂缓 / E30随机子空间与E31确定性provider均未开放真实密码训练 / E32小SPN精确标签被ID边际解释 / E32b train-only matched contrast过门 / 仅E33合成SPN神经比较开放

## 一句话创新

不再要求神经网络逐样本复原具体密文，而是给定密码结构、轮数、
活动输入集合和输出线性掩码，预测该掩码跨整个输入集合、跨未知密钥
是否保持 XOR 平衡：

```text
结构描述 (cipher, r, S, u)
    -> neural predictor
    -> P_K[ XOR_{x in S} <u, E_K^r(x)> = 0 ]
```

方法暂定名：

```text
Structure-Conditioned Neural Integral Property Prediction
结构条件化神经积分性质预测
```

## 三种任务必须分开

| 任务 | 一条样本 | 预测目标 | 随机基线 |
|---|---|---|---:|
| Kimura 式完整输出预测 | 一个明文/状态 | 完整输出值是否 exact match | 4-bit 时 `1/16` |
| 单样本输出掩码预测 | 一个明文/状态 | `<u,E_K(x)>` 的 `0/1` | `1/2` |
| 本方法的积分性质预测 | 一个结构化输入集合 `S` | `XOR_{x in S}<u,E_K(x)>` 的跨密钥平衡概率 | 非积分区约 `1/2` |

固定 `u=1111`、只 XOR 单个输出 nibble 的四个 bit 属于第二行，
不是积分。只有再跨结构化输入集合 `S` 求 XOR，才进入第三行。

## 数据对象

以 PRESENT-80 为例，一条结构由以下字段组成：

```text
cipher            = PRESENT-80
rounds            = 5
active_nibble     = 2                  # 从最低有效 nibble 起 0-based
fixed_context     = 其余 15 个 nibble
active_values     = 0,1,...,15
output_nibble     = 7
output_mask       = 1111
```

它展开成 16 个明文：活动 nibble 遍历全部值，其他 nibble 在集合内
不变。对密钥 `K`，标签观测为：

```text
q(K,S,u) = XOR_{x in S} <u, E_K^r(x)> in {0,1}
```

`q=0` 只说明该密钥、该结构、该掩码的一次观测平衡。要称为确定性
积分性质，需要对所有密钥恒为零；有限随机密钥只能给出经验支持，
不能代替符号证明或完整搜索。

## 为什么预测概率而不是直接贴“积分/非积分”标签

前置审计已经发现，同一个 PRESENT r5 结构在不同密钥下可能得到
不同 `q`。如果输入不含密钥，却把每把密钥的 `q` 当成确定性结构
标签，网络会面对完全相同输入的矛盾标签。

首轮实验采用更诚实的统计目标：

```text
同一结构在多把随机密钥下重复出现
网络输入不含密钥
BCE 最优输出 = P_K[q=1 | structure]
平衡概率       = 1 - P_K[q=1 | structure]
```

这不是确定性积分发现的终点，而是先判断结构信息是否包含可泛化的
跨密钥 parity 概率信号。

## 输入表示

首轮只用最小、可审查的结构向量：

```text
active_nibble one-hot   16
output_nibble one-hot   16
output_mask one-hot     15
fixed plaintext bits    64（活动 nibble 清零）
---------------------------------------------
total                  111 features
```

PRESENT 和 r5 在首轮固定，不作为特征，避免网络只靠密码名或轮数分类。
网络不接收最终 16 个密文或其输出 nibble；否则任务退化成直接计算 XOR。

## 与已有工作的边界

- Kimura 等工作预测具体密文/明文并评价完整输出 exact match；本方法
  改为预测结构化集合上的密码分析性质。
- Hou 等神经线性攻击已经证明神经网络可学习输入/输出 XOR 分布，
  因而固定单掩码二分类本身不是新概念。
- Watanabe 等工作指出输出预测网络可能捕获积分 `ALL/BALANCE`，但未
  将“结构描述到跨密钥积分掩码概率”冻结为独立监督任务。

因此当前可主张的是新的任务构造候选，不是“首次用神经网络处理
XOR”，也不是已经发现了新积分。

## 论文贡献成立所需证据

必须依次证明：

1. **标签有效：** 不向输入泄露最终输出，训练/验证/测试密钥互斥。
2. **结构有效：** 非线性结构模型优于相同输入的线性模型和标签打乱控制。
3. **泛化有效：** 在未见结构、未见密钥、不同 seed 上保持优势。
4. **密码分析有效：** 预测结果能找到经典基线漏掉的高平衡概率掩码，
   或减少积分搜索/验证成本。
5. **跨结构有效：** 至少覆盖两个 SPN，并最终与 ARX/Feistel 对比，
   才能支持平台级方法声明。

## 预审计结果

每轮随机 160 个结构、每结构 32 把密钥的 PRESENT 审计为：

| 轮数 | `q=1` 比例 | 32 把密钥全零结构 | 密钥相关结构 |
|---:|---:|---:|---:|
| 1--4 | `0.0000` | 每轮 `160/160` | 每轮 `0/160` |
| 5 | `0.2699` | `14/160` | `146/160` |
| 6 | `0.4898` | `0/160` | `160/160` |
| 7 | `0.4930` | `0/160` | `160/160` |

混合轮数训练会产生轮数捷径：1--4 轮全零，6--7 轮近随机。因此第一
轮神经实验只使用 r5，测试结构交互能否解释中间过渡区。

## 当前裁决边界

首轮实验最多支持：

```text
PRESENT r5 的结构字段包含/不包含可泛化的跨密钥积分 parity 概率信号
```

它不能直接支持：

```text
发现新的确定性积分
优于 MILP/SAT/ division property 等经典方法
完成开题报告全部轮间输出预测创新
达到正式或论文规模
```

## 下一步

首轮候选 test AUC 为 `0.6553`，相对同输入线性模型和标签打乱控制分别
提高 `+0.0344` 和 `+0.1070`，说明结构排序信号存在；但结构概率 MAE
只比线性改善 `0.0124`，未达到冻结的 `0.02` 门槛。因此路线保持，
但不得扩样或远程训练。

下一步执行实验计划中冻结的 E1 独立校准/标签噪声审判：增加独立
calibration split，并用 256 把新密钥重估同一 test 结构的标签稳定性。
只有校准后的概率误差过门，才进入多 seed、几何组合留出与经典积分
基线比较。

E1 已于 2026-07-16 完成。MLP 的 256-key 校准 MAE 为 `0.09580`，比
线性模型改善 `0.01533`，且 AUC 优势保持 `+0.03440`；但它未达到
`0.09` 的绝对 MAE 门槛，同一结构的 32-key 与 256-key 观测率 MAE
为 `0.05988`，也未达到 `0.05` 稳定性门槛。

因此当前方法边界进一步收紧：结构表示包含可泛化排序信号，但 32-key
点概率不够稳定，不能直接作为“是否平衡”的精确概率输出。下一步优先
复用 256-key 标签做不确定性感知排序与 top-k 候选筛选审判；只有这种
密码分析效用也优于线性与打乱控制，才值得做新的 seed/几何留出确认。

E2 已复用 E1 的 128 个 test 结构和 256-key stability 标签完成。结构
MLP 的 Spearman 为 `0.68543`，比同输入线性模型高 `0.09337`；它选出的
top-16 结构平均平衡率为 `0.92212`，比全局高 `0.20551`、比线性 top-16
高 `0.04932`。打乱标签控制的 top-16 反而比全局低 `0.02106`，四个
预注册门槛全部通过。

因此当前最准确的创新表述是：网络尚不能可靠输出精确积分概率，但能在
固定 PRESENT r5 协议下，根据活动位置、输出位置/掩码和固定上下文，
优先筛出跨新密钥更可能保持 XOR 平衡的候选结构。下一步只更换为独立
seed1，原样复验排序和 top-16 效用；通过后再做几何组合留出，当前不
启动远程 GPU，也不声称发现确定性积分。

E3 已按同预算 seed1 完成，四门再次通过。两颗 seed 的 MLP Spearman
分别为 `0.68543/0.78236`，top-16 平均平衡率为 `0.92212/0.96313`；
相对线性 top-16 的优势为 `+0.04932/+0.04761`，而打乱标签控制相对
全局为 `-0.02106/+0.00400`。联合裁决为
`innovation2_integral_ranking_utility_two_seed_confirmed`。

因此项目现在已有“双 seed 随机未见结构上的候选排序效用”证据。下一步
的关键问题不再是重复加 seed 或放大训练量，而是把 active nibble、输出
nibble、输出 mask 三元组在 train/test 间完全留出，判断模型是否学到可
迁移的结构关系，而非记住常见位置/掩码组合。该几何留出仍应先在本地
同预算审判；通过前不启动远程 GPU。

## E4 结果与 2026 论文边界复审

E4 已于 `2026-07-16` 完成。训练、validation、calibration、test 的
`(active_nibble, output_nibble, output_mask)` 三元组完全互斥，结构 MLP
在 128 个未见 geometry 和每个结构 256 把新密钥上的 Spearman 为
`0.825454`，相对线性提高 `+0.139636`；MLP top-16 平衡率为
`0.950439`，相对线性 top-16 提高 `+0.065430`，四个冻结排序门全过。

最新文献把创新边界进一步收紧：

- Zhang 等在 EUROCRYPT 2026 的 *Neural-inspired Advances in Integral
  Cryptanalysis* 已采用神经特征提出积分输出组合，再用前缀/后缀 split
  search 逐候选验证。因此“神经候选生成 + 传统积分验证”不能作为本项目
  的首创点。
- Hwang 等的 kernel 路线从多把随机密钥的 parity matrix 提取经验平衡
  空间，并用独立密钥做后选择验证；它明确区分 `B_true` 与 `B_emp`，有限
  密钥结果仍不是无条件证明。
- Wang、Hadipour、Gerhalter 的 *On Extending Integral Distinguishers*
  用精确后缀 ANF、前缀 monomial-prediction oracle 和左核给出可靠但不完备
  的 Split-and-Cancel 证书。其 PRESENT 结果使用 60/63 个活动位和一次
  最后一轮精确展开，与本项目“4 活动位、固定上下文、跨密钥排序”的任务
  不同。

因此可答辩的新意应表述为：**结构条件输入、geometry-disjoint 泛化和
同预算候选富集评估**，而不是首次把神经网络用于积分分析，也不是首次把
神经候选交给符号验证。

## E5 路线调整

原拟直接做精确认证，但 readiness 审计发现当前环境没有 Z3、PySAT、
MILP、Sage 或 S-box Analyzer；五轮活动单项式支持上界对 E4 的全部 128
个候选都返回未知，不能提供有区分力的证书。用有限密钥穷举替代 SAT 并
改称“精确”会造成错误论文结论。

E5 因此冻结为 `4096` 把全新密钥的同预算候选富集验证：固定 E4 模型与
候选，不重训；比较 MLP、线性、无标签 P-layer 可达性启发式和固定随机
各 16 个候选。零失败候选只报告二项分布 95% 单侧失败率上界，不称为
确定性积分。E5 通过后停止机械实验扩展，进入论文方法、对照、限制和图表
写作；精确 SAT/Split-and-Cancel 认证保留为条件允许时的增强工作。

## E5-E6 最终裁决

E5 在 4096 把全新密钥上按预注册门通过：MLP top-16 平均平衡率为
`0.956604`，相对线性、P-layer 可达性和固定随机分别提高
`+0.062210/+0.107971/+0.154449`，且有 8 个结构零次观察到失衡。

但 8 个零失败结构全部位于 `output_nibble=0`。E6 随后只用冻结训练 split
构造输出位置边际先验，并加入与 MLP 输出位置直方图完全相同的线性/随机
对照。结果为：

```text
MLP mean balance                    = 0.956604004
train output-position prior         = 0.941787720
position-matched linear             = 0.950653076
position-matched random             = 0.919967651

MLP - position prior                = +0.014816284  < +0.03
MLP - position-matched linear       = +0.005950928  < +0.02
MLP - position-matched random       = +0.036636353  >= +0.03
```

最终 decision 为 `innovation2_integral_position_prior_explains_enrichment`。
这推翻了“E5 已足以证明神经结构交互优势”的强解释，但不否定 E0-E5 的
原始数值。当前最严格、可写入毕业论文的结论是：

1. 项目建立了不泄漏密钥、结构和 geometry 的结构条件积分 parity 预测与
   top-k 评测流程，并在双 seed、geometry holdout 和 4096 fresh keys 上
   完整运行。
2. MLP 相对弱线性/随机/P-layer 对照表现出明显候选富集。
3. 更强的训练输出位置先验和位置匹配线性基线解释了主要收益；当前 111-bit
   表示没有建立独立于输出位置的非线性结构交互优势。
4. 有限密钥零失败不是确定性积分证明，也不能与 Split-and-Cancel、SAT、
   MILP 或 division-property 结果等同。

因此同一 PRESENT r5 数据上停止继续调模型、选择器和门槛。创新2作为
**可复现的任务构造、严格控制链和负结果边界**进入毕业论文；若未来需要
恢复正向神经方法声明，必须换新 benchmark 并在实验前加入位置边际强基线，
而不是继续利用已观察的 4096-key 结果调参。

## E7：辅助高轮锚点与主任务边界

用户进一步明确，创新2的最终比较轴不是只完成一个低轮方法案例，而是：

```text
我们的神经方法在多少轮上仍能完成有用区分或预测？
与主流工作的最高轮数相比处于哪里？
```

不同任务不要求同条件比较轮数，但报告必须并列给出任务类型、输入规模、
数据复杂度、密钥条件、指标和证据强度。按本地原文，PRESENT 的直接神经
积分参照是 Wu/Guo 的 r8 accuracy `0.5732`；确定性积分参照为 r9，最新
Split-and-Cancel 条件/弱密钥结果到 r10。Kimura 对 small PRESENT-[4]
原 S-box 的 r5 ciphertext prediction 属于 16-bit toy cipher；Singh 的
31-round emulation 删除了 AddRoundKey，均不能替代 keyed PRESENT-80 r8
主流 anchor。

当前 111-bit 结构描述任务不能机械升到 r6/r7：预审计中两轮的 `q=1`
比例已经为 `0.4898/0.4930`，且 r5 的表观收益在 E6 被输出位置先验解释。
Wu/Guo 的积分 multiset、`InvP/InvS` 数据表示和 PRESENT r5/r7/r8 轮数梯
可以回答“主流二分类协议在多少轮仍有信号”，但它的标签是结构集合与随机
集合的类别，不是输出掩码是否平衡。因此 H0 只能作为辅助 round-reach 锚点，
不能替代本创新的结构条件积分输出性质预测。

H0 seed0 已在 `2^21` 总训练行、50 epochs 的论文参考规模近似上得到 r8
round-reach 信号；随后启动的同协议 seed1 因用户再次明确主任务边界而取消，
未完成指标不得解释。创新2主线回到“结构描述 -> 跨密钥平衡概率/排序”。
由于单活动 nibble 的 r6 标签近随机，下一步先审计增加活动 nibble 数能否在
r6 形成既非全平衡、也非全随机、并且保留位置先验残差的可学习过渡区；通过
后再训练网络，不直接扩远程数据量。

执行计划：

```text
docs/experiments/innovation2-present-high-round-integral-neural-anchor-plan.md
docs/experiments/innovation2-present-r6-output-property-transition-audit-plan.md
```

## E8-E27：从噪声概率标签转向跨密钥 kernel 标签

PRESENT r6 过渡区审计首先纠正了一个关键统计错误：有限密钥下观测到的
balance-rate 方差包含二项抽样噪声，不能直接当作结构可学习信号。去除
有限密钥噪声后，随机结构的跨密钥概率残差不足以支持继续训练。项目因此
不再从任意结构的有限密钥频率直接制造监督标签，而是切换到 Hwang 2026
的线性代数定义：

```text
对同一输入结构 S，在多把密钥下计算完整输出 XOR word
每把密钥形成 parity row
多行组成 GF(2) matrix M
输出 mask u 为稳定平衡方向 iff u 属于跨密钥 joint kernel ker(M)
```

这个变化没有改变创新2目标；它把原来噪声较大的概率标签，替换为经过
discovery/validation 密钥分半、basis 回代和 fresh-key 检查的经验 kernel
membership 标签。有限密钥 joint kernel 仍不是全密钥证明，但比未经校正的
32-key balance rate 更适合做神经训练前的标签 readiness。

### PRESENT 与 SKINNY 的停止边界

PRESENT 7轮成功复现了论文 kernel，并一度从不同活动块、固定上下文和
拓扑几何获得多个经验签名；但后续控制依次发现：

```text
context/mask identity 与 bit pattern 可高 AUC 解释标签；
mask-disjoint 捷径可泛化到未见 mask；
扩大到 fresh keys 后部分 context kernel 不复现；
去除 context/mask 加性边际后的 balance-rate interaction 接近零。
```

因此 PRESENT r7 inactive-context 标签表已停止，不能进入神经训练。
SKINNY-64/64 的7轮和8轮论文 kernel 也已精确复现，但相邻 pair、底行 pair
和单活动 cell 的稳定 kernel/签名覆盖不足，当前同样保持 `hold`。这些负结果
说明“复现一个积分 kernel”与“形成可泛化的结构条件标签族”是两道不同门。

### SPECK E25-E27 当前主线

SPECK32/64 提供了与 PRESENT 不同的 ARX 结构试验场。E25 在论文结构
`{5,6}` 固定、其余30 bit完整活动、每行精确 `2^30` 明文的协议下，以
32把 discovery 和32把 validation 密钥复现了：

```text
6轮 joint rank/nullity = 23/9，kernel 等于 Hwang Table 7 九维空间
7轮 joint rank/nullity = 31/1，kernel = {0x02050204}
位置控制 {0,1} 的7轮 joint nullity = 0
```

E26 随后验证固定值 `00/01/10/11` 在6/7轮共享完全相同的 joint kernel。
所以 fixed value 是 nuisance context，不是标签变量，四种值不得重复计作
四类训练数据。E27 只移动两个相邻固定 bit，在两个16-bit word内筛选全部
30个位置；先用8把 paired keys 做精确 screen，再对冻结规则选出的最多8个
候选补到64-key。当前远程 run 为：

```text
i2_speck32_hwang_positions_gpu0_20260717
source commit = 41d60a1b73c2018a09b2cfae7a9ccc44ca256d9f
status = hold / narrow_position_family
training_performed = false
```

E27 即使满足“稳定正位置至少4个、采样负位置至少8个、覆盖两个word”的
数量门，也只证明同一目标 mask 在位置族上有变化，不能自动开放神经训练。
进入下一阶段前还必须检查不同位置的完整 kernel 签名，并构造真正的
`position x output-mask` 网格。若所有正位置都只有同一个一维 kernel，任务
可能退化为记忆正位置集合，仍不满足结构条件输出预测的创新目标。

E27 的最终结果正好落入该边界：只有 `{5,6}` 与 `{6,7}` 在完整64-key证据下
稳定，二者 joint kernel 都是同一个一维 `{0x02050204}`，且高 word 没有
8-key screen 命中。因此 E28 位置×mask标签宽度门不执行；当前证据只说明
论文结构存在一个局部相邻平移，不足以形成可训练标签族。下一步改测由
SPECK真实 `ROR7(x)+y` lane关系定义的16个跨word pair，并用 offset-minus-one
的16个pair作同预算控制。只有真实拓扑 family 明确超过错位控制，才继续
构造多mask标签；否则停止 SPECK 固定pair路线。

E27-N 已完成该停止审判。真实 `{y_i,x_(i+7)}` 与错位
`{y_i,x_(i+6)}` 两组各16个pair在同一8把Phase C筛选密钥上均为 `0/16`
命中；没有候选进入64-key补全。本地对256个 `2^30` 精确枚举行的缓存、计时、
密钥、Phase C冻结SHA和GF(2)裁决重算全部通过。因此SPECK固定pair路线已停止，
不得继续扫描其他offset或把E25论文锚点复制成神经训练标签。

这一结果连同PRESENT坐标context捷径和SKINNY坐标位置族过窄，说明下一个benchmark
不能继续只枚举“哪些坐标bit/cell活动”。E30转向文献允许的一般线性/仿射输入子空间：
在PRESENT-80 7轮固定16维、`64+64`密钥预算和完整输出parity，只改变子空间orientation，
先判断是否形成跨密钥稳定且多样的joint kernel。该路线与Carlet等对k维affine space
积分传播的形式化对象一致，但本项目仍只把有限密钥kernel当经验标签readiness；
不会把它写成确定性证明，也不会在标签宽度与组外捷径门之前训练网络。

E30现已完成。四个坐标锚点的joint nullity分别为 `8/4/4/5`，证明协议和已知信号
有效；32个RREF去重的随机16维orientation却全部为零kernel，不同非零签名也是零。
因此随机orientation benchmark停止，不改seed、维度或轮数继续挑选。E15此前已经
覆盖四个坐标块的P-layer零/一/二次轨道并裁决多样性不足，所以后续也不能重复把
P-layer orbit包装成新的“拓扑保持变换”。

候选生成必须转为有密码学约束的确定性提供者。初步代码审计发现：CLAASP当前公开
仓库同时含PRESENT-80模型和monomial-prediction模块，但该模块硬依赖Gurobi license，
测试也因此全部skip；Beyne/Verbauwhede的代数转移矩阵公开仓库不依赖商业求解器，
并附带PRESENT 9轮的预计算结果，但其对象是
`sum_x r'(x,F(x)) = constant` 的广义积分性质。它不能未经转换就当作本项目当前的
“一个仿射输入集合 + 一个线性输出mask是否平衡”标签。

下一门先冻结候选提供者契约：输出必须明确给出输入集合、线性输出mask、确定性/经验
标签语义、负样本证明范围和bit order。只有完全同目标的候选才能进入标签atlas；
高阶输出单项式或多项广义关系只能作为扩展任务单独报告。

E31完整审计最终为hold：CLAASP-MP当前缺Gurobi运行时；ATM的305个singleton中虽有
198个线性输出候选，但constant值未知、负类不完备，公开8文件union rank为468而非
论文报告的470。它们不能直接作为当前0/1标签。

E32随后在16-bit合成SPN、16个S-box/P-layer组合和全部256把toy key上生成57344个
精确标签。正负宽度、签名和cipher interaction均充足，但通用
`round+structure+mask`边际在三种heldout达到约0.984--0.987 AUC，因此原始标签表
禁止训练。

E32b只用9个train topology选择其内部同时有正负的589个base cell，完全不读取
heldout标签。重裁决保留9424行与336种16-topology模式，并把unseen-S、unseen-P、
dual-unseen最强ID边际降至 `0.775693/0.742532/0.726528`，全部低于冻结停止线。
因此只开放E33合成SPN上的GraphGPS/SCGT比较；真实PRESENT/GIFT/SKINNY神经训练仍未
开放，必须等待合成模型的topology attribution与后续真实密码迁移门。

E33现已完成。GraphGPS真实拓扑的dual-unseen均值为`0.682672`，低于ID边际
`0.726528`；错误P-layer控制反而达到`0.752444`。SCGT为`0.726947`，只与边际持平。
label-shuffle为`0.465781`，训练流程控制正常。最终裁决为
`innovation2_small_spn_topology_predictor_not_ready`，真实密码迁移门继续关闭。

源码归因审计发现E33同时使用绝对bit、nibble和lane embedding，可能破坏并行S-box cell
重标号对称性。下一步仅开放E33-R：冻结数据与训练预算，改成cell-equivariant位置表示，
并要求真实P-layer超过错误P-layer和ID边际。这个实验是表示缺陷审计，不是扩大网络搜索；
若仍失败，不测试更深GraphGPS、更多epoch或远程真实密码训练。

E33-R已经完成。确定性cell重标号最大logit误差为`8.94e-08`，但真实P-layer的
dual-unseen AUC仅`0.711548`，仍低于ID边际`0.726528`，并与错误P-layer`0.708831`
基本持平。裁决为`innovation2_small_spn_cell_equivariance_repair_not_ready`。这说明
修复绝对坐标依赖能改善E33，却仍不足以建立真实拓扑贡献；equivariant SCGT和真实密码
迁移继续关闭。

下一次只允许测试round-shared neural algorithmic reasoner：同一个cell-equivariant图
处理器按实际轮数循环2--5次，替代固定三个不共享block。理由来自源码语义错配及后验分轮
诊断，而不是追逐新模型名称。数据、split、40 epochs、seed0/1、wrong-P和label-shuffle
全部冻结；未超过ID边际和wrong-P各`0.03`就停止合成GraphGPS/looped路线。

E34 round-shared reasoner已完成。真实P-layer dual均值为`0.683643`，低于ID边际
`0.726528`、E33-R静态锚点`0.711548`和错误P-layer`0.708540`。裁决为
`innovation2_small_spn_round_shared_reasoner_not_ready`；合成GraphGPS/looped家族停止，
真实密码迁移仍关闭。

下一候选只开放一个关系算子方向：小型Cipher Edge-Token Transformer。它把16条有向
P-layer边和4个S-box relation显式编码成token，并由output-mask query参与attention，测试
edge-edge/query-edge交互是否是当前neighbor-gather模型缺失的能力。数据、split、true/wrong-P
和label-shuffle继续冻结；若仍不超过错误拓扑，停止该合成benchmark上的网络架构搜索，
回到标签族和任务定义审计。

E35的首次wrong-P实现随后被审计为协议无效：跨variant roll把heldout P3换成train-seen
P2，不能作为同难度控制。E35b使用固定destination-cell rotation保持每个variant/P-family
身份后从头重跑。CETT true、公平控制和label-shuffle的dual均值为
`0.671767/0.664944/0.484934`；true只领先公平控制`0.006823`且低于ID边际`0.054761`。

因此当前合成SPN上的神经架构搜索关闭，真实密码迁移继续关闭。下一步是E36不训练审计：
验证matched标签是否具有足够的P-layer条件敏感性和组外正负宽度。若标签门不过，应更换
标签族、候选提供者或任务定义，不能继续用更多神经网络结构掩盖benchmark不可识别性。

E36确定性审计已经通过：589个matched cell中585个在训练块具有P-layer敏感性，521个
存在train S×P interaction，579个在完整4x4拓扑网格存在interaction；S3下dual P-effect
有391个，目标正负为234/157。标签本身具有足够P-layer信号。

但当前训练只有3个独立P-layer随机permutation；5301行只是9个S×P图上的重复查询。
因此下一步不是恢复CETT或GraphGPS训练，而是E37扩展到`4 S-box x 16 P-layer`，使用
12个train P和4个heldout P重新生成全key精确标签与train-only matched benchmark。
扩展标签族通过边际、interaction和公平控制门之后，才允许重新打开最小网络矩阵。


## 当前神经训练开放门

下一次神经训练必须同时满足以下条件：

1. 至少一个密码/轮数下存在多个跨密钥稳定的输入结构和多个非零输出 mask；
2. 完整 `structure x mask` 标签网格同时包含正负类，且不是固定值重复样本；
3. position identity、mask identity、重量、加性和 bitwise 线性基线的 accuracy
   与 AUC 均低于冻结停止线；
4. position-disjoint、mask-disjoint 和 dual-disjoint 切分中，简单基线不能
   泛化解释标签；
5. discovery/validation/fresh-test 密钥互斥，经验 kernel 或 balance-rate 标签
   在新密钥上稳定；
6. 神经候选必须与同输入线性模型、标签打乱、结构打乱和 deterministic kernel
   baseline 同预算比较。

满足这些门后，网络输入仍是结构描述而不是密文集合：

```text
active/fixed bit mask
fixed context（仅当实验证明其影响标签）
round count / cipher family
output mask
ARX/SPN/Feistel topology features
```

目标可以是 mask membership、跨密钥平衡概率或候选排序；都必须与
“积分 multiset 对随机 multiset 二分类”分开报告。后者只保留为辅助
round-reach anchor，不能替代创新2主任务。
