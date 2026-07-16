# 创新2 r5 先导章节结论：结构条件积分候选预测及位置先验边界

**冻结日期：** 2026-07-16
**证据范围：** PRESENT-80 r5，本地 CPU，E0-E6
**推荐状态：** 停止同一 r5 数据调参；保留为论文先导章节；创新2高轮目标未完成

## 轮数目标纠正

本文档只冻结 `PRESENT-80 r5` 的方法、结果和负归因边界，不能代表
创新2已经达到主流区分轮数。项目最终比较轴是：神经方法在多少轮上仍能
产生高于强基线的区分或预测信号。不同任务可以并列比较最高轮数，但必须
同时标注任务、数据复杂度、密钥条件和证据类型。

当前本地原文中的主要参照为：

| 路线 | 密码 | 最高轮数 | 证据 |
|---|---|---:|---|
| 本项目结构条件积分概率先导实验 | PRESENT-80 | 5 | E6 表明主要收益由输出位置先验解释 |
| Wu/Guo 积分神经区分器 | PRESENT-80 | 8 | r8 accuracy `0.5732`，经验神经区分 |
| 既有确定性积分 | PRESENT | 9 | key-independent，数据 `2^60`/`2^63` |
| Split-and-Cancel 条件结果 | PRESENT | 10 | 至少四分之一密钥空间的确定性弱密钥性质 |

因此，r5 E0-E6 可以形成毕业论文中的任务构造、评测方法和失败边界章节，
但创新2的高轮实证仍需以 PRESENT r8 主流神经积分协议为首要目标。冻结
计划见 `docs/experiments/innovation2-present-high-round-integral-neural-anchor-plan.md`。

## 建议章节标题

```text
面向分组密码积分性质的结构条件候选预测与强基线审计
```

英文可写为：

```text
Structure-Conditioned Candidate Prediction for Integral Properties
with Strong-Baseline Auditing
```

## 研究问题

给定 PRESENT 的活动输入位置、固定上下文、输出位置和非零 4-bit 输出掩码，
能否不向模型提供密钥或最终密文，预测该结构在未知密钥下更可能保持 XOR
平衡，并用模型分数缩小候选搜索空间？

一条标签为：

```text
q(K,S,u) = XOR_{x in S} <u, E_K^5(x)>
```

其中 `S` 包含活动 nibble 的全部 16 个取值。`q=0` 表示该密钥下观察平衡，
不是对所有密钥的确定性证明。

## 方法实现

输入为 111 bit：

```text
active nibble one-hot   16
output nibble one-hot   16
output mask one-hot     15
fixed plaintext bits    64
```

候选模型为 64-hidden-unit MLP；基础对照包括同输入线性模型、打乱标签 MLP、
P-layer 可达性、固定随机、训练输出位置先验，以及位置分布匹配的线性和随机
选择器。训练、validation、calibration、test、stability 密钥互斥；E4 将
`(active_nibble, output_nibble, output_mask)` geometry 在各 split 间完全
留出。

## 核心实验结果

### E2-E3：随机未见结构的双 seed 排序

| 指标 | seed0 | seed1 |
|---|---:|---:|
| MLP Spearman | 0.685426 | 0.782359 |
| MLP top-16 平衡率 | 0.922119 | 0.963135 |
| MLP-linear top-16 | +0.049316 | +0.047607 |
| shuffled-global top-16 | -0.021057 | +0.003998 |

双 seed 冻结门通过，说明模型在弱对照下具有稳定排序效用。

### E4：未见 geometry 泛化

| 指标 | 结果 |
|---|---:|
| MLP Spearman | 0.825454 |
| linear Spearman | 0.685818 |
| MLP-linear Spearman | +0.139636 |
| MLP top-16 平衡率 | 0.950439 |
| linear top-16 平衡率 | 0.885010 |
| MLP-linear top-16 | +0.065430 |

四个 geometry-holdout 排序门全部通过。概率校准的 32-key/256-key 标签稳定性
MAE 为 `0.051849`，略高于 `0.05` 门槛，因此只主张排序，不主张精确概率。

### E5：4096 把全新密钥

| 选择器 | top-16 平均平衡率 | 最低平衡率 | 零观察失衡结构 |
|---|---:|---:|---:|
| 结构 MLP | 0.956604 | 0.841797 | 8 |
| 同输入线性 | 0.894394 | 0.715576 | 2 |
| P-layer 可达性 | 0.848633 | 0.542725 | 1 |
| 固定随机 | 0.802155 | 0.501465 | 2 |

E5 预注册门全部通过。零失败结构的 95% 单侧 q1 上界为 `0.000731113`，但
有限随机密钥不能证明对所有密钥平衡。

### E6：强位置先验审计

| 选择器 | top-16 平均平衡率 | MLP 差值 |
|---|---:|---:|
| 结构 MLP | 0.956604 | - |
| 训练输出位置先验 | 0.941788 | +0.014816 |
| 位置匹配线性 | 0.950653 | +0.005951 |
| 位置匹配随机 | 0.919968 | +0.036636 |

MLP 没有超过相对位置先验 `+0.03` 和相对位置匹配线性 `+0.02` 的冻结门。
最终裁决为：

```text
innovation2_integral_position_prior_explains_enrichment
```

## 论文可以主张什么

可以主张：

1. 构造并实现了结构条件积分 parity 候选预测任务，明确区分单样本输出 parity、
   跨结构积分观测和所有密钥确定性积分。
2. 建立了密钥互斥、geometry 留出、独立校准、双 seed、fresh-key 复验和
   post-hoc 强基线审计的可复现实验流程。
3. 证明弱对照会高估 MLP 的密码分析价值：MLP 相对线性/随机看似显著，
   但训练输出位置先验和位置匹配线性解释了主要增益。
4. 给出一个有实践价值的评测结论：神经积分候选工作必须加入位置边际或
   其他结构边际强基线，否则 top-k 高平衡率不能归因于非线性结构学习。

不能主张：

- 当前 MLP 独立优于简单输出位置规则；
- 已发现新的确定性积分区分器；
- 4096 把密钥零失败等价于数学证明；
- 首次提出神经积分候选或神经候选加传统验证；
- 达到 Split-and-Cancel、SAT/MILP 或 division-property 的精确性。

## 与论文工作的关系

- Kimura 等的 output prediction 以具体输出恢复为目标；本任务预测结构集合的
  parity 统计性质。
- Zhang 等 EUROCRYPT 2026 已使用神经特征提出积分组合并做 split-search
  验证，所以“神经 + 积分验证”不是本项目首创。
- Hwang 等 kernel 方法区分经验平衡空间与真实平衡空间，支持本项目对有限
  密钥证据的保守解释。
- Wang、Hadipour、Gerhalter 的 Split-and-Cancel 通过精确后缀 ANF、前缀
  monomial oracle 和左核给出可靠证书；本项目没有实现或声称等价能力。

## 建议论文结构

```text
1. 问题定义与三种输出预测任务的区别
2. 结构条件输入、标签和网络
3. 密钥/geometry 无泄漏实验协议
4. E0-E4：从可行性到几何泛化
5. E5：4096 fresh-key 候选富集
6. E6：输出位置先验强基线审计
7. 讨论：为何弱基线会制造神经优势
8. 局限：有限密钥、单密码单轮数、无精确认证
```

## 答辩时的一句话

```text
我不是把一个被位置先验解释的结果包装成神经突破，而是建立了一个可复现
的积分候选预测流程，并通过更强对照证明：在 PRESENT 五轮任务上，输出
位置先验解释了主要收益，因此后续神经积分研究必须先消除这类结构捷径。
```

这是一项可以写入毕业论文的完整 r5 先导研究和方法学贡献，但不是新的
密码攻击，也不是创新2高轮目标的完成证据。

## 主要产物

```text
docs/experiments/innovation2-present-r5-structure-conditioned-integral-parity-feasibility-plan.md
docs/experiments/innovation2-present-r5-fresh-key-enrichment-plan.md
docs/experiments/innovation2-present-r5-output-position-prior-audit-plan.md
outputs/local_diagnostic/i2_present_r5_integral_parity_geometry_holdout_ranking_seed0/
outputs/local_diagnostic/i2_present_r5_integral_fresh_key_enrichment_4096_seed0/
outputs/local_diagnostic/i2_present_r5_integral_position_prior_audit_4096_seed0/
```
