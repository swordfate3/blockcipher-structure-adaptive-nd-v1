# Innovation 1 PRESENT 更高轮次研究策略

**日期：** 2026-07-05

**状态：** 新任务研究蓝图 / r9 weak-probe 条件路线准备中

**范围：** Innovation 1 的 SPN/PRESENT 更高轮次推进。本文回答“如果目标不是只在 r7/r8 同协议下微调，而是尽量推进 PRESENT 可验证轮数，研究路线应该怎么改”。具体 r9 weak-probe 运行计划放在 `docs/experiments/innovation1-present-r9-weak-probe-plan.md`。

## 1. 研究目标

用户的新任务是：

```text
推进更高轮次研究。
```

这里的“更高轮次”不能理解成简单把 `rounds` 从 8 改到 9、10、11 然后硬跑。PRESENT 的高轮 real-vs-random 神经信号会迅速接近随机，盲目增加轮数很容易得到一堆不可解释的 0.50x 结果。

因此本任务的目标应定义为：

```text
围绕 PRESENT/SPN，建立 r8 -> r9 -> r10 的条件式 round-extension ladder，
验证 SPN-aware 数据表示、pair-set evidence pooling、curriculum / transfer、
multi-query aggregation 哪一种路线最可能把弱单样本信号推到更高轮。
```

它不是一个单独的模型实验，而是一个研究阶段。

## 2. 文献边界

当前可稳妥引用的公开锚点：

| 文献 | PRESENT 相关结论 | 对本项目的作用 |
|---|---|---|
| Jain/Kohli/Mishra 2021 | 深度学习差分区分器可到 PRESENT-80 6 轮 | 说明 PRESENT/SPN 标准 DL 区分本来就比 SPECK 路线更难 |
| Zhang/Wang 2022 | PRESENT-80 r7 Case2 `m=16` accuracy `0.7205` | 本项目当前严格同协议 r7 外部 anchor |
| 本项目综述整理 | 存在一些 r8 或更高轮 PRESENT 线索，但常涉及多 pair、E=D、related-key、advanced features 或很弱优势 | 高轮 claim 必须按 taxonomy 分开，不可直接混报 |

本项目不能直接写：

```text
我们要首次做到 PRESENT r8/r9。
```

更严谨的表述是：

```text
我们在 strict encrypted-random-plaintext negatives、Zhang/Wang Case2 m=16、
同协议同尺度证据链下，探索 SPN-structure-adaptive 表示是否能推进
PRESENT real-vs-random neural distinguisher 的可验证轮数。
```

## 3. 为什么不能直接冲 r10

如果 r8 还没有完成 retrieved / validated / gated，直接开 r9/r10 会有三个问题：

1. **缺少条件证据。** 不知道 r8 是还有单样本信号、只剩聚合信号，还是已经接近随机。
2. **不可归因。** r9/r10 失败时无法判断是轮数太高、训练策略不对、数据表示不够，还是模型容量不合适。
3. **浪费 GPU。** 当前 r7 S-box prior 和 r8 round-extension 已经分别占用 watcher/GPU 资源，更高轮应该准备好但等待 gate 触发。

所以本阶段采用：

```text
prepare now, launch conditionally
```

而不是：

```text
launch everything now
```

## 4. 高轮路线假设

### 4.1 单样本弱信号路线

假设：

```text
r9 仍存在微弱但可学习的单样本 real-vs-random 信号。
```

第一候选：

```text
present_nibble_invp_only_spn_only
```

原因：

```text
r7 两个 1M/class seed 已经支持 InvP/P-layer aligned representation；
r8 首轮也把它作为主候选。若 r9 仍有信号，最可能先在这个表示上出现。
```

风险：

```text
r9 单样本 AUC 可能只有 0.50x，262144/class 诊断方差会很大。
```

### 4.2 Pair-set / multi-query 聚合路线

假设：

```text
r9 单个 sample 的 score 很弱，但多个 pair 或多个 query 的 log-odds 聚合仍可形成应用级证据。
```

候选：

```text
present_nibble_invp_pair_consistency_spn_only
frozen single-pair score aggregation
multi-query score aggregation
```

约束：

```text
这类结果只能写 application-level evidence，不能写 raw single-sample SOTA。
```

### 4.3 Curriculum / transfer 路线

假设：

```text
r7/r8 学到的 SPN cell filter、InvP active pattern 或 pair evidence pooling
可以迁移到 r9，比随机初始化更稳定。
```

候选训练方式：

```text
r7 checkpoint -> r8 fine-tune -> r9 fine-tune
r6/r7/r8 mixed-round curriculum -> r9
r8 positive route checkpoint -> r9 weak probe
```

约束：

```text
训练策略变化必须和模型/数据结构变化分开记录。
```

### 4.4 高轮输入差分搜索路线

假设：

```text
当前 Zhang/Wang difference member 0 不一定是 r9/r10 最优输入差分。
```

如果 r8/r9 都弱，下一步应考虑：

```text
固定模型与协议，搜索 PRESENT/SPN 的 high-round candidate input differences；
先做小规模筛选，再对候选差分做 262144/class 验证。
```

这属于 benchmark/difference 变化，必须单独开研究路线，不能和模型结构创新混在一起。

## 5. Round-Extension Ladder

### Stage H0：r8 watcher 结果

当前已启动：

```text
i1_present_r8_round_extension_262k_seed0_gpu0_20260704
```

它是 r9/r10 的 gate source。没有它，不启动 r9 medium run。

### Stage H1：r9 local smoke

目的：

```text
只证明 r9 配置、模型 forward、数据生成、metric 路径能跑。
```

允许现在做：

```text
CPU tiny smoke
```

不允许说：

```text
r9 有效或无效
```

### Stage H2：r9 262144/class weak probe

触发条件：

```text
r8 262144/class 完成 retrieved / validated / gate-note，
且 best candidate AUC > 0.52 或者 r8 pair-set/multi-query 显示应用级信号。
```

矩阵保持 3 行：

| Row | Model | Role |
|---:|---|---|
| 0 | `present_zhang_wang_keras_mcnd` | r9 same-budget baseline |
| 1 | `present_nibble_invp_only_spn_only` | strongest SPN representation |
| 2 | `present_nibble_invp_pair_consistency_spn_only` | pair-set weak-signal pooling candidate |

Gate：

| Result | Decision |
|---|---|
| best candidate AUC `<= 0.505` | 不再从零训练推 r10；改做 curriculum / difference search |
| AUC `0.505 - 0.52` | 只算 near-random weak trace；优先多 seed 方差或 aggregation，不升 1M |
| AUC `> 0.52` 且超过 baseline | r9 weak positive，考虑 seed1 或 curriculum-scale |
| AUC `> 0.55` 且超过 baseline `+0.005` | 强诊断，准备 1M/class seed0 |

### Stage H3：r9 1M/class 或 r9 curriculum

只有 H2 支持时才进入。

可选路径：

```text
Path A: from-scratch r9 1M/class
Path B: r8 checkpoint -> r9 fine-tune
Path C: r7/r8/r9 curriculum
```

优先级：

```text
如果 H2 AUC > 0.55：Path A
如果 H2 AUC 0.505-0.55：Path B/C
```

### Stage H4：r10 weak probe

r10 只在以下条件之一成立后启动：

```text
1. r9 262144/class 有 clear weak positive；
2. r9 curriculum / transfer 明显超过 from-scratch；
3. multi-query aggregation 在 r9 已经稳定高于随机并有应用价值。
```

否则 r10 不作为训练任务启动，只保留为研究方向。

## 6. 当前行动

本轮应做：

```text
1. 写本研究蓝图。
2. 写 r9 weak-probe 实验计划。
3. 准备 r9 smoke CSV 和 r9 262144/class conditional CSV。
4. 跑本地解析/结构测试；可选 tiny CPU smoke。
5. 提交并推送。
6. 不启动 r9 remote，等待 r8 watcher gate。
```

这让项目继续向高轮目标前进，同时不破坏现有 r7/r8 远程闭环。

## 6.1 2026-07-05 更新：r8 已触发 r9 weak-probe

`i1_present_r8_round_extension_262k_seed0_gpu0_20260704` 已经 retrieved /
validated / plotted / gate-note generated。结果显示：

```text
best_model = present_nibble_invp_pair_consistency_spn_only
best_auc = 0.552908501064
baseline_auc = 0.540348751209
delta_vs_baseline = +0.012559749855
decision = support_scale_r8_to_1m_seed0
```

这改变当前高轮路线的状态：

```text
H0 r8 watcher result = complete
H1 r9 smoke = passed
H2 r9 262144/class weak probe = launchable
```

新的研究解释：

```text
r8 信号不是 InvP-only 单独保留下来的，而是 pair-set consistency 明显更强。
因此高轮推进的主要候选暂时应从“单样本 InvP-only”转向
“InvP-aligned pair-set evidence pooling”。
```

下一步：

```text
启动 r9 262144/class weak-probe；
并行准备 r8 1M pair-set confirmation；
如果 r9 仍有弱正信号，再决定 r9 seed1 / r8 pair-set seed1 /
frozen aggregation control / curriculum-transfer 哪个优先。
```

## 7. 参考来源

- Jain, Kohli, and Mishra, *Deep Learning based Differential Distinguisher for Lightweight Block Ciphers*, arXiv:2112.05061: https://arxiv.org/abs/2112.05061
- Zhang and Wang, *Improving Differential-Neural Distinguisher Model For DES, Chaskey, and PRESENT*, arXiv:2204.06341: https://arxiv.org/abs/2204.06341
- 本项目综述：`docs/research/spn_structured_nn_research_plan.md`
- 本项目 r8 阶梯：`docs/research/innovation1-present-round-extension-research-plan.md`
