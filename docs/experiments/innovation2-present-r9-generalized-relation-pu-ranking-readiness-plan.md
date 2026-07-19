# 创新2 E98：PRESENT九轮广义积分关系正例-未标注排序就绪审判

日期：2026-07-19

状态：已完成 / 暂缓 / E99关闭

## 1. 研究问题

E97证明：在冻结语义和本地复杂度上限内，当前开放provider不能为真实PRESENT-80五轮查询生成
可用于二分类监督学习的非平凡严格正负标签。E98不绕过这个结论，也不再把“没有出现在论文结果中”
解释成负类，而是审查一个更窄的九轮任务：

```text
冻结ATM九轮结果中的已知广义积分关系，能否在来源分组完全隔离后，
形成一个不被文件、重量、位置或坐标重合捷径支配的正例-未标注候选排序基准？
```

输入是广义关系候选及其指数、支撑和位置摘要；输出只允许是候选排序分数。已知ATM关系是正例，
其余匹配候选一律称为`unlabeled`，不得称为不平衡、非积分或严格负例。

## 2. 冻结来源与声明边界

```text
source repository = AlgebraicTransitionMatrices
source commit     = b2ffbb2bf0ef8f2ffabe3203896006874aa1c40b
cipher layers     = PRESENT round function
rounds            = 9
key model         = independent 64-bit round keys
result files      = 8 R9-complex-oracle-*.pkl
loader            = project RestrictedUnpickler
```

E98必须同时报告：序列化关系数、并集GF(2)秩、论文维数、来源频率、singleton/multi-term数、
留出关系数和坐标重合数。独立轮密钥下对所有赋值成立的正关系只有在轮数与bit convention另行验证后，
才可限制到PRESENT-80调度密钥；本实验不完成该迁移验证。未知常数不得解释成零平衡。

## 3. 分组与无泄漏协议

以8个结果文件作为8个来源组，逐文件留一：

```text
train positives   = 其余7个文件中的关系并集
heldout positives = 仅存在于当前留出文件、且未在train出现的完整关系
```

同时记录heldout关系的每个坐标是否在train支撑中出现。完整关系在train出现时必须从heldout删除；
若有效留出正例不足、只有少数组有正例、或支撑坐标泄漏无法排除，任务不得进入神经训练。

## 4. 未标注候选池

每个heldout positive形成一个独立排序池。对该关系的所有64位输入指数和输出指数同步循环平移
`1..63`位，生成确定性位置变体，并执行以下过滤：

```text
exclude exact duplicate of the heldout positive
exclude every relation in the complete known-positive union
deduplicate canonical GF(2) coordinate sets
```

该变换逐候选严格保持：关系项数、输入指数重量多重集、输出指数重量多重集、不同输入/输出指数数、
输入和输出内部异或距离、二乘二Cartesian关系形状。每池目标为`1 positive + >=31 unlabeled`。
候选只是边缘匹配的未知关系；E98不验证其密码学真值。

## 5. 同预算基线与指标

每个排序池冻结比较：

```text
deterministic hash random
relation size
input/output exponent weight
exact relation frequency in training files
training coordinate frequency
training support overlap
input/output absolute bit-position summary
```

报告`Recall@1`、`Recall@5`、平均倒数排名、top-5 enrichment，并单列最佳非随机捷径。
这些指标只衡量已知正例恢复，不计算二分类accuracy/AUC，也不把unlabeled当作0标签训练。

## 6. 预注册裁决门

```text
advance_to_E99_local_neural_ranking:
  8个冻结文件和SHA256全部重放；
  470个序列化关系及论文维数契约无未解释漂移；
  至少6/8个留出组各有>=8个严格未见正例；
  总heldout positives >=64；
  heldout完整关系及支撑坐标均不泄漏到train；
  每个正例有>=31个严格边缘匹配的unlabeled候选；
  unlabeled池不包含任何已知positive；
  最佳非随机捷径Recall@5 <=0.50且MRR <=0.35。

hold:
  来源秩/维数契约仍未解释，或有效组/正例宽度不足，或存在不可去除的泄漏，
  或候选边缘无法匹配，或简单捷径已主导恢复。
  -> 不训练E99，不启动远程GPU；优先寻找更多独立九轮正关系来源或可验证关系生成器。

protocol_invalid:
  commit、文件集、hash、安全反序列化、候选确定性或已知正例排除失败。
  -> 只修协议，不解释科学结果。
```

阈值在读取最终审计结果前冻结。它们要求的不只是“能拼出候选”，而是足够多的独立组和正例，
避免24条同族关系之类的小样本被误包装成高轮神经突破。

## 7. 执行与产物

```text
run_id  = i2_present_r9_generalized_relation_pu_ranking_readiness_20260719
device  = local CPU
epochs  = 0
seeds   = none
training = no
remote   = no
```

输出目录：

```text
outputs/local_audits/i2_present_r9_generalized_relation_pu_ranking_readiness_20260719/
```

产物必须包含`results.jsonl`、`gate.json`、`summary.json`、`metadata.json`、`progress.jsonl`、
`source_hashes.json`、`folds.csv`、`ranking_baselines.csv`和`curves.svg`。完成后更新本文件正式结果，
运行`visual-qa-redraw`，并刷新`outputs/00_RECENT_RESULTS.md/json`。

## 8. 正式结果

权威run：

```text
i2_present_r9_generalized_relation_pu_ranking_readiness_20260719
```

冻结ATM commit、8个文件名、8个SHA256、安全反序列化、470个去重序列化关系和并集秩468均
成功重放。8个文件总共序列化了3039条关系引用，但去重后只有470条；其中446条在多个文件
重复，只有24条只属于一个文件。

逐文件留一结果：

| 留出split | 严格未见正例 | 达到每组8条 |
|---|---:|---:|
| 1-5-3 | 0 | 否 |
| 1-6-2 | 3 | 否 |
| 1-7-1 | 18 | 是 |
| 2-4-3 | 0 | 否 |
| 2-5-2 | 0 | 否 |
| 2-6-1 | 3 | 否 |
| 3-4-2 | 0 | 否 |
| 3-5-1 | 0 | 否 |

24条留出正例全部是4项multi-term关系，没有singleton；它们的96个关系坐标均未在对应train
关系中出现。候选构造也有效：每条正例生成62--63条同步循环平移的unlabeled关系，已知正例
混入为0，关系项数、输入/输出指数重量、不同指数数、异或距离和Cartesian形状漂移均为0。

同预算基线中最强的是绝对bit位置：

```text
best shortcut Recall@5 = 0.166666667
best shortcut MRR      = 0.099406779
shortcut gate          = pass
```

因此当前失败不是简单捷径已经把正例找完，而是统计和来源宽度不够：只有3/8组存在任何留出
正例，只有1/8组达到8条，总正例24条，均低于预注册的6组、每组8条、总计64条门槛。同时
470个序列化关系的并集GF(2)秩仍为468，与论文报告维数470不一致。

冻结裁决：

```text
status   = hold
decision = innovation2_present_r9_pu_ranking_benchmark_not_ready
E99      = closed
training = no
remote   = no
```

## 9. 证据范围与推荐下一步

E98证明的是“当前公开ATM结果不足以组成可靠的九轮神经排序基准”，不是“九轮广义积分关系不可
学习”。尤其不能把446条跨文件重复关系反复计作独立验证样本，也不能把循环平移候选当作负类。

下一步不训练E99。先做一个确定性来源审计，精确定位470个序列化关系中的两条GF(2)依赖，检查
它来自跨split基底合并、坐标约定还是结果文件问题；该审计只有在把470/468差异解释清楚后，才
能解除维数契约阻塞。即使秩问题解决，仍需新增至少5个达到8条严格留出正例的独立来源组，并使
总留出正例达到64，才重新运行E98。禁止机械扩充循环平移候选、重复采样现有24条正例、把
unlabeled改名为negative、直接训练E99或启动远程GPU。

可视化`curves.svg`已通过`visual-qa-redraw`：最终SVG按`2460 x 1338`像素渲染检查，标题、说明、
坐标轴、标签、阈值线、图例和底部裁决均无重叠、裁切、缺字或含义歧义。
