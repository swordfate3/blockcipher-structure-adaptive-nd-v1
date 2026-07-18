# 创新2 E79：GIFT-64四轮r3-only平衡谱算子30轮seed1确认计划

日期：2026-07-19

状态：已完成 / pass / GIFT-64四轮双seed神经拓扑归因确认

## 1. 研究问题

E78 seed0真实GIFT P validation AUC为`0.913111`，相对independent、corrupted和E77公平ridge
分别领先`+0.341831/+0.138398/+0.169615`。E79只检验该正式归因是否在独立训练seed1复现。

## 2. 唯一变量

```text
changed = training seed 0 -> 1
```

以下全部不变：

```text
E75 labels/split/620 observed coordinates
r3-only 13-dimensional input
independent / corrupted shift1 / true GIFT P
4,795 parameters each
hidden=32, steps=2, dropout=0.10
30 epochs, batch=8
AdamW lr=1e-3, weight_decay=1e-4
best checkpoint by validation AUC
local CPU, no checkpoint transfer
```

必须读取并验证E78 gate/results/hash，确认seed0状态pass、三行存在、全部候选/关系/协议门通过。

## 3. seed1与联合门

seed1独立质量门：

```text
true-P validation AUC >= 0.80
true-P train - validation AUC <= 0.15
true-P - E77 true-P topology-expanded ridge >= +0.03
```

seed1拓扑归因门：

```text
true-P - independent >= +0.03
true-P - corrupted   >= +0.03
```

联合门要求E78 seed0和E79 seed1逐seed全部通过；平均值只报告，不用于挽救任何单seed失败。

全部通过：

```text
decision = innovation2_gift64_r3_only_two_seed_confirmed
```

否则：

```text
decision = innovation2_gift64_r3_only_seed_not_replicated
```

失败后不得追加seed、调参或更换错误P。通过后也只证明同一个GIFT-64四轮严格unit-profile任务上的
双seed神经拓扑归因，不是高轮、跨密码零样本泛化、攻击或SOTA。

## 4. 运行与产物

```text
run_id = i2_gift64_r4_r3_only_profile_operator_seed1_20260719
output = outputs/local_diagnostic/i2_gift64_r4_r3_only_profile_operator_seed1_20260719
```

必须生成`results.jsonl`、`history.csv`、三个checkpoint、`gate.json`、`summary.json`、
`metadata.json`、`progress.jsonl`和中文`curves.svg`，刷新最近结果索引并执行真实像素
`visual-qa-redraw`。完成后更新架构排名与下一步建议。

## 5. 实际结果

E75/E78来源、hash、620坐标、三行参数公平、masked loss、cell等变与seed1训练协议全部通过。

逐seed validation AUC：

```text
seed0:
  independent = 0.571280
  corrupted   = 0.774714
  true        = 0.913111

seed1:
  independent = 0.569719
  corrupted   = 0.784599
  true        = 0.911030
```

逐seed真实P增益：

```text
true - independent = +0.341831 / +0.341311
true - corrupted   = +0.138398 / +0.126431
true - E77 ridge   = +0.169615 / +0.167534
```

双seed平均：

```text
mean true AUC             = 0.912071
mean true - independent   = +0.341571
mean true - corrupted     = +0.132414
mean true - E77 ridge     = +0.168574
```

seed1候选、拓扑、联合和全部协议门均通过：

```text
status   = pass
decision = innovation2_gift64_r3_only_two_seed_confirmed
remote   = no
```

因此可冻结GIFT-64四轮结果：在严格全称正类/具体反例负类、structure-disjoint checkerboard和
一元边际0.5控制下，r3-only真实P算子在两颗seed都稳定超过独立node、same-family错误P和信息范围
公平的确定性拓扑ridge。

证据边界必须保留：每颗seed都在GIFT标签上从头训练；没有PRESENT checkpoint迁移或零样本测试；
任务只有四轮。它不是高轮积分区分器、跨密码泛化、攻击或SOTA。

`curves.svg`已通过`visual-qa-redraw`最终1600x848像素检查，无字体重叠、裁剪、缺字、图例冲突、
门线歧义或不可读内容。

## 6. 推荐下一步

停止同一GIFT benchmark继续扩容量、加seed或机械扩结构。下一步E80应先做无训练的PRESENT/GIFT
方法级证据综合：统一列出严格标签语义、结构库差异、参数、双seed AUC与真实P控制margin，形成可写
论文的claim boundary。随后只在第三真实SPN（优先SKINNY-64）通过同级严格标签与反捷径门后才训练；
若第三密码标签仍不可用，创新2以PRESENT+GIFT双密码方法证据收束，不用模型搜索替代标签缺口。
