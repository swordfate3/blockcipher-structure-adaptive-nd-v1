# 创新2 E90：RECTANGLE-80四轮r3-only平衡谱算子30轮seed0正式归因计划

日期：2026-07-19

状态：已完成 / `hold` / 真实P相对错误P差0.000354未过门

## 1. 研究问题

E89两轮readiness中，公平ridge local/corrupted/true为`0.687869/0.774292/0.824682`，神经
independent/corrupted/true为`0.662404/0.757646/0.805034`。真实P同时领先两类神经控制，且只比
更强公平ridge低`0.019649`，全部门通过。

E90只检验：在不改变E88数据、cell-major适配、r3输入、模型容量和控制的条件下，30轮训练后真实
RECTANGLE P层能否稳定超过独立节点、错误P层和公平确定性true-P ridge。

## 2. 冻结来源与矩阵

```text
profile source = E88 192结构严格unit profile
readiness      = E89 pass
fair anchor    = E89 true-P topology-expanded ridge 0.8246824849
```

三行从相同seed0随机初始化重新训练，不恢复E89 checkpoint：

```text
independent = 4,795参数，本节点更新
corrupted   = 4,795参数，同cell + destination-cell shift1错误P
true        = 4,795参数，同cell + 真实RECTANGLE P
```

训练协议：

```text
input dimensions = 13 (r3-only)
hidden / steps    = 32 / 2
dropout           = 0.10
batch size        = 8 structures
epochs            = 30
optimizer         = AdamW, lr=1e-3, weight_decay=1e-4
seed / device     = 0 / local CPU
```

E88的3192个observed坐标、structure split、cell-major重排和checkpoint选择规则全部冻结。每行按
validation AUC选择最佳epoch，不使用E89权重或后验选择错误P。

## 3. 正式门

协议门：E88/E89来源、hash和历史裁决，cell-major适配、三行同参数、masked loss、cell等变、
true/corrupted拓扑不同、30轮完成、指标有限，且模型不读取certificate/witness/parity/label状态。

候选质量门：

```text
true-P validation AUC >= 0.80
true-P train - validation AUC <= 0.15
true-P - E89 fair true-P ridge >= +0.03
```

拓扑归因门：

```text
true-P - independent >= +0.03
true-P - corrupted   >= +0.03
```

全部通过：

```text
status   = pass
decision = innovation2_rectangle80_r3_only_neural_gain_attributed
next     = 完全相同30轮seed1确认
```

任一候选或拓扑门失败：保留E88标签与E89两轮/公平ridge证据，关闭RECTANGLE正式神经路线；不得
增加hidden、steps、epochs、attention或更换较弱错误P。协议失败则修复实现，不解释AUC。

## 4. 运行与产物

```text
run_id = i2_rectangle80_r4_r3_only_profile_operator_attribution_seed0_20260719
output = outputs/local_diagnostic/i2_rectangle80_r4_r3_only_profile_operator_attribution_seed0_20260719
```

必须生成`results.jsonl`、`history.csv`、三个checkpoint、`gate.json`、`summary.json`、
`metadata.json`、`progress.jsonl`和中文`curves.svg`，刷新最近结果索引并执行真实像素
`visual-qa-redraw`。即使通过也只构成RECTANGLE-80四轮seed0正式归因，不是双seed、7轮论文复现、
跨密码checkpoint迁移、高轮区分器、攻击、远程规模或SOTA结论。

## 5. 实际结果与裁决

E88/E89来源、全部hash与历史门、cell-major适配、3192个配平坐标、三行参数公平、masked loss、
cell重标号和30轮协议全部通过。三行均为`4795`参数，真实P最佳checkpoint出现在epoch 27。

30轮最佳validation AUC：

```text
independent node    = 0.726366  (best epoch 12)
corrupted P         = 0.861051  (best epoch 29)
true RECTANGLE P    = 0.890696  (best epoch 27)
E89 fair true ridge = 0.824682
```

正式差值：

```text
true - independent = +0.164331
true - corrupted   = +0.029646
true - E89 ridge   = +0.066014
true train - validation = +0.004165
```

候选质量三门全部通过，真实P也显著超过独立节点和公平ridge；但预注册拓扑门要求
`true - corrupted >= +0.030000`，实际只为`+0.029646`，短少`0.000354`。正式裁决严格保持：

```text
status   = hold
decision = innovation2_rectangle80_r3_only_topology_not_attributed
seed1    = no
remote   = no
```

不得把三位小数四舍五入后的`0.030`解释为过门，也不得事后放宽阈值、挑选更弱错误P或增加epoch。
E89的两轮正向readiness与E90的高绝对质量仍是有效诊断，但当前无类型cell均值的r3-only算子没有
获得预注册的正式RECTANGLE拓扑归因。

## 6. 推荐下一步

不运行seed1，不机械调hidden、steps、dropout、epoch或错误P。若继续RECTANGLE神经结构探索，
下一问题必须针对现有模型的具体结构盲点：RECTANGLE四个S盒lane对应不同ShiftRow位移
`[0,1,12,13]`，而当前算子把同cell四个lane直接求均值，丢失了row/lane类型。先做一个无训练的
row-typed表示与公平确定性基线审计；只有它证明row identity提供错误P无法解释的新增信息，才允许
另立同参数或严格容量配平的`Row-Typed Shift Operator`两轮readiness。否则保留PRESENT/GIFT双密码
正式方法和RECTANGLE标签/贴线诊断，不继续网络枚举。

`curves.svg`已执行`visual-qa-redraw`。首次像素检查发现`0.029646`被三位小数显示成`+0.030`，
可能误导为过门；改为四位小数并明确标注`+0.0296 < +0.0300`后重新渲染。最终2156x1142像素
无文字重叠、裁剪、遮挡、缺字、图例冲突或裁决歧义。
