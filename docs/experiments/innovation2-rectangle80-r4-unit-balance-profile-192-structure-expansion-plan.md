# 创新2 E88：RECTANGLE-80四轮严格unit平衡谱192结构容量复核计划

日期：2026-07-19

状态：已完成 / `pass` / 允许RECTANGLE r3-only本地神经readiness

## 1. 研究问题

E87已经在96个结构上通过RECTANGLE-80最终版四轮严格unit-profile标签、matching和反捷径门。
E88不改变密码、轮数或标签，只回答：

```text
把结构库从96扩到192后，严格标签宽度、structure-disjoint覆盖和反捷径性质能否稳定保持，
从而足以开放RECTANGLE r3-only神经结构readiness？
```

E88不训练神经网络，不启动远程GPU。

## 2. 唯一变量与同预算锚点

```text
anchor structures         = 96  (E87)
candidate structures      = 192 (E88唯一变量)

cipher / rounds           = final RECTANGLE-80 / 4
active dimension          = 8 coordinate bits
outputs                   = 64 unit bits
witness keys              = 16 deterministic 80-bit keys
offsets per structure     = 8
checkerboard attempts     = 64
structure seed            = 20260718
key seed                  = 8701
offset seed               = 18701
split                     = structure index mod 4
execution                 = local CPU / no training
```

结构生成顺序必须保证E88前96个结构与E87逐项相同。E88必须读取E87本地产物并验证：

```text
E87 status = pass
E87 decision = innovation2_rectangle80_unit_profile_ready
first 96 structure definitions equal
first 96 x 64 ternary labels equal
first 96 x 64 x 39 prefix features equal
```

任一重放失败都属于协议无效，不解释扩容指标。

## 3. 标签与benchmark冻结

三态标签保持不变：

```text
positive = sound ANF-support上界中不存在完整8变量单项式
negative = 具体80-bit key与inactive offset使unit cube XOR=1
unknown  = 当前证书与witness bank都未裁决，不参与matching或训练
```

不得增加key或offset、把unknown当negative、改变四轮、降低门槛或重新定义split。checkerboard仍要求
每个选中structure和output bit内部正负精确平衡。

## 4. 预注册门与裁决

沿用E87全部门槛：

```text
raw positive/negative                         each >= 256
resolved positive prevalence                  in [0.10, 0.90]
mixed structures                              >= 32
distinct ternary signatures                   >= 4
matched train positive/negative                each >= 150
matched validation positive/negative           each >= 50
matched total/validation structures            >= 32 / 8
matched validation output bits                 >= 16
strongest validation unary marginal AUC        <= 0.65
duplicate edges / structure delta / bit delta  = 0
```

全部协议、宽度和反捷径门通过：

```text
status   = pass
decision = innovation2_rectangle80_unit_profile_expansion_ready
next     = RECTANGLE r3-only三行两轮神经readiness
```

神经矩阵只允许：

```text
independent node operator
true RECTANGLE P-layer message operator
same-family fair-corrupted RECTANGLE P-layer operator
```

协议有效但任一宽度或反捷径门失败：

```text
status   = hold
decision = innovation2_rectangle80_unit_profile_expansion_not_ready
next     = 关闭当前RECTANGLE四轮unit-profile神经路线，不继续384结构机械扩容
```

协议重放失败则`fail`并先修复，不得解释扩容结果。E88不能支持神经性能、7轮论文复现、高轮区分器、
跨密码泛化、攻击或SOTA结论。

## 5. 运行与产物

```text
run_id = i2_rectangle80_r4_unit_balance_profile_192_structures_20260719
anchor = outputs/local_audits/i2_rectangle80_r4_unit_balance_profile_readiness_20260719
output = outputs/local_audits/i2_rectangle80_r4_unit_balance_profile_192_structures_20260719
```

产物沿用E87：`structures.json`、`atlas.jsonl`、`profile_targets.npy`、`profile_observed.npy`、
`prefix_features.npy`、`matched_unit_contrast.csv`、`results.jsonl`、`gate.json`、`summary.json`、
`metadata.json`、`progress.jsonl`、`curves.svg`和`visual_qa_passed.marker`。完成后刷新最近结果索引，
并对最终SVG执行真实像素`visual-qa-redraw`检查。

## 6. 实际结果与裁决

E88按冻结协议完成：

```text
raw positive / negative / unknown = 9995 / 1791 / 502
resolved positive prevalence      = 0.8480400475
mixed structures                  = 192 / 192
distinct ternary signatures       = 191

train matched                     = 1208 / 1208
validation matched                = 388 / 388
matched total structures          = 192
validation structures / outputs   = 48 / 41
```

E87锚点状态和裁决、前96个结构定义、`96 x 64`三态标签和`96 x 64 x 39`前缀特征逐项
重放通过。最终版零向量、S-box ANF、轮常量、P-layer、16项向量/标量对拍和24个抽样negative
witness也全部通过。

checkerboard控制为：

```text
duplicate edges / structure delta / output delta = 0 / 0 / 0
global / output-bit / active-bit / strongest AUC  = 0.5 / 0.5 / 0.5 / 0.5
```

正式裁决：

```text
status   = pass
decision = innovation2_rectangle80_unit_profile_expansion_ready
remote   = no
training = no
```

这证明RECTANGLE-80四轮严格unit-profile数据在192结构上保持sound、宽、structure-disjoint且无
一元位置捷径，足以开放下一阶段本地神经readiness；它本身不是神经结果。推荐下一步只运行
RECTANGLE r3-only三行两轮矩阵：`independent / true P-layer / same-family corrupted P-layer`，
固定E88数据、13维第3轮前缀输入、共享node operator容量和相同训练预算。若真实P不能同时超过两类
控制，则关闭RECTANGLE神经分支，不进入30轮、seed1或远程GPU；只有两轮门通过才预注册30轮seed0。

`curves.svg`已按`visual-qa-redraw`渲染为2156x1102像素检查，未发现标题/标签重叠、裁剪、遮挡、
缺字、结构歧义或不可读内容。
