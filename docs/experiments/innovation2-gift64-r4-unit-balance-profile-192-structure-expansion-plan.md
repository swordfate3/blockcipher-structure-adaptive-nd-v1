# 创新2 E75：GIFT-64四轮严格unit平衡谱192结构容量复核计划

日期：2026-07-19

状态：已完成 / pass / 允许GIFT r3-only本地神经readiness

## 1. 研究问题

E74的密码学协议、raw标签宽度和反捷径门全部通过，但96结构的2x2 checkerboard只有
`92/92`条train正负边和`28/28`条validation正负边，低于冻结`150/150`和`50/50`门。
同一标签矩阵按output bit独立配平的理论上界仅为train 318边、validation 100边，而门要求总边
300与100，说明当前结构库几乎没有matching余量。

E75只回答：保持标签语义、有限witness预算和所有门不变时，把结构库从96扩到192，能否形成
稳健的严格GIFT-64四轮unit-profile benchmark。它不训练神经网络。

## 2. 单变量与同协议锚点

```text
anchor structures        = 96  (E74)
candidate structures     = 192 (E75唯一变量)
coordinate nibble pairs  = 24  (不变)
deterministic random     = 168 (新增后96个结构)

cipher / rounds          = GIFT-64 / 4
active dimension         = 8 coordinate bits
outputs                  = 64 unit bits
witness keys             = 16 deterministic 128-bit keys
offsets per structure    = 8
structure/key/offset seed= 与E74完全相同
split                    = structure index mod 4
checkerboard attempts    = 64
execution                = local CPU / no training
```

构造顺序必须保证E75前96个结构与E74逐项相同。E75必须读取E74本地产物并验证：

```text
first 96 structure definitions equal
first 96 x 64 ternary labels equal
first 96 x 64 x 39 prefix features equal
E74 anchor gate status = hold
E74 anchor decision = innovation2_gift64_unit_balance_profile_not_ready
```

任一重放失败都属于协议无效，不解释E75宽度。

## 3. 标签与控制冻结

三态标签继续为：

```text
positive = sound active-variable ANF support上界不存在完整8变量单项式
negative = 具体scheduled 128-bit key与inactive offset使unit cube XOR=1
unknown  = 当前证书与witness bank均未裁决，不参与训练
```

不得增加key或offset、不得把unknown改为negative、不得降低门槛。train/validation仍分别由标签盲
2x2 checkerboard选择，使每个selected structure和output bit内部正负精确平衡。

## 4. 预注册门与裁决

沿用E74全部协议、宽度和反捷径门：

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

全部通过：

```text
decision = innovation2_gift64_unit_balance_profile_expansion_ready
next = GIFT r3-only Prefix-Guided Profile Operator最小三行两轮readiness
```

三行固定为：

```text
independent node MLP
true GIFT P-layer message operator
same-family fair-corrupted GIFT P-layer operator
```

若协议有效但宽度或反捷径仍失败：

```text
decision = innovation2_gift64_unit_balance_profile_expansion_not_ready
next = 关闭当前GIFT四轮unit-profile迁移；转SKINNY严格标签或新的sound表示
```

不得继续384结构机械扩展，也不得用本地readiness结果宣称跨密码泛化、高轮区分器、攻击或SOTA。

## 5. 运行与产物

```text
run_id = i2_gift64_r4_unit_balance_profile_192_structures_20260719
anchor = outputs/local_audits/i2_gift64_r4_unit_balance_profile_readiness_20260718
output = outputs/local_audits/i2_gift64_r4_unit_balance_profile_192_structures_20260719
```

产物沿用E74的`atlas.jsonl`、`matched_unit_contrast.csv`、`structures.json`、三个NumPy数组、
`results.jsonl`、`gate.json`、`summary.json`、`metadata.json`、`progress.jsonl`和`curves.svg`。
完成后必须刷新最近结果索引并对SVG执行真实像素`visual-qa-redraw`检查。

## 6. 实际结果

E75按冻结协议完成：

```text
raw positive / negative / unknown = 6087 / 3286 / 2915
resolved positive prevalence      = 0.649419
mixed structures                  = 168 / 192
distinct ternary signatures       = 187

train matched                     = 248 / 248
validation matched                = 62 / 62
matched total structures          = 143
validation structures / outputs   = 33 / 39
```

所有锚点重放均通过：E74状态与裁决一致，前96个结构定义、`96 x 64`三态标签和
`96 x 64 x 39`前缀特征逐项相同。官方GIFT向量、S-box ANF、四轮vectorized/scalar、
24个negative witness标量复验、shape与split检查也全部通过。

checkerboard仍满足：

```text
duplicate edges / structure delta / output delta = 0 / 0 / 0
global / output-bit / active-bit validation AUC   = 0.5 / 0.5 / 0.5
```

E74失败的两个门已在不改阈值下通过：

```text
train each class >= 150       248 / 248
validation each class >= 50   62 / 62
```

正式裁决：

```text
status   = pass
decision = innovation2_gift64_unit_balance_profile_expansion_ready
remote   = no
```

这只证明GIFT-64四轮严格unit-profile数据与反捷径控制足以支持本地训练，不证明神经网络有效、
跨密码泛化、高轮区分器或攻击。下一步按预注册只运行E76两轮三行readiness，并同时报告r3-only
与完整39维train-only ridge基线；若readiness不过门，不进入30轮。

`curves.svg`已通过`visual-qa-redraw`：最终SVG渲染为1600x812像素，未发现标题/标签重叠、
裁剪、缺字、图例冲突、坐标歧义或不可读内容。
