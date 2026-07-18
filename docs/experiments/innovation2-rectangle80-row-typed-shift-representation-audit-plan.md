# 创新2 E91：RECTANGLE row-typed ShiftRow表示无训练审计计划

日期：2026-07-19

状态：已完成 / `pass` / 允许容量配平的Row-Typed Shift Operator readiness

## 1. 研究问题

E90真实P层30轮AUC为`0.890696`，超过独立节点和公平ridge，但相对same-family错误P只有
`+0.029646`，比预注册门少`0.000354`。不得后验调门槛或seed1。

现有算子将同一RECTANGLE S盒column的四个row/lane直接求均值，但ShiftRow对四行使用不同位移
`[0,1,12,13]`。E91不训练模型，只回答：

```text
显式保留row/lane类型，是否提供当前无类型cell均值和错误P/错误row控制无法解释的新增信息？
```

## 2. 冻结来源

```text
profile source = E88 192结构严格unit profile
formal source  = E90 hold
split/labels   = 完全重放E88
training       = none
```

必须验证E88全部来源门，以及E90状态`hold`、裁决
`innovation2_rectangle80_r3_only_topology_not_attributed`、候选质量门全部通过、唯一失败的关系门为
`true_minus_corrupted_at_least_0p03`、三行30轮完成和来源hash。

## 3. 同预算确定性表示

在E88 cell-major索引下，对每个matched `(structure, output node)`读取第3轮13维前缀：

```text
local13       = 目标node
cell_mean13   = 同column四个row的均值
predecessor13 = 指定P层前驱node
```

先重放E89的无类型39维ridge：

```text
untyped = concat(local13, cell_mean13, predecessor13)
```

row-typed 117维表示：

```text
typed_local52 = one_hot(target row, 4) outer local13
typed_pred52  = one_hot(target row, 4) outer predecessor13
typed117      = concat(cell_mean13, typed_local52, typed_pred52)
```

固定五行、全部训练集标准化、`lambda=1e-3`：

```text
untyped_true39
untyped_corrupted39
typed_true117
typed_corrupted117
wrong_row_typed_true117
```

错误row控制保持117维和相同行边际，但对每个结构使用
`wrong_row = (true_row + 1 + structure_index mod 3) mod 4`，破坏跨结构一致row语义；不读取标签。

## 4. 预注册门

协议门：来源、cell-major适配、shape、有限值、训练标准化、维数和E89 untyped true ridge逐值重放。

机制门：

```text
typed true - untyped true >= +0.01
typed true - typed corrupted >= +0.03
typed true - wrong-row typed true >= +0.01
```

全部通过：

```text
decision = innovation2_rectangle80_row_typed_representation_ready
next     = 容量严格配平的Row-Typed Shift Operator两轮readiness
```

任一机制门失败：

```text
decision = innovation2_rectangle80_row_typed_representation_not_ready
next     = 不训练row-typed网络；保留E88/E89/E90证据并停止RECTANGLE结构枚举
```

不得降低margin、换错误row规则、追加特征或根据结果训练网络。协议失败只修复实现。

## 5. 运行与产物

```text
run_id = i2_rectangle80_row_typed_shift_representation_audit_20260719
output = outputs/local_audits/i2_rectangle80_row_typed_shift_representation_audit_20260719
```

生成`results.jsonl`、`gate.json`、`summary.json`、`metadata.json`、`progress.jsonl`和中文
`curves.svg`，刷新最近结果索引并执行`visual-qa-redraw`。E91不构成神经收益、7轮论文复现、高轮、
攻击、远程规模或SOTA结论。

## 6. 实际结果与裁决

E88/E90来源、cell-major重排、3192个matched坐标、E90唯一失败门和E89 untyped true ridge逐值
重放全部通过。五个ridge只使用训练集标准化，矩阵有限且维数符合预注册。

validation AUC：

```text
untyped true39       = 0.824682
untyped corrupted39  = 0.774292
typed true117        = 0.838964
typed corrupted117   = 0.801639
wrong-row typed true = 0.821740
```

机制差值：

```text
typed true - untyped true      = +0.014282
typed true - typed corrupted   = +0.037325
typed true - wrong-row typed   = +0.017224
```

三项预注册门`+0.01/+0.03/+0.01`全部通过：

```text
status   = pass
decision = innovation2_rectangle80_row_typed_representation_ready
training = no
remote   = no
```

这说明row identity的价值不能由额外维数、错误P或同维错误row语义解释，足以开放一个新神经结构
readiness；E91本身没有训练网络。

推荐下一步E92只实现`Row-Typed Shift Operator`：保留四个lane类型，在消息更新中使用共享的
low-rank row调制；candidate、row-corrupted control和untyped anchor必须严格同参数量，数据、13维
输入、hidden、steps、两轮、seed0与E89相同。候选必须同时超过E89 untyped true神经锚点、同参数
错误row、错误P和E91 typed ridge防退化门，才允许30轮。不得直接用四套独立网络扩大容量。

`curves.svg`已通过`visual-qa-redraw`最终2156x1134像素检查，无文字重叠、裁剪、遮挡、缺字、
图例冲突、阈值歧义或不可读内容。
