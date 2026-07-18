# 创新2 E51：PRESENT四轮CGPR 30轮seed0正式残差归因计划

日期：2026-07-18

状态：完成 / hold / CGPR候选未过门

## 1. 研究问题

E50确认CGPR满足零残差等价、ridge冻结、参数公平、正确/错误P内部表示不同和有限训练契约。
E51正式判断：在E43严格structure-disjoint标签上，正确P-layer pair-state残差是否真正超过
确定性ridge、同预算prefix-only非线性残差和fair-corrupted-P残差。

两轮E50 AUC不用于修改架构、阈值或选择分支。E51是当前E43四轮新网络枚举的终局门。

## 2. 冻结数据与模型

```text
cipher             = PRESENT-80
rounds             = 4
source             = E43 matched checkerboard
train              = 800 rows（400/400）
validation         = 236 rows（118/118）
split              = structure-disjoint
base features      = E45 r1/r2/r3 39维ANF-prefix
base               = train-only ridge lambda 1e-3（冻结）
ridge AUC           = 0.6860815857512209
residual bound      = 0.25 * tanh(raw residual)
pair processor      = E44 triangle
hidden              = 16
path rank           = 2
dropout             = 0.10
epochs              = 30
batch               = 32
optimizer           = AdamW(lr=1e-3, weight_decay=1e-4)
checkpoint          = best validation AUC
seed                = 0
device              = local CPU
```

不得改变E43 label/split、39维定义、ridge拟合、残差上限、模型宽度、P-layer控制、metric或
checkpoint协议。禁止第4轮full-cube oracle、certificate、witness、key/offset parity和
validation拟合。

## 3. 冻结矩阵

```text
0. E45 ANF-prefix ridge anchor                  只读/重新核验
1. ridge + prefix-only residual                同预算非线性容量控制
2. ridge + true-P pair-state triangle residual CGPR候选
3. ridge + fair-corrupted-P triangle residual  错误transport控制
```

三行残差模型保持相同seed、optimizer、batch、epoch、checkpoint和残差上限；prefix/pair有效
参数差继续不得超过`1%`。true/corrupted pair参数形状与初始化相同，只改变P-layer buffer。

## 4. 正式门

协议门重新核验E43/E44/E45/E49/E50 run id、decision、hash、ridge AUC、零等价、冻结base、
参数、禁用输入、拓扑embedding delta、30轮完成与有限metric。

候选门：

```text
true CGPR validation AUC              >= 0.70
true CGPR - E45 ridge                 >= 0.02
```

残差归因门：

```text
true CGPR - prefix-only residual       >= 0.02
true CGPR - fair-corrupted-P residual  >= 0.03
```

全部通过：

```text
decision = innovation2_present_cgpr_topology_attributed
next     = 同矩阵本地seed1确认
```

候选门失败：停止CGPR和E43四轮新网络枚举。候选过但不超过prefix-only：说明收益来自确定性
prefix上的普通非线性修正，不归因pair-state；保留为控制结果但不升主创新。候选过但不超过
错误P：撤回拓扑归因并停止pair路线。任何协议门失败只修协议，不解释AUC。

任何hold均不增加hidden、path-rank、残差上限、epoch或新processor，不运行seed1、r5或远程
GPU。只有全部正式门通过才允许本地seed1。

## 5. 产物与声明范围

```text
outputs/local_diagnostic/i2_present_r4_cgpr_neural_attribution_seed0_20260718/

results.jsonl
history.csv
gate.json
summary.json
metadata.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

声明范围：PRESENT-80四轮、E43严格标签、本地seed0的CGPR正式残差与拓扑归因；不是高轮
积分区分器、新攻击、远程规模结果或SOTA。

## 6. 2026-07-18实际结果

权威run：

```text
i2_present_r4_cgpr_neural_attribution_seed0_20260718
```

E43/E44/E45/E49/E50 source、hash、标签、split、ridge、零残差、冻结base、参数、禁用输入、
拓扑embedding和三行30轮流程检查全部通过。结果：

| 行 | 最佳epoch | train AUC | validation AUC |
|---|---:|---:|---:|
| E45 ANF-prefix ridge | 0 | `0.777216` | `0.686082` |
| ridge + prefix-only residual | 2 | `0.769416` | `0.703174` |
| ridge + true-P pair residual | 2 | `0.777325` | `0.685938` |
| ridge + fair-corrupted-P pair residual | 2 | `0.777325` | `0.685938` |

冻结差值：

```text
true pair - ridge          = -0.000144
true pair - prefix-only    = -0.017236
true pair - corrupted pair = +0.000000
```

true pair没有达到`0.70`，没有超过ridge`0.02`，没有超过prefix-only`0.02`，也没有超过错误P
`0.03`。协议有效但候选门首先失败：

```text
status   = hold
decision = innovation2_present_cgpr_candidate_not_ready
seed1    = no
remote   = no
```

prefix-only在seed0达到`0.703174`，但它只是同预算容量控制，优势相对ridge为`0.017093`，低于
预告的`0.02`实质margin，且不含新的pair/topology贡献。不得把它后验升格成主创新或直接跑
seed1。

## 7. 推荐下一步

停止CGPR和E43四轮新网络枚举。具体关闭：MSPN扩容、identity token、Monomial Transformer、
NBFNet、CGPR调参、prefix residual seed1、更长epoch、r4新processor和远程GPU。保留：

```text
E45/E48确定性ANF/degree解释锚点  validation AUC约0.69
E44真实神经pair-state锚点       validation AUC 0.561979
E47--E51受控失败边界             说明神经传播/残差未增加组外价值
```

下一瓶颈不是四轮架构，而是五轮严格标签覆盖。执行E52无训练审计：比较当前sound ANF support
提供者与更精确、仍可验证的division-property/monomial-certificate提供者，判断PRESENT-80五轮
能否生成足够正负类、structure-disjoint、边际匹配的数据。只有五轮标签达到与E43同级宽度、
证书/反例复验和反捷径门，才重新开放网络；否则把创新2冻结为四轮方法学案例与负结果边界。

最终`curves.svg`按`visual-qa-redraw`渲染为`1800×941`像素检查；四行正式AUC、`0.70`门、
训练/验证差距、三个控制差值、裁决与证据范围无重叠、裁切、缺字或掩盖正确/错误P相同的
视觉编码，已记录`visual_qa_passed.marker`。
