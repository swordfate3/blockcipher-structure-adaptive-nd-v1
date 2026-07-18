# 创新2 E51：PRESENT四轮CGPR 30轮seed0正式残差归因计划

日期：2026-07-18

状态：计划冻结 / 待执行

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
