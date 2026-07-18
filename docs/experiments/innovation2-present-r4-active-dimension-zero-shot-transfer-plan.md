# 创新2 E70：PRESENT四轮unit-profile算子跨活动维度零样本迁移计划

日期：2026-07-18

状态：已完成 / hold / 严格标签未就绪，不解释迁移AUC

## 1. 研究问题

E68确认的`Prefix-Guided Nodewise Profile Operator`只在8个活动bit的coordinate cube上训练和
验证。E70测试其64-node/shared-round归纳偏置能否在不重新训练的情况下迁移到：

```text
4-bit active cube  = 单个活动nibble
12-bit active cube = 三个活动nibble
```

目标仍是PRESENT-80四轮、64个unit输出bit是否对所有key和inactive offset保持XOR平衡。

## 2. 严格标签提供者

每个维度固定16个coordinate structures。标签语义与E43一致：

```text
positive = full active-variable monomial不在sound ANF support over-approximation
negative = 具体PRESENT-80 key和inactive offset使实际unit parity为1
unknown  = 当前证书/反例都无法裁决，不作为negative
```

负类witness bank固定8个确定性key、每structure 4个inactive offset；抽样negative必须用标量
`Present80.encrypt`重放。正类scope仍为所有key/offset，不由witness数量决定。

## 3. 39维跨维度前缀契约

E68输入每个输出node的r1/r2/r3 39维ANF前缀。E70保持宽度不变：

```text
support mean/max/sum/union 以2^active_dimension归一化
degree bins               = degree 0..7 + degree>=8
```

在8-bit fixture上必须与E65保存的39维前缀逐值一致；4-bit高degree bins自然为0，12-bit的
degree>=8折叠到最后一bin。不得新增dimension embedding或修改E68模型参数。

## 4. 零样本矩阵

每个维度先用checkerboard选择正负、structure/output-bit边际精确平衡的transfer evaluation
rows。然后直接读取E67/E68的6个checkpoint：

```text
seed0/seed1 x independent / true-P / fair-corrupted-P
```

同时将E65 train-only ANF-prefix ridge直接应用到新维度，作为确定性迁移锚点。所有模型不做
梯度、不选择epoch、不微调标准化或参数。

## 5. 预注册门

标签与协议门：

```text
每个维度raw positive/negative                  each >= 128
每个维度matched positive/negative              each >= 40
每个维度matched structures/output bits         >= 8 / >= 16
每个structure/output class delta               0
negative scalar replay                          100%
8-bit prefix fixture replay max error           <= 1e-12
checkpoint source/hash/model state              pass
```

零样本迁移门：

```text
每个dimension两seed mean true-P AUC             >= 0.60
每个dimension每seed true-P AUC                  >= 0.55
四个dimension-seed组合mean(true-independent)    >= 0.03
mean(true-corrupted)                            >= 0.03
mean(true-ANF transfer ridge)                   >= 0.02
```

全部通过得到cross-dimension transfer evidence；标签门失败只说明新维度benchmark未就绪；模型
门失败则保留E68的8-bit in-domain结果，不增加容量或在新维度微调。

## 6. 执行与边界

```text
run_id = i2_present_r4_active_dimension_zero_shot_transfer_20260718
output = outputs/local_audits/i2_present_r4_active_dimension_zero_shot_transfer_20260718
execution = local exact label audit + checkpoint inference
training = no
remote GPU = no
```

E70不是高轮区分器、新攻击或SOTA。通过后才考虑把活动维度作为显式条件做统一训练；失败则
不把8-bit模型泛化为任意积分结构。

## 7. 2026-07-18实际结果

E43/E65/E67/E68 source run id、decision、SHA-256与8-bit 39维前缀逐值重放全部通过，前缀
最大绝对误差为`0.0`。本次未训练网络，只读取既有checkpoint。

严格标签结果：

| 活动维度 | provider完成结构 | positive | negative | unknown | matched rows |
|---|---:|---:|---:|---:|---:|
| 4-bit | `16/16` | 0 | 0 | 1024 | 0 |
| 12-bit | `0/16` | 0 | 0 | 1024 | 0 |

4-bit的sound support over-approximation没有证明任何positive，冻结的`8 keys x 4 offsets`
witness bank也没有找到negative，因此全部保持`unknown`。这不是“全部平衡”或“模型失败”。

12-bit的16个结构都在第四轮同一类S-box支持组合处达到：

```text
candidate combinations = 4,741,632
frozen hard cap        = 2,000,000
completed structures   = 0/16
```

实现已在每次笛卡尔组合前检查硬上限，记录round/nibble/output/local-term/structure并继续形成
gate，不再无界增长或异常退出。由于两个维度都没有正负匹配行，结果文件中的`0.5`只是不进入
解释的空集占位值；最终图明确隐藏这类AUC。

```text
status   = hold
decision = innovation2_present_active_dimension_transfer_labels_not_ready
training = no
remote   = no
```

## 8. 裁决与推荐下一步

E70只说明当前E43式strict provider不能审判4/12-bit跨维度迁移，不否定E68在8-bit活动cube
上的双seed域内结果。禁止提高组合cap追结果、把`unknown`改成negative、对4/12-bit微调、训练
dimension embedding或启动远程规模。

下一神经结构实验回到E65/E68同一8-bit严格标签、同一structure-disjoint split和30轮本地预算，
只测试一个新假设：把目前一次拼接的r1/r2/r3前缀改为共享权重的显式round-recurrent处理器。
E68双seedtrue-P checkpoint是只读同预算锚点；候选必须同时带错误轮序和错误P-layer控制。只有
候选逐seed不低于E68超过预注册容差，并对两个控制均有稳定增益，才保留显式轮序结构；否则
保持E68为当前第一名，不增加hidden、epoch或远程规模。

最终`curves.svg`经`visual-qa-redraw`渲染为`1900 x 998`像素检查；标题、标签宽度、provider
完成度、组合cap、裁决和范围均无重叠、裁切、缺字或AUC占位歧义。
