# 创新2 E84：SKINNY-64五轮真实拓扑ridge引导稀疏神经残差计划

日期：2026-07-19

状态：已完成 / hold / 当前SKINNY神经搜索收束

## 1. 研究问题

E83确认真实SKINNY稀疏图的显式39维展开ridge达到`0.862045`，显著超过local与错误图；但端到端
SLPO两轮true AUC只有`0.794524`，没有超过独立或错误图。E84不增加epoch或容量，只回答：

```text
把train-only真实拓扑ridge冻结为base score后，
小型稀疏图神经分支能否学习到超过该强确定性基线的有界残差？
```

暂定名为`True-Ridge-Guided Sparse Residual`（TRG-SR）。它是待验证组合，不预先宣称有效创新。

## 2. 冻结来源

```text
label source = E82五轮严格unit-profile标签
route source = E83稀疏算子readiness gate
```

E84必须验证E83为`hold / innovation2_skinny64_sparse_profile_readiness_not_passed`，并重算：

```text
true sparse39 ridge validation AUC = 0.8620446578265508
```

不得直接读取validation标签调ridge、残差结构或阈值。

## 3. 冻结base score

以E83相同的39维显式特征：

```text
local13 + same-cell mean13 + true sparse predecessor mean13
```

只在train observed edge上拟合`lambda=1e-3` ridge，保存mean、scale、intercept和weights为
`requires_grad=False` buffer。对每个结构的64个node生成base score。零残差时，模型输出必须逐项
等于ridge score，误差不超过`1e-7`；训练后ridge buffer最大变化必须为0。

## 4. 神经残差

残差分支保持E83的4,795可学习参数、13维r4-only输入、hidden 32和2步共享块。最终输出：

```text
logit = frozen_true_ridge_score + 0.25 * tanh(neural_residual)
```

残差head零初始化，使epoch0严格等于ridge。固定`0.25`边界防止神经分支任意覆盖确定性基线。

## 5. 四行矩阵

| 行 | base | residual relation | 训练 |
|---|---|---|---|
| true ridge anchor | frozen true sparse39 ridge | off | no |
| independent residual | 同一true ridge | independent | yes |
| true sparse residual | 同一true ridge | true SKINNY graph | yes |
| corrupted sparse residual | 同一true ridge | same-degree wrong graph | yes |

三条神经行使用相同base、参数、初始化、batch、optimizer和epoch；只有残差relation不同。这比让每条
控制使用不同ridge base更严格，因为任何margin只能来自神经残差关系。

## 6. 协议

```text
epochs       = 2
seed         = 0
batch size   = 8 structures
optimizer    = AdamW
lr           = 1e-3
weight decay = 1e-4
residual cap = 0.25
device       = cpu
```

每条训练行允许epoch0作为best checkpoint，防止训练把ridge基线破坏后仍被误报；但推进必须有正增益。

## 7. 门控

协议门：

1. E82/E83来源、hash、标签、r4切片和真实/错误图重放；
2. ridge AUC逐项复现E83；
3. 零残差三条神经行与ridge最大误差`<=1e-7`；
4. ridge buffers冻结且训练后变化为0；
5. 三条神经行参数均4,795，相对spread为0；
6. true/corrupted residual embedding不同，cell重标号等变；
7. 输出残差绝对值不超过0.25；
8. 无certificate/witness/parity/label buffer，数值有限。

readiness门：

```text
true residual - ridge anchor       >= 0.02 AUC
true residual - independent        >= 0.03 AUC
true residual - corrupted          >= 0.03 AUC
true train AUC - validation AUC     <= 0.15
```

## 8. 裁决与下一步

```text
protocol invalid:
  status = fail
  next   = 修复ridge、零残差、冻结、图或训练协议

readiness未通过:
  status = hold
  next   = 以E82标签 + true ridge 0.862045收束SKINNY分支

全部通过:
  status = pass
  next   = 预注册30轮seed0真实/错误/独立残差正式归因
```

失败后不得继续加hidden、消息步数、残差上限、epoch或远程预算。

## 9. 产物与边界

```text
run_id = i2_skinny64_r5_true_ridge_sparse_residual_readiness_seed0_20260719
output = outputs/local_smoke/i2_skinny64_r5_true_ridge_sparse_residual_readiness_seed0_20260719

results.jsonl
history.csv
gate.json
summary.json
metadata.json
progress.jsonl
checkpoints/*.pt
curves.svg
visual_qa_passed.marker
```

E84仍是两轮本地readiness，不是正式神经结果、高轮攻击、跨密码迁移或SOTA。

## 10. 实际结果

运行：

```text
run_id = i2_skinny64_r5_true_ridge_sparse_residual_readiness_seed0_20260719
device = cpu
epochs = 2
seed   = 0
```

四行validation AUC：

```text
true sparse39 ridge anchor = 0.8620446578
independent residual       = 0.8678364579
true sparse residual       = 0.8673792105
corrupted sparse residual  = 0.8672267947
```

门控差值：

```text
true residual - ridge anchor = +0.0053345527  < +0.02
true residual - independent  = -0.0004572474  < +0.03
true residual - corrupted    = +0.0001524158  < +0.03
```

全部协议门通过：零残差与ridge逐项误差为0，训练后ridge buffer最大变化为0，三条神经行均为
4,795个可学习参数，cell重标号最大误差为`1.19e-7`，真实图与错误图的embedding确实不同。
因此失败不能归因于base不一致、弱初始化、参数不公平、图未生效或实现协议无效。

## 11. 裁决与证据支持的下一步

```text
status       = hold
decision     = innovation2_skinny64_true_ridge_residual_not_ready
formal_seed0 = false
remote_scale = false
```

三条神经残差都只比同一ridge底座增加约`0.005` AUC，真实SKINNY图没有超过独立关系，且只比
same-degree错误图高`0.00015`。可观测增益应解释为通用校正，而不是可归因的真实拓扑神经增益。

证据支持的下一步是以E82五轮严格标签和E83真实稀疏拓扑ridge `0.862045`保留SKINNY确定性结果，
停止当前SKINNY神经结构搜索。不得继续增加epoch、hidden、消息步数、残差上限或远程规模。
创新2的神经方法主线仍由PRESENT/GIFT双真实SPN的r3-only profile operator承担；下一步应做
方法级claim综合并排序新的sound标签/真实密码候选，而不是把SKINNY hold扩大为整个创新2失败。
