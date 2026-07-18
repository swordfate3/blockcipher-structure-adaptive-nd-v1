# 创新2 E66：PRESENT四轮prefix引导逐节点平衡谱算子readiness计划

日期：2026-07-18

状态：已完成 / pass / 允许E67正式seed0

## 1. 研究问题

E65把E43严格unit-mask标签重排为每个活动结构的64维masked balance profile，并得到：

```text
ANF r1--r3 prefix ridge validation AUC = 0.793611
true topology reachability AUC         = 0.704306
fair-corrupted topology AUC            = 0.695694
static set AUC                         = 0.514722
```

E66测试一个单变量结构假设：在每个输出bit都有安全ANF前缀表示时，让64个输出节点沿SPN关系
交换消息，能否形成一次前向输出完整平衡谱的可训练算子。

```text
Prefix-Guided Nodewise Profile Operator

active structure
  -> 每个output bit的r1/r2/r3 ANF-prefix node feature
  -> 共享参数node block执行2步
  -> 64个balance logits
  -> 只在E65 observed coordinates上计算masked BCE
```

E66只判断实现和短训练是否就绪，不用两轮AUC声称网络有效。

## 2. 冻结数据与安全前缀

```text
source E65 = i2_present_r4_unit_balance_profile_readiness_20260718
cipher     = PRESENT-80
rounds     = 4
target     = 96 x 64 profile_targets.npy
observed   = 96 x 64 profile_observed.npy
train      = 50 structures / 356 observed coordinates
validation = 18 structures / 120 observed coordinates
split      = structure-disjoint
```

每个node的39维前缀完全复用E45/E65定义：r1/r2/r3的support mean/max/sum/union和degree0--8
分布。允许它作为公开可计算的安全确定性前端；禁止第4轮full-cube count、最终certificate、
witness、key/offset parity、label派生输入或validation拟合。E66必须逐行复现E65已保存的
observed-edge前缀特征。

## 3. 模型和唯一变量

所有神经行使用同一个`PrefixGuidedProfileOperator`：

```text
input width   = 39
hidden        = 32
shared steps  = 2
dropout       = 0.10
output        = 64 logits
```

共享block对每个输出node拼接`self / same-S-box-cell mean / P-predecessor`后更新。三种模式
参数形状完全相同，只改变关系来源：

```text
independent       cell/self和P/self都退化为本node，容量控制
true-P mixer      same-cell + 正确P-layer predecessor
corrupted-P mixer same-cell + fair-corrupted P-layer predecessor
```

因此唯一候选增量是跨输出node的SPN关系消息；不加入绝对output embedding、大型Transformer、
额外prefix维度或标签特征。

## 4. 两轮固定预算

```text
epochs       = 2
batch        = 8 structures
seed         = 0
optimizer    = AdamW(lr=1e-3, weight_decay=1e-4)
checkpoint   = best validation observed-edge AUC
device       = local CPU
```

矩阵：

```text
0. E65 ANF-prefix ridge                      只读锚点
1. independent shared node block            同容量控制
2. true-P prefix-guided profile mixer        候选
3. fair-corrupted-P profile mixer            错误拓扑控制
```

三行神经模型使用相同初始化seed、batch顺序、optimizer、参数量和masked loss。

## 5. Readiness门

协议与模型contract：

```text
E43/E65 source run id、decision、hash、数组和split重放       pass
full prefix tensor                           96 x 64 x 39
observed E65 prefix replay max error          <= 1e-12
model output                                 batch x 64
masked BCE等于显式observed-edge BCE           <= 1e-7
三种mode参数量完全相同                        yes
true/corrupted同权重logit max delta           >= 1e-6
cell重标号等变误差                            <= 1e-6
final certificate/witness buffers             absent
logit/loss/gradient                            finite
三行均完成2 epochs                            yes
```

短训练只设防退化门：

```text
independent validation AUC in [0.55, 0.95]
true-P validation AUC      in [0.55, 0.95]
corrupted validation AUC   in [0.35, 0.95]
```

通过只允许另建E67 30轮seed0正式计划。E67才要求true-P达到至少`0.78`、领先independent
至少`0.03`并领先corrupted至少`0.03`；具体门在E67训练前再次冻结。readiness失败则先修实现
或停止优化不足的结构，不根据两轮validation调hidden、step、dropout或学习率。

## 6. 边界与产物

E66是本地小数据readiness，不运行seed1、不使用远程GPU、不迁移r5。它不恢复RCCA，不改变
E43/E65标签，也不能写成高轮积分区分器或新攻击。

```text
run_id = i2_present_r4_prefix_guided_profile_operator_readiness_seed0_20260718
output = outputs/local_smoke/i2_present_r4_prefix_guided_profile_operator_readiness_seed0_20260718
```

## 7. 2026-07-18实际结果

E43/E65 source run、decision、hash、`96 x 64`数组、476条matched观察坐标、`50/18`
structure split和全部observed前缀逐行重放通过；前缀最大误差为`0.0`。

模型contract：

```text
output shape                         = 4 x 64
independent / true / corrupted params= 5679 / 5679 / 5679
masked BCE explicit max error        = 0.0
cell relabel max error               = 1.94e-7
true/corrupted same-weight logit delta= 0.105503
logit/loss/gradient finite           = pass
forbidden named state                = absent
```

两轮最佳checkpoint：

| 模式 | train AUC | validation AUC | validation accuracy |
|---|---:|---:|---:|
| independent node | `0.712378` | `0.717778` | `0.658333` |
| true-P profile mixer | `0.790904` | `0.799167` | `0.716667` |
| fair-corrupted-P mixer | `0.698933` | `0.692222` | `0.641667` |

```text
true - independent = +0.081389
true - corrupted   = +0.106944
```

两轮结果只用于防退化，但三行均在预注册范围内，正确P候选同时显示明显的同预算关系差异：

```text
status   = pass
decision = innovation2_present_profile_operator_readiness_passed
remote   = no
```

## 8. 裁决与推荐下一步

建立E67 30轮seed0正式归因，代码、数据、39维prefix、hidden32、steps2、dropout0.10、batch8、
optimizer和三行矩阵全部冻结。正式比较E65 prefix ridge、independent、true-P和corrupted-P；
候选必须同时达到`AUC >= 0.78`、领先independent至少`0.03`并领先corrupted至少`0.03`。
任何正式门失败都停止该结构，不增加容量、epoch或远程规模；全部通过才允许同矩阵seed1确认。

最终`curves.svg`经`visual-qa-redraw`渲染为`2200 x 1165`像素检查；标题、两轮AUC、log契约、
参数说明、裁决与证据范围均无重叠、裁切、缺字或模糊归因。
