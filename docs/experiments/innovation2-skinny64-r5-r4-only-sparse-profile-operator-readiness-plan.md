# 创新2 E83：SKINNY-64五轮r4-only稀疏线性层平衡谱算子readiness计划

日期：2026-07-19

状态：已完成 / hold / 两轮本地神经readiness

## 1. 研究问题

E82已经用严格标签确认SKINNY-64/64五轮存在宽、配平且抗一元捷径的unit-output平衡谱：

```text
raw positive / negative / unknown = 4080 / 1208 / 856
matched train                      = 584 / 584
matched validation                 = 162 / 162
```

PRESENT/GIFT的确认方法使用一对一P-layer前驱。SKINNY的线性层是`ShiftRows + MixColumns`，每个
输出bit有1、2或3个前驱，不能用一个伪造permutation代替。E83只回答：

```text
把一对一P-layer消息改成无参数的稀疏多前驱均值聚合后，
真实SKINNY线性图是否在两轮训练中优于独立node和同入度错误图？
```

暂定候选名：`Sparse Linear-Layer Profile Operator`（SLPO）。它只是待审判结构，不预先宣称创新成立。

## 2. 冻结来源与输入

来源：

```text
i2_skinny64_r5_unit_balance_profile_transition_20260719
status   = pass
decision = innovation2_skinny64_r5_unit_balance_profile_transition_ready
```

必须重放gate、metadata、structures、targets、observed、matched CSV与prefix hash。输入只取E82
`96 x 64 x 52`前缀的最后13维，即目标五轮前的r4 support profile；不输入label、certificate、
witness、parity或结构ID。

## 3. 单变量架构改变

共享参数部分与PRESENT/GIFT r3-only算子完全相同：

```text
input_dim       = 13
hidden_dim      = 32
shared steps    = 2
dropout         = 0.10
parameter count = 4,795
output          = 64 logits
```

每步仍拼接：

```text
current node state
same-S-box-cell mean context
linear-layer predecessor context
```

唯一结构变化是第三项：

```text
PRESENT/GIFT = one predecessor selected by inverse P-layer
SKINNY       = normalized mean over the exact ShiftRows+MixColumns predecessor set
```

稀疏邻接无可学习参数。真实图必须由公开`shift_rows`和`mix_columns`对64个basis bit逐项构造；
每个target的入度属于`{1,2,3}`，总边数128。

## 4. 三行公平矩阵

只训练三行，预算完全相同：

| 行 | linear context |
|---|---|
| independent | 不使用跨节点关系；cell和linear context均退化为自身 |
| true | 真实SKINNY ShiftRows+MixColumns稀疏前驱图 |
| corrupted | 固定轮换source cell，保持128边、每target入度和cell内lane，但破坏真实连接 |

错误图不得改变标签、split、输入或参数。true与corrupted初始共享可学习参数时必须产生不同logit。

## 5. 公平确定性基线

E76曾暴露单节点ridge信息范围不足，因此E83预先计算三种train-only ridge：

```text
local13
true sparse expansion      = local + cell mean + true predecessor mean
corrupted sparse expansion = local + cell mean + corrupted predecessor mean
```

所有axis选择、标准化与拟合只用train结构，固定应用到validation。

## 6. 训练协议

```text
epochs       = 2
seed         = 0
batch size   = 8 structures
optimizer    = AdamW
lr           = 1e-3
weight decay = 1e-4
device       = cpu
checkpoint   = best validation AUC per row
```

这是readiness，不是正式训练；不得因结果贴线临时加epoch、seed或hidden。

## 7. 协议与推进门

协议必须通过：

1. E82来源、hash、`96x64x52`前缀和1,492条observed edge逐项重放；
2. r4-only shape为`96x64x13`且等于columns `39:52`；
3. true/corrupted均128边、逐target入度相同、lane保持且邻接不同；
4. 模型输出`batch x 64`，三行参数均4,795；
5. masked loss等于显式observed loss；
6. 同时重标号cell、features和邻接时最大logit误差不超过`1e-6`；
7. 无label/certificate/witness命名状态，logit/loss/gradient有限。

公平确定性门：

```text
true sparse ridge - local ridge     >= 0.03
true sparse ridge - corrupted ridge >= 0.03
```

两轮神经门：

```text
true validation AUC                 >= 0.65
true - independent                  >= 0.03
true - corrupted                    >= 0.03
true - true sparse ridge            >= -0.03
```

## 8. 裁决

```text
protocol invalid:
  status = fail

公平ridge未归因真实图:
  status = hold
  next   = 停止当前SLPO，不用神经训练掩盖确定性图错配

神经门未通过:
  status = hold
  next   = 保留E82标签，停止SLPO正式训练

全部通过:
  status = pass
  next   = 预注册30轮seed0正式归因，再按门决定seed1
```

## 9. 产物与边界

```text
run_id = i2_skinny64_r5_r4_only_sparse_profile_operator_readiness_seed0_20260719
output = outputs/local_smoke/i2_skinny64_r5_r4_only_sparse_profile_operator_readiness_seed0_20260719

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

E83只说明五轮严格标签上的两轮本地结构readiness。它不是正式神经增益、高轮积分区分器、
checkpoint跨密码迁移、攻击或SOTA；无论pass/hold均不远程。

## 10. 完成结果

运行：

```bash
MPLCONFIGDIR=/tmp/matplotlib UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/train-innovation2-skinny64-sparse-profile-operator-readiness \
  --run-id i2_skinny64_r5_r4_only_sparse_profile_operator_readiness_seed0_20260719 \
  --profile-root outputs/local_audits/i2_skinny64_r5_unit_balance_profile_transition_20260719 \
  --output-root outputs/local_smoke/i2_skinny64_r5_r4_only_sparse_profile_operator_readiness_seed0_20260719
```

来源重放、`96x64x52 -> 96x64x13`切片、1,492条observed edge、模型输出、masked loss、
4,795参数公平、128边、入度分布、lane保持、错误图、cell重标号等变和数值有限性全部通过。

公平ridge：

| 输入范围 | validation AUC |
|---|---:|
| local13 | `0.763146` |
| corrupted sparse39 | `0.739217` |
| true sparse39 | **`0.862045`** |

真实图展开相对local和错误图为`+0.098899/+0.122828`，确定性拓扑归因门全部通过。

两轮神经：

| 关系 | validation AUC | accuracy |
|---|---:|---:|
| independent | `0.800069` | `0.706790` |
| corrupted | `0.788961` | `0.697531` |
| true | `0.794524` | `0.700617` |

真实图相对independent为`-0.005544`，相对错误图仅`+0.005563`，并比true sparse ridge低
`0.067520`。所以绝对AUC虽超过0.65，但三项关系/强基线门失败。

```text
status   = hold
decision = innovation2_skinny64_sparse_profile_readiness_not_passed
formal   = no
remote   = no
```

## 11. 可视化与验证

`curves.svg`分开显示神经AUC、train-only ridge和正负归因margin，没有隐藏负margin。最终
1600x818像素通过`visual-qa-redraw`，标题、旋转标签、柱值、门线、裁决和证据范围均无重叠、
裁切或歧义，已记录`visual_qa_passed.marker`。最近结果索引刷新通过，E83为`001`。

验证：

```text
focused pytest  = 131 passed
protocol checks = all true
ridge gates     = all true
neural gates    = 1 / 4 true
result rows     = 3
```

## 12. 证据支持的推荐下一步

E83不能靠加epoch升级，因为预注册两轮门已经失败。它同时提供了明确正信号：真实图的显式39维
展开是当前最强可验证表示。下一候选E84应是`True-Ridge-Guided Sparse Residual`：先用train-only
true sparse39 ridge生成冻结base score，零残差必须逐项复现ridge；再训练小型真实图残差，并与
ridge-only、无图残差和错误图残差比较。

只有真实图残差在同预算下超过冻结ridge至少`0.02`、超过错误图残差至少`0.03`且不过拟合，才保留
神经残差并进入正式seed0。否则以E82标签加`0.862045`确定性拓扑基线收束SKINNY分支，不继续
增加消息步数、hidden、epoch或远程预算。
