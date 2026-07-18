# 创新2 E85：PRESENT/GIFT拓扑参数化共享Profile Operator两轮readiness计划

日期：2026-07-19

状态：已完成 / pass / 开放E86固定30轮seed0归因

## 1. 研究问题

E73与E79已经证明相同的r3-only Prefix-Guided Nodewise Profile Operator归纳偏置，在
PRESENT-80四轮与GIFT-64四轮两套严格unit-balance profile标签上分别训练时，均能稳定超过
independent node和same-family错误P控制。但它们仍是两套独立checkpoint，不能回答：

```text
一套共享参数能否把cipher P-layer当作运行时结构输入，
同时学习PRESENT与GIFT的严格输出平衡谱，而不是分别记住两个密码？
```

E85候选名为`Topology-Parameterized Shared Profile Operator`（TPSPO）。名称只描述待验证结构，
readiness通过前不宣称新方法已经成立。

## 2. 文献与本地证据边界

当前文献中，AutoND/DBitNet提供cipher-agnostic架构与自动训练流水线；Zahednejad--Lyu把同一积分
神经框架应用到PRESENT、RECTANGLE、LBLOCK和SPECK，并在每个密码上分别训练或分别做transfer
learning。现有本地全文、2026-07-10跨SPN检索和本次多任务检索尚未找到以下直接先例：

```text
同一组参数 + 运行时真实SPN拓扑输入
-> 联合预测PRESENT/GIFT严格、structure-disjoint的unit-balance profile
-> 同时使用独立关系与same-family错误拓扑归因控制
```

“尚未找到”不是完整的新颖性证明。E85只测试方法可行性，不作论文首创、攻击、高轮或SOTA声明。

## 3. 冻结来源

| 角色 | run id |
|---|---|
| PRESENT严格四轮profile | `i2_present_r4_unit_balance_profile_readiness_20260718` |
| PRESENT结构atlas | `i2_present_r4_universal_balance_atlas_20260718` |
| GIFT严格四轮profile | `i2_gift64_r4_unit_balance_profile_192_structures_20260719` |
| PRESENT两轮独立readiness锚点 | `i2_present_r4_r3_only_profile_operator_readiness_seed0_20260718` |
| GIFT两轮独立readiness锚点 | `i2_gift64_r4_r3_only_profile_operator_readiness_seed0_20260719` |
| 双密码方法裁决 | `i2_cross_spn_r3_profile_operator_method_synthesis_20260719` |
| SKINNY分支收束裁决 | `i2_skinny64_r5_true_ridge_sparse_residual_readiness_seed0_20260719` |

所有来源必须记录SHA-256并重放run id、status、decision、split、shape、标签和r3切片。E85不修改
validation数据、标签、负类、unknown、structure-disjoint split或AUC计算。

## 4. 唯一结构变量

E73/E79的模型把一个cipher P-layer固定为buffer。E85只把它改为forward时提供的运行时有向置换：

```text
13维r3-only node feature
  -> 共享input projection
  -> 2步共享cell/P-layer message block
  -> 共享node output head

forward(features, inverse_player)
```

同一个模型依次接收PRESENT或GIFT的`inverse_player`，全部可学习参数共享。禁止添加cipher ID、
cipher embedding、专属head、adapter、FiLM、attention、额外hidden或额外消息步数。这样通过时，
收益只能来自共享算子对不同真实拓扑的处理能力，而不是显式密码身份分支。

## 5. 三行公平矩阵

| 行 | PRESENT关系 | GIFT关系 | 参数 |
|---|---|---|---:|
| independent shared | identity/self | identity/self | 4,795 |
| true-topology shared | true PRESENT P | true GIFT P | 4,795 |
| corrupted-topology shared | fair wrong PRESENT P | fair wrong GIFT P | 4,795 |

三行使用相同初始化、数据顺序、optimizer、batch、epoch和loss权重，只改变relation。错误P沿用
E73/E76的same-family确定性生成器，必须保持64-bit permutation且不同于真实P。

## 6. 同预算训练

```text
epochs       = 2
seed         = 0
batch size   = 8 structures
optimizer    = AdamW
lr           = 1e-3
weight decay = 1e-4
device       = cpu
```

PRESENT有50个train structure，即每轮7个batch；GIFT有110个，即每轮14个batch。共享模型每轮
恰好消费两套数据各一次，共21次更新；两轮共42次更新。这与分别训练E73和E76两套readiness时的
总batch/update预算一致，但TPSPO只有一套4,795参数，而独立方案合计9,590参数。

为了不让GIFT因batch更多自动主导，PRESENT batch loss乘`21/(2*7)=1.5`，GIFT batch loss乘
`21/(2*14)=0.75`；每个密码每轮累计loss权重相同。batch schedule按冻结seed确定性打乱。

## 7. 协议门

1. E65/E75/E73/E76/E80/E84来源run id、decision、status和hash全部匹配；
2. PRESENT/GIFT输入分别严格为`96x64x13`与`192x64x13`，且等于原39维的第26--38列；
3. 两套structure-disjoint split和matched labels逐项重放；
4. 三行参数均为4,795，初始化逐参数相同，无cipher ID/adapter/专属head状态；
5. 每行每轮PRESENT/GIFT batch数为`7/14`，两轮总更新数为42；
6. 同一模型能分别接受两个真实P-layer，输出shape与masked loss正确；
7. 对PRESENT和GIFT，真实/错误拓扑均改变logit，cell重标号误差`<=1e-6`；
8. 无certificate/witness/parity/full-cube/label buffer，logit、loss和gradient有限。

## 8. Readiness门

独立两轮true锚点：

```text
PRESENT = 0.8341666667
GIFT    = 0.7606659729
```

共享模型必须同时满足：

```text
PRESENT true AUC                  >= 0.7841666667  # anchor - 0.05
GIFT true AUC                     >= 0.7106659729  # anchor - 0.05
每个密码 true - independent       >= 0.03
每个密码 true - corrupted         >= 0.03
两密码macro true AUC              >= 0.75
```

门槛只决定是否值得做30轮归因；两轮AUC本身不作正式收益结论。

## 9. 裁决与下一步

```text
协议失败:
  status = fail
  next   = 修复来源、动态拓扑、共享参数、公平预算或等变协议

readiness未通过:
  status = hold
  next   = 保留E73/E79两套独立模型，关闭当前共享参数分支

全部通过:
  status = pass
  next   = E86固定30轮seed0共享模型正式归因；通过后才运行seed1
```

失败后不得通过添加cipher embedding、adapter、hidden、attention、epoch或远程规模挽救。E85是
小数据结构级readiness，必须本地运行，不使用远程GPU。

## 10. 计划产物

```text
run_id = i2_present_gift_r4_topology_parameterized_shared_profile_operator_readiness_seed0_20260719
output = outputs/local_smoke/i2_present_gift_r4_topology_parameterized_shared_profile_operator_readiness_seed0_20260719

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

证据边界：PRESENT/GIFT四轮、严格unit-balance profile、两轮本地readiness；不是零样本迁移、
高轮积分区分器、密钥恢复、攻击、远程规模、完整新颖性证明或SOTA。

## 11. 实际运行与结果

真实运行严格使用第6节预算，三行各完成两轮和42次更新。全部来源、hash、标签、split、r3切片、
动态拓扑、参数公平、cell等变和数值协议门通过。结果为：

| 关系 | PRESENT validation AUC | GIFT validation AUC | macro AUC |
|---|---:|---:|---:|
| independent | 0.601111 | 0.502862 | 0.551986 |
| corrupted topology | 0.689167 | 0.693809 | 0.691488 |
| true topology | **0.856944** | **0.762747** | **0.809846** |

逐密码margin与独立readiness锚点差值：

```text
PRESENT true - independent = +0.255833
PRESENT true - corrupted   = +0.167778
PRESENT true - anchor      = +0.022778

GIFT true - independent    = +0.259886
GIFT true - corrupted      = +0.068939
GIFT true - anchor         = +0.002081
```

模型三行均为4,795参数，初始参数逐项误差为0；PRESENT/GIFT真实P-layer不同且不进入`state_dict`，
没有cipher ID、adapter、FiLM或专属head。真实/错误拓扑在两密码上都改变logit，cell重标号最大误差
分别为`1.79e-7/2.38e-7`。因此当前readiness增益可以归因到运行时真实拓扑，而不是显式密码身份分支。

## 12. 裁决与证据支持的下一步

```text
status   = pass
decision = innovation2_shared_profile_operator_readiness_passed
remote   = no
```

所有预注册readiness门均通过。E85仍只有seed0、两轮，不能升级为稳定共享模型结论。证据支持的唯一
下一步是E86：保持同一4,795参数模型、两套严格数据、21 batch/epoch公平schedule、真实/错误/独立
三行和seed0，只把训练从2轮改为30轮。E86必须分别对PRESENT和GIFT比较E73/E79独立模型锚点，
同时维持逐密码真实拓扑归因；通过后才运行完全相同的seed1。不得加入cipher embedding、adapter、
hidden、attention或远程规模。

可视化`curves.svg`已按2166x1133像素渲染并通过`visual-qa-redraw`：无文字重叠、裁切、锚点遮挡、
margin映射歧义或零样本迁移误述，记录于`visual_qa_passed.marker`。
