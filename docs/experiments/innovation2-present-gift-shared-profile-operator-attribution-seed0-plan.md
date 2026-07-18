# 创新2 E86：PRESENT/GIFT共享Profile Operator 30轮seed0正式归因计划

日期：2026-07-19

状态：已完成 / hold / 共享参数分支关闭

## 1. 研究问题

E85两轮readiness证明一套4,795参数的Topology-Parameterized Shared Profile Operator（TPSPO）
能在PRESENT与GIFT上同时超过独立关系和same-family错误拓扑，并保持各自独立两轮锚点质量。E86只回答：

```text
把完全相同的共享模型训练到30轮后，
它能否在两个密码上分别接近独立训练的正式模型，
同时保留真实拓扑相对独立和错误拓扑的可归因增益？
```

E86不是新架构搜索，不增加任何模块或容量。

## 2. 冻结来源与锚点

| 角色 | run id | true validation AUC |
|---|---|---:|
| E85共享readiness | `i2_present_gift_r4_topology_parameterized_shared_profile_operator_readiness_seed0_20260719` | P `0.856944` / G `0.762747` |
| E73 PRESENT独立30轮seed0 | `i2_present_r4_r3_only_profile_operator_attribution_seed0_20260718` | `0.945556` |
| E78 GIFT独立30轮seed0 | `i2_gift64_r4_r3_only_profile_operator_attribution_seed0_20260719` | `0.913111` |

E86重新加载E65/E75严格profile与E85相同的路线来源；所有gate、results与关键数据记录SHA-256。
E85必须为`pass / innovation2_shared_profile_operator_readiness_passed`且全部协议、readiness门通过。

## 3. 唯一变化与禁止变化

```text
only change = epochs 2 -> 30
```

继续冻结：

```text
input         = 每节点13维r3-only前缀
nodes         = 64
hidden        = 32
message steps = 2
parameters    = 4,795
seed          = 0
batch size    = 8 structures
optimizer     = AdamW
lr            = 1e-3
weight decay  = 1e-4
device        = cpu
```

禁止添加cipher ID、embedding、adapter、FiLM、专属head、attention、额外hidden、额外消息步数、
checkpoint transfer或后验阈值。validation标签不得用于选择结构、超参数或门槛。

## 4. 三行与同预算schedule

三行仍为：

```text
independent shared
true-topology shared
corrupted-topology shared
```

每行每轮完整消费PRESENT 7个batch和GIFT 14个batch，共21次更新；30轮共630次更新。PRESENT
loss权重固定1.5，GIFT固定0.75，使每个密码每轮累计权重相同。三行使用相同初始化与schedule，
只改变relation。最佳checkpoint按两密码validation AUC的macro平均选择，但正式门逐密码独立判断，
不能用一边的高分补偿另一边失败。

## 5. 协议门

1. E85来源、hash、status、decision、协议和readiness门全部通过；
2. E73/E78来源run id、decision、status、30轮seed0与true AUC逐项匹配；
3. E65/E75数据、标签、unknown、split、r3切片与E85逐项重放；
4. 三行参数均4,795，初始参数相同，无cipher专属可学习状态；
5. 每行30轮，每轮batch为`7+14`，总更新630；
6. 动态真实/错误P-layer、masked loss、cell等变和有限数值协议继续通过；
7. 三条best checkpoint和完整history均写入产物。

## 6. 正式质量与归因门

每个密码分别满足：

```text
PRESENT true AUC >= 0.915556  # E73 seed0 true - 0.03
GIFT true AUC    >= 0.883111  # E78 seed0 true - 0.03

true - independent >= 0.03
true - corrupted   >= 0.03
train AUC - validation AUC <= 0.15
```

联合压缩门：

```text
macro true AUC >= 0.899333  # mean(E73, E78) - 0.03
shared parameters = 4,795
separate anchor parameters = 9,590
```

## 7. 裁决与下一步

```text
协议失败:
  status = fail
  next   = 修复E85/锚点来源、30轮schedule、动态拓扑或产物协议

任一密码质量或拓扑门失败:
  status = hold
  next   = 保留E73/E79两套独立模型，关闭共享参数分支

全部通过:
  status = pass
  next   = E87运行完全相同的30轮seed1联合复现
```

失败后不得通过adapter、容量、epoch、后验权重或远程预算挽救。通过seed0也不能直接升级为稳定结论，
必须由seed1复现。

## 8. 计划产物与边界

```text
run_id = i2_present_gift_r4_topology_parameterized_shared_profile_operator_attribution_seed0_20260719
output = outputs/local_diagnostic/i2_present_gift_r4_topology_parameterized_shared_profile_operator_attribution_seed0_20260719

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

证据范围仍是PRESENT/GIFT四轮严格unit-balance profile上的联合训练seed0归因；不是零样本迁移、
未见密码泛化、高轮区分器、攻击、远程规模、完整新颖性证明或SOTA。

## 9. 实际结果

三行均完成30轮、每轮`7+14`个batch、每行630次更新。全部E85/E73/E78来源、严格profile、r3切片、
共享参数、动态P-layer、loss、cell等变、history、checkpoint和数值协议门通过。

| 关系 | PRESENT validation AUC | GIFT validation AUC | macro AUC |
|---|---:|---:|---:|
| independent | 0.647083 | 0.516909 | 0.581996 |
| corrupted topology | 0.864722 | 0.748439 | 0.806581 |
| true topology | **0.950278** | **0.859521** | **0.904900** |

逐密码正式比较：

```text
PRESENT true - independent = +0.303194
PRESENT true - corrupted   = +0.085556
PRESENT true - E73 anchor  = +0.004722  # 质量门通过
PRESENT train - validation = +0.020622

GIFT true - independent    = +0.342612
GIFT true - corrupted      = +0.111082
GIFT true - E78 anchor     = -0.053590  # 低于允许的-0.03
GIFT train - validation    = +0.015088
```

macro true仅比两套独立锚点平均低`0.024434`，通过联合平均门；但逐密码门明确禁止用PRESENT高分
补偿GIFT失败。GIFT true在完整30轮history中的最高AUC就是epoch30的`0.859521`，所以失败不是
macro checkpoint恰好选错epoch造成的。

## 10. 裁决与证据支持的下一步

```text
status   = hold
decision = innovation2_shared_profile_operator_quality_not_retained
seed1    = no
remote   = no
```

真实拓扑相对独立与错误拓扑的四项margin全部通过，说明一个共享算子确实保留了两种SPN拓扑的
可归因信号；但同一组参数在正式预算下没有保住GIFT质量。因此E85是有价值的正向readiness，E86
则把方法边界定位为“共享表示可行、完全共享参数不足以替代两套独立模型”。

按预注册计划不运行seed1，不添加cipher adapter、专属head、hidden、attention、epoch或远程预算。
保留E73/E79两套独立双seed模型作为创新2当前正式神经方法证据；TPSPO只作为受控负结果和后续论文
讨论中的参数共享边界。下一实验应回到新的sound标签/真实密码或方法级论文综合，不在同一联合任务上
继续调多任务权重。

可视化`curves.svg`以2166x1133像素通过`visual-qa-redraw`：两密码锚点、质量下限、四个拓扑margin、
GIFT失败和hold裁决均清楚，无文字重叠、裁切或macro掩盖逐密码失败，记录于
`visual_qa_passed.marker`。
