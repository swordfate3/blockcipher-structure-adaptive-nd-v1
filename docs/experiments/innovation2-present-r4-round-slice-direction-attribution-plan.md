# 创新2 E72：PRESENT四轮平衡谱前缀轮切片方向归因计划

日期：2026-07-18

状态：已完成 / hold / r3稳定主导，关闭轮递归分支

## 1. 研究问题

E71两轮readiness中，预注册候选`r1->r2->r3` AUC为`0.716667`，反向控制
`r3->r2->r1`却为`0.867222`。39维布局已经确认没有切片错误，但反向序列把r1放在最靠近
输出头的位置，所以该结果不能直接解释为“从输出向输入回推有效”。

E72不训练新网络，审计E65/E68稳定方法的真实轮切片依赖，决定是否还有理由设计新的轮序结构。

## 2. 冻结来源与两类证据

```text
data/split       = E65严格unit profile，50/18 structure-disjoint
features         = r1/r2/r3各13维，E65逐值重放
checkpoints      = E67 seed0 true-P / E68 seed1 true-P
baseline AUC     = 0.953056 / 0.961389，必须逐值重放
training         = no neural gradients
```

确定性证据：每次只使用一个13维round slice，在E65 train observed edges上拟合ridge，直接评估
同一validation edges。标准化和权重只由train拟合。

checkpoint证据：对每个round slice，用50个train structures在每个output node上的均值替换该
13维段，再读取冻结true-P checkpoint。记录：

```text
ablation drop = intact validation AUC - neutralized validation AUC
```

均值中和不读取validation分布、标签或oracle；另外记录全39维ridge作为E65重放锚点。

## 3. 协议与方向门

协议门：E65/E67/E68 run id、decision、hash、shape、split、prefix、checkpoint、baseline AUC与
全39维ridge `0.793611`全部重放通过。

对r1/r2/r3分别得到：

```text
single-round ridge AUC
seed0 ablation drop
seed1 ablation drop
```

只有同时满足以下条件，才认为存在可执行的主导轮信号：

```text
三组证据的argmax是同一个round
top ridge AUC - second ridge AUC               >= 0.03
该round seed0/seed1 ablation drop              each >= 0.03
每seed top drop - second drop                  each >= 0.02
```

若主导轮为r1，下一候选不是简单反向GRU，而是保留正确轮序并给早轮提供直接skip-to-head的
`Round-Salience Gated Residual`；若主导轮为r3，则反向控制优势与稳定checkpoint依赖矛盾，
轮递归路线关闭；若三组证据不一致，同样关闭。

## 4. 产物与范围

```text
run_id = i2_present_r4_round_slice_direction_attribution_20260718
output = outputs/local_audits/i2_present_r4_round_slice_direction_attribution_20260718
device = local CPU
remote = no
```

E72只解释PRESENT-80四轮、8-bit活动cube严格unit balance profile的特征方向，不是新网络收益、
高轮区分器、跨维度证据、攻击或SOTA。

## 5. 2026-07-18实际结果

E65/E67/E68 run id、decision、hash、前缀、split、完整39维ridge与双seed checkpoint AUC全部
逐值重放通过：

```text
full 39-d ridge AUC = 0.793611
seed0 intact AUC    = 0.953056
seed1 intact AUC    = 0.961389
```

单轮train-only ridge：

| slice | validation AUC |
|---|---:|
| r1 | `0.636111` |
| r2 | `0.672500` |
| r3 | `0.799444` |

双seed切片中和：

| neutralized slice | seed0 AUC / drop | seed1 AUC / drop |
|---|---:|---:|
| r1 | `0.927500 / +0.025556` | `0.951667 / +0.009722` |
| r2 | `0.958611 / -0.005556` | `0.960556 / +0.000833` |
| r3 | `0.597500 / +0.355556` | `0.690000 / +0.271389` |

ridge、seed0与seed1的argmax全部是r3，top-second间隔和双seedr3 drop全部超过预注册门。方向
证据本身非常稳定，但它指向本来就位于正向序列末端的r3，不能解释E71反向序列的高分。因此：

```text
status   = hold
decision = innovation2_present_round_direction_not_confirmed
neural readiness = no
remote = no
```

## 6. 裁决与推荐下一步

E71反向轮序`0.867222`重分类为两轮优化现象，不开放backward recurrent、early-round skip、
30轮、seed1或远程规模。E68继续保持架构第一。

E72同时暴露一个更直接、低风险的问题：r3单轮ridge `0.799444`略高于完整39维ridge
`0.793611`，而双seed模型几乎不依赖r2。下一步E73应先做`r3-only Prefix-Guided Profile
Operator`同预算readiness，比较true-P、independent和fair-corrupted-P；候选只删除r1/r2输入，
不改消息结构、标签、split或optimizer。若r3-only不能保持E68 readiness质量则停止；若能保持，
优先得到更简单、更可解释的方法，而不是继续增加新架构容量。

最终`curves.svg`经`visual-qa-redraw`渲染为`1900 x 1013`像素检查；单轮ridge、双seed切片
drop、`0.03`门、三组排序、r3主导裁决与证据范围均无重叠、裁切或方向歧义。
