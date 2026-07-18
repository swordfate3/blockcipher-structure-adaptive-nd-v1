# 创新2 E80：PRESENT/GIFT r3-only平衡谱算子方法级综合计划

日期：2026-07-19

状态：已完成 / pass / 无新训练

## 1. 研究问题

E73已经在PRESENT-80四轮严格8-bit cube单位输出平衡谱上完成双seed归因；E79又在
GIFT-64四轮独立构造的严格标签上完成相同归纳偏置的双seed归因。E80只回答：

```text
同一r3-only Prefix-Guided Nodewise Profile Operator方法，
是否已经在两个真实64-bit SPN上分别获得可归因、可复核的证据；
以及第三个真实SPN是否已经具备同级标签条件。
```

E80不训练模型、不迁移checkpoint、不更改任何已完成实验的标签、split、控制或指标。

## 2. 冻结来源

| 角色 | run id | 必须读取 |
|---|---|---|
| PRESENT双seed方法证据 | `i2_present_r4_r3_only_profile_operator_seed1_20260718` | `gate.json`、`metadata.json` |
| GIFT双seed方法证据 | `i2_gift64_r4_r3_only_profile_operator_seed1_20260719` | `gate.json`、`metadata.json` |
| SKINNY r7论文kernel | `i2_skinny64_r7_hwang_kernel_readiness_768keys_seed0_20260717` | `gate.json` |
| SKINNY r8论文kernel | `i2_skinny64_r8_hwang_kernel_readiness_768keys_seed0_20260717` | `gate.json` |
| SKINNY r8相邻pair宽度 | `i2_skinny64_r8_adjacent_pair_kernel_diversity_128keys_seed0_20260717` | `gate.json` |
| SKINNY r8底行pair闭合 | `i2_skinny64_r8_bottom_row_pair_closure_128keys_seed0_20260717` | `gate.json` |
| SKINNY r7单cell宽度 | `i2_skinny64_r7_single_cell_geometry_128keys_seed0_20260717` | `gate.json` |
| 真实SPN标签迁移门 | `i2_real_spn_pair_state_transfer_readiness_20260718` | `gate.json` |

所有来源记录SHA-256；run id、status、decision或关键协议字段不匹配时，E80为协议失败。

## 3. 方法同一性与归因门

PRESENT和GIFT必须同时满足：

1. 四轮、64-bit状态、64个输出节点；
2. `r3-only`每节点13维输入、hidden 32、共享message steps 2；
3. independent/true/fair-corrupted三行参数均为4,795；
4. seed0和seed1各训练30 epochs；
5. 每颗seed的真实P AUC分别比independent和same-family错误P至少高`0.03`；
6. 两个来源gate为`pass`，且其内部协议检查全部通过；
7. 两套模型使用各自密码的严格标签重新训练，不把GIFT结果解释为PRESENT checkpoint迁移。

如果任一密码没有逐seed归因，不能形成双密码方法证据，E80为`hold`。

## 4. 可比性边界

必须在结构化产物和图中显式记录：

```text
PRESENT structures = 96,  observed matched edges = 476
GIFT    structures = 192, observed matched edges = 620
```

两个数据集的密码、结构库、标签分布和可观测边不同。因此AUC只允许在各自密码内部比较
真实P与控制；禁止用`0.9467 > 0.9121`宣称PRESENT模型强于GIFT模型。

允许的方法级结论仅为：

> 相同13维r3输入、64-node共享算子和真实cipher P-layer归纳偏置，在PRESENT与GIFT两套
> 独立严格标签上分别训练后，均在两颗seed稳定超过独立node和same-family错误P。

不允许写成零样本跨密码泛化、checkpoint迁移、高轮积分区分器、攻击、远程规模或SOTA。

## 5. 第三SPN门

SKINNY证据必须分成两层：

1. E20/E21论文kernel复现是否通过；
2. E22/E23/E24/E42是否已经形成宽、稳定、structure-disjoint且抗边际捷径的训练标签库。

论文kernel复现通过不等于神经标签就绪。只要E42的`ready_label_family_count == 0`，或
E22/E23/E24仍只有少数稳定结构，第三密码训练保持关闭。

## 6. 冻结裁决

```text
if 任一来源或协议不匹配:
    status   = fail
    decision = innovation2_cross_spn_method_synthesis_protocol_invalid
elif PRESENT或GIFT未逐seed通过真实P归因:
    status   = hold
    decision = innovation2_cross_spn_r3_profile_method_not_confirmed
elif SKINNY同级标签尚未就绪:
    status   = pass
    decision = innovation2_cross_spn_r3_profile_method_confirmed_skinny_labels_not_ready
else:
    status   = pass
    decision = innovation2_cross_spn_r3_profile_method_confirmed_third_spn_ready
```

当前证据预期落入第三条，但以生成gate为准。

## 7. 产物

```text
run_id = i2_cross_spn_r3_profile_operator_method_synthesis_20260719
output = outputs/local_audits/i2_cross_spn_r3_profile_operator_method_synthesis_20260719

results.jsonl
evidence_matrix.csv
source_hashes.json
gate.json
summary.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

`results.jsonl`按PRESENT、GIFT、SKINNY三行记录方法证据或标签就绪状态。`curves.svg`使用
分密码面板展示各自seed内的真实P与控制，不画跨密码高低连线，并单列SKINNY标签门。

## 8. 下一步分支

若双密码方法通过、SKINNY标签未就绪：

```text
next_question = 能否为SKINNY-64构造与PRESENT/GIFT同语义的四轮严格unit-profile标签
same_budget_anchor = E75 GIFT 192结构标签门
only_change = cipher topology/key schedule/strict provider改为SKINNY-64
planned_scale = 96结构readiness，必要时冻结扩展到192结构
training = no，标签门通过前禁止网络训练
advance_gate = 正负类、unknown、structure-disjoint checkerboard与边际控制全部通过
stop_gate = 宽度不足或可被结构/位置边际解释
do_not_pursue = PRESENT/GIFT同benchmark继续加seed、扩模型、加层或机械远程规模
```

只有SKINNY严格标签门通过，才开放同一4,795参数三行readiness。若标签仍不可用，创新2先以
PRESENT+GIFT双密码四轮方法证据收束，再排序新的sound标签表示；不能用更多网络搜索替代标签缺口。

## 9. 完成结果

真实运行：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python \
  scripts/run-innovation2-cross-spn-r3-profile-operator-synthesis \
  --run-id i2_cross_spn_r3_profile_operator_method_synthesis_20260719 \
  --present-root outputs/local_diagnostic/i2_present_r4_r3_only_profile_operator_seed1_20260718 \
  --gift-root outputs/local_diagnostic/i2_gift64_r4_r3_only_profile_operator_seed1_20260719 \
  --skinny-r7-root outputs/local_audits/i2_skinny64_r7_hwang_kernel_readiness_768keys_seed0_20260717 \
  --skinny-r8-root outputs/local_audits/i2_skinny64_r8_hwang_kernel_readiness_768keys_seed0_20260717 \
  --skinny-adjacent-root outputs/local_audits/i2_skinny64_r8_adjacent_pair_kernel_diversity_128keys_seed0_20260717 \
  --skinny-bottom-row-root outputs/local_audits/i2_skinny64_r8_bottom_row_pair_closure_128keys_seed0_20260717 \
  --skinny-single-cell-root outputs/local_audits/i2_skinny64_r7_single_cell_geometry_128keys_seed0_20260717 \
  --real-spn-root outputs/local_audits/i2_real_spn_pair_state_transfer_readiness_20260718 \
  --output-root outputs/local_audits/i2_cross_spn_r3_profile_operator_method_synthesis_20260719
```

所有冻结来源run id、status、decision、SHA-256、内部协议和方法契约检查通过。重新从原始训练行
计算的证据为：

| 密码 | seed0 true / independent / wrong-P | seed1 true / independent / wrong-P | mean true-ind | mean true-wrong |
|---|---|---|---:|---:|
| PRESENT-80 r4 | `0.945556 / 0.657500 / 0.820000` | `0.947778 / 0.671944 / 0.879444` | `+0.281944` | `+0.096944` |
| GIFT-64 r4 | `0.913111 / 0.571280 / 0.774714` | `0.911030 / 0.569719 / 0.784599` | `+0.341571` | `+0.132414` |

两个密码逐seed均超过两类控制至少`0.03`，共享13维r3输入、hidden 32、2步共享消息传递和
4,795参数；PRESENT使用96结构/476条可观测边，GIFT使用192结构/620条可观测边。后者差异已
写入gate和图，两个mean true AUC不作直接横向排名。

SKINNY E20/E21论文kernel复现通过，但E22/E23/E24标签宽度或闭合仍为hold，E42
`ready_label_family_count = 0`。最终裁决：

```text
status   = pass
decision = innovation2_cross_spn_r3_profile_method_confirmed_skinny_labels_not_ready
training = no
remote   = no
```

## 10. 可视化与验证

`curves.svg`按PRESENT、GIFT和SKINNY三面板绘制。首次像素检查发现图例占用柱形绘图区，随后
改为顶部共享图例；最终1600x852像素输出通过`visual-qa-redraw`，无字体重叠、裁切、缺字、
数值遮挡、图例冲突或跨密码AUC误导，并记录`visual_qa_passed.marker`。

验证：

```text
targeted pytest = 115 passed
E80 plot tests  = 5 passed
source_checks   = all true
method_checks   = all true
result_rows     = 3
result index    = pass, E80 is 001
```

## 11. 证据支持的推荐下一步

E80确认了方法归纳偏置在两个真实SPN上的独立证据，所以继续PRESENT/GIFT同benchmark加层、
加seed或换网络名称的信息增益很低。真正阻止第三密码验证的是标签，不是算子容量。

下一步执行E81：SKINNY-64四轮严格8-bit cube unit-profile标签readiness。以GIFT E75的
192结构标签门为same-budget anchor，先运行96结构；只改变密码轮函数、key schedule与拓扑。
E81不得训练网络。只有全称正类、具体反例负类、unknown保留、structure-disjoint checkerboard
和行列边际控制全部通过，才开放SKINNY的4,795参数三行两轮readiness；否则停止当前SKINNY
unit-profile构造并回到sound标签表示排序。
