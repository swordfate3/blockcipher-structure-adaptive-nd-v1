# 创新2 E98-C：PRESENT九轮ATM支撑与旋转轨道双互斥PU就绪门

日期：2026-07-20

状态：完成 / pass / E99本地训练恢复开放

## 1. 触发原因与研究问题

E98-B按共享完整`(u,v)`坐标构造六个positive互斥组，但E99训练前门发现同步64-bit旋转候选轨道
可以跨组。同一unlabeled关系若同时出现在train和test，单关系打分网络可能记住它在训练中不是
target并人为提高test positive排名。

E98-C回答：

```text
同时把“共享完整坐标”和“属于同一个同步旋转轨道”的positive合成不可拆组件后，
能否仍构造6个配平fold，并让所有train/test示例的精确关系零重合、候选宽度充足、
且简单位置/频率捷径不主导？
```

## 2. 冻结来源与构造

```text
source run  = i2_present_r9_atm_support_component_pu_readiness_20260720
source gate = pass / innovation2_present_r9_atm_support_component_pu_ready
source hash = 2f3f3d0cce46d3e786a39899ed87949eddb6c614deb52e16c8aaca623a5c0cb9
positive    = E98-A正确消元保留的468个canonical独立关系
orbit id    = 一个关系全部64个同步输入/输出旋转中canonical坐标最小者
component   = shared-coordinate边与same-orbit边并集的连通分量
packing     = 按(-component size, canonical id)排序后贪心放入最小组
groups      = 6
```

每fold的train candidate排除test positive support，test candidate排除train positive support；全部候选
排除470个已知positive。因为同一旋转轨道不可拆，train/test候选集合还必须精确关系零重合。

## 3. 同预算确定性基线

复用E98-B全部排序基线和指标：hash随机、组号、关系项数、指数重量、已见关系、训练坐标频率、训练
支撑重合和绝对bit位置。每个pool为1个已知positive加全部合格unlabeled；只报告Recall@1、
Recall@5、MRR和Top-5 enrichment，不计算accuracy/AUC。

## 4. 冻结裁决门

`advance_to_E99_revised`要求：

```text
E98-B hash/status/decision重放通过；
468个独立positive、删除2个依赖关系；
6组各至少64条、最大最小差<=1；
same-orbit组件不跨组、shared-coordinate组件不跨组；
每fold train/test所有relation（含unlabeled）精确重合=0；
train candidate与test positive support重合=0；
test candidate与train positive support重合=0；
全部候选与470个已知positive重合=0；
每个train/test positive至少31个unlabeled；
最佳非随机捷径Recall@5<=0.50且MRR<=0.35。
```

通过只允许修订E99来源gate后做本地神经排序；远程仍关闭。

`hold`：轨道组件过大、不能配平、候选不足、全示例泄漏或捷径主导。停止当前公开语料九轮神经
路线，不通过删检查、减少fold或把unlabeled改成negative修补。

`protocol_invalid`：来源、正确基、轨道canonical化、组件或候选确定性失败。只修协议。

## 5. 执行与产物

```text
run_id   = i2_present_r9_atm_support_rotation_orbit_pu_readiness_20260720
device   = local CPU
training = no
remote   = no
output   = outputs/local_audits/i2_present_r9_atm_support_rotation_orbit_pu_readiness_20260720/
```

产物包括`results.jsonl`、`gate.json`、`summary.json`、`metadata.json`、`progress.jsonl`、
`groups.csv`、`components.csv`、`fold_audit.csv`、`ranking_baselines.csv`、`curves.svg`和
`visual_qa_passed.marker`。完成后记录正式结果、刷新最近结果索引并给出E99是否可恢复的明确建议。

## 6. 正式结果

执行时间：2026-07-20。

```text
status   = pass
decision = innovation2_present_r9_atm_support_orbit_pu_ready
training = no
remote   = no
```

冻结468个独立positive形成368个同步旋转轨道；与共享坐标边合并后得到352个不可拆组件：

```text
component size 1: 267
component size 2:  67
component size 3:  11
component size 4:   3
component size 5:   2
component size 6:   2
```

最大组件仅6条，确定性装箱仍得到`78,78,78,78,78,78`。六折最小候选宽度为：

```text
train pool minimum unlabeled = 55
test pool minimum unlabeled  = 51
```

所有关键泄漏检查均为0：旋转轨道跨组、train/test全部关系（含unlabeled）精确重合、候选碰对侧
positive support、候选命中470个已知positive。边缘统计不匹配也为0。

最强非随机捷径仍为绝对bit位置：

```text
Recall@5 = 0.128205128
MRR      = 0.119001193
```

低于冻结停止线`0.50/0.35`。产物位于：

```text
outputs/local_audits/i2_present_r9_atm_support_rotation_orbit_pu_readiness_20260720/
```

`curves.svg`通过`visual-qa-redraw`的2500×1345像素检查，无文字重叠、裁切、缺字、含糊标题或
误导坐标轴。

### 下一步

E98-C证明可以在不牺牲宽度的情况下修复E98-B遗漏的候选轨道泄漏。下一步更新E99来源为本gate
`ebebd137a90c53ea9a45c0f3af8a30b02803d9f1e395f38e4d822bbd31523568`，冻结新的六个
support+orbit组件组，执行原定两seed、六折、40 epochs的本地神经排序矩阵。

远程仍关闭。E99若不稳定超过本门最强位置锚点及summary/coordinate/label-shuffle控制，停止或本地
重设计；不得用更多epoch、删泄漏检查或把unlabeled改称negative来放宽裁决。
