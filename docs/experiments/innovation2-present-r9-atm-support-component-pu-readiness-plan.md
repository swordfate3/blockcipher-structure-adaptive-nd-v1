# 创新2 E98-B：PRESENT九轮ATM支撑组件互斥PU排序就绪门

日期：2026-07-20

状态：完成 / pass / 只开放E99本地神经排序

## 1. 研究问题

E98-A确认冻结8文件语料不是470维，而是470个去重关系张成的468维GF(2)空间。原E98按来源
文件留出只得到24条独占正例，但同一关系跨文件出现不代表它必须从语料删除。E98-B回答：

```text
把468个canonical独立正关系按“共享精确坐标”连成不可拆支撑组件后，
能否在不拆组件的前提下构造6个关系数配平、train/heldout支撑坐标完全互斥的组，
并为每个heldout正例保留足够的边缘匹配unlabeled候选？
```

这是同一公开ATM语料内部的结构泛化门，不宣称独立论文来源，也不生成负类。

## 2. 冻结构造

```text
source gate = E98-A pass / public_merge_count_not_rank
positive bank = canonical coordinate order下GF(2)消元保留的468个独立关系
component edge = 两个关系共享至少一个完整(u,v)坐标
component rule = 连通分量不可拆
group count = 6
packing = component按(-size, canonical id)排序，贪心放入当前最小组
```

候选仍使用E98的64位输入/输出指数同步循环平移`1..63`，但追加过滤：候选的任何完整坐标都不得
出现在该fold的train support中。同步平移只构造边缘匹配unlabeled，不假设它保持密码学性质。

每个候选必须保持关系项数、输入/输出重量多重集、不同指数数、内部异或距离和Cartesian形状；
必须排除全部470个已知关系。指标只用已知正例恢复的Recall@K/MRR，不计算二分类accuracy/AUC。

## 3. 同预算基线

冻结复用E98基线：确定性hash随机、group id、关系项数、指数重量、train关系频率、train坐标频率、
train支撑重合和绝对bit位置。group id及所有train支撑基线必须在同一pool内无法区分类别。

## 4. 裁决门

```text
advance_to_E99_local:
  E98-A gate hash/status/decision通过；
  canonical独立正关系恰好468，删除依赖元素恰好2；
  所有支撑组件内部完整、组件间坐标互斥；
  6组各至少64条正例，最大最小组差<=1；
  每折train/heldout relation、component、support coordinate重合均为0；
  每个正例至少31个过滤后unlabeled；
  unlabeled中已知positive为0，边缘漂移为0；
  最佳非随机捷径Recall@5<=0.50且MRR<=0.35。
  -> 只开放E99本地PU排序；远程仍关闭。

hold:
  组件过窄/过大、不能配平、候选不足、泄漏或捷径主导。
  -> 不训练E99，停止当前九轮神经路线。

protocol_invalid:
  冻结来源、消元、组件或候选确定性失败。
  -> 只修协议。
```

## 5. 执行

```text
run_id   = i2_present_r9_atm_support_component_pu_readiness_20260720
device   = local CPU
training = no
remote   = no
output   = outputs/local_audits/i2_present_r9_atm_support_component_pu_readiness_20260720/
```

产物：`results.jsonl`、`gate.json`、`summary.json`、`metadata.json`、`progress.jsonl`、
`groups.csv`、`components.csv`、`candidate_pools.csv`、`ranking_baselines.csv`、`curves.svg`和
`visual_qa_passed.marker`。完成后更新正式结果并刷新最近结果索引。

## 6. 正式结果

执行时间：2026-07-20。

```text
status   = pass
decision = innovation2_present_r9_atm_support_component_pu_ready
training = no
remote   = no
```

正确消元从470个去重关系中保留468个canonical独立正关系、删除2个依赖元素。共享完整
`(u,v)`坐标得到452个不可拆支撑组件，其中444个大小为1、8个大小为3。确定性贪心装箱得到：

```text
group sizes = 78, 78, 78, 78, 78, 78
max train/heldout relation overlap   = 0
max train/heldout component overlap  = 0
max train/heldout coordinate overlap = 0
```

468个正例各自形成一个排序pool。过滤后的同步旋转候选数为`51..63`；候选与全部470个已知
positive重合为0，边缘统计不匹配为0，重复执行漂移为0。它们仍只能称`unlabeled`，不能称负例。

最强非随机捷径是绝对bit位置：

```text
Recall@5 = 0.132478632
MRR      = 0.120060420
```

低于冻结停止线`Recall@5<=0.50`和`MRR<=0.35`。其余非随机捷径Recall@5为
`0.096153846`，MRR为`0.081816420`。因此任务没有被关系项数、训练频率、支撑重合等简单规则
直接垄断。

产物位于：

```text
outputs/local_audits/i2_present_r9_atm_support_component_pu_readiness_20260720/
```

`curves.svg`经`visual-qa-redraw`实际渲染为2500×1351像素检查，未发现文字重叠、裁切、缺字、
含糊标题、不可辨曲线或误导坐标轴。

## 7. 证据裁决与下一步

E98-B回答的是“现有九轮公开正关系能否构成一个没有明显泄漏的PU排序数据门”，答案为可以；
它本身不是神经结果，也没有产生新九轮关系或严格负类。

推荐下一步是立即预注册并执行E99本地六折神经排序门：固定当前468个正例、组件分组、候选池和
排序指标，只改变关系编码/模型。精简矩阵为最强确定性捷径、summary-only控制、coordinate-set
模型和topology-aware候选；按fold报告Recall@1、Recall@5、MRR和相对确定性基线的提升。

E99只有在六折聚合优于最强捷径、不是summary控制可解释、无单折崩溃且重复seed方向一致时，
才允许设计远程扩大。禁止把unlabeled改称negative、禁止改用二分类accuracy/AUC、禁止跳过本地门
直接上远程GPU，也禁止把同语料内部六折称为独立论文复现或PRESENT-80密钥调度证据。

## 8. 事后训练前限定（2026-07-20）

E99实现前增加“所有train/test示例的精确关系也不得重合”检查后，support-only六折出现部分train
pool候选数降为0。原因不是正例/坐标组件泄漏，而是同步旋转候选的等价轨道跨组：不同组positive
可产生相同unlabeled关系。若直接训练，网络可能记住测试候选身份并把它们压低。

因此本门原始`pass`只对已预注册的positive关系/组件/support互斥和捷径检查有效；它不足以单独开放
E99训练。E99现由E98-C“support + rotation-orbit组件互斥”新门接管。E98-B产物与原裁决保持不变，
不做事后改写。
