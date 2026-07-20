# 创新2 E105：PRESENT九轮ATM `(3,3,3)` 来源留出零适配排序门

日期：2026-07-20

状态：Freeze通过 / 12个E99 checkpoint已冻结 / Evaluate等待E104生成门

## 1. 研究问题

E99在ATM公开的八个PRESENT九轮split上发现，`coordinate_deepsets`对已知正关系相对同步旋转
unlabeled候选有稳定排序信号。E104正在生成公开语料缺失的`(3,3,3)` split。E105只回答：

```text
只用公开八个split训练并冻结的E99坐标集合模型，
不接触、不适配、不选择于新(3,3,3)关系，
能否把独立生成来源中的正关系排在其同步旋转候选前列？
```

这不是二分类、PRESENT-80密钥调度、区分器、攻击、论文结果复现或SOTA比较。

## 2. 输入与来源门

```text
training source = ATM公开8个R9 split的470条序列化关系/468维独立基
training model  = E99 coordinate_deepsets
training seeds  = 0, 1
training folds  = E98-C固定6折，每折只用其余5组
heldout source  = E104本地检索并验证通过的R9 (3,3,3) relations.json
key model       = independent 64-bit round keys
```

E105必须验证公开八文件hash、E99 gate/summary hash、E104 gate/relations hash、冻结ATM source
hash、完整64-bit模型hash、search parameter hash及`generation_passed.marker`；并在输出中生成本次
输入hash manifest。E104处于running、fallback-retrieved、
resource-cap、hold或fail时不得读取其关系进行评估。

来源留出还必须覆盖候选身份，而不只是正关系文件身份。Evaluate必须按冻结E98-C/E99协议重建
六个fold的实际训练pool，并要求每条E104正关系及其全部同步旋转未标注候选都没有作为正例或
未标注样本出现在任一fold训练pool中。任一精确关系身份交集非零均为协议无效，不解释排名。

## 3. 两阶段冻结协议

E99没有保存checkpoint，因此不能把现有指标文件冒充已冻结权重。E105分成两个严格分离的阶段：

### 3.1 Freeze阶段

- 只读取公开八个split、E99 summary/gate和E98-C来源；
- 按E99固定的6折、40 epochs、batch 32、Adam `lr=0.002`、`weight_decay=0.0001`、seed 0/1
  重放12个`coordinate_deepsets`最终epoch模型；
- 每个fold的Recall@1、Recall@5、MRR、loss和参数量必须复现E99记录；
- 保存12个state dict、逐文件SHA-256和不可变checkpoint manifest；
- Freeze阶段不接受E104路径，也不得读取任何`(3,3,3)`产物。

这里的“零适配”是指新来源不参与训练。由于历史E99未落盘权重，公开语料训练需确定性重放一次；
只有重放门通过并写出manifest后，Evaluate阶段才能启动。

### 3.2 Evaluate阶段

- 只加载已冻结checkpoint manifest和E104验证关系；
- 打开12个checkpoint并核对内部model/seed/fold/state dict、文件hash及重建fold的位置目标；
- 不创建optimizer、不调用`backward()`、不更新权重；
- 每条heldout正关系与其同步64-bit旋转候选形成一个PU排序pool；
- 候选排除公开及E104全部已知正关系，但不得把剩余unlabeled称作负例；
- 重建六个E99 fold的完整训练样本集合，要求heldout评估pool与每个fold均零精确关系重叠；
- 每个模型独立评分；每个seed内对6个fold模型的pool内标准化分数取均值，规则预先冻结；
- 同时计算六fold集成的`absolute_position`、训练坐标频率和训练支撑重合三条确定性锚点；
- Recall@5与MRR分别取三条锚点中的最大值，不能只选择对神经模型最有利的位置规则；
- 另报随机Top-5期望、Recall@1、Recall@5、MRR和Top-5 enrichment；
- 单独报告新关系与公开训练关系、训练坐标及旋转轨道的重合，不能把来源留出夸大为坐标互斥。

## 4. 冻结裁决门

`source_heldout_signal_confirmed`要求：

```text
E104 generation gate = pass，全部来源/参数/artifact hash通过；
E99 checkpoint replay = 12/12通过，权重manifest在读取E104前冻结；
六个E99 fold训练pool与E104评估pool的精确关系身份交集均为0；
heldout独立关系 >= 32，每个pool至少31个unlabeled；
两seed均 Recall@5 >= 0.50；
两seed均 MRR >= 0.40；
两seed均 Top-5 enrichment >= 5.0；
两seed均相对三条确定性规则的逐指标最大值：Recall@5 +0.20、MRR +0.15；
两seedRecall@5差 <= 0.10；
评估期间optimizer/backward/权重写入次数 = 0。
```

`diagnostic_only`：E104关系数少于32或候选宽度不足，只报告逐关系排名，不作来源泛化结论。

`source_shift_not_confirmed`：协议有效且关系宽度充足，但任一seed未过信号门。停止当前E99坐标模型的
高轮外部来源路线，不通过增加epoch、模型宽度或读取heldout调参补救。

`protocol_invalid`：任何输入hash、公开语料重放、候选正例排除、权重冻结或零更新检查失败。只修
协议，不解释指标。

## 5. 执行路径与产物

Freeze使用本地CPU，规模约为E99中最快的坐标模型子矩阵，不需要远程GPU。Evaluate也是本地CPU。

```text
freeze run = i2_present_r9_atm_e99_coordinate_checkpoint_replay_seed0_seed1_20260720
eval run   = i2_present_r9_atm_split333_source_heldout_ranking_seed0_seed1_20260720
```

Freeze至少输出checkpoint manifest、12个权重、复现指标、gate、progress和metadata。Evaluate至少输出
`results.jsonl`、逐关系rank CSV、summary、gate、metadata、progress和中文可视化。可视化必须经过
`visual-qa-redraw`像素检查；每个完成的结果阶段都刷新`outputs/00_RECENT_RESULTS.md/json`。Evaluate的
metadata还必须记录本地Git revision、E105任务/运行器/绘图/后处理源码SHA-256，以及仅覆盖这些
源码的scoped Git状态；无关工作树改动不得混入该状态。

## 6. 下一步边界

- E105通过：保留“九轮ATM通用坐标关系排序可跨公开来源迁移”的有限结论，再设计独立算法/密码
  来源复核；仍不得称为PRESENT九轮神经区分器。
- E105失败：停止E99坐标身份路线，回到关系表示或生成机制，不用heldout数据继续训练。
- E104未通过：E105 Evaluate保持关闭；按E104的probe/resource/protocol裁决处理，不伪造heldout。

禁止并行启动R10、其他R9 split、PRESENT-80攻击训练或以E104结果调参。

## 7. 正式结果

### 7.1 Freeze阶段

2026-07-20本地CPU完成公开语料checkpoint重放：

```text
status                 = pass
decision               = innovation2_present_r9_e99_coordinate_checkpoints_frozen
public source checks   = 8/8 pass
checkpoint replay      = 12/12 exact metric match
checkpoint files       = 12
heldout source read    = false
E104 relations read    = false
```

全部公开ATM文件hash、冻结commit、470条序列化关系、468维联合秩、E99 summary/gate hash均通过。
每个seed的六个`coordinate_deepsets`最终epoch模型都复现原E99逐折Recall@1、Recall@5、MRR、loss、
rank边界和参数量；12个checkpoint分别写入SHA-256 manifest。Freeze没有E104命令行参数，输出manifest
也明确记录`heldout_source_read=false`，因此新`(3,3,3)`关系未用于训练或选择。

产物：

```text
outputs/local_readiness/
i2_present_r9_atm_e99_coordinate_checkpoint_replay_seed0_seed1_20260720/
```

当前裁决只开放Evaluate readiness，不构成新的神经性能结果。下一步严格等待E104
`generation_passed`及来源/模型/参数hash检索通过，再加载已冻结权重做零适配排序；E104若进入
resource cap、hold或fail，Evaluate保持关闭。
