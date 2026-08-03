# 图表计划

## 已生成图表

| 图号 | 文件 | 内容 | 数据状态 |
|---|---|---|---|
| 图 1 | `figures/fig_method_framework.svg` | 公开结构、运行时说明书、uKNIT/Dialga 差异化专家与同协议对照评估 | 方法总图；已完成像素验收 |
| 图 2 | `figures/fig_uknit_k1bz_published_architecture_comparison.svg` | uKNIT K1-BS/K1-BZ 2048/class：结构专家、AutoND、MCND 与 Liu Conv2D AUC | 本地闭环架构诊断，gate=`hold`；不能代替主规模比较 |
| 图 3 | `figures/fig_uknit_k1u_semantic_position.svg` | uKNIT K1-U 65536/class：正确/错误 S 盒与位置不变控制 | 原始回收并重裁决；语义有效、位置必要性未获支持 |
| 图 4 | `figures/fig_uknit_k1bv_boundary.svg` | uKNIT K1-BV r6 AUC、pair 增益与 S 盒语义差值 | 远程 gate=`hold`，归档待闭环 |
| 图 5 | `figures/fig_dialga_d2_same_checkpoint.svg` | Dialga D2 同一检查点替换结构说明书 AUC | 本地闭环机制证据 |
| 图 6 | `figures/fig_dialga_dmc2_auc.svg` | Dialga DMC2 262144/class：正确拓扑、扰动拓扑和 AutoND AUC 与差值 | 原始回收并本地重裁决；本文最高规模，非公开论文精确复现 |

## 投稿前唯一待生成主图

K1-CA/K1-CB 回收后，从同一合并结果表机械生成 uKNIT 五模型 `262144/class` 主图，内容包括结构专家、AutoND/DBitNet、MCND、Liu Conv2D 和 Gohr-style ResNet 的逐 seed AUC、均值/范围及相对 AutoND 差值。该图必须与主表共享一个数据源，不能手工抄录；生成后执行独立像素验收。最终编排时应把它放在 RQ1 正文位置，当前图 2 的小样本先导图可移至附录，避免先导诊断占据主结果图位。

## 表 1：实验协议总览

列：算法、轮数、pair 数、samples/class、seed、模型、负样本、验证密钥、状态。

## 图表审计要求

每张图必须标注：

```text
rounds
samples/class
pairs_per_sample
seed
metric
negative_mode
是否 gate 闭环
```

当前六张正文 SVG 均已在本地生成并完成像素级检查。实验图数据来自项目中已核验的 `results.jsonl`、`gate.json` 或 `progress.jsonl` 摘要；方法图只描述已实现接口、差异化专家与同协议评估，不暗示共享权重或自动生成任意 SPN 最优网络。K1-U/DMC2 保留原始回收限定，K1-BV 保留归档未闭环限定，DMC2 还保留无独立最终测试限定。协议无效的 K1-BT 图和用户终止的 DFC1 局部结果均不进入正文。
