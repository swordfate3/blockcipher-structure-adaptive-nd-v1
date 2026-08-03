# 图表计划

## 已生成图表

| 图号 | 文件 | 内容 | 数据状态 |
|---|---|---|---|
| 图 1 | `figures/fig_uknit_k1bz_published_architecture_comparison.svg` | uKNIT K1-BS/K1-BZ 2048/class：结构专家、AutoND、MCND 与 Liu Conv2D AUC | 本地闭环架构诊断，gate=`hold` |
| 图 2 | `figures/fig_uknit_k1u_semantic_position.svg` | uKNIT K1-U 65536/class：正确/错误 S 盒与位置不变控制 | 原始回收并重裁决；语义有效、位置必要性未获支持 |
| 图 3 | `figures/fig_uknit_k1bv_boundary.svg` | uKNIT K1-BV r6 AUC、pair 增益与 S 盒语义差值 | 远程 gate=`hold`，归档待闭环 |
| 图 4 | `figures/fig_dialga_d2_same_checkpoint.svg` | Dialga D2 同一检查点替换结构说明书 AUC | 本地闭环机制证据 |
| 图 5 | `figures/fig_dialga_dmc2_auc.svg` | Dialga DMC2 262144/class：正确拓扑、扰动拓扑和 AutoND AUC 与差值 | 原始回收并本地重裁决；本文最高规模，非正式基准 |

## 计划补充图表

## 图 6：整体框架图

标题：运行时结构描述驱动的异构 SPN 神经区分框架

```text
密码规范 / 结构配置
→ 结构编译器
→ 运行时结构说明书
→ 结构解释器
→ 差异化专家：位置直方图专家 / GF(2) 拓扑专家 / 通用密文对专家
→ 门控融合
→ 神经区分输出
```

## 图 7：uKNIT 结构专家信息流

```text
16 对 64-bit 密文
→ 多阶段逆线性 / 逆 S盒
→ 阶段 × cell 的 16-bin 直方图
→ 位置保持结构残差
→ 区分概率
```

重点标出：阶段、cell 位置、S盒语义。

## 图 8：Dialga 拓扑专家信息流

```text
4 对 128-bit 密文
→ bit 三通道特征
→ 非连续 cell 聚合
→ GF(2) 源 bit 到目标 bit 消息传递
→ 异构轮循环
→ 区分概率
```

重点标出：非连续 cell、GF(2) 接线图、正确/错误拓扑替换。

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

当前五张 SVG 图均在本地生成，数据来自项目中已核验的 `results.jsonl`、`gate.json` 或 `progress.jsonl` 摘要。K1-BZ、K1-U、K1-BV、D2 和 DMC2 已按主稿引用版本完成像素级检查；K1-U/DMC2 保留原始回收限定，K1-BV 保留归档未闭环限定，DMC2 还保留无独立最终测试限定。协议无效的 K1-BT 图和用户终止的 DFC1 局部结果均不进入正文。
