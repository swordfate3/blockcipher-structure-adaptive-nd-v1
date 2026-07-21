# 创新2 OPM1：PRESENT三轮易预测输出位粗粒度结构基线审计

日期：2026-07-22

状态：本地确定性审计完成 / 粗粒度结构基线不足以解释易预测位

## 1. 研究问题

OP10冻结了八个PRESENT三轮易预测真实密文输出位：

```text
MSB-first位置 = [0, 2, 8, 10, 32, 34, 40, 42]
```

OPA1--OPB1表明网络架构会显著改变这些位置的预测能力，但尚未回答一个更简单的机制问题：这些位置
是否仅仅因为反向输入依赖锥更小，或者落在明显更简单的PRESENT S-box输出坐标上？

OPM1不训练神经网络，只计算全部64个输出bit的精确反向锥宽和四个S-box输出坐标的布尔函数基线。

## 2. 冻结协议

```text
cipher                  = PRESENT-80
rounds                  = 1, 2, 3
all output positions    = 64 MSB-first positions
selected positions      = [0, 2, 8, 10, 32, 34, 40, 42]
cone baseline           = exact inverse P-layer plus complete source S-box nibble
S-box baselines         = balance, ANF degree, nonlinearity, ANF term count
neural training         = no
sample classification   = no
```

MSB-first位置`j`必须转换为整数内部bit `63-j`，再通过最后一轮inverse P-layer恢复该密文位置对应的
S-box源bit；源bit模4才是LSB-first S-box输出坐标。禁止直接用密文内部bit模4替代inverse P-layer。

## 3. 裁决门

如果同时满足：

```text
全部64个三轮输出bit的输入锥宽相同；
八个selected bit的三轮锥宽不小于其他bit；
selected S-box坐标不具有全体坐标中唯一最低的ANF degree；
selected与全部S-box坐标具有相同nonlinearity；
四个S-box输出坐标全部平衡；
```

则裁决为：

```text
innovation2_present_r3_selected_output_not_explained_by_coarse_structure_baselines
```

这只否定“更小依赖范围”或“单个S-box坐标明显更简单”这两个粗粒度解释，不证明易预测性没有密码结构
原因，也不归因具体网络。若任一粗粒度基线出现selected特有分离，则只进入新的受控机制计划，不允许
后验重选输出位。

## 4. 产物与验证

```text
outputs/local_audits/
  i2_output_prediction_opm1_present_r3_selected_output_structural_baseline_audit_20260722/
    results.jsonl
    metadata.json
    summary.json
    gate.json
    progress.jsonl
    artifact_manifest.json
    validation.json
```

本审计不生成图，因此不触发`visual-qa-redraw`。完成后必须刷新`outputs/00_RECENT_RESULTS.md/json`。

## 5. 结果后的下一动作

- 粗粒度基线不能解释：保留OPC1冻结输出位与四行矩阵，等待其正式结果；论文把易预测位机制写成
  未完全解释，不再用锥宽或单S-box坐标复杂度作充分解释。
- 粗粒度基线出现分离：只预注册针对该单一结构量的匹配控制，不增加远程数据、epoch或输出位。

## 6. 正式结果与裁决

本地冻结审计已经完成：

```text
run_id = i2_output_prediction_opm1_present_r3_selected_output_structural_baseline_audit_20260722
all output rows = 64
selected rows = 8
round1 cone width set = [4]
round2 cone width set = [16]
round3 cone width set = [64]
selected S-box output coordinates, LSB-first = [1, 3]
selected last-round S-box source bits = [63, 55, 31, 23, 61, 53, 29, 21]
status = pass
decision = innovation2_present_r3_selected_output_not_explained_by_coarse_structure_baselines
```

通过最后一轮inverse P-layer回溯后，PRESENT四个S-box输出坐标全部平衡，nonlinearity均为`4`；
ANF degree依次为`[2,3,3,3]`，八个selected bit只使用坐标`1/3`，两者degree均为`3`，并不是唯一
最低degree坐标。全部64个输出bit的
反向输入锥宽均按`4 -> 16 -> 64`增长，八个selected bit没有更小的三轮依赖范围。

全部protocol、execution和coarse-baseline checks为真。`results.jsonl`为64行，manifest包含5个核心
产物，`validation.json`重新计算size与SHA-256后返回`status=pass`。本审计没有生成图。

因此，八个易预测位置不能仅用“输入依赖锥更窄”或“落在明显更简单的单S-box输出坐标”解释。这个
结论不排除更细的多轮代数谱、固定密钥函数或优化动力学原因，也不归因任何网络。证据支持的下一动作是
保持OPC1四行矩阵和冻结输出位不变，等待正式结果；不启动新输出位搜索或额外远程训练。
