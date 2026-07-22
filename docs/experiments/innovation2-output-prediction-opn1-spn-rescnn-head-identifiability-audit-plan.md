# 创新2 OPN1：SPN-ResCNN全局输出头拓扑可识别性审计

日期：2026-07-22

状态：本地确定性审计完成 / 最后路由可被全局输出头吸收

## 1. 研究问题

OPC1在普通ResCNN的三个残差阶段之间插入固定P-layer，并用全局
`Linear(252 * 64, 8)`输出头预测八个真实密文bit。该输出头同时读取全部64个最终位置，因此必须先
回答一个不依赖训练结果的问题：最后一次固定位置重排是否能被线性头的列重排完全吸收？

OPN1不训练模型，也不读取OPC1中间或最终AUC。它只审计实际实现的结构可识别性。

## 2. 冻结协议

```text
模型                    = OPC1普通ResCNN与SPN-ResCNN实际类
正式channels / outputs = 252 / 8
位置映射                = identity、PRESENT exact-P、OPC1 fixed wrong-P
输入头                  = 64个明文bit
输出头                  = Flatten + Linear(channels * 64, 8)
数值验证                = 固定随机hidden与head，比较路由前后重参数化输出
neural training         = no
sample classification   = no
```

对位置排列`p`，最后路由与线性头满足：

```text
y = W * flatten(h[:, :, p]) + b
  = W_p * flatten(h) + b
```

其中`W_p[:, :, p] = W[:, :, :]`。审计必须在float64下对exact-P与wrong-P都验证最大绝对误差不超过
`1e-12`，并确认普通ResCNN与混合模型的输出头形状完全相同、每个输出都连接全部64个最终位置。

## 3. 裁决门

若协议与数值等价检查全部通过，裁决为：

```text
innovation2_spn_rescnn_final_routing_absorbable_by_global_head
```

该结论只说明第三个、位于最后残差阶段之后的P-layer不能单独通过当前全局线性头识别。前两个P-layer
位于后续非线性残差阶段之前，本审计不声称它们也能被输出头吸收。因此OPC1矩阵、阈值、输出位与当前
远程任务保持不变；若OPC1的exact-P超过wrong-P，差异只能来自前两个阶段间路由及其后续非线性处理，
不能归因于最后一次路由本身。

若数值等价失败，则只修复审计推导或核对实际flatten顺序，不据此修改正在运行的OPC1。

## 4. 产物

```text
outputs/local_audits/
  i2_output_prediction_opn1_present_r3_spn_rescnn_head_permutation_identifiability_audit_20260722/
    results.jsonl
    metadata.json
    summary.json
    gate.json
    progress.jsonl
    artifact_manifest.json
    validation.json
```

本审计不生成图，因此不触发`visual-qa-redraw`。完成后刷新`outputs/00_RECENT_RESULTS.md/json`。

## 5. 结果后的下一动作

- OPC1通过：仍按冻结计划做全新固定密钥原样确认，但论文只把exact/wrong差异归因到前两个可识别的
  阶段间路由，不把最后路由列为独立贡献。
- OPC1失败或hold：保留ResCNN锚点并停止当前混合路线；下一模型假设优先检查位置绑定或共享选定位
  输出头，不通过增加深度、数据、epoch、错误P或输出位置绕过结果。

## 6. 正式结果与裁决

本地冻结审计已经完成：

```text
run_id = i2_output_prediction_opn1_present_r3_spn_rescnn_head_permutation_identifiability_audit_20260722
formal head = Linear(16128, 8)
connected final positions per output = 64
audited mappings = identity / exact PRESENT P / fixed wrong P
maximum absolute equivalence error = 1.9895196601282805e-13
status = pass
decision = innovation2_spn_rescnn_final_routing_absorbable_by_global_head
```

普通ResCNN、exact-P混合模型和wrong-P混合模型都使用相同的`Flatten + Linear(252 * 64, 8)`输出
头。三种映射都是完整64位置排列；对最后路由执行`W_p[:, :, p] = W[:, :, :]`后，未路由hidden与
重参数化head的输出在float64下最大绝对误差仅`1.99e-13`，低于冻结的`1e-12`门。

全部protocol和execution checks通过，`validation.json`重新计算五个核心产物的size与SHA-256后为
`status=pass`。该审计不生成图，也没有读取OPC1训练分数。

因此，最后一个位于第三残差阶段之后的P-layer在当前全局head下不可单独识别；前两个路由之后仍有
非线性残差阶段，本审计没有证明它们可被吸收。OPC1继续原样运行，但任何exact/wrong性能差异只能
归因到前两个阶段间路由及后续非线性处理。若OPC1未过门，下一模型假设优先约束输出头的位置绑定，
不机械扩大当前混合模型。
