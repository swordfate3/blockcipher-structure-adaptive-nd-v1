# 创新2 OPK1：PRESENT三轮 key-blind 跨密钥目标稳定性审计计划

日期：2026-07-22

状态：正式审计完成 / key-blind零样本密钥泛化目标不稳定

## 1. 研究问题

OP10--OPA3在多把独立固定秘密密钥上重复了相同八输出任务，但每把密钥都重新生成训练数据并重新
训练模型。它们证明的是固定协议跨密钥可重复，不是“同一个模型在完全未见秘密密钥上零适配预测”。

若模型输入只有明文`P`而不包含密钥、密钥身份或任何已知明密文校准对，那么跨独立密钥时，同一
明文对应的真实密文bit可能根本不是稳定标签。OPK1只回答：

> 对PRESENT三轮和冻结八个真实密文输出bit，即使参考密钥与评估密钥共享完全相同的1024个明文，
> 使用参考密钥上每个`(P, bit)`的经验频率作为最有利的key-blind查表分数，能否预测完全未见密钥
> 上的真实输出值并超过全局多数基线？

这是目标可识别性审计，不是神经网络实验，不测试四轮、XOR、真假样本分类或已知密钥输入。

## 2. 冻结协议

```text
cipher                         = PRESENT-80
rounds                         = 3
selected MSB positions         = [0,2,8,10,32,34,40,42]
shared plaintexts              = 1024 unique deterministic 64-bit values
reference keys                 = 256 unique deterministic 80-bit keys
held-out evaluation keys       = 256 unique deterministic 80-bit keys
key overlap                    = 0
reference observations         = 262144 plaintext-key pairs
evaluation observations        = 262144 plaintext-key pairs
neural training                = none
sample classification          = false
device                         = local CPU
```

共享同一组明文是刻意给key-blind方法的最有利条件：它无需泛化到新明文，只需判断参考密钥上同一个
明文的输出频率能否转移到未见密钥。若这一条件仍接近随机，则更严格的“新密钥且新明文”不会由当前
证据自动成立。

## 3. 冻结基线

对每个共享明文`P`和输出位置`b`，只使用256把参考密钥计算：

```text
score(P,b) = mean_K_reference bit_b(E_K^3(P))
```

随后把同一`score(P,b)`用于256把未见评估密钥的全部真实标签，报告：

```text
held-out-key AUC
threshold accuracy at score >= 0.5
evaluation majority accuracy
accuracy - majority
Brier score
reference/evaluation global prevalence
evaluation-key per-plaintext prevalence deviation from 0.5
```

全局多数只使用评估集总体标签比例计算，是描述性常数基线。不得使用评估密钥的逐明文频率生成预测，
因为那会把测试标签泄漏进分数。

## 4. 正式门与裁决

全部PRESENT向量、位序、密钥/明文唯一性、密钥零重合、真实密文回放、结果数量和SHA256产物必须先
通过。

若八bit平均满足：

```text
mean directional AUC = mean(max(AUC, 1-AUC)) <= 0.510
mean accuracy - majority <= +0.005
每个bit directional AUC <= 0.515
每个bit evaluation prevalence in [0.48, 0.52]
```

则裁决为：

```text
innovation2_present_r3_key_blind_zero_shot_target_not_stable
```

含义是当前`P -> C_selected8`任务只有在固定秘密密钥条件下才是确定函数；论文中的“跨密钥泛化”
必须写成多把独立固定密钥上的逐密钥训练重复性，不能写成单模型对未见密钥零适配预测。

若查表分数超过门，只能裁决为“发现候选跨密钥偏差，需要全新密钥集独立复验”，不得直接训练大模型
或声称跨密钥泛化。

## 5. 证据边界与下一步

通过不意味着所有跨密钥学习都不可能。下列任务仍是不同问题：

- 输入显式包含密钥；
- 输入包含少量已知明密文support set，用于推断固定密钥上下文；
- 元学习模型在新密钥上经过校准或微调；
- 当前采用的每把固定密钥分别训练、在未见明文上测试。

通过时，创新2主线继续保留最后一种固定密钥真实输出预测，OPB1/OPC1的模型裁决不受影响。以后若要
主张更强泛化，应优先预注册“support-set条件预测”而不是无密钥、无校准的key-blind零样本任务。

## 6. 产物

```text
run_id = i2_output_prediction_opk1_present_r3_key_blind_target_stability_audit_20260722
output = outputs/local_audits/i2_output_prediction_opk1_present_r3_key_blind_target_stability_audit_20260722/

results.jsonl
summary.json
gate.json
metadata.json
progress.jsonl
artifact_manifest.json
validation.json
```

不生成图。完成后刷新最近结果索引，更新开题承诺矩阵、创新2论文边界和本计划的实际指标与下一推荐
动作。

## 7. 正式结果与裁决

冻结的`1024`个共享明文、`256`把参考密钥和`256`把零重合评估密钥已经完整运行：

```text
reference observations                    = 262144
held-out-key evaluation observations       = 262144
mean AUC                                   = 0.500543701
mean directional AUC                       = 0.501219550
maximum per-bit directional AUC            = 0.502176345
mean accuracy - evaluation majority        = -0.000466347
evaluation prevalence range                = [0.499279022, 0.501678467]
mean exact selected8 majority per plaintext = 0.018711090
```

八个输出bit的方向化AUC分别均不超过`0.502177`，远低于冻结上限`0.515`；所有评估集总体标签比例都
位于`[0.48,0.52]`，参考密钥逐明文频率在未见密钥上没有稳定预测作用。密钥集合各自唯一、严格零
重合，1024个明文完全相同且唯一；参考分数不读取评估标签，真实标签抽查回放、八行结果、全部指标和
SHA256产物验证均通过。

正式gate为：

```text
status = pass
decision = innovation2_present_r3_key_blind_zero_shot_target_not_stable
protocol / execution / stability checks = all true
artifact SHA256 validation = pass
```

因此论文中的泛化结论冻结为：同一八输出位置和训练协议在多把独立固定秘密密钥上经逐密钥重新训练后
具有重复性；当前没有、也不应声称一个只看明文的模型能零适配预测完全未见密钥。若继续更强泛化，
下一研究对象应是带少量已知明密文support set的密钥上下文条件预测，并另行预注册；不训练无密钥、
无校准的key-blind网络。OPB1和OPC1固定密钥主线保持不变。
