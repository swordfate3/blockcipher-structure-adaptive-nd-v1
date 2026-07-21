# 创新2：PRESENT完整轮间状态预测可识别性审计计划

日期：2026-07-22

状态：已完成 / pass / 完整轮间状态临界轮不可识别

## 1. 开题报告来源与冲突

《分组密码智能化检测系统设计与分析》开题报告提出两组同时存在的要求：

1. 第1页：用ResNet、时间序列网络、Transformer等预测密文输出比特，比较预测精度、收敛速度、
   泛化能力和达到随机猜测的临界攻击轮数；
2. 第4页：现有研究缺少“上一轮密文 -> 下一轮密文”的轮间预测；
3. 第11--12页：把“明文 -> 最终密文”重构为“中间轮状态 -> 下一轮状态”的迭代预测模式；
4. 第17页：把神经网络密文输出预测作为分组密码安全检测的第二项创新。

完整中间状态轮间预测与“扩散到随机猜测的临界轮”可能并不兼容。PRESENT常规轮为：

```text
Y = P(S(X xor K_r))
```

其中`X`是完整64-bit轮输入状态，`Y`是S-box和P-layer后的完整64-bit下一状态，`K_r`是当轮64-bit
子密钥。因为S-box和P-layer都是双射，只需一个完整转移对即可确定：

```text
K_r = X xor S_inverse(P_inverse(Y))
```

如果该等式在真实PRESENT-80密钥调度和未见状态上成立，那么`state_r -> state_(r+1)`任务测量的是
单轮子密钥可识别性，而不是累计扩散或随机猜测临界轮。继续比较LSTM、ResNet或Transformer会把一个
确定性100%基线包装成神经网络问题。

## 2. 审计问题

本审计只回答：

> 在已知PRESENT轮函数、固定未知PRESENT-80主密钥、完整当前/下一内部状态可见的条件下，一个校准
> 转移是否足以恢复每轮64-bit子密钥，并在全部31个常规轮上100%预测未见转移？

这是协议/可识别性审计，不训练神经网络，不使用OPB1远程GPU，不替代当前`plaintext -> r-round
ciphertext selected bits`输出预测主任务。

## 3. 冻结协议

```text
cipher                         = PRESENT-80
master-key schedule            = actual Present80._update_key
master keys                    = 16 deterministic unique 80-bit keys
regular round indices          = 1..31
calibration transitions        = 1 per (master key, round index)
held-out transitions           = 256 per (master key, round index)
held-out transition total      = 16 * 31 * 256 = 126976
final whitening observations   = 1 per master key
held-out full encryptions       = 16 * 256 = 4096
device                         = local CPU
neural training                = none
```

每把主密钥使用257个互异明文：第一个明文提供31个校准轮转移和最终whitening观察；其余256个明文
只用于未见状态预测和完整加密重构。每个明文都通过真实PRESENT轮函数追踪，不用任意伪造中间状态。

## 4. 确定性基线

对每把主密钥和每个常规轮：

```text
1. 读取一个校准(X_r, Y_r)
2. recovered_K_r = X_r xor S_inverse(P_inverse(Y_r))
3. 验证recovered_K_r等于真实密钥调度的K_r
4. 对256个未见X_r预测Y_r = P(S(X_r xor recovered_K_r))
5. 统计逐转移exact match
```

PRESENT在31个常规轮后还有最终子密钥异或。对同一个校准明文：

```text
K_32 = state_after_round31 xor ciphertext
```

用31个恢复子密钥和恢复的最终whitening key重构256个未见明文的完整31轮密文，并与
`Present80(rounds=31).encrypt`逐块比较。

## 5. 正式门

只有下列条件全部成立才通过协议审计：

```text
official PRESENT zero-key vector passes
16/16 master keys unique and actual schedule used
496/496 regular round keys recovered exactly
16/16 final whitening keys recovered exactly
126976/126976 held-out next states predicted exactly
4096/4096 held-out full encryptions reconstructed exactly
one calibration transition per (key, round) is sufficient
all result/artifact counts and hashes are finite and complete
```

通过裁决固定为：

```text
decision = innovation2_present_full_state_next_round_criticality_not_identifiable
```

如果任何等式或真实加密重构失败，状态为`fail`，必须修复轮边界、密钥调度、P/S逆映射或状态定义，
不得把失败解释为需要训练神经网络。

## 6. 证据范围与下一步

通过时可以写：

> 对PRESENT完整内部状态、固定未知子密钥和已知轮函数，完整轮间转移只需一对样本即可代数恢复当轮
> 子密钥，因此不能用该协议测量累计扩散导致的随机猜测临界轮。

不能写：

- 神经网络输出预测普遍无意义；
- 明文到多轮密文输出也能由一个样本恢复；
- 部分输出bit、隐藏内部状态或跨密钥预测同样退化；
- 该审计是密码攻击轮数或SOTA结果。

审计通过后的任务边界是：继续使用当前`P -> E_K^r(P)`真实密文输出值预测来测量累计多轮函数；若
研究轮间动态，只能隐藏部分状态、限制校准信息或改变跨密钥协议，并重新证明随机猜测临界轮可识别。
不得按原“完整state_r -> 完整state_(r+1)”直接训练多架构模型。

## 7. 产物与结果处理

```text
run_id = i2_present_next_round_full_state_identifiability_audit_20260722
output = outputs/local_audits/i2_present_next_round_full_state_identifiability_audit_20260722/

results.jsonl
summary.json
gate.json
metadata.json
progress.jsonl
artifact_manifest.json
validation.json
```

本审计不生成图，因此不触发SVG视觉门。完成后必须刷新`outputs/00_RECENT_RESULTS.md/json`，把实际
结果、裁决和下一推荐动作写回本文，并更新开题报告承诺到当前证据的矩阵。

## 8. 正式结果与裁决

正式审计已按冻结协议完成：

```text
master keys                         = 16 / 16 unique
regular rounds                      = 31
result rows                         = 496 / 496
regular round keys exact            = 496 / 496
final whitening keys exact          = 16 / 16
held-out next states exact          = 126976 / 126976
held-out next-state exact rate      = 1.000000000
held-out full encryptions exact     = 4096 / 4096
held-out full-encryption exact rate = 1.000000000
calibration transitions             = 1 per (key, round)
protocol checks                     = all true
execution checks                    = all true
artifact manifest / SHA256 validation = pass, 5/5 core artifacts
status                              = pass
decision = innovation2_present_full_state_next_round_criticality_not_identifiable
```

所有257个明文轨迹（每把密钥一个校准、256个未见）均与真实`Present80(rounds=31).encrypt`一致，
不是用任意状态或简化密钥调度构造的合成捷径。结果证明：只要完整轮输入、完整轮输出和公开PRESENT
轮结构可见，一个转移对就揭示该轮完整64-bit子密钥；对最终whitening也只需一次状态/密文异或。

产物：

```text
outputs/local_audits/i2_present_next_round_full_state_identifiability_audit_20260722/
  results.jsonl
  summary.json
  gate.json
  metadata.json
  progress.jsonl
  artifact_manifest.json
  validation.json
```

最近结果索引已刷新，本审计为完成时的`001`。

## 9. 证据支持的下一动作

保留当前主任务：

```text
输入  = 未见明文P
目标  = E_K^r(P)的完整密文、冻结单bit或冻结多bit函数值
密钥  = 固定未知秘密密钥，按独立密钥做发现/确认
```

停止直接实施：

```text
完整state_r -> 完整state_(r+1)
然后用神经网络准确率下降定义扩散临界轮
```

理由不是该方向训练困难，而是确定性基线在全部31轮恒为100%，没有“随轮数回到随机猜测”的可测
量。若论文仍要研究轮间动态，下一份计划必须先改变可见信息，例如只给部分状态、隐藏输出坐标或做
严格跨密钥预测，并先证明确定性/查表基线不会再次使任务退化。当前优先等待OPB1真实输出预测正式
结果，不启动另一组完整轮间状态神经训练。
