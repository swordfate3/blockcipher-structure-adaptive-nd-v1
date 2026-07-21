# 创新2：PRESENT轮间八输出bit部分子密钥可识别性审计计划

日期：2026-07-22

状态：正式审计完成 / 部分子密钥恢复退化确认

## 1. 来源问题

完整`state_r -> state_(r+1)`审计已经证明：一个完整状态对即可恢复PRESENT当轮全部64-bit子密钥，
因此该协议不能测量扩散到随机猜测的临界轮。一个自然修订是只公开下一状态的少量输出bit。

当前真实输出主线冻结的八个MSB-first位置为：

```text
[0, 2, 8, 10, 32, 34, 40, 42]
```

在PRESENT一个常规轮中，这八个位置经过逆P-layer后对应四个S-box输出nibble，每个nibble各公开两个
输出bit。只要完整当前轮状态仍可见，多组样本可能足以枚举并唯一确定四个4-bit子密钥nibble。

本审计回答：

> 在完整当前状态、冻结八个下一轮输出bit、公开PRESENT轮函数和固定未知轮密钥下，最多16个校准
> 转移是否足以恢复控制这八个输出的16-bit部分子密钥，并100%预测未见八bit输出？

如果成立，则“完整当前状态 -> 八个下一状态bit”仍主要是部分子密钥恢复，不适合作为累计扩散临界轮
任务。它不否定当前`plaintext -> r-round ciphertext selected bits`主任务，因为后者看不到真实中间状态。

## 2. 冻结几何

对每个MSB-first输出位置`m`：

```text
destination integer bit = 63 - m
source S-box output bit  = P_inverse(destination)
```

冻结映射应为：

| 输出MSB位置 | 逆P源integer bit | S-box nibble | S-box输出bit角色 |
|---:|---:|---:|---:|
| 0 | 63 | 15 | 3 |
| 32 | 61 | 15 | 1 |
| 2 | 55 | 13 | 3 |
| 34 | 53 | 13 | 1 |
| 8 | 31 | 7 | 3 |
| 40 | 29 | 7 | 1 |
| 10 | 23 | 5 | 3 |
| 42 | 21 | 5 | 1 |

因此目标只依赖当轮子密钥的nibble`[15, 13, 7, 5]`，合计16-bit。审计代码必须从真实P-layer动态
推导并核对该表，不手工把结果写死后直接通过。

## 3. 冻结协议

```text
cipher                         = PRESENT-80
master-key schedule            = actual Present80._update_key
master keys                    = 16 deterministic unique 80-bit keys
regular round indices          = 1..31
selected next-state bits       = [0,2,8,10,32,34,40,42] MSB-first
maximum calibration pairs      = 16 per (master key, round)
held-out transitions           = 256 per (master key, round)
key-round instances            = 16 * 31 = 496
subkey-nibble instances        = 16 * 31 * 4 = 1984
held-out transition total      = 16 * 31 * 256 = 126976
device                         = local CPU
neural training                = none
```

每把主密钥使用272个互异明文：前16个只用于候选子密钥枚举，其余256个只用于未见八bit预测。每个
明文都用真实PRESENT-80密钥调度追踪31轮；候选器只能读取完整当前状态和冻结八bit观察，不能读取
完整下一状态、真实轮密钥或未选择输出bit。

## 4. 确定性候选基线

对每个`(master key, round, affected nibble)`：

```text
candidates = {0,1,...,15}

for each calibration pair:
    x = current_state[nibble]
    observed = two selected next-state bits sourced from this nibble
    keep k only when projected_bits(SBOX[x xor k]) == observed
```

记录候选集合第一次缩小为唯一值所需的最小样本数；唯一候选还必须等于真实密钥调度的当轮key
nibble。四个nibble都恢复后，仅用这16-bit部分子密钥预测256个未见当前状态对应的八bit输出。

## 5. 正式门

```text
official PRESENT vector passes
selected8 inverse-P geometry matches the frozen four-nibble/two-role table
16/16 master keys unique and actual schedule used
1984/1984 actual key nibbles remain consistent with every calibration observation
1984/1984 key nibbles become unique within at most 16 calibration pairs
1984/1984 unique candidates equal the actual round-key nibble
126976/126976 held-out selected8 vectors predicted exactly
candidate code receives no unselected next-state bits or actual round key
496 result rows, progress, metadata, gate, summary and checksummed artifact validation complete
```

若全部通过，裁决为：

```text
innovation2_present_selected8_next_round_is_partial_subkey_recovery_not_diffusion_criticality
```

若候选没有全部唯一，只能报告实际可识别比例和候选宽度；不得自动训练神经网络或放宽16样本上限。

## 6. 证据边界和下一步

通过时可以写：

> 对完整PRESENT当前内部状态和冻结八个下一状态bit，少量校准样本可恢复控制这些输出的四个子密钥
> nibble；该任务主要测量部分子密钥可识别性，而不是累计多轮扩散。

不能写：

- 未见明文到三轮密文八bit也退化为少量样本子密钥恢复；
- 所有部分状态协议均无意义；
- 已恢复完整64-bit轮密钥；
- 这是神经攻击轮数或SOTA结果。

若通过，轮间动态要继续就必须进一步隐藏当前内部状态、采用严格跨密钥函数学习，或定义不能被局部
子密钥枚举解决的受限观测。当前GPU训练仍只保留OPB1端到端真实密文输出主线。

## 7. 产物

```text
run_id = i2_present_next_round_selected8_partial_subkey_identifiability_audit_20260722
output = outputs/local_audits/i2_present_next_round_selected8_partial_subkey_identifiability_audit_20260722/

results.jsonl
summary.json
gate.json
metadata.json
progress.jsonl
artifact_manifest.json
validation.json
```

不生成图，不触发SVG视觉门。完成后刷新最近结果索引，更新创新2权威边界、开题承诺矩阵和本计划的
实际结果及下一推荐动作。

## 8. 正式结果与裁决

冻结协议已经完整运行：

```text
master keys                         = 16/16
regular rounds                      = 31
key-round result rows               = 496/496
affected key nibbles                = [15,13,7,5]
subkey-nibble instances             = 1984/1984
actual nibbles remain consistent    = 1984/1984
unique within <=16 calibration      = 1984/1984
unique candidate equals actual      = 1984/1984
held-out selected8 exact            = 126976/126976
held-out selected8 exact rate       = 1.0
maximum samples to unique           = 11
mean samples to unique              = 3.115423387
```

逆P几何从真实`Present80.inverse_permutation_layer`动态推导并与冻结表完全一致。候选更新函数的可见
参数只有`current_nibble`、两个已公开角色bit、角色编号和当前候选集合；不接收完整下一状态、未选择
输出bit或真实轮密钥。真实轮密钥只在候选恢复结束后用于审计比对。

正式gate为：

```text
status = pass
decision = innovation2_present_selected8_next_round_is_partial_subkey_recovery_not_diffusion_criticality
protocol checks = all true
execution checks = all true
artifact SHA256 validation = pass
```

因此开题提出的“完整当前内部状态 -> 八个下一状态bit”仍不适合作为随机猜测临界轮实验。它测量的是
四个局部轮密钥nibble的可识别性，而不是多轮累计扩散。该结果不否定`plaintext -> E_K^r(P)`真实
密文输出预测，因为当前主任务没有看到任何真实中间状态。

证据支持的下一动作是：不训练完整当前状态版本的selected8轮间神经网络；继续OPB1端到端真实密文
输出主线。若以后恢复轮间动态研究，必须先预注册更严格的部分当前状态可见性或跨密钥零适配协议，并
重新运行确定性可识别性门。不得仅换神经网络、增加轮数或减少输出bit来绕过本裁决。
