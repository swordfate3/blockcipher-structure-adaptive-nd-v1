# 创新2 E53：PRESENT五轮开放3SDP标签提供者readiness计划

日期：2026-07-18

状态：Phase A exact oracle完成 / GLPK trail枚举器待实现

## 1. 研究问题

E52证明P0 sound active-variable support在PRESENT-80五轮完全饱和，无法产生正类；CLAASP-MP
源码语义最接近目标，但当前缺`bitstring`、`gurobipy`和已验证Gurobi license。本机Sage 9.5
的`GLPKBackend`已通过binary MILP fixture。E53测试一个问题：能否在不降低全密钥、全inactive
offset语义的前提下，用开放后端实现可复验的3SDP/monomial-trail奇偶证书。

E53不训练网络，不使用远程GPU，不把普通2SDP trail absence自动当作消去感知证书。

## 2. 固定目标与唯一变量

```text
cipher          = PRESENT-80
target rounds   = 5
active dimension= 8
output query    = 64-bit nonzero linear mask
positive        = masked cube XOR对所有key与inactive offset恒为0
negative        = 具体(key, inactive offset)上的masked XOR=1反例
fixed data      = E52 structures, masks, witness bank, splits and width gates
only change     = P0 support overapprox -> open cancellation-aware provider
```

## 3. Phase A：正确性fixture

先实现最小provider contract，不直接跑五轮：

```text
A0  PRESENT S-box ANF与P-layer bit order对拍
A1  1轮单bit/单nibble cube exact Boolean ANF正负fixture各>=8
A2  2轮小cube exact Boolean ANF正负fixture各>=8
A3  full superpoly保留key与全部非cube plaintext变量
A4  multi-bit output mask按GF(2)异或superpoly
A5  solver trail枚举结果按奇偶消去，证书可序列化并独立回代
```

必须区分：不可达、偶数条trail消去、奇数条trail存在、timeout和provider error。只返回一个
`balanced=True`且无法回放的求解结果不通过。

## 4. Phase B：冻结子集覆盖门

Phase A全部通过后，才运行E52冻结池的第一个固定子集：

```text
structures      = E52 indices 0..15
masks           = E52 indices 0..63
candidate pairs = 1024
anchor          = 同子集P0标签
negative bank   = 16 keys x 8 offsets/structure
device          = local CPU
timeout         = 逐query记录，不并入unknown
```

推进门：

```text
positive > 0
negative > 0
P1 positive - P0 positive > 0
all sampled positive certificates independently recheck
all sampled negative witnesses scalar recheck
timeout/error separated
bit-order and round convention pass
```

若通过，下一实验才是完整`96 x 300` E52池。若不通过，不换网络、不加样本、不使用经验标签。

## 5. 同预算控制与停止线

```text
control 1 = E52 P0 support overapprox
control 2 = exact Boolean ANF fixture oracle，仅用于1--2轮校准
control 3 = trail顺序重排后奇偶结果不变
control 4 = 故意取消GF(2)消去的existence-only实现必须被fixture识别
```

以下任一成立即停止当前开放provider：只能表达2SDP可达性；无法保留inactive变量；无法组合
multi-bit mask；枚举不能证明完整；fixture与exact ANF不一致；子集没有新增严格正类。不得用
远程GPU或更长timeout掩盖语义错误。

## 6. 预期产物与后续网络门

```text
outputs/local_audits/i2_present_r5_open_3sdp_provider_readiness_20260718/

provider_manifest.json
fixtures.jsonl
certificates.jsonl
witnesses.jsonl
results.jsonl
gate.json
summary.json
metadata.json
progress.jsonl
curves.svg
visual_qa_passed.marker
```

E53若未通过，五轮神经搜索继续关闭。E53与后续完整池均通过后，先运行确定性shortcut/
feature attribution，再冻结最多3行的首个五轮网络矩阵：确定性baseline、E44最强简洁神经
锚点、一个基于标签几何证据选择的结构候选。不得先验指定GraphGPS或Transformer为创新结论。

## 7. Phase A exact oracle正式结果

权威run：

```text
i2_present_r5_open_3sdp_exact_anf_phase_a_20260718
```

实现保留了完整`64 plaintext + 80 key = 144`个符号变量，并准确实现PRESENT-80真实key
schedule、S-box ANF、P-layer和final whitening。没有固定key，也没有把inactive plaintext固定为0。

```text
r1 total output ANF monomials          = 1907
r2 total output ANF monomials          = 4352830
r1 exact-vs-scalar vectors             = 8 / 8 pass
r2 exact-vs-scalar vectors             = 4 / 4 pass
r1 strict positive / negative fixtures = 8 / 8
r2 strict positive / negative fixtures = 8 / 8
multi-bit output-mask fixtures         = 4 / round
positive scalar rechecks               = all pass
negative concrete witnesses            = all pass
```

多bit mask按所选output bit完整superpoly的GF(2)异或计算；所有fixture均复核component XOR与
combined superpoly一致。非零superpoly同时确认保留key变量与inactive plaintext变量。

S-box transition完整枚举结果：

```text
candidate (input exponent, output exponent) = 256
at least one raw trail exists               = 166
odd trail parity / exact nonzero coefficient= 90
existence-only false positives              = 76
maximum even-cancelled raw trail count      = 228
```

全部trail按GF(2)奇偶得到的transition与直接S-box exact ANF逐项一致，且改变输出坐标展开顺序
不改变结果。故普通“存在一条trail”控制被明确否定，不能作为五轮正负标签。

Sage 9.5的`GLPKBackend` binary fixture通过，但GLPK trail enumerator尚未实现，五轮`16x64`
子集没有执行。Phase A裁决：

```text
status       = pass
decision     = innovation2_present_r5_open_3sdp_exact_oracle_ready
training     = no
remote_scale = no
```

下一步只实现GLPK trail枚举器，并要求它逐项复现本run的全部一、二轮fixture和76个消去控制。
通过前不执行五轮子集、不训练网络、不使用远程GPU。
