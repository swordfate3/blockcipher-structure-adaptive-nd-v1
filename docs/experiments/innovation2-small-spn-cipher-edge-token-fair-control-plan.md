# 创新2 E35b：Cipher Edge-Token Transformer公平拓扑控制重裁决

日期：2026-07-18

状态：已完成 / hold / 合成神经架构搜索关闭

## 1. 纠正原因

原E33--E35的`topology_mode=shuffled`实现为：

```python
np.roll(player_array, shift=1, axis=0)
```

variant顺序是`s0p0,s0p1,s0p2,s0p3,...,s3p3`，训练与heldout为：

```text
train         = S0..S2 x P0..P2
unseen-P      = S0..S2 x P3
dual-unseen   = S3 x P3
```

因此旧控制把所有heldout `P3`替换成训练见过的`P2`，同时把每个S-box组的`P0`替换成
`P3`。旧wrong-P控制改变了拓扑，也改变了train/heldout分布难度，不是公平归因控制。

原E35的真实CETT仍低于ID边际，所以“未通过”方向没有被该错误推翻；但
`true P - rolled P = -0.151122`不能解释为网络偏好错误拓扑。原E35重分类为
`innovation2_small_spn_cipher_edge_token_protocol_invalid`。

## 2. 公平控制

E35b对每个variant自己的P-layer应用同一个固定destination-cell旋转：

```text
destination cell 0 -> 1
destination cell 1 -> 2
destination cell 2 -> 3
destination cell 3 -> 0
lane role保持不变
```

形式为`wrong_P = q compose P`，其中`q`是固定非平凡permutation。它满足：

```text
每一行仍来自同一个variant/P-family
P3控制仍由P3派生，不替换成P0/P1/P2
每个wrong P仍是合法16-bit permutation
标签、split、S-box、active、mask全部不变
```

## 3. 冻结矩阵与预算

从头重跑，避免混合旧checkpoint：

| 行 | seed |
|---|---|
| CETT true P | 0,1 |
| CETT fair-corrupted P | 0,1 |
| CETT label shuffle | 0 |

模型、token、数据、cell split和训练预算完全沿用E35：

```text
37 tokens
hidden64
3 Transformer layers
4 heads
FFN128
dropout0.10
AdamW, lr1e-3, weight_decay1e-4
batch128
40 epochs
```

readiness仍使用hidden32、2层、8 epochs三行seed0。

## 4. 控制与表示门

```text
token count = 37
cell relabeling max logit error <= 1e-6
heldout P3 control is not any train P0/P1/P2
every corrupted row is a permutation of 0..15
matrix uses topology_mode=corrupted, not shuffled
```

## 5. 裁决门

沿用E35冻结门：

```text
label-shuffle dual <= 0.60
true P两seed dual均 > 0.726528
true P mean dual >= 0.756528
true P mean dual >= fair-corrupted P mean + 0.03
unseen-S mean >= 0.765693
unseen-P mean >= 0.732532
```

通过才准备真实密码迁移readiness；任何有效失败都关闭当前合成benchmark上的神经架构
搜索并返回标签/任务设计。不得因修复控制而改变true行门槛。

## 6. 产物

```text
smoke = outputs/local_smoke/i2_small_spn_cipher_edge_token_fair_control_smoke_seed0_20260718/
full  = outputs/local_diagnostic/i2_small_spn_cipher_edge_token_fair_control_seed0_seed1_20260718/
```

## 7. 实际执行与结果

readiness：

```text
run_id = i2_small_spn_cipher_edge_token_fair_control_smoke_seed0_20260718
decision = innovation2_small_spn_cipher_edge_token_readiness_passed
token_count = 37
cell relabeling max error = 5.960464477539063e-08
fair_control_heldout_avoids_train_players = true
all_corrupted_players_are_permutations = true
```

完整重裁决：

```text
run_id = i2_small_spn_cipher_edge_token_fair_control_seed0_seed1_20260718
rows   = 5
epochs = 40
parameters = 168321
```

| 方法 | unseen-S | unseen-P | dual-unseen |
|---|---:|---:|---:|
| ID边际 | 0.775693 | 0.742532 | 0.726528 |
| CETT true P | 0.829973 | 0.649425 | 0.671767 |
| CETT fair-corrupted P | 0.828069 | 0.692038 | 0.664944 |
| CETT label shuffle | 0.560215 | 0.592531 | 0.484934 |

逐seed dual：

```text
true P           = 0.695197 / 0.648338
fair-corrupted P = 0.650097 / 0.679791
```

true两seed与原E35完全复现。公平控制把旧rolled-P的`0.822889`降到`0.664944`，验证旧
高控制分数来自分布偏置；但true只领先公平控制`+0.006823`，仍比ID边际低`-0.054761`，
两seed均未过基线。

最终裁决：

```text
status       = hold
decision     = innovation2_small_spn_cipher_edge_token_not_ready
remote_scale = false
```

因此CETT没有建立真实P-layer贡献，当前合成benchmark上的GraphGPS、SCGT、round-shared
reasoner和edge-token Transformer架构搜索关闭。下一步不再训练新网络，先执行E36拓扑
标签可识别性审计：按固定S-box比较P0--P3标签翻转率、P-layer条件interaction和train-only
matched选择后的P敏感cell宽度。只有标签本身存在足够、组外可验证的P-layer交互，才允许
重新设计benchmark；否则创新2返回确定性候选提供者或不同任务标签，而非继续搜架构。
