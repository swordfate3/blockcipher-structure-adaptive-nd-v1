# Innovation 1 PRESENT r8 Integral / Inverse-Round Feature Screen 计划

**日期：** 2026-07-05

**状态：** smoke passed / 65536-class screen prepared / remote assets prepared / not launched

**研究蓝图：** `docs/research/innovation1-present-higher-round-strategy.md`

## 1. 研究问题

如果目标是推进 PRESENT 更高轮次，继续只在 Zhang/Wang Case2 `m=16`
raw differential pair 协议里微调网络可能太慢。公开高轮线索里，最相关的方向是：

```text
integral/multiset data structure
inverse permutation / inverse S-box previous-round features
partial-decryption style feature engineering
```

本 screen 问一个很窄的问题：

```text
在 strict encrypted-random-plaintext negatives 下，PRESENT r8 的
plaintext-integral-nibble / inverse-round feature 表示是否比 raw pair 表示
更容易暴露 SPN higher-round signal？
```

这不是 Zhang/Wang 同协议模型提升实验，也不是 Wu/Guo 2024 integral-neural
PRESENT 的正式复现。它是一个 high-round data-representation screen。

## 2. 唯一变化假设

固定：

```text
cipher = PRESENT-80
rounds = 8
seed = 0
pairs_per_sample = 16
sample_structure = plaintext_integral_nibble
integral_active_nibble = 0
negative_mode = encrypted_random_plaintexts
difference_profile = present_zhang_wang2022_mcnd
train_key = 0x00000000000000000000
validation_key = 0x11111111111111111111
checkpoint_metric = val_auc
```

变化：

```text
feature representation / compatible model view
```

矩阵：

| Row | Model | Feature encoding | Role |
|---:|---|---|---|
| 0 | `present_nibble_invp_pair_consistency_spn_only` | `ciphertext_pair_bits` | integral-nibble raw pair anchor |
| 1 | `present_matrix_trail_hybrid_pairset_invp` | `present_pair_xor_paligned_cell_matrix_bits` | InvP/P-layer aligned matrix candidate |
| 2 | `present_matrix_trail_hybrid_pairset_invp_sinv` | `present_pair_xor_paligned_sinv_cell_matrix_bits` | InvP + structural inverse-S candidate |

Row 1 和 Row 2 是最关键对比：同一 matrix/pairset 模型下，只增加 structural
inverse-S previous-round view，观察是否有更强高轮信号。

## 3. 配置

Smoke：

```text
configs/experiment/innovation1/innovation1_spn_present_r8_integral_inverse_feature_screen_smoke.csv
```

Screen：

```text
configs/experiment/innovation1/innovation1_spn_present_r8_integral_inverse_feature_screen_65k_seed0.csv
```

Prepared remote assets：

```text
run_id = i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu0_20260705
configs/remote/innovation1_spn_present_r8_integral_inverse_feature_screen_65k_seed0_gpu0_20260705.json
configs/remote/generated/run_i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu0_20260705.cmd
configs/remote/generated/monitor_i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu0_20260705.sh
```

当前状态：

```text
prepared / not launched
```

当前不启动远程、不占用 GPU，因为已有 watcher-managed 任务：

```text
i1_present_r9_weak_probe_262k_seed0_gpu0_20260705
i1_present_r8_pairset_1m_seed0_gpu1_20260705
```

结果 ready 后使用 route-specific advance：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/advance-integral-inverse-feature-result \
  --results outputs/remote_results/<run_id>/results/<run_id>.jsonl \
  --output-dir outputs/remote_results/<run_id> \
  --run-id <run_id> \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_integral_inverse_feature_screen_65k_seed0.csv \
  --expected-rows 3 \
  --update-plan-doc docs/experiments/innovation1-present-r8-integral-inverse-feature-screen-plan.md
```

Advance artifacts：

```text
<run_id>_local_result_gate.json
<run_id>_integral_inverse_feature_gate.json
<run_id>_postprocess_summary.json
<run_id>_postprocess_summary.md
<run_id>_advance_summary.json
<run_id>_curves.svg
<run_id>_history.csv
```

## 4. 证据等级

```text
smoke = 只证明路径能跑
65536/class = screen / diagnostic only
262144/class = medium diagnostic
1000000/class = paper-scale single-seed diagnostic
```

本 screen 即使 InvP+Sinv 明显更好，也只能说明：

```text
inverse-round data representation 值得进入 262144/class confirmation
```

不能写：

```text
PRESENT r8 正式突破
优于 Zhang/Wang Case2
复现 Wu/Guo 8-round integral-neural result
SOTA
```

## 5. Gate

若未来启动 `65536/class` screen：

| 结果 | 决策 |
|---|---|
| InvP+Sinv AUC `>= raw anchor + 0.01` 且 `>= InvP matrix + 0.005` | 准备 262144/class confirmation |
| InvP matrix 或 InvP+Sinv 只有弱正 `0 < delta < 0.01` | 作为 weak candidate，等待 r8/r9 active gates |
| 所有 inverse-round candidate `<= raw anchor` | 暂停该数据结构路线 |
| validate / plan alignment 失败 | 不看指标，先修复 |

## 6. 下一步分支

如果 screen 正向：

```text
1. 写 262144/class confirmation 计划；
2. 加 route-specific postprocess；
3. 只确认最强 inverse-round 表示 vs raw anchor；
4. 再考虑 pure Wu/Guo integral reproduction 或 1M/class。
```

如果 screen 不正向：

```text
不继续扩大这条 hybrid integral route；
回到 r9 weak-probe / r9 curriculum / r9 difference screen。
```

## 7. Claim Scope

本路线服务“更高轮次研究”的数据结构探索：

```text
high-round data representation screen
not same-protocol Zhang/Wang differential evidence
not Zhang/Wang same-protocol model evidence
not formal route evidence
not publication-style claim
```
