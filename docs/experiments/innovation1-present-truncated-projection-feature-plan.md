# Innovation 1 PRESENT Truncated / Projection Feature Screen 计划

**日期：** 2026-07-05

**状态：** planned / smoke config prepared / 65536-class screen config prepared / remote assets prepared / not launched

## 1. 研究问题

PRESENT r8/r9 的差分神经信号可能不是均匀分布在完整 ciphertext pair
输入上。继续把完整 16-pair raw ciphertext 全部交给一个网络，可能让网络在大量
无关位里寻找非常稀疏的 SPN 残差信号。

本 screen 验证一个更窄的问题：

```text
在不改变标签、负样本、validation key、sample structure、metric 的前提下，
只改变每个样本被网络观察到的比特投影，是否能提高 PRESENT r8 的信噪比？
```

这里的“截断”不是减少 `samples_per_class`，而是截断/投影每个样本的观察维度。

## 2. 唯一变化假设

固定：

```text
cipher = PRESENT-80
rounds = 8
seed = 0
pairs_per_sample = 16
sample_structure = zhang_wang_case2_official_mcnd
negative_mode = encrypted_random_plaintexts
difference_profile = present_zhang_wang2022_mcnd
train_key = 0x00000000000000000000
validation_key = 0x11111111111111111111
checkpoint_metric = val_auc
model_key = mlp
```

变化：

```text
feature_encoding / selected_bit_indices
```

第一轮故意使用通用 `mlp`，不使用现有 `present_nibble_*` SPN 模型，因为那些模型
内部假定单个 pair 仍是 128-bit raw ciphertext pair。投影后单 pair 宽度改变，
直接接入会把 pair 边界解释错。这个 screen 先验证数据表示本身是否含有更强信号。

## 3. 矩阵

| Row | Feature view | Encoding | Projection | Role |
|---:|---|---|---|---|
| 0 | full raw pair | `ciphertext_pair_bits` | none | same-model full-input anchor |
| 1 | raw pair P-column projection | `ciphertext_pair_bits` | 8 structure-selected nibbles from both ciphertexts | tests whether removing low-prior cells improves SNR |
| 2 | full InvP(delta) | `ciphertext_xor_spn_paligned_bits` | second 64-bit half only | tests inverse-P aligned delta as compact view |
| 3 | InvP(delta) P-column projection | `ciphertext_xor_spn_paligned_bits` | same 8 structure-selected nibbles inside InvP(delta) | strongest truncated-differential prior |

The selected nibble set is structure-defined before seeing results:

```text
nibbles = [0, 3, 4, 7, 8, 11, 12, 15]
```

It covers alternating PRESENT nibble columns after bit-order conversion, rather than
being selected from validation performance.

## 4. 配置

Smoke：

```text
configs/experiment/innovation1/innovation1_spn_present_truncated_projection_feature_screen_smoke.csv
```

Screen：

```text
configs/experiment/innovation1/innovation1_spn_present_truncated_projection_feature_screen_65k_seed0.csv
```

Prepared remote assets：

```text
run_id = i1_present_r8_truncated_projection_feature_screen_65k_seed0_gpu0_20260705
configs/remote/innovation1_spn_present_truncated_projection_feature_screen_r8_65k_seed0_gpu0_20260705.json
configs/remote/generated/run_i1_present_r8_truncated_projection_feature_screen_65k_seed0_gpu0_20260705.cmd
configs/remote/generated/monitor_i1_present_r8_truncated_projection_feature_screen_65k_seed0_gpu0_20260705.sh
```

当前状态：

```text
prepared / not launched
```

不把 smoke 结果解释为模型能力；smoke 只验证配置解析、selected-bit 投影、训练入口和
结果校验能跑通。

## 5. 弱网络融合分支

多个弱网络可以提高准确率，但前提是它们的错误不完全相关。本路线把 ensemble 作为
条件分支，而不是默认堆模型：

```text
if 至少两个 projection rows 都有 weak-positive AUC:
    准备 checkpoint/prediction ensemble 计划
else:
    不做 ensemble，先换投影规则或回到 SPN-aware architecture
```

候选融合方式：

| Fusion | 含义 | 适用条件 |
|---|---|---|
| probability mean | 平均多个模型概率 | 快速 sanity check |
| logit mean / sum logodds | 平均或求和 logits | 二分类区分器更自然 |
| calibrated weighted ensemble | 用 validation AUC/logloss 学小权重 | 需要固定 validation，避免 test leakage |
| stacking meta-classifier | 以多个模型输出为特征再训练小模型 | 只能在严格分离的 validation split 上做 |

第一版 ensemble 必须保存每个弱模型在同一 validation set 上的预测或 checkpoint，
否则无法判断互补性。不能只看 JSONL 里的最终 AUC 来“想象融合会变强”。

当前已准备最小可执行工具：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/evaluate-projection-ensemble \
  --plan configs/experiment/innovation1/innovation1_spn_present_truncated_projection_feature_screen_smoke.csv \
  --device cpu \
  --epochs 1 \
  --batch-size 8 \
  --hidden-bits 16 \
  --output outputs/smoke/i1_projection_ensemble_smoke.json
```

该工具按同一个 plan 逐行训练弱模型，固定同一个 validation split，然后输出：

```text
rows[]                  = 单模型指标和输入投影
ensembles[].mode        = probability_mean / logit_mean / auc_weighted_logit_mean
best_single             = 最强单模型
best_ensemble           = 最强融合
delta_best_ensemble_vs_single_auc
```

`auc_weighted_logit_mean` 只给 `AUC > 0.5` 的弱正模型赋权；若没有任何弱正模型，
自动退回均匀权重，防止为了提高指标而人为挑权重。这个 ensemble 结果仍然只是
same-validation diagnostic，后续必须用新 seed 或独立 confirmation split 验证。

## 6. Gate

`65536/class` screen 是 diagnostic only：

| 结果 | 决策 |
|---|---|
| 任一 projection AUC `>= full raw anchor + 0.01` | 准备 `262144/class` confirmation |
| 至少两个 projection AUC `> 0.505` 且均不低于 anchor | 准备 weak-model ensemble smoke |
| projection 全部 `<= anchor` | 停止本投影规则，不扩大；改投影先验 |
| validate-results 失败 | 不看指标，先修复 plan alignment |

若 `262144/class` 仍正向，才考虑：

```text
r8 1M/class paper-scale single-seed
r9 262144/class projection diagnostic
checkpoint/prediction ensemble confirmation
```

## 7. Claim Scope

本路线只能产生：

```text
truncated/projection data-representation diagnostic evidence
```

不能写：

```text
PRESENT r8 正式突破
PRESENT r9 已解决
ensemble 证明优于 Zhang/Wang
SOTA
```

除非后续达到 1M/class、多 seed、同协议、retrieved、validated、plan-aligned 的正式证据等级。
