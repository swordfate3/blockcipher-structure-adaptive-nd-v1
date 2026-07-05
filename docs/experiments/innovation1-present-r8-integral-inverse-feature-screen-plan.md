# Innovation 1 PRESENT r8 Integral / Inverse-Round Feature Screen 计划

**日期：** 2026-07-05

**状态：** retry1 retrieved / validated / postprocessed / stop current InvP-Sinv matrix route

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

run_id = i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_20260705
configs/remote/innovation1_spn_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_20260705.json
configs/remote/generated/run_i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_20260705.cmd
configs/remote/generated/monitor_i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_20260705.sh
```

当前状态：

```text
GPU1 launch selected / local tmux monitor required after launch
```

启动依据：

```text
projection feature screen completed with weak_projection_candidate_hold
best projection AUC = 0.502584959846
full raw anchor AUC = 0.501484732144
delta = +0.001100227702
weak ensemble candidates = 0
decision = do not scale projection yet; move to next high-round data-representation screen
```

一次有界 GPU 检查显示：

```text
GPU0 busy, GPU1 free enough for 65536/class screen
```

因此本次启动 GPU1 版本，不占用正在运行的 r9 curriculum GPU0 任务。

历史暂缓原因：

```text
原计划曾因以下 watcher-managed 任务占用/等待而暂缓启动：
i1_present_r9_weak_probe_262k_seed0_gpu0_20260705
i1_present_r8_pairset_1m_seed0_gpu1_20260705
```

## Remote Launch Record

<!-- integral-inverse-feature-launch:i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_20260705:start -->

**Run ID：**

```text
i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_20260705
```

**状态：**

```text
launched / running / local tmux watcher handoff
```

**启动时间：** 2026-07-05 17:25 Asia/Shanghai

**启动依据：**

```text
projection feature screen gate = weak_projection_candidate_hold
GPU1 bounded check = available enough for this 65536/class screen
remote readiness = pass
source commit = 413d2a9 experiment: prepare integral inverse screen on gpu1
```

**远程启动路径：**

```text
G:\lxy\run_i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_20260705.cmd
G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_20260705
```

**启动确认：**

```text
started.marker present
readiness.txt present
r8_integral_inverse_feature_screen_progress.jsonl present
dataset_cache progress observed
strict negative_mode = encrypted_random_plaintexts
dataset cache root = G:\lxy\blockcipher-structure-adaptive-nd-runs\r8_integral_inverse_feature_screen_cache
```

**Watcher：**

```text
monitor_i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_20260705
```

说明：第一次 detached `start /b` invocation 返回成功但未创建 run artifacts；随后使用
`cmd.exe /c call G:\lxy\run_i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_20260705.cmd`
确认进入脚本并启动成功。该启动确认经验已记录到 `.learnings/ERRORS.md` 和 `AGENTS.md`。

<!-- integral-inverse-feature-launch:i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_20260705:end -->

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

## 7. 2026-07-05 中途协议审查

当前远程 run 仍在 watcher-managed running 状态，完整 `3/3` matrix 结果尚未
ready。已同步的第 1 行是 raw integral anchor：

```text
run_id = i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_20260705
row = 0 / 3
model = present_nibble_invp_pair_consistency_spn_only
feature_encoding = ciphertext_pair_bits
sample_structure = plaintext_integral_nibble
negative_mode = encrypted_random_plaintexts
val_auc = 0.9999795751646161
val_calibrated_accuracy = 0.998504638671875
status = partial row only, not final matrix gate
```

源码和本地 `2048/class` generator 审查说明：

```text
positive 样本两路 plaintext 都保持 active-nibble integral set；
negative 样本左路保持 integral set，右路是 encrypted random plaintexts；
对每个样本 16 个 pair 的 ciphertext pair-xor 做 pair 维度按位 XOR，
positive 的 xor_hw = 0.0，negative 的 xor_hw ≈ 31.974；
简单阈值 xor_hw <= 0 在本地审查样本上 accuracy = 1.0。
```

该审查已工具化为本地 CLI：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/audit-integral-parity-signal \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_integral_inverse_feature_screen_smoke.csv \
  --row-index 0 \
  --samples-per-class 16 \
  --seed 7 \
  --output /tmp/i1_integral_parity_audit_smoke.json
```

验证输出：

```text
positive_pair_xor_parity_hw.zero_rate = 1.0
negative_pair_xor_parity_hw.zero_rate = 0.0
best_threshold = xor_hw <= 0
best_threshold.accuracy = 1.0
interpretation = parity_statistic_alone_nearly_separates_classes
```

解释：

```text
Row 0 的强指标首先证明 plaintext-integral-nibble 数据结构含有强
integral/multiset parity signal；
不能把该 partial row 解释成 InvP/Sinv matrix 架构突破；
不能把该 partial row 写成 Zhang/Wang same-protocol 提升。
```

完整结果 ready 后：

```text
1. 仍必须等待 3/3 rows、validate-results、route-specific postprocess；
2. 若 Row 1 / Row 2 没有超过 Row 0 raw integral anchor，
   本路线归类为 integral data-construction route，而不是 inverse-round feature gain；
3. 若 Row 1 / Row 2 超过 Row 0，才讨论 InvP/Sinv 在 integral route 上的增益；
4. 后续若扩大到 262144/class，需要同时保留协议/control audit。
```

## 8. Claim Scope

本路线服务“更高轮次研究”的数据结构探索：

```text
high-round data representation screen
not same-protocol Zhang/Wang differential evidence
not Zhang/Wang same-protocol model evidence
not formal route evidence
not publication-style claim
```

## 9. Failure / Repair Record

<!-- integral-inverse-feature-failure:i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_20260705:start -->

### i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_20260705

**状态：**

```text
failed / partial / do not postprocess
```

**本地 retrieval 证据：**

```text
failed_marker = logs/i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_20260705_failed.marker
results_jsonl_line_count = 1 / 3
completed row = Row 0 raw integral anchor only
missing rows = Row 1 InvP matrix, Row 2 InvP+Sinv matrix
```

**失败原因：**

```text
torch.OutOfMemoryError during Row 1 / architecture_rank=1
model = present_matrix_trail_hybrid_pairset_invp
feature_encoding = present_pair_xor_paligned_cell_matrix_bits
input_bits = 4096
batch_size = 2048
device = cuda:1
```

stderr 摘要：

```text
CUDA out of memory. Tried to allocate 2.00 GiB.
GPU 1 total = 47.99 GiB
free at failure = 571 MiB
allocated by PyTorch = 43.25 GiB
reserved by PyTorch = 2.59 GiB
```

解释：

```text
这不是模型路线失败，也不是 inverse-round feature 负结果；
这是 batch_size=2048 对 4096-bit matrix feature 的显存失败。
该 run 只有 Row 0 partial evidence，因此不能执行 3-row gate，
也不能把 Row 0 的 AUC=0.9999795751646161 写成完整 screen 结论。
```

**Repair run：**

```text
run_id = i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705
remote_config = configs/remote/innovation1_spn_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705.json
launcher = configs/remote/generated/run_i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705.cmd
monitor = configs/remote/generated/monitor_i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705.sh
```

**Repair launch record：**

<!-- integral-inverse-feature-retry-launch:i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705:start -->

```text
status = launched / running / local watcher handoff / not ready for postprocess
readiness status = pass
expected_rows = 3
started.marker = present
progress jsonl = present
results jsonl = present, partial 2 / 3 rows
monitor first local sync = 2026-07-05T18:31:31+08:00
latest checked monitor state = running
source commit = 6c279b3ea9490f5e84ec67b0bd8d04cb01783d05
postprocess_allowed = false
needs_main_thread_intervention = false
heartbeat = fresh
```

解释：

```text
retry1 已经修复原始 OOM run 的启动/资源门，并由 watcher 持续检索本地
artifacts。但当前只有 2 / 3 result rows，因此不能执行 integral/inverse
feature gate，也不能把 partial rows 解释成完整 screen 结论。完整 3-row
matrix ready 后再运行第 3 节的 advance-integral-inverse-feature-result 命令。
```

<!-- integral-inverse-feature-retry-launch:i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705:end -->

唯一训练资源变化：

```text
batch_size: 2048 -> 512
PYTORCH_CUDA_ALLOC_CONF = expandable_segments:True
```

保持不变：

```text
plan
rounds
seed
sample_structure
negative_mode
train_key / validation_key
checkpoint_metric
dataset_cache_root
expected_rows = 3
```

Repair launch gate：

```text
1. 本地 check-remote-readiness 必须 pass；
2. git commit/push 后从 pushed commit 启动；
3. 启动后做一次 bounded artifact confirmation；
4. local tmux watcher 接管；
5. 若 retry1 再 OOM，不继续堆 GPU，改为 matrix rows 拆分或更小 batch 的单-row plan。
```

<!-- integral-inverse-feature-failure:i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_20260705:end -->

## Retrieved Integral / Inverse-Feature Result

<!-- integral-inverse-feature-postprocess:i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705:start -->
### i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705 Integral / Inverse-Feature Result

| Field | Value |
|---|---|
| Run ID | `i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705` |
| Postprocess status | `pass` |
| Validation status | `pass` |
| Raw anchor AUC | `0.999995831400` |
| InvP matrix AUC | `0.513465017546` |
| InvP+Sinv AUC | `0.505787684582` |
| Best row | `Present-RawPair-IntegralNibble-r8-65k` |
| Best feature | `ciphertext_pair_bits` |
| Delta Sinv vs raw AUC | `-0.494208146818` |
| Delta Sinv vs InvP AUC | `-0.007677332964` |
| Decision | `stop_integral_inverse_feature_screen_for_now` |
| Action | `return_to_r9_weak_probe_curriculum_or_difference_screen` |
| Next action branch | `stop_integral_inverse_feature_for_now` |
| Claim scope | `PRESENT r8 integral/inverse-round feature screen only; 65536/class is not Zhang/Wang same-protocol model evidence, paper-scale evidence, or breakthrough evidence` |
| Results JSONL | `outputs/remote_results/i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705/results/i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705.jsonl` |
| Validation report | `outputs/remote_results/i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705/i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705_local_result_gate.json` |
| Integral/inverse gate | `outputs/remote_results/i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705/i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705_integral_inverse_feature_gate.json` |
| Summary JSON | `outputs/remote_results/i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705/i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705_postprocess_summary.json` |
| Summary Markdown | `outputs/remote_results/i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705/i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705_postprocess_summary.md` |
<!-- integral-inverse-feature-postprocess:i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705:end -->
