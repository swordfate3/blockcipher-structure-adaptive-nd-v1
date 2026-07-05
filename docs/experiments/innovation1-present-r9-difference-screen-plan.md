# Innovation 1 PRESENT r9 输入差分筛选计划

**日期：** 2026-07-05

**状态：** launched / running / local watcher handoff / awaiting 7-row postprocess gate

**Run ID：** `i1_present_r9_difference_screen_65k_seed0_gpu0_20260705`

**配置：**

```text
configs/experiment/innovation1/innovation1_spn_present_r9_difference_screen_65k_seed0.csv
configs/remote/innovation1_spn_present_r9_difference_screen_65k_seed0_gpu0_20260705.json
configs/remote/generated/run_i1_present_r9_difference_screen_65k_seed0_gpu0_20260705.cmd
configs/remote/generated/monitor_i1_present_r9_difference_screen_65k_seed0_gpu0_20260705.sh
```

## 1. 研究问题

当前 r7/r8 主线主要在 Zhang/Wang 2022 Case2 `m=16` 输入差分上比较模型结构。推进 r9/r10 时，如果仍固定同一个输入差分，失败结果无法区分：

```text
1. r9 本身没有可学习信号；
2. 当前网络没学到 SPN 结构；
3. Zhang/Wang r7 友好的输入差分并不适合更高轮。
```

本实验单独测试第 3 个问题：

```text
在固定模型、固定训练协议、固定 strict negative 定义时，PRESENT r9 是否存在比 Zhang/Wang 0x9 更有利的输入差分种子？
```

这是 **数据构造 / benchmark 搜索**，不是同协议模型提升实验。

## 2. 唯一变化假设

固定：

```text
model_key = present_nibble_invp_pair_consistency_spn_only
rounds = 9
seed = 0
samples_per_class = 65536
pairs_per_sample = 16
sample_structure = zhang_wang_case2_official_mcnd
negative_mode = encrypted_random_plaintexts
checkpoint_metric = val_auc
lr_scheduler = official_cyclic
validation_key = 0x11111111111111111111
```

唯一变化：

```text
difference_profile / difference_member
```

候选差分：

| Row | Difference | Role |
|---:|---|---|
| 0 | `present_zhang_wang2022_mcnd:0` | 当前 r7/r8 主线参考 |
| 1 | `present_wang_jain2021:0` | Wang/Jain high-probability candidate |
| 2 | `present_wang_jain2021:1` | Wang/Jain high-probability candidate |
| 3 | `present_wang_jain2021:2` | Wang/Jain high-probability candidate |
| 4 | `present_wang_jain2021:3` | Wang/Jain high-probability candidate |
| 5 | `present_autond_dbitnet2023_highround:0` | high-round screen seed |
| 6 | `present_entropy2026_gohr:0` | high-round screen seed |

## 3. 证据等级

本实验是 `65536/class` screen：

```text
screen / diagnostic only
not formal evidence
not paper-scale
not same-protocol model-improvement evidence
not breakthrough evidence
```

它只决定是否把某个输入差分候选送入 `262144/class` 确认。不能用于写“r9 已经成功”。

## 4. Gate

结果 ready 后 watcher 自动执行：

```text
validate-results
plot-results
gate-difference-screen by difference_profile:difference_member
```

手工或 monitor-health 统一 postprocess 入口：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/postprocess-difference-screen \
  --plan configs/experiment/innovation1/innovation1_spn_present_r9_difference_screen_65k_seed0.csv \
  --results outputs/remote_results/i1_present_r9_difference_screen_65k_seed0_gpu0_20260705/results/i1_present_r9_difference_screen_65k_seed0_gpu0_20260705.jsonl \
  --output-dir outputs/remote_results/i1_present_r9_difference_screen_65k_seed0_gpu0_20260705 \
  --run-id i1_present_r9_difference_screen_65k_seed0_gpu0_20260705 \
  --expected-rows 7 \
  --update-plan-doc docs/experiments/innovation1-present-r9-difference-screen-plan.md
```

决策规则：

| 条件 | 动作 |
|---|---|
| best 仍是 `present_zhang_wang2022_mcnd:0` | 继续当前差分，不把差分搜索作为优先路线 |
| 所有候选 `AUC <= 0.505` | 停止本轮 r9 差分筛选，转 curriculum / aggregation |
| best 非 Zhang/Wang 且 `delta_auc >= +0.01` | 推进该差分到 `262144/class` confirmation |
| best 非 Zhang/Wang 且 `0 < delta_auc < +0.01` | 弱候选，优先重复或升到 `262144/class` 小确认 |
| 结果不完整、validate 失败、plan 不对齐 | 不分析指标，先写 failure/repair 记录 |

## 5. 远程规则

远程运行必须满足：

```text
remote = lxy-a6000
remote root = G:\lxy\blockcipher-structure-adaptive-nd-runs
dataset cache = G:\lxy\blockcipher-structure-adaptive-nd-runs\r9_difference_screen_cache
launcher = cmd.exe /c
source = pushed GitHub commit
monitor = local tmux watcher / sub-agent retrieval
```

历史启动约束：

```text
i1_present_r9_weak_probe_262k_seed0_gpu0_20260705
i1_present_r8_pairset_1m_seed0_gpu1_20260705
```

启动条件：

```text
1. 当前 active watcher 至少释放一块 GPU，或用户明确要求停止/替换；
2. 本计划、CSV、remote config、launcher、monitor 已提交并推送；
3. scripts/check-remote-readiness 对 remote config 通过；
4. 启动后由 tmux watcher 接管，不由主线程循环轮询。
```

## Remote Launch Record

<!-- r9-difference-screen-launch:i1_present_r9_difference_screen_65k_seed0_gpu0_20260705:start -->

**Run ID：**

```text
i1_present_r9_difference_screen_65k_seed0_gpu0_20260705
```

**状态：**

```text
launched / running / local watcher handoff / not ready for postprocess
```

**本地检索到的启动证据：**

```text
readiness status = pass
expected_rows = 7
started.marker = present
progress jsonl = present
results jsonl = present, partial 3 / 7 rows
monitor first local sync = 2026-07-05T18:34:56+08:00
latest checked monitor state = running
source commit = 877d419ea8f57bce41b77ce9c86de4a97373bf97
```

**当前 gate：**

```text
postprocess_allowed = false
needs_main_thread_intervention = false
heartbeat = fresh
```

解释：

```text
该 run 已启动并由 watcher 检索本地 artifacts，但当前只有 3 / 7 result rows。
因此不能执行 difference-screen gate，也不能把 partial rows 解释成 r9 差分
筛选结论。完整结果 ready 后再运行本文件第 4 节的 postprocess 命令。
```

<!-- r9-difference-screen-launch:i1_present_r9_difference_screen_65k_seed0_gpu0_20260705:end -->

Automatic next-action source:

```text
scripts/postprocess-r9-weak-probe writes
<seed0_run_id>_next_action_readiness.json.

If the r9 weak-probe gate returns:

baseline_best_or_candidate_not_above_baseline

that artifact points directly to this difference-screen remote config with
should_launch_remote=true after local readiness and generated launcher/monitor
checks pass. This branch exists because a baseline-best r9 result says "do not
scale the current SPN candidate"; it does not by itself prove that a new input
difference is better.
```

## 6. 后续分支

如果 screen 找到强候选：

```text
r9 difference candidate 65536/class -> 262144/class confirmation -> 1M/class single-seed -> multi-seed
```

如果 screen 没有强候选：

```text
回到 r9 weak-probe / r8-to-r9 curriculum / pair-set aggregation 路线；
不继续盲目扩大差分搜索矩阵。
```
