# Innovation 1 PRESENT r8 轮数推进阶梯实验计划

**日期：** 2026-07-04

**状态：** prepared / pending local verification / pending remote readiness

**研究蓝图：** `docs/research/innovation1-present-round-extension-research-plan.md`

**实验配置：** `configs/experiment/innovation1/innovation1_spn_present_round_extension_r8_262k_seed0.csv`

## 1. Research Question

在保持 Zhang/Wang 2022 Case2 `m=16`、strict `encrypted_random_plaintexts`、同训练协议、同样本规模和同 metric 的前提下：

```text
PRESENT-80 r8 是否还存在可学习的 real-vs-random 神经区分信号？
如果存在，InvP/P-layer aligned representation 或 pair-set consistency
是否比 Zhang/Wang-style MCND 更能保留这个信号？
```

这不是 r7 attribution 的重复实验，而是 round-extension ladder 的第一步。

## 2. Same-Budget Baseline

同预算 baseline：

```text
present_zhang_wang_keras_mcnd
```

固定协议：

| Field | Value |
|---|---|
| Cipher | `PRESENT-80` |
| Rounds | `8` |
| Difference profile | `present_zhang_wang2022_mcnd` |
| Difference member | `0` |
| Sample structure | `zhang_wang_case2_official_mcnd` |
| Pairs/sample | `16` |
| Negative mode | `encrypted_random_plaintexts` |
| Train key | `0x00000000000000000000` |
| Validation key | `0x11111111111111111111` |
| Key rotation | `0` |
| Scheduler | `official_cyclic` |
| Learning rate | `0.0001` |
| Max learning rate | `0.002` |
| Checkpoint metric | `val_auc` |
| Restore best checkpoint | `true` |
| Early stopping | patience `8`, min delta `0.0001` |
| Primary metric | `val_auc` |
| Secondary metric | calibrated accuracy, accuracy, loss |

## 3. Single Hypothesis

唯一变化假设：

```text
在 r8 更弱信号下，SPN/P-layer 对齐表示和 pair-set evidence pooling
比普通 Zhang/Wang-style MCND 更能保留可学习区分信号。
```

首轮不同时改变：

```text
negative mode
sample structure
difference profile
validation key
checkpoint metric
metric computation
```

## 4. First Planned Run

建议 run id：

```text
i1_present_r8_round_extension_262k_seed0_gpu0_20260704
```

启动状态：

```text
prepared only
```

远程启动前必须完成：

```text
1. 本地配置解析 / readiness smoke。
2. scoped commit。
3. git push。
4. 从 pushed commit 启动远程。
5. 检查 active r7 S-box prior run 不被中断；如 GPU0 空闲可用，优先使用 GPU0。
6. 启动本地 tmux watcher 或子 agent 监控/拉回，不由主线程循环轮询。
```

远程建议参数：

| Field | Value |
|---|---|
| Device | `cuda:0` if available, otherwise readiness gate chooses another GPU |
| Samples/class | `262144` |
| Expected rows | `3` |
| Epochs | `30` |
| Batch size | `2048` |
| Train eval interval | `0` |
| Dataset cache | disk-backed cache under `G:\lxy` |
| Project path | `G:\lxy\blockcipher-structure-adaptive-nd` |
| Run root | `G:\lxy\blockcipher-structure-adaptive-nd-runs` |
| Shell | `cmd.exe /c` |

## 5. Matrix

| Row | Model | Role |
|---:|---|---|
| 0 | `present_zhang_wang_keras_mcnd` | same-budget Zhang/Wang-style r8 baseline |
| 1 | `present_nibble_invp_only_spn_only` | current strongest SPN/P-layer aligned representation |
| 2 | `present_nibble_invp_pair_consistency_spn_only` | tests whether pair-set evidence pooling helps under weaker r8 signal |

不加入 S-box prior gate、DDT graph、active auxiliary、trail-family 或更多 control。第一轮只回答 r8 是否有弱信号、以及现有最强结构表示是否保留信号。

## 6. Evidence Level

```text
262144/class = medium diagnostic only
```

允许说：

```text
r8 diagnostic signal exists / weak / absent under this protocol.
```

不允许说：

```text
PRESENT r8 已正式突破。
某路线已正式优于全部文献。
r8 完全失败。
```

正式路线判断至少需要：

```text
1000000/class
multiple seeds
retrieved / validated / plan-aligned
必要 attribution 或 aggregation controls
```

## 7. Gate

Gate 是下一步决策规则，不是论文结论。

| Result | Decision |
|---|---|
| best candidate AUC `<= 0.52` | stop this from-scratch r8 route; plan curriculum / transfer or new data representation |
| `0.52 <` best candidate AUC `< 0.55` and candidate beats baseline | weak signal; repeat seed1 or test r7->r8 curriculum before 1M |
| best candidate AUC `>= 0.55` and beats baseline by `+0.005` | support scaling to 1M/class seed0 |
| pair-consistency beats InvP-only but not frozen aggregation control | mark application-level / aggregation signal only; do not claim architecture improvement |
| baseline beats both candidates | r8 structure route not supported at this scale; inspect whether training, representation, or pair-set control should change |

## 8. Postprocess

Expected artifacts after remote retrieval:

```text
outputs/remote_results/i1_present_r8_round_extension_262k_seed0_gpu0_20260704/results/i1_present_r8_round_extension_262k_seed0_gpu0_20260704.jsonl
outputs/remote_results/i1_present_r8_round_extension_262k_seed0_gpu0_20260704/*curves.svg
outputs/remote_results/i1_present_r8_round_extension_262k_seed0_gpu0_20260704/*history.csv
```

Generic validation:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --results outputs/remote_results/i1_present_r8_round_extension_262k_seed0_gpu0_20260704/results/i1_present_r8_round_extension_262k_seed0_gpu0_20260704.jsonl \
  --plan configs/experiment/innovation1/innovation1_spn_present_round_extension_r8_262k_seed0.csv \
  --expected-rows 3
```

Generic plot:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-results \
  --results outputs/remote_results/i1_present_r8_round_extension_262k_seed0_gpu0_20260704/results/i1_present_r8_round_extension_262k_seed0_gpu0_20260704.jsonl \
  --output outputs/remote_results/i1_present_r8_round_extension_262k_seed0_gpu0_20260704/i1_present_r8_round_extension_262k_seed0_gpu0_20260704_curves.svg \
  --history-csv outputs/remote_results/i1_present_r8_round_extension_262k_seed0_gpu0_20260704/i1_present_r8_round_extension_262k_seed0_gpu0_20260704_history.csv \
  --title i1_present_r8_round_extension_262k_seed0_gpu0_20260704
```

当前没有新增专用 postprocess 脚本。结果回来后先用通用 validate/plot 和人工 gate 记录；如果 r8 有信号，再实现专用 gate/postprocess。

## 9. Relation To Active r7 S-box Prior Run

当前已知 active run：

```text
i1_sbox_prior_gate_r7_262k_seed0_gpu1_20260703
```

关系：

```text
1. 不停止、不覆盖、不重启 active r7 run。
2. 它是 r7 mechanism validation。
3. 本计划是 r8 round-extension ladder。
4. 如果 GPU 资源不足，优先等待 watcher 拉回 r7 结果。
5. 如果 GPU0 空闲且 readiness 通过，可以并行启动 r8 diagnostic。
```

## 10. Next Branches

### If r8 is weak or absent

写入结果分析后，转向：

```text
r7 -> r8 curriculum
r7 checkpoint -> r8 fine-tune
mixed-round training
更强 SPN-aware feature representation
```

### If r8 has weak positive signal

推进：

```text
seed1 diagnostic
or 1M/class seed0 if AUC >= 0.55 and baseline margin sufficient
```

### If pair-set is strongest

必须补：

```text
frozen single-pair score aggregation control
multi-query application-level evidence label
```

### If InvP-only is strongest

推进：

```text
InvP-only r8 1M/class seed0
then seed1 / attribution controls if seed0 supports
```

## 11. Launch Record

**Launch time:** 2026-07-04 23:50 +08:00

**Launch status:** launched / watcher-managed / no result evidence yet

**Launch commit:**

```text
02fcb066c865005c6d1c05e30b76372015b95ed2
```

**Remote config:**

```text
configs/remote/innovation1_spn_present_round_extension_r8_262k_seed0_gpu0_20260704.json
```

**Remote launcher source:**

```text
configs/remote/generated/run_i1_present_r8_round_extension_262k_seed0_gpu0_20260704.cmd
```

**Remote launcher uploaded to:**

```text
G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_round_extension_262k_seed0_gpu0_20260704\run_i1_present_r8_round_extension_262k_seed0_gpu0_20260704.cmd
```

The uploaded launcher is the committed launcher file from the pushed commit.
The launcher itself clones/pulls the GitHub `main` branch into the run-owned
source directory before training:

```text
G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_round_extension_262k_seed0_gpu0_20260704\source
```

**Task Scheduler command:**

```text
cmd.exe /c G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_round_extension_262k_seed0_gpu0_20260704\run_i1_present_r8_round_extension_262k_seed0_gpu0_20260704.cmd
```

**GPU selection:**

```text
cuda:0
```

Bounded pre-launch GPU check showed GPU0 had no active training Python process
and GPU1 was occupied by the active r7 S-box prior run. This r8 run was
therefore launched on GPU0 without interrupting:

```text
i1_sbox_prior_gate_r7_262k_seed0_gpu1_20260703
```

**Local watcher:**

```text
tmux session = monitor_i1_present_r8_round_extension_262k_seed0_gpu0_20260704
script = configs/remote/generated/monitor_i1_present_r8_round_extension_262k_seed0_gpu0_20260704.sh
```

Watcher responsibilities:

```text
1. scp logs/results from G:\lxy\blockcipher-structure-adaptive-nd-runs\<run_id>
2. wait for 3 JSONL rows
3. run scripts/validate-results
4. run scripts/plot-results
5. write <run_id>_gate_note.json
```

**Local retrieval target:**

```text
outputs/remote_results/i1_present_r8_round_extension_262k_seed0_gpu0_20260704
```

**Readiness checks before launch:**

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_project_structure.py -k "present_r8_round_extension or zhang_wang_262k_official_cyclic"
```

Result:

```text
3 passed, 242 deselected
```

```text
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness \
  --config configs/remote/innovation1_spn_present_round_extension_r8_262k_seed0_gpu0_20260704.json
```

Result:

```text
status = pass
expected_rows = 3
checked_invariants include medium_scale_dataset_cache
```

**Claim scope:**

```text
This launch is not result evidence.
It is a 262144/class single-seed medium diagnostic.
No r8 success/failure or breakthrough claim is allowed until watcher retrieval,
validation, plotting, and gate-note generation complete.
```

## 12. Retrieved r8 Diagnostic Result

**Retrieval time:** 2026-07-05 03:05 +08:00

**Status:** retrieved / validated / plotted / gate-note generated

**Evidence level:**

```text
262144/class single-seed medium diagnostic only
```

**Validation:**

```text
status = pass
expected_rows = 3
result_rows = 3
field_mismatches = []
```

Artifacts:

```text
outputs/remote_results/i1_present_r8_round_extension_262k_seed0_gpu0_20260704/results/i1_present_r8_round_extension_262k_seed0_gpu0_20260704.jsonl
outputs/remote_results/i1_present_r8_round_extension_262k_seed0_gpu0_20260704/i1_present_r8_round_extension_262k_seed0_gpu0_20260704_validation.json
outputs/remote_results/i1_present_r8_round_extension_262k_seed0_gpu0_20260704/i1_present_r8_round_extension_262k_seed0_gpu0_20260704_gate_note.json
outputs/remote_results/i1_present_r8_round_extension_262k_seed0_gpu0_20260704/i1_present_r8_round_extension_262k_seed0_gpu0_20260704_curves.svg
outputs/remote_results/i1_present_r8_round_extension_262k_seed0_gpu0_20260704/i1_present_r8_round_extension_262k_seed0_gpu0_20260704_history.csv
```

Metrics:

| Model | Accuracy | Calibrated accuracy | AUC | Loss | Role |
|---|---:|---:|---:|---:|---|
| `present_zhang_wang_keras_mcnd` | 0.518814 | 0.528740 | 0.540349 | 0.694303 | same-budget r8 baseline |
| `present_nibble_invp_only_spn_only` | 0.500000 | 0.511150 | 0.514940 | 0.693147 | InvP-only representation |
| `present_nibble_invp_pair_consistency_spn_only` | 0.537376 | 0.537937 | 0.552909 | 0.688330 | pair-set consistency candidate |

Gate note:

```text
decision = support_scale_r8_to_1m_seed0
best_model = present_nibble_invp_pair_consistency_spn_only
best_auc = 0.552908501064
baseline_auc = 0.540348751209
delta_vs_baseline = +0.012559749855
```

Interpretation:

```text
The r8 diagnostic has a real positive signal, but the signal is carried by
pair-set consistency rather than InvP-only. This supports continuing the
round-extension ladder and triggers the planned r9 weak-probe condition.
```

Claim scope:

```text
This is not formal r8 success or a breakthrough claim.
It is a 262144/class single-seed diagnostic that justifies r9 weak-probe and
later r8 1M/class confirmation if the route remains promising.
```

## 13. r8 Pair-Set 1M Confirmation Plan

**Status:** launched / watcher-managed / no result evidence yet

The r8 diagnostic gate supports scaling the pair-set route, but the claim is
still diagnostic until a paper-scale confirmation exists. The next r8
confirmation run therefore uses a lean two-row matrix:

| Row | Model | Role |
|---:|---|---|
| 0 | `present_zhang_wang_keras_mcnd` | same-budget r8 1M baseline |
| 1 | `present_nibble_invp_pair_consistency_spn_only` | pair-set consistency candidate |

Config:

```text
configs/experiment/innovation1/innovation1_spn_present_pairset_r8_1m_seed0.csv
```

Remote assets:

```text
configs/remote/innovation1_spn_present_pairset_r8_1m_seed0_gpu1_20260705.json
configs/remote/generated/run_i1_present_r8_pairset_1m_seed0_gpu1_20260705.cmd
configs/remote/generated/monitor_i1_present_r8_pairset_1m_seed0_gpu1_20260705.sh
```

Gate:

| Result | Decision |
|---|---|
| pair-set AUC - baseline AUC `>= +0.005` | support r8 pair-set 1M confirmation; prepare seed1 or frozen aggregation control |
| pair-set AUC - baseline AUC `> 0` but `< +0.005` | weak paper-scale positive; repeat seed or run controls |
| pair-set AUC `<= baseline` | stop or rethink this scale route |

Claim scope:

```text
1000000/class seed0 is paper-scale single-seed evidence only.
It is not formal multi-seed route evidence and not a breakthrough claim.
```

Launch record:

| Field | Value |
|---|---|
| Launch time | 2026-07-05 08:15 +08:00 |
| Launch commit | `0b8279c` |
| Remote task | `i1_present_r8_pairset_1m_seed0_gpu1_20260705` |
| Remote launcher uploaded to | `G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_pairset_1m_seed0_gpu1_20260705\run_i1_present_r8_pairset_1m_seed0_gpu1_20260705.cmd` |
| Task Scheduler command | `cmd.exe /c G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r8_pairset_1m_seed0_gpu1_20260705\run_i1_present_r8_pairset_1m_seed0_gpu1_20260705.cmd` |
| GPU | `cuda:1` |
| Local watcher | `monitor_i1_present_r8_pairset_1m_seed0_gpu1_20260705` |
| Local retrieval target | `outputs/remote_results/i1_present_r8_pairset_1m_seed0_gpu1_20260705` |

Pre-launch verification:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_project_structure.py -k "present_r8_pairset_1m or present_r9_weak_probe or present_r8_round_extension"
5 passed, 243 deselected

UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness --config configs/remote/innovation1_spn_present_pairset_r8_1m_seed0_gpu1_20260705.json
status = pass
expected_rows = 2
checked_invariants include medium_scale_dataset_cache

bash -n configs/remote/generated/monitor_i1_present_r8_pairset_1m_seed0_gpu1_20260705.sh
pass

rg -n "cmd\.exe /k" <r8-pairset-1m remote assets>
no matches
```

Launch interpretation:

```text
This is launched evidence only, not result evidence.
The watcher is responsible for retrieval, validate-results, plot-results, and
gate-note generation. Do not claim r8 pair-set confirmation until results are
retrieved, validated, and plan-aligned.
```

## 14. Prepared r8 Pair-Set Aggregation Control

**Status:** prepared / not launched

If the r8 1M pair-set result is positive or if r9 weak-probe keeps pair-set as
the strongest high-round route, the next attribution question is whether
learned pair-set consistency beats a frozen single-pair aggregation baseline.
The prepared control plan is:

```text
docs/experiments/innovation1-present-r8-pairset-aggregation-control-plan.md
```

Prepared assets:

```text
configs/experiment/innovation1/innovation1_spn_present_pairset_aggregation_control_single_pair_r8_262k.csv
configs/experiment/innovation1/innovation1_spn_present_pairset_aggregation_control_r8_262k.csv
configs/remote/innovation1_spn_present_pairset_aggregation_control_single_pair_r8_262k_gpu0_20260705.json
configs/remote/innovation1_spn_present_pairset_aggregation_control_r8_262k_gpu0_20260705.json
configs/remote/generated/run_i1_pairset_aggregation_control_r8_262k_seed0_gpu0_20260705.cmd
configs/remote/generated/monitor_i1_pairset_aggregation_control_r8_262k_seed0_gpu0_20260705.sh
```

Purpose:

```text
Separate learned cross-pair structure from fixed aggregation of independent
single-pair InvP scores. This is required before treating pair-set consistency
as an architecture innovation rather than application-level aggregation.
```
