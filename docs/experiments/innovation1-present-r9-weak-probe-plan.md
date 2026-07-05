# Innovation 1 PRESENT r9 Weak-Probe 实验计划

**日期：** 2026-07-05

**状态：** remote assets prepared / launchable after scoped commit and GPU gate

**研究蓝图：** `docs/research/innovation1-present-higher-round-strategy.md`

**Smoke config：** `configs/experiment/innovation1/innovation1_spn_present_round_extension_r9_weak_probe_smoke.csv`

**Medium config：** `configs/experiment/innovation1/innovation1_spn_present_round_extension_r9_262k_seed0.csv`

## 1. Research Question

在 r8 round-extension 完成之后，进一步问：

```text
PRESENT-80 r9 在 strict encrypted-random-plaintext negatives、
Zhang/Wang Case2 m=16、同协议同训练预算下，是否仍有可检测的弱神经信号？
```

这个实验不追求马上证明 r9 成功，而是作为 r9/r10 高轮路线的 weak probe。

## 2. Launch Gate

原始条件：不得启动 r9 medium remote，除非先满足：

```text
i1_present_r8_round_extension_262k_seed0_gpu0_20260704
```

已经：

```text
retrieved
validated
plan-aligned
plotted
gate-note generated
```

并且至少满足一个条件：

```text
1. r8 best candidate AUC > 0.52；
2. r8 pair-set / aggregation 路线显示应用级弱信号；
3. 用户明确选择进行 r9 weak probe，并接受它只是高轮诊断。
```

否则只允许本地 smoke，不允许远程 r9 262144/class。

### Gate Satisfaction

As of 2026-07-05, the r8 gate condition is satisfied:

```text
run_id = i1_present_r8_round_extension_262k_seed0_gpu0_20260704
status = retrieved / validated / plotted / gate-note generated
validation_status = pass
best_model = present_nibble_invp_pair_consistency_spn_only
best_auc = 0.552908501064
baseline_auc = 0.540348751209
delta_vs_baseline = +0.012559749855
gate_decision = support_scale_r8_to_1m_seed0
```

Therefore the r9 weak-probe route is launchable after:

```text
1. remote readiness passes
2. scoped commit is pushed
3. GPU availability is checked once
4. local tmux watcher is started
```

## 3. Fixed Protocol

| Field | Value |
|---|---|
| Cipher | `PRESENT-80` |
| Rounds | `9` |
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

## 4. Matrix

| Row | Model | Role |
|---:|---|---|
| 0 | `present_zhang_wang_keras_mcnd` | r9 same-budget baseline |
| 1 | `present_nibble_invp_only_spn_only` | strongest known SPN/P-layer aligned representation |
| 2 | `present_nibble_invp_pair_consistency_spn_only` | pair-set weak-signal pooling candidate |

保持 3 行，不加入 S-box prior、DDT graph、active auxiliary、trail-family。r9 首轮只回答：

```text
是否还有弱信号；
弱信号更像单样本 InvP 信号，还是 pair-set 聚合信号。
```

## 5. Smoke

Smoke 只用于验证路径：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_round_extension_r9_weak_probe_smoke.csv \
  --device cpu \
  --epochs 1 \
  --batch-size 4 \
  --output /tmp/i1_present_r9_weak_probe_smoke.jsonl
```

Smoke 不产生模型能力结论。

## 6. Medium Run

Run id：

```text
i1_present_r9_weak_probe_262k_seed0_gpu0_20260705
```

状态：

```text
remote assets prepared / not yet result evidence
```

远程参数：

| Field | Value |
|---|---|
| Samples/class | `262144` |
| Expected rows | `3` |
| Epochs | `30` |
| Batch size | `2048` |
| Dataset cache | disk-backed under `G:\lxy` |
| Result sync | local tmux watcher / scp fallback |

不创建远程 launch record，除非 r8 gate 触发 r9。

Prepared remote assets:

```text
configs/remote/innovation1_spn_present_r9_weak_probe_262k_seed0_gpu0_20260705.json
configs/remote/generated/run_i1_present_r9_weak_probe_262k_seed0_gpu0_20260705.cmd
configs/remote/generated/monitor_i1_present_r9_weak_probe_262k_seed0_gpu0_20260705.sh
```

## 7. Gate

| Result | Decision |
|---|---|
| best candidate AUC `<= 0.505` | near-random；停止 from-scratch r9/r10，转 curriculum / difference search |
| AUC `0.505 - 0.52` | weak trace；不升 1M，优先 seed variance 或 aggregation |
| AUC `> 0.52` 且超过 baseline | r9 weak positive；准备 seed1 或 curriculum-scale |
| AUC `> 0.55` 且超过 baseline `+0.005` | strong diagnostic；准备 r9 1M/class seed0 |

所有 gate 都是研究决策，不是正式成功声明。

## 8. Claim Scope

允许说：

```text
r9 weak-probe diagnostic signal
r9 near-random under this protocol
r9 application-level aggregation signal
```

不允许说：

```text
r9 正式突破
r9 SOTA
PRESENT 高轮已证明有效
```

正式路线至少需要：

```text
1M/class
multiple seeds
retrieved / validated / plan-aligned
same-budget baseline
必要 attribution / aggregation controls
```

## 9. Next Branch

如果 r9 weak probe 弱或近随机：

```text
r7/r8 -> r9 curriculum / transfer
high-round input difference search
multi-query aggregation study
```

如果 r9 weak probe 明确正向：

```text
r9 seed1 diagnostic
r9 1M/class seed0
再考虑 r10 weak probe
```

### Prepared Curriculum Branch

The r8-to-r9 curriculum branch has been prepared but must not launch until this
from-scratch r9 weak-probe is retrieved, validated, plotted, gate-noted, and
plan-aligned:

```text
docs/experiments/innovation1-present-r9-curriculum-from-r8-plan.md
configs/experiment/innovation1/innovation1_spn_present_r9_curriculum_from_r8_262k_seed0.csv
configs/remote/innovation1_spn_present_r9_curriculum_from_r8_262k_seed0_gpu0_20260705.json
configs/remote/generated/run_i1_present_r9_curriculum_from_r8_262k_seed0_gpu0_20260705.cmd
configs/remote/generated/monitor_i1_present_r9_curriculum_from_r8_262k_seed0_gpu0_20260705.sh
```

It uses:

```text
8 epochs r8 pretraining + 22 epochs r9 fine-tuning
```

Claim scope:

```text
262144/class medium diagnostic only; tests training path, not a new benchmark.
```

## 10. Launch Record

**Launch time:** 2026-07-05 08:07 +08:00

**Launch status:** launched / watcher-managed / no result evidence yet

**Launch commit:**

```text
4391e2b72d6af905b56c6fc3938a9730c0acd6c0
```

**Remote config:**

```text
configs/remote/innovation1_spn_present_r9_weak_probe_262k_seed0_gpu0_20260705.json
```

**Remote launcher source:**

```text
configs/remote/generated/run_i1_present_r9_weak_probe_262k_seed0_gpu0_20260705.cmd
```

**Remote launcher uploaded to:**

```text
G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r9_weak_probe_262k_seed0_gpu0_20260705\run_i1_present_r9_weak_probe_262k_seed0_gpu0_20260705.cmd
```

**Task Scheduler command:**

```text
cmd.exe /c G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r9_weak_probe_262k_seed0_gpu0_20260705\run_i1_present_r9_weak_probe_262k_seed0_gpu0_20260705.cmd
```

**GPU selection:**

```text
cuda:0
```

Bounded pre-launch GPU check showed no active training Python process on GPU0
or GPU1. GPU0 was selected for continuity with the r8 round-extension run.

**Local watcher:**

```text
tmux session = monitor_i1_present_r9_weak_probe_262k_seed0_gpu0_20260705
script = configs/remote/generated/monitor_i1_present_r9_weak_probe_262k_seed0_gpu0_20260705.sh
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
outputs/remote_results/i1_present_r9_weak_probe_262k_seed0_gpu0_20260705
```

**Readiness checks before launch:**

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/test_project_structure.py -k "present_r9_weak_probe or present_r8_round_extension"
```

Result:

```text
4 passed, 243 deselected
```

```text
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness \
  --config configs/remote/innovation1_spn_present_r9_weak_probe_262k_seed0_gpu0_20260705.json
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
It is a 262144/class single-seed r9 weak probe.
No r9 success/failure, r10 projection, or breakthrough claim is allowed until
watcher retrieval, validation, plotting, and gate-note generation complete.
```
