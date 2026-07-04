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
