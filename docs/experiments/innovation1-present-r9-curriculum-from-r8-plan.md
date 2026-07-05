# Innovation 1 PRESENT r9 r8-to-r9 Curriculum Plan

**Date:** 2026-07-05

**Status:** launched / watcher-managed / waiting for retrieved results

**Scope:** PRESENT-80 r9, Zhang/Wang 2022 Case2 `m=16`, strict `encrypted_random_plaintexts` negatives.

## Research Question

If r9 from-scratch training is near-random or only weakly positive, can the same
SPN-aware model learn a better high-round filter by first training on the
slightly easier r8 task and then fine-tuning on r9?

This tests a training-path hypothesis, not a new benchmark:

```text
r8 pretraining may initialize SPN/P-layer or pair-set filters that remain useful
when the r9 signal is too weak for stable random-initialization training.
```

## Fixed Protocol

```text
cipher = PRESENT-80
target rounds = 9
pretrain rounds = 8
sample_structure = zhang_wang_case2_official_mcnd
negative_mode = encrypted_random_plaintexts
pairs_per_sample = 16
difference_profile = present_zhang_wang2022_mcnd
train_key = 0x00000000000000000000
validation_key = 0x11111111111111111111
checkpoint_metric = val_auc
lr_scheduler = official_cyclic
```

The optimization budget is constrained to:

```text
8 pretrain epochs on r8 + 22 target epochs on r9
```

This keeps the total epoch count close to the 30-epoch from-scratch r9 weak
probe while changing the training path.

## Matrix

```text
configs/experiment/innovation1/innovation1_spn_present_r9_curriculum_from_r8_262k_seed0.csv
```

| Row | Model | Role |
|---:|---|---|
| 0 | `present_nibble_invp_only_spn_only` | r8-to-r9 curriculum for simple InvP structural view |
| 1 | `present_nibble_invp_pair_consistency_spn_only` | r8-to-r9 curriculum for current high-round pair-set candidate |

The same-budget from-scratch reference is the completed r9 weak-probe run:

```text
i1_present_r9_weak_probe_262k_seed0_gpu0_20260705
```

## Remote Assets

```text
configs/remote/innovation1_spn_present_r9_curriculum_from_r8_262k_seed0_gpu0_20260705.json
configs/remote/generated/run_i1_present_r9_curriculum_from_r8_262k_seed0_gpu0_20260705.cmd
configs/remote/generated/monitor_i1_present_r9_curriculum_from_r8_262k_seed0_gpu0_20260705.sh
```

Readiness invariants:

```text
cmd.exe /c only
all remote artifacts under G:\lxy
disk-backed r8/r9 dataset cache
strict encrypted_random_plaintexts negatives
same Zhang/Wang Case2 sample structure
pretrain_rounds = 8
pretrain_epochs = 8
```

## Launch Gate

Do not launch while the current active high-round task is running:

```text
i1_present_r8_pairset_1m_seed0_gpu1_20260705
```

The from-scratch r9 weak-probe is now complete:

```text
retrieved / validated / plotted / gate-noted / plan-aligned
```

Retrieved r9 gate:

```text
run_id = i1_present_r9_weak_probe_262k_seed0_gpu0_20260705
best_candidate_auc = 0.502131485177
baseline_auc = 0.503853519913
decision = stop_from_scratch_r9_r10_plan_curriculum_or_difference_search
```

This satisfies the r9-side curriculum condition. The active r8 pair-set 1M
seed0 run has now been retrieved and postprocessed, and high-round arbitration
selected this curriculum branch as the next launchable action.

## Launch Record

```text
status = launched / watcher-managed
launch_time = 2026-07-05 16:11 +08:00
run_id = i1_present_r9_curriculum_from_r8_262k_seed0_gpu0_20260705
remote_process_id = 50544
remote_run_root = G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_present_r9_curriculum_from_r8_262k_seed0_gpu0_20260705
source_commit = e13e1930db2078656b98f781c191f645e27e7f05
local_watcher = monitor_i1_present_r9_curriculum_from_r8_262k_seed0_gpu0_20260705
```

Launch/readiness evidence:

```text
remote readiness = pass
cmd.exe /c launcher = used
strict negatives = encrypted_random_plaintexts
dataset cache = disk-backed under G:\lxy
started marker = present
progress = dataset_cache stage for row 1 present_nibble_invp_only_spn_only
results rows at launch check = 0 / 2
```

The local tmux watcher owns retrieval, validation, plotting, and gate-note
generation. The main thread should not SSH-poll this run; use bounded local
artifact checks or watcher outputs.

Decision rule:

| From-scratch r9 result | Action |
|---|---|
| best AUC `<= 0.505` | launch curriculum before any r10 attempt |
| best AUC `0.505-0.52` | launch curriculum or variance seed before scaling |
| best AUC `> 0.52` and pair-set strongest | prioritize r9 seed1 or r8 pair-set attribution before curriculum |
| best AUC `> 0.55` | do not launch curriculum immediately; confirm from-scratch route first |

Automatic next-action source:

```text
scripts/postprocess-r9-weak-probe now writes
<seed0_run_id>_next_action_readiness.json.

If the r9 weak-probe gate returns:

1. near_random_r9_weak_trace_check_variance_or_aggregation
2. stop_from_scratch_r9_r10_plan_curriculum_or_difference_search

that artifact points directly to this curriculum remote config with
should_launch_remote=true after local readiness and generated launcher/monitor
checks pass. This makes the weak/near-random r9 branch actionable without
opening r10 or changing the benchmark.
```

## Gate

| Curriculum result | Decision |
|---|---|
| best curriculum AUC > from-scratch best AUC by `+0.003` | support r8-to-r9 curriculum and prepare seed1 |
| curriculum tied with from-scratch | do not scale curriculum; use simpler from-scratch route or attribution control |
| curriculum worse than from-scratch | stop curriculum route |

## Claim Scope

```text
262144/class single-seed curriculum is medium diagnostic only.
It cannot support breakthrough, SOTA, or formal route claims.
It can decide whether high-round SPN/PRESENT should use curriculum/transfer.
```
