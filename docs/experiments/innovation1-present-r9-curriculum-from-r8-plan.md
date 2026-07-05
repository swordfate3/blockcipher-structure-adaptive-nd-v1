# Innovation 1 PRESENT r9 r8-to-r9 Curriculum Plan

**Date:** 2026-07-05

**Status:** prepared / not launched / wait for from-scratch r9 weak-probe

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

The same-budget from-scratch reference is the active run:

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

Do not launch while the current active tasks are running:

```text
i1_present_r9_weak_probe_262k_seed0_gpu0_20260705
i1_present_r8_pairset_1m_seed0_gpu1_20260705
```

Launch this curriculum route only after the from-scratch r9 weak-probe is:

```text
retrieved / validated / plotted / gate-noted / plan-aligned
```

Decision rule:

| From-scratch r9 result | Action |
|---|---|
| best AUC `<= 0.505` | launch curriculum before any r10 attempt |
| best AUC `0.505-0.52` | launch curriculum or variance seed before scaling |
| best AUC `> 0.52` and pair-set strongest | prioritize r9 seed1 or r8 pair-set attribution before curriculum |
| best AUC `> 0.55` | do not launch curriculum immediately; confirm from-scratch route first |

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
