# Innovation 1 PRESENT r8 Pair-Set Aggregation Control Plan

**Date:** 2026-07-05

**Status:** prepared / readiness passed / not launched / wait for active r8 pair-set 1M gate

**Scope:** PRESENT-80 r8, Zhang/Wang 2022 Case2 `m=16`, strict `encrypted_random_plaintexts` negatives.

## Research Question

The r8 round-extension diagnostic showed:

```text
present_nibble_invp_pair_consistency_spn_only AUC = 0.552908501064
present_zhang_wang_keras_mcnd AUC                  = 0.540348751209
delta                                              = +0.012559749855
```

This supports scaling the pair-set route, but it does not yet prove that the
learned pair-set model is exploiting cross-pair SPN structure. The alternative
explanation is simpler:

```text
many weak independent single-pair scores + fixed score aggregation
```

This plan tests that alternative directly.

## Hypothesis

Holding cipher, round count, difference profile, sample structure, negative
mode, keys, optimizer, scheduler, checkpoint metric, and scale fixed:

```text
If learned r8 pair-set consistency captures real cross-pair structure, it should
beat frozen single-pair InvP score aggregation by at least +0.001 AUC.
```

If it does not beat the frozen aggregation control, the r8 pair-set result
should be interpreted as aggregation/application-level evidence, not as a
network-architecture innovation claim.

## Staged Design

Stage A trains a frozen single-pair scorer:

```text
config = configs/experiment/innovation1/innovation1_spn_present_pairset_aggregation_control_single_pair_r8_262k.csv
model = present_nibble_invp_only_spn_only
rounds = 8
samples_per_class = 262144
pairs_per_sample = 1
checkpoint_output under G:\lxy\blockcipher-structure-adaptive-nd-runs
```

Stage B trains the same-scale learned pair-set rows and evaluates the frozen
scorer over 16-pair samples:

```text
config = configs/experiment/innovation1/innovation1_spn_present_pairset_aggregation_control_r8_262k.csv
models = present_nibble_invp_only_spn_only,
         present_nibble_invp_pair_consistency_spn_only
rounds = 8
samples_per_class = 262144
pairs_per_sample = 16
frozen_aggregation = sum_logodds over 16 single-pair logits
```

## Remote Assets

```text
configs/remote/innovation1_spn_present_pairset_aggregation_control_single_pair_r8_262k_gpu0_20260705.json
configs/remote/innovation1_spn_present_pairset_aggregation_control_r8_262k_gpu0_20260705.json
configs/remote/generated/run_i1_pairset_aggregation_control_r8_262k_seed0_gpu0_20260705.cmd
configs/remote/generated/monitor_i1_pairset_aggregation_control_r8_262k_seed0_gpu0_20260705.sh
```

Readiness invariants:

```text
pairset_aggregation_stage_lock
cmd.exe /c only
all remote artifacts under G:\lxy
dataset_cache_root under G:\lxy\blockcipher-structure-adaptive-nd-runs
strict encrypted_random_plaintexts negatives
same Zhang/Wang Case2 sample structure
```

## Launch Gate

Do not launch while the active r8 paper-scale task is still running:

```text
i1_present_r8_pairset_1m_seed0_gpu1_20260705
```

The r9 from-scratch weak-probe has already completed and is not an active
watcher anymore:

```text
run_id = i1_present_r9_weak_probe_262k_seed0_gpu0_20260705
status = retrieved / validated / plotted / gate-noted / plan-aligned
decision = stop_from_scratch_r9_r10_plan_curriculum_or_difference_search
```

Readiness check status:

```text
single-pair scorer remote config = pass
pair-set aggregation remote config = pass
```

Launch this control when one of these conditions is true:

```text
1. r8 1M pair-set result is positive enough that attribution is needed before
   seed1 or claim expansion.
2. r9 weak-probe shows pair-set remains the best high-round route.
3. r8 1M is weak or tied, but we need to decide whether the 262k pair-set signal
   was only fixed aggregation.
```

This is prepared evidence only until launched and retrieved.

## Gate

| Result | Decision |
|---|---|
| learned pair-set AUC >= frozen aggregation AUC + 0.001 and beats InvP anchor | support learned pair-set consistency; consider seed1 or 1M confirmation branch |
| learned pair-set within +/- 0.001 AUC of frozen aggregation | treat as aggregation/application-level evidence; do not claim architecture innovation |
| learned pair-set below frozen aggregation | stop learned pair-set route; keep frozen aggregation as diagnostic baseline |

## Claim Scope

```text
262144/class r8 control = medium diagnostic only.
It cannot prove formal r8 success, breakthrough, or SOTA.
It can decide whether pair-set consistency deserves more high-round scale.
```
