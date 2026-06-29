# Innovation 1 InvP-Centered Seed1 Fast Confirmation Plan

**Date:** 2026-06-29

**Status:** completed / documented

**Scope:** PRESENT-80 r7, Zhang/Wang 2022 Case2 `m=16`, strict encrypted-random-plaintext negatives, `262144/class` medium diagnostic.

## Motivation

The seed0 InvP-centered diagnostic completed with a weak-positive result:

```text
run_id = i1_invp_centered_r7_262k_seed0_gpu1_20260628
pair-consistency AUC = 0.792800
InvP-only AUC        = 0.792105
delta               = +0.000695
```

This is not enough to promote pair-consistency as a clear architectural improvement because the predeclared continuation gate was `+0.002` AUC. However, pair-consistency was best on AUC, calibrated accuracy, fixed-threshold accuracy, and loss. The next question is whether this small advantage survives another seed.

## Research Question

Does `present_nibble_invp_pair_consistency_spn_only` remain better than the simpler `present_nibble_invp_only_spn_only` under the same protocol at seed1?

Secondary question:

```text
Does the new fast evaluation mode reduce training wall time without changing
the experimental protocol or validation metric?
```

## Fixed Protocol

All rows preserve:

| Field | Value |
|---|---|
| Cipher | `PRESENT-80` |
| Rounds | `7` |
| Seed | `1` |
| Difference profile | `present_zhang_wang2022_mcnd` |
| Sample structure | `zhang_wang_case2_official_mcnd` |
| Pairs per sample | `16` |
| Samples per class | `262144` |
| Feature encoding | `ciphertext_pair_bits` |
| Negative mode | `encrypted_random_plaintexts` |
| Train key | `0x00000000000000000000` |
| Validation key | `0x11111111111111111111` |
| Scheduler | `official_cyclic` |
| Checkpoint metric | `val_auc` |
| Restore best checkpoint | `true` |

Do not change validation data, labels, negative mode, metric computation, keying protocol, or Zhang/Wang Case2 sample construction.

## Lean Matrix

Config:

```text
configs/experiment/innovation1/innovation1_spn_present_invp_centered_seed1_fast_r7_262k.csv
```

Rows:

| Rank | Model key | Role |
|---:|---|---|
| 0 | `present_zhang_wang_keras_mcnd` | same-protocol baseline / drift check |
| 1 | `present_nibble_invp_only_spn_only` | current simple InvP anchor |
| 2 | `present_nibble_invp_pair_consistency_spn_only` | weak-positive candidate from seed0 |

This intentionally excludes the old `DeltaC+InvP`, `DeltaC-only`, and shuffled-P controls. Those were already checked in the seed0 attribution and InvP-centered diagnostics, and this run is a lean confirmation rather than a full attribution audit.

## Fast Evaluation Mode

Remote launch should use:

```text
--train-eval-interval 0
```

Meaning:

```text
per-epoch full validation metrics remain enabled
per-epoch full train-set metrics are skipped
final validation metrics still use the restored best checkpoint
```

The primary metric remains exact `val_auc`; this option only removes repeated train-set diagnostic evaluation.

## Decision Gates

Primary comparison:

```text
pair-consistency AUC - InvP-only AUC
```

Decision:

| Condition | Interpretation | Action |
|---|---|---|
| Pair-consistency beats InvP-only by `>= +0.002` AUC | clear aggregation improvement | promote pair-consistency to 1M/class candidate |
| Pair-consistency remains best but by `< +0.002` AUC | weak but stable positive | run one more lean diagnostic seed or promote only if loss/calibrated accuracy also improves |
| Pair-consistency within `±0.001` AUC of InvP-only | effectively tied | prefer simpler InvP-only route |
| Pair-consistency below InvP-only by `> 0.001` AUC | seed0 gain likely noise | discard pair-consistency aggregation, keep InvP-only anchor |
| Baseline behaves far outside seed0 range | possible protocol/cache issue | pause interpretation and audit |

Speed diagnostic:

```text
Compare epoch seconds against seed0 models where possible. Treat this as an
execution benchmark, not cryptanalytic evidence.
```

## Execution Plan

1. Add the lean seed1 CSV matrix.
2. Run local tests and a CPU smoke with `--train-eval-interval 0`.
3. Commit and push the plan/config before remote launch.
4. Launch remote from the pushed commit under `G:\lxy`.
5. Use disk-backed shared dataset cache under `G:\lxy\blockcipher-structure-adaptive-nd-runs\shared_dataset_cache`.
6. Use `cmd.exe /c`, not `cmd.exe /k`.
7. Start a local tmux monitor to retrieve logs/results/results_archive automatically.
8. After retrieval, validate plan alignment, generate history/curves, update this document, commit, and push.

## Claim Scope

This run can support:

```text
medium diagnostic cross-seed confirmation for InvP-centered architecture choice
fast-eval execution benchmark
```

It cannot support:

```text
formal reproduction
breakthrough claim
publication-scale multi-seed conclusion
```

## Completed Remote Run

The seed1 fast confirmation completed and was retrieved by the local tmux
monitor:

| Field | Value |
|---|---|
| Run ID | `i1_invp_centered_seed1_fast_r7_262k_gpu1_retry1_20260629` |
| Source commit | `b26aebea641f42db550036680798cb61a539412e` |
| Remote root | `G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_invp_centered_seed1_fast_r7_262k_gpu1_retry1_20260629` |
| Local artifacts | `outputs/remote_results/i1_invp_centered_seed1_fast_r7_262k_gpu1_retry1_20260629/` |
| Result JSONL | `outputs/remote_results/i1_invp_centered_seed1_fast_r7_262k_gpu1_retry1_20260629/results/i1_invp_centered_seed1_fast_r7_262k_gpu1_retry1_20260629.jsonl` |
| Local gate | `outputs/remote_results/i1_invp_centered_seed1_fast_r7_262k_gpu1_retry1_20260629/i1_invp_centered_seed1_fast_r7_262k_gpu1_retry1_20260629_local_result_gate.json` |
| History CSV | `outputs/remote_results/i1_invp_centered_seed1_fast_r7_262k_gpu1_retry1_20260629/i1_invp_centered_seed1_fast_r7_262k_gpu1_retry1_20260629_history.csv` |
| Curves SVG | `outputs/remote_results/i1_invp_centered_seed1_fast_r7_262k_gpu1_retry1_20260629/i1_invp_centered_seed1_fast_r7_262k_gpu1_retry1_20260629_curves.svg` |

Gate:

```text
remote result_lines = 3
remote expected_rows = 3
local validation status = pass
local result_rows = 3
local expected_rows = 3
stderr = empty
status = completed, retrieved locally, plan-aligned
```

## Metrics

Best validation metrics by `val_auc`:

| Rank | Model | Best epoch | History epochs | Accuracy | Calibrated accuracy | AUC | Loss |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | `present_zhang_wang_keras_mcnd` | 6 | 14 | 0.700138 | 0.712681 | 0.786113 | 0.567279 |
| 1 | `present_nibble_invp_only_spn_only` | 20 | 20 | 0.718082 | 0.718239 | 0.792977 | 0.545585 |
| 2 | `present_nibble_invp_pair_consistency_spn_only` | 20 | 20 | **0.718567** | **0.718674** | **0.793216** | **0.545394** |

Deltas:

```text
pair-consistency - InvP-only:
  AUC                 = +0.000239
  accuracy            = +0.000484
  calibrated_accuracy = +0.000435
  loss                = -0.000190

InvP-only - Zhang/Wang:
  AUC                 = +0.006864
  accuracy            = +0.017944
  calibrated_accuracy = +0.005558
  loss                = -0.021694

pair-consistency - Zhang/Wang:
  AUC                 = +0.007103
  accuracy            = +0.018429
  calibrated_accuracy = +0.005993
  loss                = -0.021885
```

## Speed Result

The fast evaluation mode behaved as intended. Per-epoch full train-set metrics
were skipped:

```text
train_accuracy = null
train_auc = null
train_eval_loss = null
```

For the first completed epoch of the baseline row:

```text
current fast epoch duration = 120.9 seconds
current fast train pass     = 97.8 seconds
current fast validation     = about 21.9 seconds
```

The comparable seed0 run had:

```text
old epoch median = 164.4 seconds
old epoch mean   = 179.1 seconds
old epoch range  = 141.6 - 243.0 seconds
```

Interpretation:

```text
fast evaluation gives about 26% speedup versus the old median and about 32%
versus the old mean on the observed comparable epoch timing. This is an
execution improvement only, not cryptanalytic evidence.
```

## Decision

The pair-consistency row remained numerically best, but its advantage over the
simpler InvP-only row was only `+0.000239` AUC on seed1. The predeclared gate
treats differences within `±0.001` AUC as effectively tied.

Decision:

```text
Do not promote pair-consistency as the main route.
Keep pair-consistency as a weak auxiliary candidate only.
Promote the simpler InvP-only SPN route as the next scale-up target.
```

The stronger and more stable signal is the InvP/SPN structural view over the
Zhang/Wang baseline:

```text
seed0 pair-consistency over baseline AUC = about +0.00845
seed0 InvP-only over baseline AUC        = about +0.00776
seed1 pair-consistency over baseline AUC = about +0.00710
seed1 InvP-only over baseline AUC        = about +0.00686
```

Next action:

```text
Run present_nibble_invp_only_spn_only at 1000000/class as a single-row
paper-scale diagnostic, comparing against the already completed same-protocol
Zhang/Wang 1000000/class anchor. Do not spend the next GPU window on another
pair-consistency diagnostic unless the 1M InvP-only result is unstable.
```
