# Innovation 1 InvP-Only 1M Scale Plan

**Date:** 2026-06-29

**Status:** running remotely / tmux monitor active

**Scope:** PRESENT-80 r7, Zhang/Wang 2022 Case2 `m=16`, strict encrypted-random-plaintext negatives, `1000000/class` single-seed paper-scale diagnostic.

## Motivation

Two medium-scale InvP-centered diagnostics now agree on the useful direction:
the SPN-adapted inverse-P nibble view improves over the Zhang/Wang MCND anchor,
while the extra pair-consistency aggregation is effectively tied with the
simpler InvP-only route.

Medium diagnostic evidence:

| Run | Seed | Baseline AUC | InvP-only AUC | Pair-consistency AUC | InvP-only delta over baseline | Pair delta over InvP-only |
|---|---:|---:|---:|---:|---:|---:|
| `i1_invp_centered_r7_262k_seed0_gpu1_20260628` | 0 | 0.784347 | 0.792105 | 0.792800 | +0.007758 | +0.000695 |
| `i1_invp_centered_seed1_fast_r7_262k_gpu1_retry1_20260629` | 1 | 0.786113 | 0.792977 | 0.793216 | +0.006864 | +0.000239 |

Interpretation:

```text
The robust signal is the InvP/SPN structural representation.
The pair-consistency pooling is too small to justify the next 1M/class GPU slot.
```

## Research Question

Does the simpler `present_nibble_invp_only_spn_only` route preserve the
medium-scale advantage at `1000000/class`?

Primary comparison anchor:

```text
Completed same-protocol Zhang/Wang 1M seed0 baseline:
run_id = zhang_wang_present_r7_1m_official_cyclic_seed0_20260625
accuracy = 0.715281
calibrated_accuracy = 0.718555
AUC = 0.793897025948
loss = 0.549200775116
```

Secondary reference:

```text
Completed same-matrix p-aligned MCND 1M seed0:
run_id = i1_spn_present_mcnd_r7_1m_seed0_gpu1_retry1_20260626
present_nibble_paligned_mcnd AUC = 0.794619119358
delta over same-run baseline = +0.000708743544
```

## Planned Matrix

Config:

```text
configs/experiment/innovation1/innovation1_spn_present_invp_only_r7_1m_seed0.csv
```

Rows:

| Rank | Model key | Role |
|---:|---|---|
| 0 | `present_nibble_invp_only_spn_only` | strongest simple SPN/InvP candidate |

This is intentionally a single-row run. The same-protocol Zhang/Wang 1M seed0
baseline has already completed, so repeating it would spend GPU time without
adding a new comparison point.

Conditional seed1 confirmation config:

```text
configs/experiment/innovation1/innovation1_spn_present_invp_only_r7_1m_seed1.csv
```

This file is prepared but must not be launched until the seed0 result is
retrieved, validated, and reaches at least the weak-positive `>= +0.001` AUC
gate over the completed Zhang/Wang 1M anchor. A `>= +0.003` delta is the strong
single-seed gate; `+0.001` to `+0.003` is weaker survival that still needs
seed1 before any route-strength claim. It is intentionally single-row and
same-protocol so the only planned change is `seed = 1`.

## Fixed Protocol

| Field | Value |
|---|---|
| Cipher | `PRESENT-80` |
| Rounds | `7` |
| Seed | `0` |
| Difference profile | `present_zhang_wang2022_mcnd` |
| Sample structure | `zhang_wang_case2_official_mcnd` |
| Pairs per sample | `16` |
| Samples per class | `1000000` |
| Feature encoding | `ciphertext_pair_bits` |
| Negative mode | `encrypted_random_plaintexts` |
| Train key | `0x00000000000000000000` |
| Validation key | `0x11111111111111111111` |
| Scheduler | `official_cyclic` |
| Checkpoint metric | `val_auc` |
| Restore best checkpoint | `true` |
| Fast train eval | `--train-eval-interval 0` |

Do not change validation data, labels, negative mode, metric computation,
keying protocol, or Zhang/Wang Case2 sample construction.

## Decision Gates

Primary comparison:

```text
InvP-only 1M AUC - completed Zhang/Wang 1M AUC
```

Gates:

| Condition | Interpretation | Action |
|---|---|---|
| `>= +0.003` AUC over Zhang/Wang 1M anchor | meaningful paper-scale single-seed improvement | run 1M seed1 confirmation |
| `+0.001` to `+0.003` AUC | weak but positive scale survival | run seed1 before claiming route strength |
| within `±0.001` AUC | medium signal mostly collapses at 1M | prefer simpler baseline or revisit architecture |
| below baseline by `> 0.001` AUC | route does not survive scale | discard InvP-only as main 1M candidate |

Claim scope:

```text
1000000/class single-seed paper-scale diagnostic only.
Not formal multi-seed evidence.
Not a breakthrough claim.
```

## Execution Plan

1. Add the single-row 1M CSV matrix.
2. Add a remote config for `cuda:1`, `batch_size=1024`, shared disk-backed cache, and `train_eval_interval=0`.
3. Run a small CPU smoke using a dedicated smoke CSV or the existing seed1 smoke pattern; do not use the 1M CSV for smoke.
4. Commit and push plan/config before launch.
5. Launch from the pushed commit into a run-owned clean clone under `G:\lxy\blockcipher-structure-adaptive-nd-runs`.
6. Use `cmd.exe /c`, not `cmd.exe /k`.
7. Use a local tmux monitor to retrieve artifacts automatically.
8. After retrieval, validate plan alignment, generate history/curves locally, update this document, commit, and push.

Conditional seed1 readiness:

| Field | Value |
|---|---|
| Plan CSV | `configs/experiment/innovation1/innovation1_spn_present_invp_only_r7_1m_seed1.csv` |
| Remote config | `configs/remote/innovation1_spn_present_invp_only_r7_1m_seed1_gpu1_20260629.json` |
| Intended run ID | `i1_invp_only_r7_1m_seed1_gpu1_20260629` |
| Status | prepared / not launched |
| Launch trigger | seed0 validated AUC delta over Zhang/Wang 1M anchor `>= +0.001` |
| Strong interpretation | seed0 delta `>= +0.003`; launch seed1 as paper-scale confirmation |
| Weak-positive interpretation | seed0 delta `+0.001` to `+0.003`; launch seed1 before any route-strength claim |
| No-launch trigger | seed0 delta `< +0.001`; use the DDT graph conditional plan or return to the baseline route |

The seed1 config is preparation, not evidence. Do not report it as running or
completed unless a separate remote launch record is added.

## Remote Launch Record

The 1M InvP-only run has been launched from a pushed GitHub commit and handed
off to a local tmux monitor.

| Field | Value |
|---|---|
| Run ID | `i1_invp_only_r7_1m_seed0_gpu1_20260629` |
| Source commit | `d9454ea17c6e0446ea934a3fb7ebfd6f56720aee` |
| Plan CSV | `configs/experiment/innovation1/innovation1_spn_present_invp_only_r7_1m_seed0.csv` |
| Remote config | `configs/remote/innovation1_spn_present_invp_only_r7_1m_seed0_gpu1_20260629.json` |
| Remote root | `G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_invp_only_r7_1m_seed0_gpu1_20260629` |
| Local monitor | `tmux: monitor_i1_invp_only_1m_20260629` |
| Local retrieval root | `outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629/` |
| Device | `cuda:1` |
| Expected rows | `1` |

Launch health:

```text
remote scheduled task = blockcipher_i1_invp_only_1m_20260629
initial logs appeared = yes
progress.jsonl appeared = yes
stdout/stderr files appeared = yes
immediate failed marker = no
monitor status = running
```

Post-retrieval gate to run automatically:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_invp_only_r7_1m_seed0.csv \
  --results outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629/results/i1_invp_only_r7_1m_seed0_gpu1_20260629.jsonl \
  --expected-rows 1 \
  --output outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629/i1_invp_only_r7_1m_seed0_gpu1_20260629_local_result_gate.json
```

Post-retrieval plotting command:

```bash
MPLCONFIGDIR=/tmp/mplconfig UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-results \
  --results outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629/results/i1_invp_only_r7_1m_seed0_gpu1_20260629.jsonl \
  --output outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629/i1_invp_only_r7_1m_seed0_gpu1_20260629_curves.svg \
  --history-csv outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629/i1_invp_only_r7_1m_seed0_gpu1_20260629_history.csv \
  --title i1_invp_only_r7_1m_seed0_gpu1_20260629
```

Post-retrieval branch gate command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/gate-invp-result \
  --results outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629/results/i1_invp_only_r7_1m_seed0_gpu1_20260629.jsonl \
  --output outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629/i1_invp_only_r7_1m_seed0_gpu1_20260629_branch_gate.json
```

Equivalent one-command local postprocess:

```bash
MPLCONFIGDIR=/tmp/mplconfig UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/postprocess-invp-result \
  --plan configs/experiment/innovation1/innovation1_spn_present_invp_only_r7_1m_seed0.csv \
  --results outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629/results/i1_invp_only_r7_1m_seed0_gpu1_20260629.jsonl \
  --output-dir outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629 \
  --run-id i1_invp_only_r7_1m_seed0_gpu1_20260629 \
  --expected-rows 1
```

Result interpretation will compare against:

```text
Zhang/Wang 1M anchor AUC = 0.793897025948
p-aligned MCND 1M AUC   = 0.794619119358
```

The branch decision uses the Zhang/Wang 1M anchor as the primary reference.
The postprocess branch gate also reports `auc_delta_vs_paligned_mcnd_1m` as a
secondary context value, but that secondary delta must not by itself trigger
seed1 or DDT graph.

Conditional next step:

```text
If InvP-only 1M beats the Zhang/Wang 1M anchor by >= +0.003 AUC,
launch a 1M seed1 confirmation.

If InvP-only is weakly positive by +0.001 to +0.003 AUC, launch seed1 before
claiming route strength.

If InvP-only is tied within the weak band or below the Zhang/Wang anchor, shift
the next design iteration toward a true SPN-topology / DDT-aware graph backbone
rather than additional pair-consistency pooling.
```
