# Innovation 1 InvP-Centered Next Plan

**Date:** 2026-06-28

**Status:** completed medium diagnostic / decision recorded

**Scope:** PRESENT-80 r7, Zhang/Wang 2022 Case2 `m=16`, strict encrypted-random-plaintext negatives.

## Motivation

The SPN-only attribution experiment found that `InvP(DeltaC)` is the strongest current Innovation 1 signal.

Completed diagnostic:

```text
run_id = i1_spn_only_attr_r7_262k_seed0_gpu1_20260628
scale = 262144/class
status = completed remotely, fallback-retrieved locally, plan-aligned
```

Key AUC:

| Model | View | AUC |
|---|---|---:|
| `present_zhang_wang_keras_mcnd` | raw `C0 || C1` baseline | 0.783228 |
| `present_nibble_delta_only_spn_only` | `DeltaC` | 0.782918 |
| `present_nibble_shuffled_paligned_spn_only` | `DeltaC || shuffled(DeltaC)` | 0.784487 |
| `present_nibble_paligned_spn_only` | `DeltaC || InvP(DeltaC)` | 0.790665 |
| `present_nibble_invp_only_spn_only` | `InvP(DeltaC)` | **0.792536** |

Interpretation:

```text
The useful PRESENT/SPN signal is concentrated in inverse-P aligned ciphertext
difference tokens. Generic DeltaC and shuffled alignment are much weaker.
```

This upgrades the Innovation 1 direction from generic SPN-only to:

```text
InvP-centered SPN neural distinguisher
```

## Research Question

Can an InvP-centered pair-set consistency model improve over the current InvP-only SPN-only anchor without changing the benchmark protocol?

Current InvP-only aggregation:

```text
pair_embeddings -> mean pooling + max pooling -> classifier
```

Hypothesis:

```text
For MCND samples, the 16 pair embeddings may carry useful consistency patterns.
Adding low-cost pair-set consistency statistics and evidence pooling can improve
the distinguisher while preserving the same InvP-centered representation.
```

## Fixed Protocol

All rows must preserve:

| Field | Value |
|---|---|
| Cipher | `PRESENT-80` |
| Rounds | `7` |
| Difference profile | `present_zhang_wang2022_mcnd` |
| Sample structure | `zhang_wang_case2_official_mcnd` |
| Pairs per sample | `16` |
| Feature encoding on disk | `ciphertext_pair_bits` |
| Negative mode | `encrypted_random_plaintexts` |
| Train key | `0x00000000000000000000` |
| Validation key | `0x11111111111111111111` |
| Scheduler | `official_cyclic` |
| Checkpoint metric | `val_auc` |
| Scale | `262144/class` medium diagnostic |

Do not change validation data, labels, negative mode, metric computation, keying protocol, or Zhang/Wang Case2 sample construction.

## Proposed Model

Model key:

```text
present_nibble_invp_pair_consistency_spn_only
```

Input view:

```text
InvP(DeltaC)
```

Minimal architecture:

```text
C0 || C1
  -> DeltaC = C0 xor C1
  -> InvP(DeltaC)
  -> 16 nibble tokens per pair
  -> shared SPN token mixer
  -> 16 pair embeddings
  -> concat(
       mean(pair_embeddings),
       max(pair_embeddings),
       std(pair_embeddings),
       top-k/logsumexp evidence pooled embedding
     )
  -> classifier
  -> binary score
```

This intentionally changes only pair aggregation, not the dataset or per-pair InvP encoder.

Default options:

```text
spn_mixer_depth = 2
pooling = topk_logsumexp
top_k = 4
lse_temperature = 1.0
activation = relu
norm = layernorm
```

## Experiment Matrix

New smoke config:

```text
configs/experiment/innovation1/innovation1_spn_present_invp_centered_smoke.csv
```

New medium diagnostic config:

```text
configs/experiment/innovation1/innovation1_spn_present_invp_centered_r7_262k.csv
```

Rows:

| Rank | Model key | Role |
|---:|---|---|
| 0 | `present_zhang_wang_keras_mcnd` | same-protocol baseline |
| 1 | `present_nibble_invp_only_spn_only` | current strongest InvP-only anchor |
| 2 | `present_nibble_invp_pair_consistency_spn_only` | proposed InvP-centered consistency model |
| 3 | `present_nibble_paligned_spn_only` | old `DeltaC + InvP` anchor |
| 4 | `present_nibble_delta_only_spn_only` | DeltaC-only attribution control |
| 5 | `present_nibble_shuffled_paligned_spn_only` | shuffled-P attribution control |

## Decision Gates

Primary metric:

```text
validation AUC
```

Supporting metrics:

```text
calibrated accuracy, validation loss, fixed-threshold accuracy
```

Keep/continue conditions:

| Condition | Interpretation | Action |
|---|---|---|
| Pair-consistency AUC >= InvP-only AUC + 0.002 | aggregation improvement | run `262144/class` seed1 |
| Pair-consistency AUC within +/- 0.001 of InvP-only and lower loss | possible simplification/stability | run one more diagnostic seed before deciding |
| Pair-consistency below InvP-only by > 0.002 | aggregation not useful | keep InvP-only anchor, discard this model |
| InvP-only again beats DeltaC-only and shuffled-P | attribution stable | plan 1M/class seed0 for strongest InvP route |
| InvP-only fails against controls | attribution unstable | pause scaling and inspect seed/protocol/cache |

No formal claim is allowed from this single `262144/class` run.

## Execution Plan

1. Implement `present_nibble_invp_pair_consistency_spn_only` by reusing `_PresentNibblePAlignedSpnEncoder(view_mode="inv_p")`.
2. Add registry export and model factory branch.
3. Add tests for model construction/forward and plan protocol invariants.
4. Add smoke and `262144/class` configs.
5. Run local targeted tests and smoke.
6. If smoke passes, commit/push and launch remote `262144/class` with disk-backed shared dataset cache and tmux monitor retrieval.
7. When results are retrieved, update this document with gate, artifacts, metrics, deltas, decision, and next action.

## Claim Scope

This route can support:

```text
medium diagnostic evidence for InvP-centered architecture selection
```

It cannot yet support:

```text
formal reproduction
breakthrough claim
paper-scale multi-seed conclusion
```

## Launch Record

### 2026-06-28 Remote Medium Diagnostic

Status:

```text
completed remotely, retrieved by local tmux scp fallback, plan-aligned
```

Run metadata:

| Field | Value |
|---|---|
| Run ID | `i1_invp_centered_r7_262k_seed0_gpu1_20260628` |
| Source branch | `main` |
| Source commit at launch | `971144f` |
| Remote | `lxy-a6000` |
| Device | `cuda:1` |
| Remote run dir | `G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_invp_centered_r7_262k_seed0_gpu1_20260628` |
| Local monitor | `tmux: monitor_i1_invp_centered_20260628` |
| Expected result rows | `6` |
| Dataset cache root | `G:\lxy\blockcipher-structure-adaptive-nd-runs\shared_dataset_cache` |
| Dataset cache workers | `4` |
| Progress log | `logs\i1_invp_centered_r7_262k_seed0_gpu1_20260628_progress.jsonl` |

Launch gate:

```text
local smoke passed
tests/test_project_structure.py passed
code/config/docs pushed to GitHub
run-owned clean clone created under G:\lxy
cmd.exe /c launcher used
remote logs/progress created
initial stderr was 0 bytes after launcher entered training
```

Local smoke artifact:

```text
outputs/smoke/innovation1_invp_centered_smoke.jsonl
```

Next automatic action:

```text
tmux monitor waits for done/failed marker, retrieves logs/results/results_archive via scp,
then this document should be updated with gate status, metrics, deltas, and decision.
```

## Result Record

### 2026-06-28 Retrieved Result

Status:

```text
completed remotely
retrieved locally by tmux scp fallback
plan-aligned
not formal reproduction
not breakthrough evidence
```

Completion and retrieval:

| Field | Value |
|---|---|
| Run ID | `i1_invp_centered_r7_262k_seed0_gpu1_20260628` |
| Remote completion marker | `logs\i1_invp_centered_r7_262k_seed0_gpu1_20260628_done.marker` |
| Local monitor completion | `2026-06-28T23:41:49+08:00 done` |
| Result rows | `6` |
| Expected rows | `6` |
| Local alignment gate | `pass` |
| Remote stderr | `0 bytes` |
| Source commit at launch | `971144f` |
| Remote git status before run | `## main...origin/main` plus untracked run-local `logs/` |

Local artifacts:

| Artifact | Path |
|---|---|
| Results JSONL | `outputs/remote_results/i1_invp_centered_r7_262k_seed0_gpu1_20260628/results/i1_invp_centered_r7_262k_seed0_gpu1_20260628.jsonl` |
| Progress JSONL | `outputs/remote_results/i1_invp_centered_r7_262k_seed0_gpu1_20260628/logs/i1_invp_centered_r7_262k_seed0_gpu1_20260628_progress.jsonl` |
| Local gate | `outputs/remote_results/i1_invp_centered_r7_262k_seed0_gpu1_20260628/i1_invp_centered_r7_262k_seed0_gpu1_20260628_local_result_gate.json` |
| History CSV | `outputs/remote_results/i1_invp_centered_r7_262k_seed0_gpu1_20260628/i1_invp_centered_r7_262k_seed0_gpu1_20260628_history.csv` |
| Curves SVG | `outputs/remote_results/i1_invp_centered_r7_262k_seed0_gpu1_20260628/i1_invp_centered_r7_262k_seed0_gpu1_20260628_curves.svg` |

Metrics:

| Rank | Model | Accuracy | Calibrated Accuracy | AUC | Loss | Best Epoch | Epochs Ran |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | `present_zhang_wang_keras_mcnd` | 0.706841 | 0.710552 | 0.784347 | 0.560659 | 6 | 14 |
| 1 | `present_nibble_invp_only_spn_only` | 0.715893 | 0.716774 | 0.792105 | 0.547156 | 17 | 20 |
| 2 | `present_nibble_invp_pair_consistency_spn_only` | **0.716969** | **0.717220** | **0.792800** | **0.545899** | 19 | 20 |
| 3 | `present_nibble_paligned_spn_only` | 0.715004 | 0.716602 | 0.790835 | 0.549935 | 17 | 20 |
| 4 | `present_nibble_delta_only_spn_only` | 0.708637 | 0.708794 | 0.782918 | 0.556032 | 18 | 20 |
| 5 | `present_nibble_shuffled_paligned_spn_only` | 0.709686 | 0.710415 | 0.784487 | 0.555996 | 19 | 20 |

Key deltas:

| Comparison | AUC Delta | Interpretation |
|---|---:|---|
| Pair-consistency vs InvP-only | `+0.000695` | weak positive, below the predeclared `+0.002` continuation gate |
| Pair-consistency vs Zhang/Wang baseline | `+0.008453` | strong medium-scale diagnostic improvement over same-protocol baseline |
| Pair-consistency vs shuffled-P control | `+0.008312` | true InvP-centered route remains meaningfully above shuffled alignment |
| InvP-only vs DeltaC-only | `+0.009187` | attribution remains stable: InvP carries the useful signal |
| InvP-only vs shuffled-P control | `+0.007618` | attribution remains stable against shuffled-P |
| Old DeltaC+InvP anchor vs InvP-only | `-0.001270` | adding raw `DeltaC` still does not improve the InvP-only route |

Decision gate:

```text
Pair-consistency AUC did not beat InvP-only by +0.002, so it is not a clear
aggregation improvement. However, it is best on AUC, calibrated accuracy, fixed
threshold accuracy, and loss, and remains above baseline and shuffled-P.
```

Decision:

```text
Keep pair-consistency as weak-positive diagnostic evidence, not as a promoted
main route yet. Do not scale this 6-row matrix again. Next run should use a
lean 262144/class seed1 confirmation matrix with only:

1. present_nibble_invp_pair_consistency_spn_only
2. present_nibble_invp_only_spn_only
3. optional present_zhang_wang_keras_mcnd baseline

If pair-consistency remains best across seed1 or improves by >= +0.002 AUC,
promote it to the candidate for 1M/class. If it stays within noise, keep the
simpler InvP-only anchor as the main route.
```

Claim scope:

```text
This is medium diagnostic single-seed evidence at 262144/class. It supports
InvP-centered SPN architecture selection, but it does not support formal
reproduction, breakthrough claims, or publication-scale multi-seed conclusions.
```
