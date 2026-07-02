# Innovation 1 Training Accelerator Topology Seed1 Plan

**Date:** 2026-07-01

**Status:** failed accelerator diagnostic

**Run ID:** `i1_spn_topology_aware_network_r7_262k_seed1_gpu1_accel_bf16_20260701`

## Purpose

Run the previously planned topology-aware seed1 diagnostic through the optional
training accelerator plugin to test whether the accelerator path can execute the
same Innovation 1 protocol on the remote A6000 workstation.

This is an accelerator-path diagnostic. It is not a new cryptanalytic method and
must not be used as evidence that the topology-aware route is better unless it is
retrieved, validated, plan-aligned, and compared against the non-accelerated seed0
and route anchors.

## Background

The earlier seed1 watcher for:

```text
i1_spn_topology_aware_network_r7_262k_seed1_gpu1_20260701
```

was stopped after remote inspection showed:

```text
remote run directory: missing
remote python training process: none
GPU utilization: idle
local watcher: only looping on missing logs/results
```

Therefore the old run is classified as **not launched successfully**, not as
running, completed, failed training, or retrieved evidence.

## Research Question

Can the plugin accelerator runner execute the same topology-aware seed1 262144/class
matrix under the official Zhang/Wang MCND protocol while preserving result schema and
recording accelerator metadata?

## Protocol

```text
cipher: PRESENT-80
rounds: 7
plan: configs/experiment/innovation1/innovation1_spn_present_topology_aware_network_r7_262k_seed1.csv
expected_rows: 3
samples_per_class: 262144
pairs_per_sample: 16
negative_mode: encrypted_random_plaintexts
sample_structure: zhang_wang_case2_official_mcnd
checkpoint_metric: val_auc
loss: mse
lr_scheduler: official_cyclic
max_learning_rate: 0.002
train_eval_interval: 0
dataset_cache: shared disk cache under G:\lxy
```

## Accelerator Settings

```text
plugin command: python -m blockcipher_training_accelerator run-accelerated
speed_profile: amp-bf16
device: cuda:1
batch_size: 1024
epochs: 20
hidden_bits: 32
```

`amp-bf16` is expected to become effective only on CUDA. The result rows must record
`training.accelerator.profile`, `amp_effective`, `compile_effective`, and timing metadata.

## Remote Policy

```text
remote alias: lxy-a6000
remote root: G:\lxy\blockcipher-structure-adaptive-nd-runs
run root: G:\lxy\blockcipher-structure-adaptive-nd-runs\<run_id>
source: run-owned clean clone from pushed GitHub main
command shell: cmd.exe /c
python: F:\Anaconda\envs\DWT\torch310\python.exe
result retrieval: local tmux watcher via scp fallback
```

## Gate

The run is usable only if all are true:

1. Remote run directory exists and contains logs/results.
2. Results JSONL has exactly 3 non-empty rows.
3. `scripts/postprocess-topology-aware-result` succeeds.
4. Rows are plan-aligned with the topology-aware seed1 plan.
5. Each row contains `training.accelerator.profile = amp-bf16`.
6. No NaN/Inf or schema break appears in metrics/history.

## Claim Scope

If successful, this run can support only:

```text
accelerator path can execute topology-aware seed1 diagnostic
```

It cannot by itself establish:

```text
formal topology-aware route evidence
speedup over baseline
publication-level accuracy conclusion
```

Speedup requires a later same-protocol baseline-vs-accelerated timing comparison and
`quality-gate` comparison.

## Next Action

Launch the run from the pushed commit, start a local tmux watcher, and wait for retrieved
artifacts before analyzing metrics or attempting further accelerator profiles.

## 2026-07-02 Retrieved Status

The remote accelerator run launched and produced local artifacts, but it did not
complete the planned 3-row matrix.

```text
remote status: failed
local watcher: failed at 2026-07-01T23:00:52+08:00
results rows: 1 / 3
progress rows: 330
failed marker: present
remote python process after check: none
GPU utilization after check: GPU0 0%, GPU1 0%
source revision: 5337230ef1d364859576ecaec17d1698c74fe5c4
remote git status before run: clean main tracking origin/main
```

Completed row:

```text
model: present_nibble_invp_only_spn_only
samples_per_class: 262144
seed: 1
speed_profile: amp-bf16
best_epoch: 18
val_auc / auc: 0.793222204898484
accuracy: 0.7070960998535156
calibrated_accuracy: 0.7185859680175781
row_duration_seconds: 1321.692667
```

The run failed at row 2:

```text
model: present_nibble_invp_p_layer_graph_spn_only
event: accelerated_train_start
speed_profile: amp-bf16
error: scatter(): Expected self.dtype to be equal to src.dtype
location: src/blockcipher_nd/models/common/components.py, weights.scatter_(...)
```

Interpretation:

- This is **not** a completed topology-aware seed1 result.
- This is **not** validated plan-aligned evidence for the 3-row topology matrix.
- The single completed InvP-only row is useful as a partial diagnostic, but the run
  fails the planned gate because only 1/3 rows exist.
- The failure indicates that the `amp-bf16` accelerator profile is not safe for all
  current topology-aware graph/pooling models. Under autocast, the scatter target and
  source tensors can land in different dtypes.

Decision:

Do not continue using `amp-bf16` for topology-aware graph rows until the dtype path is
made safe and covered by a smoke/quality gate. The next accelerator attempt should use
an FP32-preserving plugin profile, preferably `dataloader`, so that dataset/cache and
loader-side acceleration can be tested without changing model math. If AMP is revisited,
the fix should be treated as accelerator tooling work and must pass a graph-model AMP
smoke before being used in a meaningful remote run.
