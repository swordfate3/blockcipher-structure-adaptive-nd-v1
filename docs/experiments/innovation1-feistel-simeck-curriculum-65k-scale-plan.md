# Innovation 1 SIMECK Curriculum 65536/Class Scale Plan

**Status:** remote package preparation authorized by two-seed local gate

## Question

Does the SIMECK64/128 r14-to-r15 true-relation curriculum retain its absolute
signal and controlled advantage when only the sample scale changes from
`8192/class` to `65536/class`?

## Local Evidence

| Seed | curriculum true | shuffled | target scratch | true-shuffled | true-scratch |
|---:|---:|---:|---:|---:|---:|
| 0 | `0.696612916` | `0.503923842` | `0.511121858` | `+0.192689074` | `+0.185491058` |
| 1 | `0.701582395` | `0.504952087` | `0.593733866` | `+0.196630308` | `+0.107848529` |

Both seeds pass the frozen `0.55/+0.02/+0.01` local gates. SIMON is excluded
because its seed0 curriculum AUC `0.534241434` missed the absolute signal gate.

## Paper-Protocol Boundary

Lu et al.'s basic protocol uses `2e7` total training samples, `2e6` test
samples, 120 epochs, batch 30000, a cyclic learning rate, an SE-ResNet, and
five repeated runs. This scale probe uses `131072` total training rows per
stage, only `0.65536%` of the paper training rows. It retains the project
pair-pool network, strict encrypted-random-plaintext negatives, 10 total
epochs, and AUC. It is a medium diagnostic, not an exact or paper-scale
reproduction, and paper accuracy is not directly comparable to project AUC.

## Frozen Matrix

```text
plan = configs/experiment/innovation1/innovation1_feistel_simeck_curriculum_scale_probe_65536_seed0.csv
run_id = i1_feistel_simeck_curriculum_65k_seed0
device = remote A6000 GPU0
seed = 0
```

Three roles:

```text
true relation: r14 5 epochs -> r15 5 epochs
shuffled relation: r14 5 epochs -> r15 5 epochs
true relation scratch: r15 10 epochs
```

Common protocol:

```text
samples_per_class       = 65536
train total/stage       = 131072
validation total        = 131072
fresh test total        = 262144 x 3 repeats
pairs_per_sample        = 8
input_difference        = (0x00000000, 0x00000040)
negative_mode           = encrypted_random_plaintexts
key_rotation_interval   = 1
key row indexing        = global_dataset_row
batch / hidden bits     = 128 / 32
loss / optimizer        = MSE / Adam
learning rate / L2      = 1e-4 / 1e-5
optimizer transition    = reset_each_stage
checkpoint              = best val_loss restored per stage
```

## Disk And Remote Requirements

All source, cache, logs, results, and archives stay under:

```text
G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_feistel_simeck_curriculum_65k_seed0
```

The runner must use a disk-backed dataset cache with `features.npy`,
`labels.npy`, metadata, chunk progress, parameter-matched reuse, chunk size
8192, and four generation workers. The launch must use `cmd.exe /c`, a clean
run-owned clone pinned to a pushed commit, a local tmux retrieval monitor, and
a verified result branch/archive gate.

## Scale Gate

Fresh-test AUC must satisfy all four conditions:

```text
curriculum true                                  >= 0.65
curriculum true - shuffled                       >= +0.10
curriculum true - equal-epoch scratch             >= +0.05
curriculum true - local seed0 anchor 0.696612916 >= -0.02
```

If all pass, run the identical `65536/class` seed1 matrix before any stronger
claim or larger scale. If the first three pass but the scale-preservation gate
fails, hold remote scale and retain only local two-seed evidence. Otherwise
stop this curriculum scale route. No rescue sweep is authorized.

## Required Artifacts

```text
results.jsonl
validation.json
gate.json
progress.jsonl
git_revision.txt
git_status_before_run.txt
gpu_info.txt
torch_info.txt
plan.csv
remote_config.json
SHA256SUMS
```

The local monitor must retrieve, verify hashes, re-run alignment and the scale
gate, render the Chinese-readable SVG/CSV, refresh the recent-results index,
and leave a verified retrieval marker before the result is reported.
