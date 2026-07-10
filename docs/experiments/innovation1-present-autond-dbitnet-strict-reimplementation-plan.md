# Innovation 1 PRESENT AutoND/DBitNet Strict Reimplementation Plan

**Date:** 2026-07-10

**Status:** R0 passed; R1 completed, fallback-retrieved, and plan-aligned; R2 blocked pending audit

**Claim scope:** published-baseline audit, not an Innovation 1 novelty result

## Goal

Reimplement the AutoND/DBitNet PRESENT single-pair pipeline closely enough to
audit the published r8/r9 baseline while preserving the project's strict
encrypted-random-plaintext negative definition.

This is deliberately separate from the Zhang/Wang Case2 multi-pair task and
from any new typed-SPN candidate experiment.

## Why This Is Not Called an Exact Reproduction

Bellini et al. describe the negative class as ciphertext pairs produced from
random plaintext pairs. The current public AutoND repository at commit
`8d66e0a43a9e6fbbb3624d68296c1b77aa2da779` instead replaces negative-class
ciphertext bits with random bits in `main.py::make_train_data`.

The project evidence rules require:

```text
negative_mode = encrypted_random_plaintexts
```

Therefore this experiment is a strict-protocol reimplementation audit. It can
test whether the published architecture and curriculum transfer to the
project's benchmark, but it cannot claim byte-for-byte reproduction of the
authors' data generator.

## Research Question

Under standard single-key PRESENT-80 with one ciphertext pair per sample and
input difference `0x000000000d000000`, does a faithful DBitNet architecture
trained with the AutoND round curriculum recover a stable r8 signal and any r9
signal under strict negatives?

## Frozen Protocol

```text
cipher                  = PRESENT-80
round curriculum        = r5 -> r6 -> r7 -> r8 -> r9
pairs_per_sample        = 1
feature_encoding        = ciphertext_pair_bits
input_difference        = 0x000000000d000000
negative_mode           = encrypted_random_plaintexts
sample_structure        = independent_pairs
train_key               = 0x00000000000000000000
validation_key          = 0x11111111111111111111
key_rotation_interval   = 0
loss                    = MSE over sigmoid probability
optimizer               = Adam with AMSGrad
learning_rate           = 0.001
lr_scheduler            = none
batch_size              = 5000 for remote diagnostic
checkpoint_metric       = val_accuracy
restore_best_checkpoint = true
```

The separate train and validation keys are a project-side generalization gate.
They are held constant across every curriculum stage.

## Architecture Contract

The new `autond_dbitnet2023` model must follow the public AutoND source rather
than the existing simplified `dbitnet_dilated_cnn`:

```text
input width             = 128 bits (C || C')
input normalization     = (x - 0.5) / 0.5
dilation schedule       = [63, 31, 15, 7, 3]
initial channels        = 32
channel increment       = 16 per wide-narrow block
wide convolution        = Conv1D kernel 2, valid, dilation d, ReLU, BatchNorm
narrow convolution      = Conv1D kernel 2, causal, dilation 1, ReLU
block merge             = narrow + wide residual, then BatchNorm
flattened width         = 96 channels x 9 positions
prediction head         = 256 -> 256 -> 64 -> 1
dense hidden layers     = Linear, BatchNorm, ReLU
dense L2 coefficient    = 1e-5, including output layer
```

The model returns logits because the project trainer owns the sigmoid used by
MSE and metric evaluation.

## Curriculum Contract

The matrix schema gains an optional JSON list:

```text
pretrain_round_sequence = [5, 6, 7, 8]
```

For a target r9 row, the same model instance trains on each listed round in
order and then trains on r9. Each stage receives a separately seeded,
round-specific, disk-backed train and validation dataset. Results must retain
per-stage round, metrics, epochs, and checkpoint metadata.

Existing scalar `pretrain_rounds` behavior remains supported and is treated as
a one-element sequence.

## Experiment Ladder

### R0: CPU readiness smoke

```text
rows              = 1
seed              = 0
target            = r9
samples_per_class = 128
stage epochs      = 1
target epochs     = 1
batch_size        = 64
device            = cpu
```

R0 validates construction, forward/backward execution, strict negatives,
five-stage curriculum progress, cache reuse, JSONL output, plotting, and plan
alignment. It is not accuracy evidence.

### R1: remote medium diagnostic

```text
rows              = 1
seed              = 0
target            = r9
samples_per_class = 65536
stage epochs      = 10
target epochs     = 10
batch_size        = 5000
device            = selected free A6000 GPU
```

R1 uses disk-backed datasets and progress logging under
`G:\lxy\blockcipher-structure-adaptive-nd-runs`. It is a medium diagnostic,
not formal training and not a ceiling claim.

### Conditional later ladder

Only if R1 is protocol-valid and learns the lower rounds:

```text
R2 = 262144/class, 40 epochs per stage, seed0
R3 = 1000000/class, 40 epochs per stage, seeds 0 and 1
```

R3 is the first project-scale evidence suitable for a formal route comparison.
The authors' `10^7` total training rows correspond to `5000000/class` in this
balanced project terminology and remain a later exact-scale option, not the
first remote launch.

## R1 Gate

R1 passes readiness only when all integrity checks pass:

```text
model_key                  = autond_dbitnet2023
input_bits                 = 128
dilations                  = [63,31,15,7,3]
optimizer.amsgrad          = true
negative_mode              = encrypted_random_plaintexts
pairs_per_sample           = 1
pretraining round sequence = [5,6,7,8]
result rows                = 1
```

Research interpretation uses stage metrics:

```text
r5 accuracy >= 0.75 and r6 accuracy >= 0.60:
  basic pipeline sanity supported

r7 accuracy >= 0.52:
  lower-round signal supported

r8 accuracy > 0.505:
  allow R2 design review

r8 accuracy <= 0.505:
  stop scale-up and audit implementation/protocol mismatch

r9:
  report diagnostically; no positive requirement at 65536/class
```

The published targets are reported as two inconsistent views and are not used
as hard R1 gates:

```text
summary table: r8 0.5546, r9 0.5092
appendix runs: r8 about 0.5106/0.5120, r9 about 0.5012/0.5018
```

Accuracy is the primary literature-comparison metric. AUC remains a
project-side supplementary metric and must not be numerically conflated with
published accuracy.

## Implementation Tasks

1. Add failing unit tests for the AutoND dilation schedule, block geometry,
   dense head, logits, and dense-only L2 auxiliary loss.
2. Add a failing matrix/curriculum test for `[5,6,7,8]` and per-stage metadata.
3. Implement only the model and curriculum behavior needed by those tests.
4. Add smoke and 65536/class CSV plans.
5. Run focused tests, R0 training, result validation, and plot generation.
6. Commit and push the scoped implementation and plan.
7. Generate an R1 remote config, `cmd.exe /c` launcher, and local tmux monitor.
8. Run remote readiness, perform one bounded launch confirmation, and leave
   result retrieval to the local monitor.

## Planned Artifacts

R0:

```text
outputs/local_smoke/i1_present_autond_dbitnet_strict_smoke_seed0/results.jsonl
outputs/local_smoke/i1_present_autond_dbitnet_strict_smoke_seed0/progress.jsonl
outputs/local_smoke/i1_present_autond_dbitnet_strict_smoke_seed0/curves.svg
outputs/local_smoke/i1_present_autond_dbitnet_strict_smoke_seed0/history.csv
outputs/local_cache/i1_present_autond_dbitnet_strict_smoke_seed0/
```

R1 after retrieval:

```text
outputs/remote_results/i1_present_autond_dbitnet_strict_65k_seed0_<gpu>_20260710/
```

## R0 Readiness Result

The CPU smoke completed on 2026-07-10 with the planned five-stage execution:

```text
pretraining rounds = [5, 6, 7, 8]
target round       = 9
epochs per round   = 1
samples_per_class  = 128
result rows        = 1
plan alignment     = pass
```

The persisted model and protocol integrity fields are:

```text
model_key         = autond_dbitnet2023
input_bits        = 128
dilations         = [63, 31, 15, 7, 3]
output_width      = 9
output_channels   = 96
flattened_width   = 864
dense L2          = 1e-5
optimizer.amsgrad = true
loss              = mse
negative_mode     = encrypted_random_plaintexts
pairs_per_sample  = 1
```

One-epoch readiness metrics were:

```text
r5 accuracy=0.4765625 AUC=0.511474609375
r6 accuracy=0.5000000 AUC=0.463867187500
r7 accuracy=0.4765625 AUC=0.442871093750
r8 accuracy=0.5000000 AUC=0.494628906250
r9 accuracy=0.5000000 AUC=0.507080078125
```

These tiny one-epoch scores are not research evidence and are not compared
against the R1 gates. R0 only establishes that the architecture, dense-only L2,
strict data protocol, disk-backed per-round caches, ordered curriculum,
checkpoint restoration, progress events, JSONL result, validation, and plotting
paths execute end to end. A second run reused all ten round/split caches.
After seeding model construction from the task seed, two consecutive complete
R0 runs produced the identical result SHA-256
`c595634e58b80094e391648465ba654b5ee97733298c78dc20d2300b4aa378af`.

R0 artifacts:

```text
outputs/local_smoke/i1_present_autond_dbitnet_strict_smoke_seed0/results.jsonl
outputs/local_smoke/i1_present_autond_dbitnet_strict_smoke_seed0/progress.jsonl
outputs/local_smoke/i1_present_autond_dbitnet_strict_smoke_seed0/curves.svg
outputs/local_smoke/i1_present_autond_dbitnet_strict_smoke_seed0/history.csv
outputs/local_cache/i1_present_autond_dbitnet_strict_smoke_seed0/
```

## R1 Launch Package

The medium diagnostic is packaged as:

```text
run_id       = i1_present_autond_dbitnet_strict_65k_seed0_gpu1_20260710
device       = cuda:1
source       = pushed main commit recorded by the remote launcher
remote root  = G:\lxy\blockcipher-structure-adaptive-nd-runs\<run_id>
cache root   = <remote root>\dataset_cache
result sync  = local tmux monitor with SCP fallback retrieval
```

The remote readiness gate locks the strict single-pair protocol, AutoND
difference, MSE/Adam/AMSGrad settings, `[5,6,7,8]` curriculum, one result row,
and disk-backed cache. The local monitor requires the remote done marker and
one result row before it validates, plots, and writes the R1 integrity/decision
artifact.

Launch handoff on 2026-07-10:

```text
source commit       = 9491e47c72e977ebc4c465061d95cc030339db88
remote source       = run-owned clean clone, main aligned to source commit
scheduled command   = cmd.exe /c <tracked launcher>
bounded confirmation = launch_env log present
local tmux monitor  = i1_autond_dbitnet_65k_seed0_gpu1_20260710
monitor state       = running; completion/result retrieval pending
```

The bounded confirmation did not yet observe a `started.marker`, so this record
does not promote the run to completed remotely or retrieved. The monitor owns
subsequent waiting and fallback retrieval.

## R1 Retrieved Result

The local monitor fallback-retrieved the completed run on 2026-07-10 and
finished validation, plotting, and gate generation:

```text
run_id             = i1_present_autond_dbitnet_strict_65k_seed0_gpu1_20260710
source commit      = 9491e47c72e977ebc4c465061d95cc030339db88
remote run time    = 665.94 seconds (11 minutes 6 seconds)
local postprocess  = completed about 13 minutes 48 seconds after first sync
result rows        = 1 / 1
plan alignment     = pass
retrieval status   = fallback-retrieved from the G:\lxy run-owned directory
claim scope        = 65536/class single-seed medium diagnostic only
```

Protocol integrity passed:

```text
model_key          = autond_dbitnet2023
input_bits         = 128
dilations          = [63,31,15,7,3]
amsgrad            = true
negative_mode      = encrypted_random_plaintexts
pairs_per_sample   = 1
round_sequence     = [5,6,7,8]
train storage      = disk
validation storage = disk
```

Restored-best-checkpoint validation metrics were:

| Round | Accuracy | AUC | Best epoch |
| --- | ---: | ---: | ---: |
| r5 | 0.634368896 | 0.758460025 | 10 |
| r6 | 0.576644897 | 0.603231360 | 3 |
| r7 | 0.510253906 | 0.515467749 | 3 |
| r8 | 0.501113892 | 0.499572861 | 1 |
| r9 | 0.504653931 | 0.502980342 | 6 |

R1 did not pass the planned lower-round sanity or r8 advancement gates:

```text
r5 required >= 0.75; observed 0.634368896
r6 required >= 0.60; observed 0.576644897
r7 required >= 0.52; observed 0.510253906
r8 required >  0.505; observed 0.501113892
```

The adjudication is:

```text
decision = stop_and_audit_lower_round_pipeline
remote_scale = no
R2 262144/class = blocked
```

This is not a ceiling claim: `65536/class` is about 76 times smaller per class
than the public AutoND `10^7`-row training set, uses strict encrypted-random-
plaintext negatives, and has one seed. Before any larger run, audit the remaining
reimplementation differences, particularly checkpoint selection (`val_accuracy`
versus the public code's `val_loss` checkpoint), optimizer-state handling across
rounds, held-out validation key, and public random-ciphertext behavior as an
explicitly labeled ablation only.

The remote launcher calculated and enforced the one-row condition, but its
archived `result_gate.txt` lost the displayed numeric values because adjacent
Windows redirection parsed the trailing `1` as a file descriptor. Independent
local plan validation passed with `result_rows=1`, `expected_rows=1`, and no
errors. The tracked launcher has been corrected to put spaces before `>`/`>>`;
this logging defect does not change the retrieved metrics.

Retrieved artifacts:

```text
outputs/remote_results/i1_present_autond_dbitnet_strict_65k_seed0_gpu1_20260710/results/
outputs/remote_results/i1_present_autond_dbitnet_strict_65k_seed0_gpu1_20260710/logs/
outputs/remote_results/i1_present_autond_dbitnet_strict_65k_seed0_gpu1_20260710/i1_present_autond_dbitnet_strict_65k_seed0_gpu1_20260710_validation.json
outputs/remote_results/i1_present_autond_dbitnet_strict_65k_seed0_gpu1_20260710/i1_present_autond_dbitnet_strict_65k_seed0_gpu1_20260710_gate.json
outputs/remote_results/i1_present_autond_dbitnet_strict_65k_seed0_gpu1_20260710/i1_present_autond_dbitnet_strict_65k_seed0_gpu1_20260710_curves.svg
outputs/remote_results/i1_present_autond_dbitnet_strict_65k_seed0_gpu1_20260710/i1_present_autond_dbitnet_strict_65k_seed0_gpu1_20260710_history.csv
```

## Decision Boundaries

- This audit does not reopen DDT input exploration.
- This audit does not continue the corrected E1 graph architecture.
- Random-ciphertext negatives may be used only as a separately labeled
  ablation and are not part of R0 or R1.
- No r8/r9 ceiling claim is allowed from R1 or R2.
- A successful baseline audit calibrates the next typed-SPN candidate; it is
  not itself the Innovation 1 method contribution.
