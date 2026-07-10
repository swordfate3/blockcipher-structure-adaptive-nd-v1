# Innovation 1 PRESENT AutoND/DBitNet Strict Reimplementation Plan

**Date:** 2026-07-10

**Status:** R0/R1/C1 completed; R1A-C2 local readiness and remote medium-diagnostic package verified, pending pushed-commit launch; paper-scale reproduction not yet run

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

## Recommended Next Step: R1A Protocol Attribution Audit

Do not launch R2 at `262144/class`. R1 missed not only the r8 advancement gate
but also every r5-r7 lower-round sanity gate, so additional samples cannot yet
be interpreted as a test of r8 data scarcity. The next experiment slot should
first determine whether a bounded reimplementation/protocol mismatch explains
the weak curriculum transfer.

The source-contract audit against public AutoND commit
`8d66e0a43a9e6fbbb3624d68296c1b77aa2da779` found two concrete differences:

```text
public checkpoint selection       = best val_loss
R1 checkpoint selection           = best val_accuracy

public optimizer transition       = one compiled Adam/AMSGrad object reused
                                    across rounds; weights loaded from the prior
                                    round's best-val_loss checkpoint
R1 optimizer transition           = a new Adam/AMSGrad object per round stage
```

The public code does not recompile the model or instantiate a new optimizer in
its round loop. Its `clear_session()` call does not explicitly replace the
live model or optimizer objects. The current PyTorch trainer creates a new
optimizer on each `train_binary_classifier` call, so its transition is
explicitly recorded as `reset_each_stage`.

R1A-C1 adjudicates checkpoint selection first because `val_loss` is directly
specified by the public `ModelCheckpoint`, is already supported by the local
trainer, and can be changed without modifying optimization dynamics. Optimizer
carry is reserved for a later C2 only if C1 fails; C1 must not combine both
differences.

The R1A-C1 research question is:

```text
Can best-val_loss checkpoint transfer restore the lower-round R1 gates
under the strict encrypted-random-plaintext benchmark, without changing the
DBitNet architecture or increasing the dataset size?
```

Frozen R1A-C1 plans:

```text
local readiness:
  plan              = configs/experiment/innovation1/innovation1_spn_present_autond_dbitnet_r1a_valloss_smoke_seed0.csv
  samples_per_class = 128
  epochs_per_round  = 2

remote diagnostic:
  plan              = configs/experiment/innovation1/innovation1_spn_present_autond_dbitnet_r1a_valloss_65k_seed0.csv
  samples_per_class = 65536
  seed              = 0
  epochs_per_round  = 10
```

The only R1-to-R1A-C1 training-protocol change is:

```text
checkpoint_metric = val_accuracy -> val_loss
```

Execute R1A-C1 in this order:

1. Run the local CPU readiness smoke. Require two epochs so best-val-loss
   selection and restoration execute non-trivially; do not interpret its
   accuracy as research evidence.
2. Keep architecture, input difference, negatives, keys, curriculum order,
   optimizer reset behavior, and all other training settings frozen.
3. If the smoke and artifact gates pass, run one same-budget R1A-C1 diagnostic at
   `65536/class`, seed 0, and 10 epochs per round. Compare it to the completed
   R1 anchor; do not combine checkpoint, optimizer, key, or negative-generation
   changes in one row.
4. Use same-key validation only as a separately labeled generalization control
   if the strict held-out-key result remains weak. Use public random-ciphertext
   negatives only as an author-protocol ablation; they cannot replace strict
   encrypted-random-plaintext evidence.

### R1A-C1 Local Readiness Result

The two-epoch CPU smoke completed on 2026-07-10:

```text
samples_per_class          = 128
round sequence             = [5,6,7,8] -> 9
checkpoint_metric          = val_loss
selected checkpoint        = best for every stage
optimizer_state_transition = reset_each_stage
result rows                = 1 / 1
plan alignment             = pass
plot/history generation    = pass
```

The tiny validation metrics are readiness diagnostics only:

```text
r5 accuracy=0.4609375 AUC=0.5625000
r6 accuracy=0.5078125 AUC=0.5004883
r7 accuracy=0.4765625 AUC=0.3823242
r8 accuracy=0.5000000 AUC=0.6044922
r9 accuracy=0.5000000 AUC=0.4899902
```

Artifacts:

```text
outputs/local_smoke/i1_present_autond_dbitnet_r1a_valloss_smoke_seed0/results.jsonl
outputs/local_smoke/i1_present_autond_dbitnet_r1a_valloss_smoke_seed0/progress.jsonl
outputs/local_smoke/i1_present_autond_dbitnet_r1a_valloss_smoke_seed0/validation.json
outputs/local_smoke/i1_present_autond_dbitnet_r1a_valloss_smoke_seed0/curves.svg
outputs/local_smoke/i1_present_autond_dbitnet_r1a_valloss_smoke_seed0/history.csv
outputs/local_cache/i1_present_autond_dbitnet_r1a_valloss_smoke_seed0/
```

Recommendation after the readiness gate: proceed to the single R1A-C1 remote
`65536/class`, seed-0 diagnostic on the idle `cuda:1`. Do not add optimizer
carry, same-key validation, random-ciphertext negatives, another seed, or a
larger scale to this run.

### R1A-C1 Remote Package

```text
run_id       = i1_present_autond_dbitnet_r1a_valloss_65k_seed0_gpu1_20260710
device       = cuda:1
GPU gate     = 150 MiB / 49140 MiB, 0% utilization at bounded pre-launch check
remote root  = G:\lxy\blockcipher-structure-adaptive-nd-runs\<run_id>
cache root   = <remote root>\dataset_cache
result sync  = local tmux monitor with SCP fallback retrieval
```

Tracked launch artifacts:

```text
configs/remote/innovation1_spn_present_autond_dbitnet_r1a_valloss_65k_seed0_gpu1_20260710.json
configs/remote/generated/run_i1_present_autond_dbitnet_r1a_valloss_65k_seed0_gpu1_20260710.cmd
configs/remote/generated/monitor_i1_present_autond_dbitnet_r1a_valloss_65k_seed0_gpu1_20260710.sh
```

The local readiness report passes all nine invariants, including the one-row
plan, strict-negative protocol, medium-scale disk cache, and AutoND protocol
lock. The monitor validates and plots the retrieved row, verifies `val_loss`
for the target and all curriculum stages, verifies `reset_each_stage`, and
writes per-round accuracy/AUC deltas versus the completed R1 anchor.

Launch handoff on 2026-07-10:

```text
source commit        = 96a8f4c4f9f69a2090fdbc3137c0071c03cc1e38
remote source        = run-owned clean clone, main aligned to source commit
scheduled task       = i1_autond_r1a_valloss_65k_gpu1_20260710
scheduled command    = cmd.exe /c <tracked launcher>
bounded confirmation = launch_env, Git, GPU/Torch, and readiness logs present
started.marker       = not yet observed at the single bounded check
local tmux monitor   = i1_autond_r1a_valloss_65k_seed0_gpu1_20260710
monitor state        = active; completion/result retrieval pending
```

The launch is therefore recorded as launcher-started and watcher-managed, not
as completed remotely or retrieved. The main thread must not SSH-poll it; the
local monitor owns subsequent waiting, SCP fallback retrieval, validation,
plotting, R1 delta generation, and final gate creation.

### R1A-C1 Retrieved Result

The local watcher retrieved and postprocessed the completed run on 2026-07-10:

```text
run_id             = i1_present_autond_dbitnet_r1a_valloss_65k_seed0_gpu1_20260710
source commit      = 96a8f4c4f9f69a2090fdbc3137c0071c03cc1e38
remote run time    = 720.63 seconds (12 minutes 0.63 seconds)
result rows        = 1 / 1
plan alignment     = pass
retrieval status   = fallback-retrieved from the G:\lxy run-owned directory
watcher completion = postprocess_done at 2026-07-10T17:39:22+08:00
claim scope        = 65536/class single-seed medium diagnostic only
```

Every integrity check passed, including the AutoND model geometry, strict
negative definition, `[5,6,7,8]` curriculum, best-`val_loss` checkpoint for
every stage, and `reset_each_stage` optimizer transition.

| Round | R1 accuracy | C1 accuracy | Accuracy delta | R1 AUC | C1 AUC | AUC delta | C1 best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| r5 | 0.634368896 | 0.517059326 | -0.117309570 | 0.758460025 | 0.729835789 | -0.028624236 | 7 |
| r6 | 0.576644897 | 0.562896729 | -0.013748169 | 0.603231360 | 0.585832156 | -0.017399204 | 2 |
| r7 | 0.510253906 | 0.505065918 | -0.005187988 | 0.515467749 | 0.508800202 | -0.006667547 | 3 |
| r8 | 0.501113892 | 0.498672485 | -0.002441406 | 0.499572861 | 0.497703894 | -0.001868967 | 3 |
| r9 | 0.504653931 | 0.502273560 | -0.002380371 | 0.502980342 | 0.505101557 | +0.002121215 | 1 |

C1 failed every frozen advancement gate:

```text
r5 required >= 0.75; observed 0.517059326
r6 required >= 0.60; observed 0.562896729
r7 required >= 0.52; observed 0.505065918
r8 required >  0.505; observed 0.498672485

decision     = stop_and_audit_lower_round_pipeline
remote_scale = no
R2           = blocked
```

Selecting best `val_loss` alone does not explain the R1 gap. It worsened both
accuracy and AUC at r5-r8 relative to the same-budget R1 anchor. The small r9
AUC increase remains near-chance diagnostic variation and does not compensate
for the lower-round regression. This is not a route-ceiling claim because the
run is still `65536/class` and single-seed.

Retrieved artifacts:

```text
outputs/remote_results/i1_present_autond_dbitnet_r1a_valloss_65k_seed0_gpu1_20260710/results/
outputs/remote_results/i1_present_autond_dbitnet_r1a_valloss_65k_seed0_gpu1_20260710/logs/
outputs/remote_results/i1_present_autond_dbitnet_r1a_valloss_65k_seed0_gpu1_20260710/i1_present_autond_dbitnet_r1a_valloss_65k_seed0_gpu1_20260710_validation.json
outputs/remote_results/i1_present_autond_dbitnet_r1a_valloss_65k_seed0_gpu1_20260710/i1_present_autond_dbitnet_r1a_valloss_65k_seed0_gpu1_20260710_gate.json
outputs/remote_results/i1_present_autond_dbitnet_r1a_valloss_65k_seed0_gpu1_20260710/i1_present_autond_dbitnet_r1a_valloss_65k_seed0_gpu1_20260710_curves.svg
outputs/remote_results/i1_present_autond_dbitnet_r1a_valloss_65k_seed0_gpu1_20260710/i1_present_autond_dbitnet_r1a_valloss_65k_seed0_gpu1_20260710_history.csv
```

### Recommended Next Step: R1A-C2 Optimizer Carry

C2 is the final training-protocol attribution recommended before closing this
strict baseline audit. Compare directly against C1 and change only:

```text
optimizer_state_transition = reset_each_stage -> carry_across_stages
```

Freeze the remaining protocol:

```text
checkpoint_metric  = val_loss
samples_per_class  = 65536
seed               = 0
epochs_per_round   = 10
round sequence     = [5,6,7,8] -> 9
architecture       = autond_dbitnet2023
input              = 128-bit C || C'
negative_mode      = encrypted_random_plaintexts
train/validation keys remain separate
```

Frozen C2 plans:

```text
local readiness:
  plan              = configs/experiment/innovation1/innovation1_spn_present_autond_dbitnet_r1a_c2_optcarry_smoke_seed0.csv
  samples_per_class = 128
  epochs_per_round  = 2

remote diagnostic:
  plan              = configs/experiment/innovation1/innovation1_spn_present_autond_dbitnet_r1a_c2_optcarry_65k_seed0.csv
  samples_per_class = 65536
  seed              = 0
  epochs_per_round  = 10
```

The local smoke passes readiness only when:

```text
r5 optimizer reused          = false
r5 optimizer step before     = 0
r6-r9 optimizer reused       = true
optimizer step before/after  = continuous and strictly increasing
checkpoint metric            = val_loss for every stage
result validation/artifacts  = pass
```

Implementation must make optimizer ownership explicit, preserve Adam/AMSGrad
slots across round stages, and record `carry_across_stages` in the result. Run
a two-epoch local smoke first; launch one same-budget remote row only if model,
optimizer, checkpoint, cache, and artifact integrity all pass. Reuse the same
r5-r8 advancement gates.

### R1A-C2 Local Readiness Result

The two-epoch CPU smoke completed on 2026-07-10:

```text
run_id              = i1_present_autond_dbitnet_r1a_c2_optcarry_smoke_seed0
samples_per_class   = 128
epochs_per_round    = 2
round sequence      = [5,6,7,8] -> 9
result rows         = 1 / 1
plan alignment      = pass
checkpoint metric   = val_loss for r5-r9
optimizer transition = carry_across_stages
claim scope         = implementation readiness only
```

The optimizer audit proves that one Adam/AMSGrad instance and its step state
continue across every curriculum transition:

| Round | Session call | State reused | Step before | Step after |
| --- | ---: | --- | ---: | ---: |
| r5 | 1 | false | 0 | 8 |
| r6 | 2 | true | 8 | 16 |
| r7 | 3 | true | 16 | 24 |
| r8 | 4 | true | 24 | 32 |
| r9 | 5 | true | 32 | 40 |

The smoke metrics are readiness noise, not research evidence:

```text
r5 AUC = 0.562500000
r6 AUC = 0.498291016
r7 AUC = 0.494628906
r8 AUC = 0.557128906
r9 AUC = 0.496582031
```

Local artifacts:

```text
outputs/local_smoke/i1_present_autond_dbitnet_r1a_c2_optcarry_smoke_seed0/results.jsonl
outputs/local_smoke/i1_present_autond_dbitnet_r1a_c2_optcarry_smoke_seed0/progress.jsonl
outputs/local_smoke/i1_present_autond_dbitnet_r1a_c2_optcarry_smoke_seed0/validation.json
outputs/local_smoke/i1_present_autond_dbitnet_r1a_c2_optcarry_smoke_seed0/curves.svg
outputs/local_smoke/i1_present_autond_dbitnet_r1a_c2_optcarry_smoke_seed0/history.csv
outputs/local_cache/i1_present_autond_dbitnet_r1a_c2_optcarry_smoke_seed0/
```

Recommendation: proceed to exactly one C2 `65536/class`, seed-0 remote medium
diagnostic against C1. Do not call that run paper-scale and do not advance to
`262144/class` or a purported reproduction from its result. After C2, freeze
this strict-medium audit and write a separate public-code-aligned paper-scale
plan with independently configurable validation size and final fresh-test
evaluation.

The GPU1 remote package passed all nine readiness invariants, including plan
alignment, training-protocol consistency, AutoND protocol lock, and disk-backed
medium-scale cache requirements:

```text
configs/remote/innovation1_spn_present_autond_dbitnet_r1a_c2_optcarry_65k_seed0_gpu1_20260710.json
configs/remote/generated/run_i1_present_autond_dbitnet_r1a_c2_optcarry_65k_seed0_gpu1_20260710.cmd
configs/remote/generated/monitor_i1_present_autond_dbitnet_r1a_c2_optcarry_65k_seed0_gpu1_20260710.sh
```

The monitor requires optimizer continuity as a hard integrity gate, compares
accuracy and AUC at r5-r9 directly against C1, and recommends a separately
planned paper-scale phase rather than mechanical strict-medium scaling.

If C2 improves the lower-round same-budget metrics, retain optimizer carry as
the source-faithful implementation. If it does not improve them, discard it as
a strict-medium improvement but retain it in any public-code reproduction,
because the public code still reuses one compiled optimizer. In neither case
may a `65536/class` result validate or refute the paper-scale claim.

After C2, the paper-scale work must be planned as a separate phase rather than
as R2/R3 mechanical scaling. It must represent `10^7` total training rows,
`10^6` total validation rows, 40 epochs per round for the r9 claim, and five
fresh `10^6`-sample evaluations. A public-code-aligned track must separately
model random-ciphertext negatives and per-row keys; the current strict
encrypted-random-plaintext, held-out-key track remains benchmark-transfer
evidence. Do not merge those tracks or call either exact while paper text and
public code disagree.

Do not reopen dense DDT or extend the held E1 graph route while completing this
baseline audit.

The strict C2 diagnostic retains these interpretation gates:

```text
r5 accuracy >= 0.75
r6 accuracy >= 0.60
r7 accuracy >= 0.52
r8 accuracy > 0.505
all protocol/artifact integrity checks = pass
```

If no audited single-variable variant restores those gates, close this strict
baseline reimplementation as an unresolved protocol/scale mismatch and return
to selecting a separately justified typed-SPN candidate. Do not respond by
mechanically increasing to R2/R3, reopening dense DDT inputs, or extending the
held E1 graph route.

## Decision Boundaries

- This audit does not reopen DDT input exploration.
- This audit does not continue the corrected E1 graph architecture.
- Random-ciphertext negatives may be used only as a separately labeled
  ablation and are not part of R0 or R1.
- No r8/r9 ceiling claim is allowed from R1 or R2.
- A successful baseline audit calibrates the next typed-SPN candidate; it is
  not itself the Innovation 1 method contribution.
