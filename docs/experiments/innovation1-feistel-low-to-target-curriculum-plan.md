# Innovation 1 Balanced-Feistel Low-to-Target Curriculum Plan

**Status:** seed0 diagnostic complete; SIMECK-only seed1 confirmation frozen

## Research Question

Can the corrected, two-seed easier-round relation signal transfer to the Lu et
al. target rounds under an equal-total-epoch comparison?

This experiment does not change the cipher benchmark, input difference, labels,
negative construction, key sampling, eight-pair grouping, relation network, or
fresh-test metric. It changes only the order of training data seen by the same
capacity model:

```text
easier round -> target round
```

## Evidence Anchor

The corrected independent-key `8192/class`, 10-epoch evidence is:

| Cipher | easier round true | easier round shuffled | target round true | target round shuffled |
|---|---:|---:|---:|---:|
| SIMON64/128 | r11 `0.658869159` | `0.500257527` | r12 `0.505085583` | `0.500342575` |
| SIMECK64/128 | r14 `0.860501059` | `0.502064635` | r15 `0.511121858` | `0.500361473` |

At easier-round epoch 5, validation AUC was already approximately `0.666` for
SIMON r11 and `0.783` for SIMECK r14. A fixed `5+5` split therefore gives the
pretraining stage a demonstrated non-random signal while retaining half of the
budget for target-round adaptation. No split sweep is authorized.

Literature and implementation references remain frozen to:

```text
Lu et al., The Computer Journal, DOI 10.1093/comjnl/bxac195
author repository commit 602c664e649a4e3e8e56dc1961efb67400f5c7fb
docs/research/innovation1-feistel-author-layout-parity-audit-20260716.md
docs/research/innovation1-feistel-paper-to-code-map-20260716.md
```

## Frozen Protocol

Common to every row:

```text
ciphers                 = SIMON64/128 and SIMECK64/128
target rounds           = SIMON r12, SIMECK r15
pretrain rounds         = SIMON r11, SIMECK r14 when enabled
input difference        = (0x00000000, 0x00000040)
pairs per sample        = 8
raw input               = 1024 ciphertext bits/sample
negative mode           = encrypted random plaintexts
key sampling            = one independent random 128-bit key/sample
key row indexing        = global_dataset_row
dataset label mode      = balanced_per_class
sample structure        = independent_pairs
seed                    = 0
loss / optimizer        = MSE / Adam
learning rate / L2      = 1e-4 / 1e-5
optimizer transition    = reset at low-to-target boundary
checkpoint              = best val_loss restored in each stage
batch / hidden bits     = 128 / 32 for diagnostic
fresh metric            = mean AUC over three independent repeats
```

The optimizer reset is part of the frozen transfer protocol. It keeps target
adaptation independent of low-round Adam moments while transferring the best
validated model weights. Both curriculum roles use the same transition.

## Three Roles Per Cipher

| Role | Representation | Training schedule | Total epochs |
|---|---|---|---:|
| curriculum true | cipher-correct relation | 5 easier + 5 target | 10 |
| curriculum shuffled | branch-swapped equal-capacity control | 5 easier + 5 target | 10 |
| target scratch | same cipher-correct architecture | 10 target only | 10 |

The target-scratch model key is an explicit registry alias of the true-relation
architecture. Its parameter count must exactly match curriculum true. It exists
only so plan/result rows remain uniquely auditable.

## Readiness

Plan:

```text
configs/experiment/innovation1/innovation1_feistel_low_to_target_curriculum_readiness_seed0.csv
```

Budget:

```text
samples_per_class       = 64
validation total        = 128
fresh test total        = 128 x 1 repeat
curriculum epochs       = 1 easier + 1 target
scratch epochs          = 2 target
batch                   = 64
purpose                 = mechanics only
```

Readiness passes only if all six rows are plan-aligned, all stages complete,
both curriculum rows per cipher record the correct lower round and `1+1`
schedule, scratch records no pretraining and two target epochs, total epochs
match, candidate/control capacities match, rotating-key metadata is corrected,
and all fresh metrics are finite.

Readiness AUC is not research evidence. A passing readiness automatically
authorizes the frozen local diagnostic below.

## Local Diagnostic

Plan:

```text
configs/experiment/innovation1/innovation1_feistel_low_to_target_curriculum_8192_seed0.csv
```

Budget per row:

```text
samples_per_class       = 8192 (16384 total train rows/stage)
validation total        = 16384 (8192/class)
fresh test total        = 32768 (16384/class) x 3 repeats
curriculum epochs       = 5 easier + 5 target
scratch epochs          = 10 target
total epochs            = 10
device                  = local CPU
```

Per-cipher advance gates use fresh-test AUC:

```text
curriculum true AUC                         >= 0.55
curriculum true - curriculum shuffled       >= +0.02
curriculum true - equal-epoch target scratch >= +0.01
```

Decisions:

| Result | Decision and next action |
|---|---|
| both ciphers pass all gates | run the identical six-row seed1 confirmation before any scale proposal |
| one cipher passes | retain only that cipher-conditional curriculum route and confirm it with seed1 |
| signal passes but attribution or scratch gain fails | reject curriculum as the cause; retain the simpler target-only result |
| neither cipher reaches signal | stop this transfer route and return to representation/difference redesign |

## Stopped Actions

Until the local diagnostic passes:

- no remote launch;
- no `16k/32k/65k` mechanical sample ladder;
- no extra curriculum split, learning-rate, epoch, or model sweep;
- no related-key, RX, polytopic, or alternate negative protocol;
- no paper-scale, exact-reproduction, SOTA, or breakthrough claim.

## Required Artifacts

Every completed readiness or diagnostic run must produce:

```text
results.jsonl
progress.jsonl
validation.json
gate.json
curves.svg
history.csv
```

The run must then be added to `outputs/00_RECENT_RESULTS.md`. The completed
diagnostic section of this document must record exact AUCs, all three gates,
the route decision, claim scope, and one executable recommended next action.

## Completed Seed0 Diagnostic

All six rows completed, passed plan alignment, used corrected global rotating-
key row indexing in target and pretraining datasets, and matched at `32225`
parameters per role. Fresh-test AUC over three `32768`-row repeats was:

| Cipher | curriculum true | curriculum shuffled | target scratch | true-shuffled | true-scratch |
|---|---:|---:|---:|---:|---:|
| SIMON r11->r12 | `0.534241434` | `0.498029904` | `0.505085583` | `+0.036211530` | `+0.029155851` |
| SIMECK r14->r15 | `0.696612916` | `0.503923842` | `0.511121858` | `+0.192689074` | `+0.185491058` |

SIMON passes both control margins but misses the frozen absolute signal gate
because `0.534241434 < 0.55`. SIMECK passes all three gates. Decision:

```text
feistel_curriculum_cipher_conditional
passing_ciphers = [simeck64]
remote_scale = no
paper_scale_or_SOTA_claim = no
```

This is single-seed local `8192/class` evidence. It supports a SIMECK-
conditional curriculum hypothesis, not a two-cipher Feistel conclusion.

## Frozen SIMECK Seed1 Confirmation

The only authorized next training is:

```text
configs/experiment/innovation1/innovation1_feistel_low_to_target_curriculum_8192_seed1_simeck.csv
```

It repeats exactly the three SIMECK roles at seed1 with `8192/class`, `5+5`
versus `10`, the same data protocol, model capacity, optimizer reset,
checkpoint rule, and three fresh-test repeats. It does not include SIMON and
does not change the curriculum split.

Seed1 passes only if:

```text
curriculum true AUC                          >= 0.55
curriculum true - curriculum shuffled        >= +0.02
curriculum true - equal-epoch target scratch >= +0.01
```

If seed1 passes, synthesize the two-seed SIMECK evidence and audit the exact
remaining author-protocol/data-scale gap before proposing one remote scale
step. If it fails, stop the curriculum route; do not rescue it with extra
splits, epochs, seeds, or intermediate sample sizes.

## Completed SIMECK Seed1 Confirmation

All three rows completed and passed the same schedule, cache, capacity, and
plan-alignment checks:

| Role | seed1 fresh AUC |
|---|---:|
| r14->r15 curriculum true | `0.701582395` |
| r14->r15 curriculum shuffled | `0.504952087` |
| r15 equal-epoch scratch | `0.593733866` |

Margins:

```text
true - shuffled = +0.196630308
true - scratch  = +0.107848529
```

Decision:

```text
feistel_curriculum_seed1_confirmation_pass
```

The SIMECK route is now confirmed at `8192/class` on two independent seeds:

| Seed | true | shuffled | scratch | true-shuffled | true-scratch |
|---:|---:|---:|---:|---:|---:|
| 0 | `0.696612916` | `0.503923842` | `0.511121858` | `+0.192689074` | `+0.185491058` |
| 1 | `0.701582395` | `0.504952087` | `0.593733866` | `+0.196630308` | `+0.107848529` |

This supports a stable SIMECK-conditional curriculum effect under the project
protocol. The substantial seed-to-seed scratch variation reinforces why the
same-seed equal-epoch control is required.

The exact remaining gap is large: Lu et al. use `2e7` total training samples,
`2e6` test samples, 120 epochs, batch 30000, cyclic learning rate, SE-ResNet,
and five repeated runs. Project results use `16384` total training rows/stage,
10 total epochs, pair-pool architecture, strict negatives, and AUC. Therefore
paper accuracy and project AUC remain non-comparable.

The evidence-backed next action is one SIMECK-only `65536/class` remote medium
scale probe with the same three roles and no other protocol change:

```text
docs/experiments/innovation1-feistel-simeck-curriculum-65k-scale-plan.md
```

It is not paper scale. It must pass the frozen scale gate before a same-scale
seed1 run is considered.
