# Innovation 1 Runtime-SPN A8 Dialga Whole-Cipher Holdout Plan

Date: 2026-07-26

```text
status = completed / hold
execution = local CPU readiness then 2048/class/source diagnostic
remote_scale = no
decision = innovation1_runtime_spn_dialga_holdout_not_supported
```

## Research Question

Can one cipher-name-free base Runtime-E4 state trained without any Dialga rows
use externally supplied cell, S-box and exact GF(2) structure to distinguish
Dialga prefix-r4, while beating same-checkpoint structural counterfactuals and
an independently trained no-topology anchor on both seeds?

A7 selected Dialga because its target-trained Runtime-E4 oracle reaches
`0.958417/0.958679`, its same-checkpoint topology margins are replicated, all
16 target atomic role-to-role GF(2) types are present in the source panel, and
it has not served as a previous whole-cipher holdout.

## One Experimental Variable

Train two parameter-matched models from the same initial state per seed:

```text
candidate = base Runtime-E4 with exact inverse-GF(2) relation_mode=true
anchor    = the same base Runtime-E4 with relation_mode=independent
```

Both roles receive exactly the same source examples, task order, batch order,
optimizer, loss, epochs and checkpoint metric. The candidate does not add the
completed unsupported typed-relation residual, relation-mass pooling, Adapter,
FiLM, MoE, cipher ID, target head or task-specific trainable state.

## Frozen Whole-Cipher Split

```text
sources = GIFT-64 r6, SKINNY-64/64 r7,
          RECTANGLE-80 r6, uKNIT prefix-r5
holdout = Dialga-128 prefix-r4
Dialga training rows = 0
Dialga checkpoint-selection rows = 0
```

The uKNIT source supplies all 16 atomic GF(2) role-pair types needed by Dialga.
Dialga's exact S-box truth table does not occur in any source cipher, so S-box
transfer remains genuinely unseen rather than a table lookup.

## Frozen Model And Budget

```text
model = base Runtime-E4, 442466 parameters for both roles
runtime descriptors = external cell membership, bit role, S-box truth table,
                      two-round exact GF(2) inverse-linear window
train/validation/target = 2048/1024/1024 per class
pairs/sample = 4 independent ciphertext pairs
negative = encrypted random plaintexts
seeds = 0, 1
epochs = 10
batch = 256
optimizer = A3 representation-L2 equalization + fixed-order PCGrad
loss = MSE
checkpoint = four-source validation macro AUC
device = local CPU diagnostic
```

This is a sub-medium local mechanism diagnostic, not formal scale,
universality, attack, SOTA or breakthrough evidence.

## Target Controls

For each seed, evaluate only after both source roles finish:

| Evaluation | Checkpoint | Dialga structure | Relation mode | A8 target steps |
| --- | --- | --- | --- | ---: |
| candidate correct | correct-trained | exact | true | 0 |
| corrupted topology | same candidate | deterministic wrong GF(2) | true | 0 |
| no topology | same candidate | exact descriptor | independent | 0 |
| wrong S-box | same candidate | GIFT table broadcast over Dialga cells | true | 0 |
| trained no-topology anchor | independent-trained | exact descriptor | independent | 0 |

The first four rows must share one exact checkpoint SHA per seed. The wrong
S-box counterfactual changes only `sbox_truth_bits`; Dialga cell membership,
bit roles, forward/inverse linear matrices and runtime window remain exact.
The fifth row is an independently trained same-budget architecture anchor.

Completed Dialga D1 correct AUC is reported as a target-trained oracle upper
reference. It is not an A8 row, does not affect checkpoint selection and is
not cross-cipher evidence.

## Readiness Gate

Before training require all of:

1. A7 is hash-valid, protocol-valid and selected Dialga;
2. the exact source panel excludes Dialga and contains four ciphers;
3. the candidate and anchor have the same `442466` parameters, state keys and
   bit-exact initial state per seed;
4. typed relation, Adapter and FiLM modes are `none`, while relation activity
   pooling remains the base `uniform` behavior;
5. all 16 Dialga atomic GF(2) relation types are covered by sources and the
   exact target S-box overlap is zero;
6. correct, corrupted, no-topology and wrong-S-box logits differ under one
   state, and correct/wrong-S-box paths preserve cell relabeling within `1e-6`;
7. source train/validation and Dialga validation disk caches are complete,
   while Dialga train cache is never referenced;
8. a synthetic source-only smoke completes both roles before target loading,
   with zero target optimizer steps;
9. outputs are finite and relevant Runtime-E4/A6/A7 tests remain green.

Any failure stops A8 and permits only repair of the failed invariant.

## Frozen Advance Gate

For both seeds require:

```text
Dialga correct AUC >= 0.55
correct - corrupted/no-topology >= +0.005
correct - wrong-S-box >= +0.005
correct - independently-trained no-topology anchor >= +0.005
candidate four-source macro >= no-topology anchor macro - 0.005
conflict projections for both training roles >= 1
all initialization, checkpoint, cache and zero-target-step checks pass
```

A full two-seed pass supports a second independent unseen-cipher result for
the base Runtime-E4 exact-structure method. It still does not establish
universal SPN adaptation or formal-scale performance.

If protocol validation passes but any functional target gate fails, classify
the second-holdout transfer as unsupported at this budget and stop. Do not
increase samples/epochs, launch remote scale, add target supervision or reopen
the closed typed-relation, relation-mass, Adapter, FiLM or MoE branches.

## Required Artifacts

Readiness and diagnostic runs must each produce JSONL results, progress,
validation, gate and summary artifacts. The diagnostic also produces history,
gradient-scale CSV and a Chinese SVG comparing the unseen Dialga controls and
source retention. The final SVG must pass rendered-pixel `visual-qa-redraw`.
Every completed run refreshes the recent-results index in the same turn.

## Completed Readiness

The zero-training readiness run completed at:

```text
outputs/local_readiness/i1_runtime_spn_dialga_holdout_a8_readiness_20260726/
status = pass
decision = innovation1_runtime_spn_dialga_holdout_readiness_passed
checks = 17/17
parameter count = 442466 for both roles
required cache files = 54/54
Dialga train cache referenced = false
training performed = false
```

The source panel covers all `16/16` Dialga atomic GF(2) relation types and has
zero exact overlap with the Dialga S-box. Correct, corrupted, no-topology and
wrong-S-box paths produced distinct logits, while correct and wrong-S-box cell
relabeling errors stayed below `1.8e-7`. Readiness therefore authorized the
frozen diagnostic; it did not supply an AUC result.

## Completed Diagnostic

The two-seed local CPU run completed at:

```text
outputs/local_diagnostic/i1_runtime_spn_dialga_holdout_a8_2048_seed0_seed1_20260726/
result rows = 26
history rows = 40 plus header
checkpoints = 4
protocol validation = pass, 17/17 checks
Dialga training rows = 0
Dialga optimizer steps = 0
```

The same shared candidate weights per seed produced the following five-cipher
validation results. The first four ciphers participated in source training;
Dialga was evaluated only after both source roles finished.

| Cipher and round | seed0 candidate AUC | seed1 candidate AUC | seed0/seed1 best accuracy | Role |
| --- | ---: | ---: | ---: | --- |
| GIFT-64 r6 | `0.538085` | `0.511875` | `0.537109 / 0.523438` | source |
| SKINNY-64/64 r7 | `0.516874` | `0.485603` | `0.524414 / 0.507812` | source |
| RECTANGLE-80 r6 | `0.725778` | `0.683919` | `0.673828 / 0.644531` | source |
| uKNIT-BC prefix-r5 | `0.520689` | `0.510944` | `0.523438 / 0.513672` | source |
| Dialga-128 prefix-r4 | `0.834955` | `0.848254` | `0.770996 / 0.773438` | zero-training holdout |

The held-out Dialga controls were:

| Seed | Correct | Corrupted GF(2) | No topology | Wrong S-box | Trained no-topology |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `0.834955` | `0.734476` | `0.508750` | `0.859363` | `0.520172` |
| 1 | `0.848254` | `0.785574` | `0.524268` | `0.848068` | `0.528065` |

Correct topology exceeded corrupted GF(2) by `+0.100479/+0.062680`, same-
checkpoint no topology by `+0.326205/+0.323986`, and the independently trained
no-topology anchor by `+0.314784/+0.320189`. Both source-macro retention checks
also passed: candidate versus no-topology source macro was
`0.575357 vs 0.540966` for seed0 and `0.548085 vs 0.532963` for seed1.

The frozen wrong-S-box gate failed on both seeds. Correct minus wrong-S-box was
`-0.024407` for seed0 and only `+0.000186` for seed1, below the required
`+0.005`. The model therefore transfers a useful GF(2)-topology-dependent
signal to unseen Dialga, but this experiment does not show that it correctly
interprets the unseen Dialga S-box primitive. High absolute Dialga AUC cannot
override that failed attribution control.

```text
status = hold
decision = innovation1_runtime_spn_dialga_holdout_not_supported
claim = zero-training Dialga GF(2) topology signal supported; composable unseen
        S-box primitive learning unsupported at this budget
remote_scale = no
```

The final SVG passed `visual-qa-redraw` after rendered-pixel inspection at
`1600x1000` and `1280x800`. The title now states the specific split verdict:
the GF(2) topology gate passed while the S-box primitive gate failed.

## Evidence-Backed Next Action

Stop A8 training and do not rescue it with more samples, epochs, target rows,
remote GPU time, target heads, typed relations, relation-mass pooling,
Adapter, FiLM or MoE. Consolidate the supported per-cipher and whole-cipher
evidence around the narrower method boundary:

```text
supported = one shared parameter geometry; runtime cell and exact GF(2)
            descriptors; replicated SKINNY attribution; RECTANGLE holdout;
            Dialga zero-training GF(2) sensitivity
unsupported = stable uKNIT transfer; unseen S-box primitive attribution;
              universal composable-SPN claim
```

Before any new training branch, perform a frozen-checkpoint S-box
identifiability synthesis across the completed five-cipher artifacts. The
exact mismatch to resolve is whether Runtime-E4's S-box descriptor path is
functionally ignored/aliased or whether the specific GIFT-to-Dialga control is
an unusually favorable alternative. That audit must use existing checkpoints
and same-topology counterfactuals only. It may justify a new method-level S-box
primitive hypothesis, but it may not reopen A8 scale-up or relax this gate.
