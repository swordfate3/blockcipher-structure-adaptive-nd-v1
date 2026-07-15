# Innovation 1 Feistel / DES-r5 Official-Raw Mapping Attribution Plan

**Date:** 2026-07-15

**Status:** 2048/class completed; canonical DES mapping attributed

## Research Question

Does the strong Zhang/Wang four-channel DES-r5 layout depend on the correct
canonical DES state mapping, or can an identical backbone learn just as well
after a fixed random 64-bit mapping?

The preceding eight-channel experiment established correct mapping over its
equal-capacity shuffled control, but rejected the added XOR channels because
seed1 underperformed the official raw baseline. This experiment removes those
channels and attributes the strongest retained model itself.

## One Variable And Controls

| Role | Model | Mapping | Parameters |
| --- | --- | --- | ---: |
| Candidate/anchor | `des_zhang_wang_official_layout` | DES preoutput restored to canonical `L_r || R_r` | 649793 |
| Attribution control | `des_zhang_wang_official_layout_shuffled` | Fixed random permutation of the same 64 ciphertext bits | 649793 |

Both models use exactly four input channels, initial kernels `(1,4,6)`, five
residual kernels `(3,5,7,9,11)`, global average pooling, and one output. The
mapping is the only changed variable.

## Frozen Protocol

```text
cipher             = DES
rounds             = 5
difference         = external 0x0000801000004000
pairs/sample       = 16 independent basic pairs
input              = 2048 raw ciphertext-pair bits
negative           = encrypted_random_plaintexts
key schedule       = independent random key per basic pair
sample structure   = zhang_wang_case2_official_mcnd
train              = 2048/class
validation         = 1024/class
fresh test         = 3 x 2048/class
seeds              = 0, 1
epochs             = 10
loss               = MSE
optimizer          = Adam
weight decay       = 8e-4
schedule           = cyclic 1e-4 -> 2e-3
checkpoint         = best val_loss restored
```

This remains a local scaled diagnostic. Zhang/Wang Case2 used about ten
million grouped training rows and reported accuracy rather than project AUC.

## Readiness

```text
plan       = configs/experiment/innovation1/innovation1_feistel_des_r5_official_raw_mapping_attribution_readiness_seed0.csv
scale      = 64/class, seed0, 2 epochs
fresh test = 1 x 64/class
```

Readiness passes only with two aligned rows, identical parameter counts,
per-pair random-key metadata, checkpoints, and
`research_decision_applied=false`. Readiness AUC is not research evidence.

Readiness completed on 2026-07-15:

```text
result rows                 = 2/2
validation errors           = []
canonical parameters        = 649793
shuffled parameters         = 649793
decision                    = feistel_des_official_raw_mapping_readiness_passed
research_decision_applied   = false
next action                 = run_des5_official_raw_mapping_attribution_2048
recent result index         = 001
```

Artifacts are under
`outputs/local_smoke/i1_feistel_des_r5_official_raw_mapping_attribution_readiness_seed0/`.

## Frozen Gate

Both seeds must independently satisfy:

```text
canonical - shuffled >= +0.005 AUC
canonical            >= 0.90 AUC
```

Complete pass:

```text
decision    = feistel_des5_official_raw_mapping_attributed
next action = retain DES canonical mapping as the first strong Feistel
              representation cell, then design the four-word SM4 recurrence gate
```

If absolute signal remains strong but the margin fails, retain the official
DES model only as a performance baseline without a mapping-attribution claim.
Any plan, capacity, key, cache, or alignment failure invalidates the run.

## Explicit Stops

- no extra XOR channels, DDT features, or other model changes;
- no DES-r6 scale-up or DES-r7 staged training;
- no mechanical DES-r5 sample increase;
- no SM4 launch until this gate is completed;
- no cross-Feistel claim from a DES-only result.

## Completed Result

The four-row run completed on 2026-07-15 with `4/4` aligned rows, identical
capacity, and no validation errors:

| Seed | Canonical raw | Shuffled raw | Canonical - shuffled |
| ---: | ---: | ---: | ---: |
| 0 | 0.968407075 | 0.922529936 | +0.045877139 |
| 1 | 0.964183966 | 0.920331796 | +0.043852170 |

The strict gate returned:

```text
signal_present       = true
topology_attributed  = true
decision             = feistel_des5_official_raw_mapping_attributed
next_action          = retain_des_mapping_cell_and_design_sm4_recurrence_gate
```

Both models have `649793` parameters. The canonical model exactly reproduced
the prior calibration AUC on both seeds, while shuffled mapping lost more than
`0.0438` AUC on each seed. This retains the canonical DES state layout as the
first strong local Feistel representation-attribution cell. It does not show
paper-scale performance or establish a rule for generalized Feistel ciphers.

Artifacts are under
`outputs/local_diagnostic/i1_feistel_des_r5_official_raw_mapping_attribution_2048_seed0_seed1/`
and the completed result is entry `001` in `outputs/00_RECENT_RESULTS.md`.

## Evidence-Backed Next Action

Retain the DES cell and move to a separate SM4 mechanism gate. Do not reuse the
DES two-half mapping: SM4 updates four ordered 32-bit words through
`X[i+4] = X[i] xor T(X[i+1] xor X[i+2] xor X[i+3] xor rk[i])`.

The next frozen plan should compare a four-word recurrence-aware candidate,
an exactly equal-capacity shuffled-word/bit control, and a Yu/Wu/Zhang-family
Conv-ResNet baseline on the same SM4-r5 single-pair task. Start with local
readiness and a small two-seed attribution gate before any large run. Keep the
2023 paper's `1,000,000` total train / `100,000` total test / 25 epoch result as
an external scale reference, not the first implementation budget.
