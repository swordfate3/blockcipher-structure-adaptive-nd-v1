# Innovation 1 Feistel / DES Official-Backbone Attribution Plan

**Date:** 2026-07-15

**Status:** readiness passed; 2048/class attribution ready to launch

## Research Question

Given that the public five-block/global-average DES layout learns DES-r5 at
`2048/class` on both seeds, does adding explicit Feistel branch interactions
improve DES-r6 under the same backbone and budget, and is any improvement
attributable to the correct branch mapping rather than model capacity?

This is a local Innovation 1 structure-attribution gate. It is not paper-scale
DES reproduction and cannot establish a general Feistel architecture rule.

## Fixed Protocol

```text
cipher             = DES
rounds             = 6
difference         = external 0x0000801000004000
pairs/sample       = 16
input              = 2048 raw ciphertext-pair bits
negative           = encrypted_random_plaintexts
key schedule       = independent random key per basic pair
sample structure   = zhang_wang_case2_official_mcnd
epochs             = 10
loss               = MSE
optimizer          = Adam
weight decay       = 8e-4
schedule           = cyclic 1e-4 -> 2e-3
checkpoint         = best val_loss restored
```

DES-r5 calibration is a prerequisite, not a result row in this matrix:

```text
seed0 fresh AUC = 0.968407075
seed1 fresh AUC = 0.964183966
decision        = feistel_des5_official_calibration_passed
```

## One Variable And Controls

| Role | Model | Purpose |
| --- | --- | --- |
| Candidate | `des_feistel_official_backbone_true` | Correct canonical branches plus four explicit within/cross-branch XOR channels |
| Attribution control | `des_feistel_official_backbone_shuffled` | Same eight channels and exactly equal capacity after a fixed random bit mapping |
| Published-layout baseline | `des_zhang_wang_official_layout` | Canonical raw `C0L,C0R,C1L,C1R` channels with the public five-block/GAP backbone |

All three use kernels `(1,4,6)` and residual kernels `(3,5,7,9,11)`. Candidate
and shuffled have `651201` parameters each; the raw baseline has `649793`.
The `1408`-parameter difference is only the extra four input channels.

## Execution

### Readiness

```text
plan       = configs/experiment/innovation1/innovation1_feistel_des_r6_official_backbone_attribution_readiness_seed0.csv
scale      = 64/class, seed0, 2 epochs
fresh test = 1 x 64/class
purpose    = three-model training/checkpoint/gate path only
```

Readiness must return
`feistel_des_official_attribution_readiness_passed` with
`research_decision_applied=false`. Its AUC values are not evidence.

Readiness completed on 2026-07-15:

```text
result rows                 = 3/3
validation errors           = []
true parameter count        = 651201
shuffled parameter count    = 651201
raw parameter count         = 649793
decision                    = feistel_des_official_attribution_readiness_passed
research_decision_applied   = false
next action                 = run_des6_official_backbone_attribution_2048
```

Artifacts are under
`outputs/local_smoke/i1_feistel_des_r6_official_backbone_attribution_readiness_seed0/`.
Readiness AUC values are not used in the research decision.

### Local Attribution Gate

```text
plan       = configs/experiment/innovation1/innovation1_feistel_des_r6_official_backbone_attribution_2048_seed0_seed1.csv
scale      = 2048/class
seeds      = 0, 1
models     = 3 per seed, 6 rows total
validation = 1024/class
fresh test = 3 x 2048/class
device     = local CPU
```

Every seed must satisfy:

```text
true - shuffled >= 0.005 AUC
true - raw       >= -0.002 AUC
true             >= 0.55 AUC
```

## Decisions

Complete pass:

```text
decision    = feistel_des6_official_branch_attribution_passed
next action = prepare DES-r6 65536/class two-seed remote diagnostic
```

Signal without topology attribution:

```text
decision    = feistel_des6_signal_without_topology_attribution
next action = retain official raw baseline and redesign branch interactions
```

No repeatable signal:

```text
decision    = feistel_des6_official_attribution_not_ready
next action = stop scale and retain only the DES-r5 mechanism calibration
```

## Explicit Stops

- no remote launch unless both seeds pass all three gates;
- no DES-r7 staged training or paper-scale mechanical increase;
- no changes to data, negative class, difference, key schedule, optimizer, or
  metric inside this architecture-only matrix;
- no SIMON/SM4 generalization from a DES-only result;
- no direct subtraction of project AUC from paper accuracy `0.7603`.

## Evidence-Backed Next Action

Run readiness locally, then continue automatically to the six-row local gate
only if readiness is plan-aligned. Generate JSONL, progress, validation, Chinese
SVG, history CSV, strict gate, and recent-result index. Use the complete
two-seed gate to decide whether a medium remote diagnostic is justified.
