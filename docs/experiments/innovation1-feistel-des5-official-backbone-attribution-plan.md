# Innovation 1 Feistel / DES-r5 Strong-Signal Attribution Plan

**Date:** 2026-07-15

**Status:** readiness passed; 2048/class attribution ready

## Research Question

On the already calibrated and strongly learnable DES-r5 task, do explicit
Feistel left/right branch interactions outperform an equal-capacity shuffled
mapping while remaining competitive with the public Zhang/Wang raw layout?

This isolates the structure-attribution question from the weak absolute signal
seen at DES-r6. It is a local `2048/class` diagnostic, not paper-scale DES
reproduction and not a general Feistel architecture rule.

## Literature And Existing Evidence

The primary anchor is Zhang and Wang, *Improving Differential-Neural
Distinguisher Model For DES, Chaskey and PRESENT* (arXiv `2204.06341`) and its
audited public `deep_net_des.py`. The project has already verified:

```text
DES-r5 official raw fresh AUC = 0.968407075 seed0, 0.964183966 seed1
DES-r6 true-shuffled margin  = +0.011802832 seed0, +0.019061764 seed1
DES-r6 true absolute AUC     = 0.524183512 seed0, 0.527223508 seed1
```

DES-r5 therefore supplies abundant learnable signal, while DES-r6 shows that
the branch mapping direction is plausible but too weak to justify scale.

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

The experiment changes only the input branch semantics. Data, difference,
negative class, key sampling, backbone, optimizer, and evaluation remain fixed.

## Lean Three-Role Matrix

| Role | Model | Purpose |
| --- | --- | --- |
| Candidate | `des_feistel_official_backbone_true` | Canonical `L/R` branches plus four explicit within/cross-branch XOR channels |
| Attribution control | `des_feistel_official_backbone_shuffled` | Fixed random bit mapping, identical eight channels and exactly equal capacity |
| Same-budget anchor | `des_zhang_wang_official_layout` | Public four-channel raw layout on the same five-block/GAP backbone |

True and shuffled have `651201` parameters each; raw has `649793`. The small
raw difference is entirely due to the candidate's four additional input
channels, so competitiveness against raw is a separate required gate.

## Readiness Gate

```text
plan       = configs/experiment/innovation1/innovation1_feistel_des_r5_official_backbone_attribution_readiness_seed0.csv
scale      = 64/class, seed0, 2 epochs
fresh test = 1 x 64/class
```

Readiness must produce three aligned rows, equal true/shuffled capacity,
per-pair random-key metadata, checkpoints, progress, and
`feistel_des_official_attribution_readiness_passed`. Its AUC values are not
research evidence and `research_decision_applied` must remain false.

Readiness completed on 2026-07-15:

```text
result rows                 = 3/3
validation errors           = []
true parameters             = 651201
shuffled parameters         = 651201
official raw parameters     = 649793
decision                    = feistel_des_official_attribution_readiness_passed
research_decision_applied   = false
next action                 = run_des5_official_backbone_attribution_2048
recent result index         = 001
```

Artifacts are under
`outputs/local_smoke/i1_feistel_des_r5_official_backbone_attribution_readiness_seed0/`.
The readiness AUC values are intentionally excluded from the research decision.

## Research Gate

Both seeds must independently satisfy:

```text
true - shuffled >= +0.005 AUC
true - raw      >= -0.002 AUC
true            >= 0.90 AUC
```

Complete pass:

```text
decision    = feistel_des5_official_branch_attribution_passed
next action = retain the first strong Feistel attribution cell and design a
              separate SM4/generalized-Feistel gate
```

Signal without topology attribution keeps the official raw model and rejects
the current explicit branch interaction. Topology without raw competitiveness
also rejects the extra branch channels. Any plan, cache, key-sampling, capacity,
or result-alignment failure invalidates the run rather than counting as a
negative research result.

## Explicit Stops

- no DES-r6 remote launch or mechanical DES-r5 scale-up from this plan;
- no DDT/trail-value input and no benchmark redesign;
- no DES-r7 staged training;
- no claim that a DES-only result establishes a Feistel-wide rule;
- no direct subtraction of project AUC from paper accuracy.

## Evidence-Backed Next Action

Run the readiness matrix locally. If it is plan-aligned, run the six-row
`2048/class` matrix as the unique writer for a new result directory, generate
JSONL/progress/validation/SVG/history/gate artifacts, refresh the numbered
result index, and apply the frozen three-part gate. The observed gate, not the
large DES-r5 raw AUC alone, decides whether explicit Feistel branch semantics
are retained for generalization.
