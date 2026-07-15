# Innovation 1 Feistel / DES Branch-Inception Plan

**Date:** 2026-07-15

**Status:** R1 completed and rejected; no remote launch

## Research Question

Under one frozen DES-6 multiple-pair protocol, does explicitly restoring and
modeling Feistel left/right branch roles provide attributable and competitive
neural-distinguisher evidence relative to a shuffled mapping control, a
Zhang/Wang-style Inception-ResNet family baseline, and a canonical pair-set
LSTM?

This is the first Feistel cell for the opening proposal's
`cipher structure -> neural architecture preference` mapping. It is not a
claim about every Feistel cipher.

## Literature And Protocol Audit

The primary anchor is Zhang/Wang 2022/2023. Its DES Case2 uses internal
difference `(0x40080000,0x04000000)`, `m in {2,4,8,16}`, kernels `(1,4,6)`,
random keys, encrypted random-plaintext negatives, `10^7` grouped training
rows, and `10^6` grouped test rows. At six rounds and `m=16`, Table 2 reports
accuracy `0.7603`. Seven rounds requires staged training and reaches only
`0.5106`, so it is excluded from the first gate.

The local literature record is
`sources/research_feistel_neural_20260715.md`. This plan adapts the paper family
to PyTorch and the project artifact protocol; it is not an exact code or
paper-scale reproduction.

## Frozen Data Contract

```text
cipher                         = DES
rounds                         = 6
internal paper difference      = 0x4008000004000000
external project difference    = 0x0000801000004000
pairs/sample                   = 16
feature                        = raw ciphertext_pair_bits (2048 bits/sample)
sample structure               = zhang_wang_case2_official_mcnd
key sampling                   = independent random key per basic pair
negative                       = encrypted_random_plaintexts
loss                           = MSE
optimizer                      = Adam
weight decay                   = 8e-4
schedule                       = cyclic 1e-4 -> 2e-3
checkpoint                     = best validation loss restored
```

The external difference compensates for this repository's explicit DES IP.
The candidate reverses the public FP and the final swap before assigning
left/right roles. The benchmark, labels, negative definition, and key sampling
are identical across rows.

## One Variable And Controls

| Role | Model key | Purpose |
| --- | --- | --- |
| Candidate | `des_feistel_branch_inception_true` | Canonical `L/R` roles plus within-pair difference and cross-branch interaction channels |
| Attribution control | `des_feistel_branch_inception_shuffled` | Same capacity with a frozen random bit mapping before branch assignment |
| Paper-family anchor | `des_zhang_wang_inception_pairset` | Canonical raw branch channels with `(1,4,6)` Inception kernels and residual blocks |
| Sequence anchor | `des_lstm_pairset` | Bidirectional LSTM over the same canonical 32 left/right positions |

The true and shuffled candidates differ only in the fixed public topology
mapping. The paper-family anchor separates the benefit of derived Feistel
interaction channels from the established multi-scale residual CNN family.

## Execution Ladder

### R0 Readiness

```text
plan    = configs/experiment/innovation1/innovation1_feistel_des_r6_branch_inception_readiness_seed0.csv
scale   = 64/class, seed0, 2 epochs, one 128-row fresh test
device  = local CPU
purpose = geometry, random-key generation, checkpoint, progress, JSONL/SVG/gate path only
```

Readiness metrics are not research evidence.

The readiness run completed with `4/4` rows, passed result validation, and
returned `feistel_des_readiness_passed` with
`research_decision_applied=false`. Its metrics are retained only as execution
evidence, not as a model-quality result.

### R1 Local Gate

```text
plan       = configs/experiment/innovation1/innovation1_feistel_des_r6_branch_inception_2048_seed0_seed1.csv
scale      = 2048/class
seeds      = 0, 1
epochs     = 10
validation = 1024/class
fresh test = 3 x 2048/class
device     = local CPU
```

Primary score is the mean fresh-test AUC. Accuracy is reported separately for
context against the paper and must not be subtracted from AUC.

The first R1 launch attempt is invalid: two local processes with different
batch sizes appended to the same `progress.jsonl`. The contaminated partial
directory is not evidence and must not be indexed. The clean rerun must have
one owning tmux session, one command, and a fresh result directory. A
parameter-matched disk cache may be reused only after array and metadata
validation.

Advance to a remote `65536/class` two-seed diagnostic only if both seeds meet:

```text
typed_true - typed_shuffled >= 0.005 AUC
typed_true >= strongest(paper_inception, lstm) - 0.002 AUC
typed_true >= 0.55 AUC
```

If signal exists but the true mapping does not beat shuffled, retain the
strongest architecture as a Feistel baseline and redesign branch attribution
locally. If the candidate is below `0.55` on either seed, do not scale.

### Conditional Remote Gate

Only after R1 passes, generate a two-seed `65536/class`, 10-epoch remote plan
with disk-backed train/validation/fresh-test caches and launch from a pushed
commit. This is a medium diagnostic, not formal or paper-scale evidence. The
paper's Case2 corresponds to approximately `5,000,000/class` grouped rows and
must remain a separate exact-reproduction plan.

## Explicit Stops

- no DES-7 staged training before DES-6 attribution and competitiveness pass;
- no `5,000,000/class` mechanical scale-up;
- no direct comparison of project AUC with paper accuracy `0.7603`;
- no claim that DES alone establishes a general Feistel architecture rule;
- no SIMON/SIMECK related-key or SM4 benchmark mixed into this first gate.

## Evidence-Backed Next Action

If R1 passes, run the conditional medium DES confirmation. If the medium result
also passes, the next research question is cross-Feistel generalization:
retain the same architecture roles and test whether branch semantics transfer
to SIMON64 under a standard-key strict-negative protocol. If R1 fails, use its
true/shuffled/paper/LSTM ordering to redesign one branch mechanism locally and
do not increase the sample count.

## R1 Completed Result

The clean single-writer rerun completed on 2026-07-15. All eight planned rows,
ten-epoch histories, and three fresh-test repeats passed plan alignment and the
strict DES result gate.

```text
result index = outputs/00_RECENT_RESULTS.md entry 001
results      = outputs/local_diagnostic/i1_feistel_des_r6_branch_inception_2048_seed0_seed1/results.jsonl
progress     = outputs/local_diagnostic/i1_feistel_des_r6_branch_inception_2048_seed0_seed1/progress.jsonl
validation   = outputs/local_diagnostic/i1_feistel_des_r6_branch_inception_2048_seed0_seed1/validation.json
curves       = outputs/local_diagnostic/i1_feistel_des_r6_branch_inception_2048_seed0_seed1/curves.svg
history      = outputs/local_diagnostic/i1_feistel_des_r6_branch_inception_2048_seed0_seed1/history.csv
gate         = outputs/local_diagnostic/i1_feistel_des_r6_branch_inception_2048_seed0_seed1/gate.json
```

Mean fresh-test AUC:

| seed | typed true | typed shuffled | paper-family adaptation | LSTM |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.499540726 | 0.499815742 | 0.494093537 | 0.505429626 |
| 1 | 0.496986469 | 0.511284153 | 0.498417497 | 0.502533913 |

Gate deltas:

```text
seed0 true - shuffled       = -0.000275016
seed1 true - shuffled       = -0.014297684
seed0 true - strongest LSTM = -0.005888899
seed1 true - strongest LSTM = -0.005547444
```

The gate returned:

```text
status                      = pass
decision                    = feistel_branch_candidate_not_ready
topology_attributed         = false
mainstream_competitive      = false
signal_present              = false
next_action                 = stop_scale_and_redesign_locally
research_decision_applied   = true
errors                      = []
```

This is a valid negative local diagnostic: the candidate did not learn a
repeatable DES-r6 signal and was worse than the shuffled control on seed1. It
does not show that Feistel-aware neural distinguishers are impossible or that
DES has no paper-scale signal. The Zhang/Wang reference trains roughly
`5,000,000/class`, while R1 used only `2048/class`, and the R1 paper-family row
was an adapted three-block mean/max model rather than the public five-block
global-average implementation.

## Executable Next Action

Do not launch DES-r6 `65536/class`, DES-r7 staged training, SIMON, or SM4 from
this result. The next question is narrower:

```text
Can the verified public five-block/global-average DES layout learn the easier
DES-r5 task at the same 2048/class local budget?
```

Use the same input difference, strict negative definition, independent
per-pair random keys, `m=16`, seeds 0 and 1, ten epochs, and three fresh tests.
Change only the network layout to the audited official backbone. Require both
seeds to exceed a frozen DES-r5 calibration threshold before returning to
DES-r6. If the calibration fails, inspect the exact training/code discrepancy
and run at most a planned local `8192/class` calibration; do not use DES-r6
remote scale as a data-scarcity rescue.
