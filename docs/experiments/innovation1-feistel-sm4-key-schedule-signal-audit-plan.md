# Innovation 1 Feistel / SM4 Key-Schedule Signal Audit Plan

**Date:** 2026-07-15

**Status:** completed; r3 calibrated, r5 baseline/scale gap identified

## Research Question

Why did all three SM4-r5 models remain near chance at `2048/class` while the
Yu/Wu/Zhang paper reports `0.999` r5 accuracy at one million total training
rows?

The completed recurrence attribution matrix showed real optimization but no
validation generalization under `key_rotation_interval=1`. Before changing the
candidate or increasing data, this audit separates three explanations:

1. the local SM4/data path cannot learn even an easier three-round task;
2. the paper-family model learns r5 only when one fixed key is shared across
   train, validation, and final test;
3. the strict rotating-key r5 task needs substantially more data than this
   local budget.

## One Model, Two Controlled Axes

All four rows use the unchanged `multiscale_dense_resnet`, identical raw
256-bit ciphertext-pair input, optimizer, loss, sample counts, and strict
encrypted-random-plaintext negatives:

| Role | Rounds | Key schedule | Purpose |
| --- | ---: | --- | --- |
| `r3_fixed` | 3 | one fixed key for all splits | easiest paper-protocol calibration |
| `r3_rotating` | 3 | a deterministic random key per row | strict cross-key positive control |
| `r5_fixed` | 5 | one fixed key for all splits | detect paper-protocol key dependence |
| `r5_rotating` | 5 | a deterministic random key per row | reproduce the failed strict anchor |

This is a planned protocol attribution study, not an architecture comparison.
The two axes are required to distinguish a round-depth problem from a
key-generalization problem; no feature, label, negative, metric, or network
parameter changes between rows.

## Frozen Protocol

```text
cipher                 = SM4
difference             = (0, 0, 0, 1)
pairs/sample           = 1
input                   = 256 raw ciphertext-pair bits
negative                = encrypted_random_plaintexts
model                   = multiscale_dense_resnet, 32 channels, 3 blocks
train                   = 2048/class
validation              = 1024/class
fresh final test        = 3 x 2048/class
seed                    = 0
epochs                  = 10
batch size              = 128
loss                    = MSE
optimizer               = Adam, fixed 1e-4
checkpoint              = best val_auc restored
fixed key               = 0x0123456789abcdeffedcba9876543210
rotating key interval   = 1 row
```

The fixed-key rows intentionally use the same key in every split because that
is the narrow protocol dependency being tested. They are calibration evidence,
not cross-key security evidence.

## Frozen Gate

A row has local signal when its three-repeat fresh-test mean AUC is at least
`0.55`. The gate maps the four booleans to an executable decision:

```text
r3 fixed + rotating pass, r5 fixed pass, r5 rotating fails
  -> fixed-key protocol dependency / r5 cross-key generalization gap
  -> next: fixed-key candidate/shuffled/baseline attribution, explicitly
           limited to paper-style fixed-key evidence

r3 fixed + rotating pass, both r5 rows fail
  -> r5 scale or paper-architecture mismatch
  -> next: closer paper baseline port before any candidate scale

r3 fixed passes but r3 rotating fails
  -> key-generalization failure already exists at low rounds
  -> next: do not scale r5; audit key-conditioned representations

r3 fixed fails
  -> local cipher/data/model calibration failure
  -> next: inspect paper input layout and dataset semantics

r5 rotating passes
  -> strict signal is unstable rather than absent
  -> next: seed1 confirmation before architecture attribution
```

Readiness is inherited from the completed three-model SM4 pipeline and the
unchanged baseline, so this audit does not consume a separate smoke matrix.

## Explicit Stops

- no recurrence-candidate tuning during this protocol audit;
- no random-ciphertext negative substitution;
- no r6-r8 sweep, remote launch, or million-row claim;
- no interpreting fixed-key signal as cross-key generalization;
- no mechanical 8192/65536 scale unless this audit identifies a specific
  data-scarcity exception.

## Evidence-Backed Next Action

Run the four-row seed0 matrix locally, validate all rows, generate curves and a
protocol gate, refresh the result index, and use the gate's exact branch to
freeze the next experiment. Do not infer the cause from the failed r5 matrix
alone.

## Completed Result

The four-row audit completed on 2026-07-15 with exact plan alignment, identical
`59809`-parameter models, and no validation errors:

| Rounds | Fixed-key fresh AUC | Rotating-key fresh AUC |
| ---: | ---: | ---: |
| 3 | 0.999994040 | 0.999999523 |
| 5 | 0.497069558 | 0.491743843 |

The gate returned:

```text
r3_fixed signal      = true
r3_rotating signal   = true
r5_fixed signal      = false
r5_rotating signal   = false
decision              = feistel_sm4_r5_scale_or_paper_architecture_gap
next_action           = port_closer_yu2023_baseline_before_candidate_scale
```

This rules out a broken SM4 primitive, difference, negative generator, and
low-round cross-key pipeline: the same implementation is almost perfectly
learnable at r3 under both key schedules. It also rules out fixed-key dependence
as the immediate explanation for the failed r5 cell. The unresolved difference
is now r5 data scale and/or the existing baseline's global pooling, which omits
the paper-described flattened two-layer 64-unit classifier.

Artifacts are under
`outputs/local_diagnostic/i1_feistel_sm4_key_schedule_signal_audit_2048_seed0/`
and the result is entry `001` in `outputs/00_RECENT_RESULTS.md`.

The next frozen experiment is
`docs/experiments/innovation1-feistel-sm4-position-preserving-paper-family-calibration-plan.md`.
It ports the closest defensible paper-family baseline before any candidate
tuning or scale increase.
