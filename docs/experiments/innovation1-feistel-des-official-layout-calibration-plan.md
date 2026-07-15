# Innovation 1 Feistel / DES Official-Layout Calibration Plan

**Date:** 2026-07-15

**Status:** 2048/class calibration completed and passed

## Why This Calibration Exists

R1 produced a valid but all-chance DES-r6 result. It cannot distinguish a weak
branch hypothesis from a training/backbone mismatch because its paper-family
row used three residual blocks, mean/max pair pooling, and an MLP head. The
public Zhang/Wang DES code instead uses five residual blocks followed by one
global average pooling operation and a sigmoid output.

Before redesigning Feistel interactions or increasing DES-r6 samples, this
calibration asks whether the audited public layout can learn the easier DES-r5
task in the same project runner.

## Paper-To-Code Contract

```text
cipher                         = DES
rounds                         = 5
external project difference    = 0x0000801000004000
internal paper difference      = (0x40080000, 0x04000000)
pairs/sample                   = 16
input after canonicalization   = (m=16, 32 positions, 4 channels)
initial kernels                = (1, 4, 6), 32 filters each
residual kernels               = (3, 5, 7, 9, 11)
aggregation                    = global mean over pair and position axes
model                          = des_zhang_wang_official_layout
negative                       = encrypted_random_plaintexts
key sampling                   = independent random key per basic pair
```

The model and data organization are mechanism-aligned with the public code.
The experiment remains a scaled calibration, not exact reproduction:

- `2048/class` rather than about `5,000,000/class` grouped training rows;
- ten epochs rather than the paper's twenty;
- batch size 256 rather than the public multi-GPU batch;
- L2 `8e-4` follows the paper DES text, while the public function actually
  passes `1e-5`; this discrepancy is recorded in the literature audit.

## Frozen Execution

### Readiness

```text
plan       = configs/experiment/innovation1/innovation1_feistel_des_r5_official_layout_readiness_seed0.csv
scale      = 64/class, seed0, 2 epochs
fresh test = 1 x 64/class
purpose    = model geometry, training, checkpoint, JSONL, gate path only
```

Readiness metrics are not research evidence. It must return
`feistel_des_official_calibration_readiness_passed` with
`research_decision_applied=false`.

Readiness completed on 2026-07-15:

```text
result index                = outputs/00_RECENT_RESULTS.md entry 001
result rows                 = 1/1
validation errors           = []
parameter count             = 649793
decision                    = feistel_des_official_calibration_readiness_passed
research_decision_applied   = false
next action                 = run_des5_official_layout_2048_two_seed_calibration
```

Artifacts are under
`outputs/local_smoke/i1_feistel_des_r5_official_layout_readiness_seed0/`.
The readiness AUC is intentionally not recorded as research evidence.

### Calibration

```text
plan       = configs/experiment/innovation1/innovation1_feistel_des_r5_official_layout_2048_seed0_seed1.csv
scale      = 2048/class
seeds      = 0, 1
epochs     = 10
validation = 1024/class
fresh test = 3 x 2048/class
device     = local CPU
```

The calibration passes only if both seeds have mean fresh-test AUC at least
`0.60`. This threshold is intentionally below the paper's near-perfect DES-r5
accuracy because the local data budget is several orders of magnitude smaller.
It is still far enough above chance to reject an accidental validation spike.

## Decisions

If both seeds pass:

```text
decision    = feistel_des5_official_calibration_passed
next action = freeze a DES-r6 2048/class three-row matrix:
              official raw baseline, official-backbone branch true,
              equal-capacity official-backbone branch shuffled
```

If either seed fails:

```text
decision    = feistel_des5_official_calibration_inconclusive
next action = inspect training curves and run at most one planned local
              DES-r5 8192/class calibration before changing DES-r6
```

## Explicit Stops

- no DES-r6 remote scale from the rejected R1 result;
- no DES-r7 staged training;
- no simultaneous feature, negative, difference, key-schedule, or optimizer
  changes;
- no claim that DES-r5 calibration establishes Feistel topology attribution;
- no direct comparison of project AUC with paper accuracy.

## Evidence-Backed Next Action

Run readiness locally. If its protocol gate passes, continue automatically to
the two-seed `2048/class` calibration using a unique tmux session and a fresh
output directory. Generate JSONL, progress, validation, SVG, history CSV,
calibration gate, and the numbered recent-result index. Use the gate decision,
not a single validation epoch, to decide whether DES-r6 attribution reopens.

## Completed Calibration Result

The two-seed calibration completed on 2026-07-15 with all plan-alignment,
history, per-pair-key, capacity, and fresh-test checks passing.

```text
result index = outputs/00_RECENT_RESULTS.md entry 001
results      = outputs/local_diagnostic/i1_feistel_des_r5_official_layout_2048_seed0_seed1/results.jsonl
progress     = outputs/local_diagnostic/i1_feistel_des_r5_official_layout_2048_seed0_seed1/progress.jsonl
validation   = outputs/local_diagnostic/i1_feistel_des_r5_official_layout_2048_seed0_seed1/validation.json
curves       = outputs/local_diagnostic/i1_feistel_des_r5_official_layout_2048_seed0_seed1/curves.svg
history      = outputs/local_diagnostic/i1_feistel_des_r5_official_layout_2048_seed0_seed1/history.csv
gate         = outputs/local_diagnostic/i1_feistel_des_r5_official_layout_2048_seed0_seed1/gate.json
```

| seed | validation AUC | mean fresh AUC | mean fresh accuracy | best epoch |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.967437744 | 0.968407075 | 0.905110677 | 6 |
| 1 | 0.952584267 | 0.964183966 | 0.909342448 | 10 |

Fresh-test AUC repeats:

```text
seed0 = 0.969305038, 0.969848871, 0.966067314
seed1 = 0.964449406, 0.962664843, 0.965437651
```

The gate returned:

```text
status                     = pass
decision                   = feistel_des5_official_calibration_passed
calibration_signal_present = true
next_action                = run_des6_official_backbone_attribution_2048
research_decision_applied  = true
errors                     = []
```

This calibration proves that the official-layout mechanism, DES adapter,
strict-negative generator, and local training path can learn a strong reduced
DES signal at `2048/class`. It does not establish DES-r6 accuracy or Feistel
topology attribution. It rejects the planned `8192/class` DES-r5 rescue and
reopens only the same-budget DES-r6 official-backbone attribution test.
