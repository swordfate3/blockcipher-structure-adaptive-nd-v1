# Innovation 1 Runtime-SPN GIFT-to-Dialga Frozen-Backbone X3 Plan

Date: 2026-07-25

## Status

```text
phase = completed local diagnostic
status = hold
decision = runtime_spn_gift_to_dialga_x3_signal_not_supported
source = completed GIFT-64 Runtime-E4 R2F seed0/seed1 checkpoints
target = completed Dialga-128 Runtime-E4 D1 seed0/seed1 data and anchors
remote_scale = prohibited
```

## Research Question

Can one unchanged Runtime-E4 parameter state learned on homogeneous 64-bit
GIFT retain useful information on heterogeneous 128-bit Dialga after training
only the existing target classifier?

X2 already showed a small GIFT-to-SKINNY frozen-backbone mechanism. Dialga is a
stronger held-out target because its block size, per-round S-box assignment and
general GF(2) linear topology differ from both GIFT and SKINNY. D7 rejected the
Runtime-E5 gated residual, so X3 returns to the supported Runtime-E4 backbone
instead of introducing another architecture.

This is target-head adaptation. It is not zero-shot transfer, full-model
fine-tuning, a Dialga attack, formal scale or universal-SPN evidence.

## Same-Budget Anchors And One Variable

The contextual target anchors are the completed D1 Runtime-E4 models trained
end to end on the exact Dialga data:

```text
seed0 correct-topology AUC = 0.958416939
seed1 correct-topology AUC = 0.958679199
```

The X3 change is parameter ownership and initialization:

```text
D1 = initialize on Dialga and train all 442466 parameters for 10 epochs
X3 = load a GIFT backbone, replace the classifier with one deterministic
     target initialization, freeze the backbone and train only the 198401-
     parameter classifier for 5 epochs
```

D1 remains contextual because X3 deliberately uses fewer trainable parameters
and epochs. The scientific same-budget controls are the three X3 roles trained
on byte-identical target data for the same five epochs.

## Frozen Evidence

Source roots:

```text
outputs/local_diagnostic/i1_rtg1_gift64_runtime_e4_late_attribution_r2f_2048_seed0
outputs/local_diagnostic/i1_rtg1_gift64_runtime_e4_late_attribution_r2f_2048_seed1
```

Per seed, use exactly the `gift64_runtime_e4_equivariant_true` and
`gift64_runtime_e4_equivariant_corrupted` restored-best checkpoints. Their
source protocol remains GIFT-64 r6, difference `0x40`, four independent pairs,
strict encrypted-random-plaintext negatives and `2048/class` training.

Target root:

```text
outputs/local_diagnostic/i1_dialga128_runtime_e4_d1_r4_2048_seed0_seed1_20260725
```

The D1 `results.jsonl`, persisted gate and validation must replay exactly. X3
must reuse the four D1 cache leaves without generating or modifying data.

## Frozen Target Protocol

For each seed:

```text
target cipher = Dialga-128 prefix-r4
target difference = 0x40
train key = 0
validation key = 0x11 repeated over 32 bytes
train = 4096 total = 2048/class
validation = 2048 total = 1024/class
pairs/sample = 4 independent ciphertext pairs
input = 1024 bits = 4 x 2 x 128
negative = encrypted random plaintext pairs
runtime window = round_start 2, rounds 2
target modes = correct or deterministic corrupted topology seed 20260725
epochs = 5
batch size = 256
optimizer = Adam, learning rate 1e-4, weight decay 1e-5
loss = MSE
checkpoint = restored best validation AUC
device = local CPU
```

## Four-Role Matrix

| Role | Frozen source backbone | Dialga target topology | Purpose |
| --- | --- | --- | --- |
| `true_source_true_target` | GIFT correct-topology best checkpoint | correct | candidate |
| `corrupted_source_true_target` | GIFT corrupted-topology best checkpoint | correct | source-topology control |
| `true_source_corrupted_target` | same GIFT correct checkpoint as candidate | corrupted | target-topology control |
| `random_source_true_target` | deterministic untrained Runtime-E4 state | correct | random-representation control |

All eight rows must replace the loaded classifier with one byte-identical
deterministic state. The source-corrupted row changes only the source state;
the target-corrupted row changes only the Dialga topology intervention.

## Readiness Gate

Readiness runs before performance training and must write durable evidence. It
passes only if:

1. Both source result files contain exactly one true and one corrupted row for
   their matching seeds and point to restored-best checkpoints with frozen
   GIFT protocol metadata.
2. The D1 source result, gate and validation hashes match and the gate replays
   as the completed two-seed pass.
3. The four exact D1 disk-cache leaves and their feature, label and metadata
   files exist with the frozen split geometry.
4. Every GIFT source state loads strictly into both required Dialga target
   structures without changing parameter names or shapes.
5. Every role has `442466` total parameters and exactly `198401` trainable
   `backbone.classifier.*` parameters.
6. Every role starts from the same classifier SHA-256; source states remain
   distinct and target structures have the expected true/corrupted hashes.
7. A real D1 batch produces finite, nonconstant representations and logits for
   every role with representation width `384`.
8. One disposable backward/optimizer step gives finite, nonzero gradients only
   to the classifier, changes the classifier hash and leaves the frozen
   backbone hash byte-identical.
9. No target data is generated, copied or modified during readiness.

If any check fails, decision is
`runtime_spn_gift_to_dialga_x3_readiness_not_supported`; stop before training.
Do not add an adapter, resize a state tensor, unfreeze the backbone, switch the
target protocol or regenerate data as a repair.

## Performance Gate

Protocol failure if the eight rows, source/checkpoint/cache hashes, strict
loads, target structure identities, common head initialization, frozen
backbone hashes, classifier-only ownership, five-epoch histories or
best-checkpoint replay are incomplete.

Each seed must independently satisfy:

```text
candidate AUC >= 0.55
candidate - corrupted-source AUC >= +0.005
candidate - corrupted-target AUC >= +0.005
candidate - random-frozen AUC >= +0.005
```

The gate also reports candidate minus the matching D1 full-target anchor
without using it as a pass threshold.

Decision routes:

- Both seeds pass: retain a small GIFT-to-Dialga cross-block-size shared
  representation result; next perform a no-new-training synthesis with X2.
- Any research margin misses: stop X3 performance scaling and retain X2 as the
  current cross-cipher boundary.
- Protocol failure: repair evidence only with all data, roles and thresholds
  frozen.

## Required Artifacts

```text
outputs/local_diagnostic/
  i1_rtg1_gift_to_dialga_frozen_backbone_target_head_x3_seed0_seed1_20260725/
    readiness.json
    results.jsonl
    history.csv
    progress.jsonl
    validation.json
    gate.json
    summary.json
    checkpoints/
    curves.svg
    visual_qa_passed.marker
```

After a completed run, refresh both recent-result indexes, update this record
with metrics and an evidence-backed next action, run `visual-qa-redraw` on the
exact SVG and commit/push only X3 files.

## Completed Result

```text
run_id = i1_rtg1_gift_to_dialga_frozen_backbone_target_head_x3_seed0_seed1_20260725
result_index = 001
readiness = pass, 14/14 checks
strict loads = 8/8 GIFT source x Dialga target combinations
performance validation = pass, 8/8 rows
protocol checks = 16/16 pass
research checks = 7/8 pass
history = 40/40 epoch rows
target cache = 12 files unchanged, 0 generation events
visual_qa_redraw = pass after replacing an overlapping threshold annotation
                      with the right-panel title threshold
```

Restored-best validation AUC:

| Seed | Correct source + correct target | Corrupted source | Corrupted target | Random frozen backbone | D1 end-to-end anchor |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.867733002 | 0.862721443 | 0.851464272 | 0.857554436 | 0.958416939 |
| 1 | 0.906326294 | 0.902811050 | 0.870974541 | 0.893280029 | 0.958679199 |

Per-seed candidate margins:

| Seed | Minus corrupted source | Minus corrupted target | Minus random backbone | Minus D1 anchor |
| ---: | ---: | ---: | ---: | ---: |
| 0 | +0.005011559 | +0.016268730 | +0.010178566 | -0.090683937 |
| 1 | +0.003515244 | +0.035351753 | +0.013046265 | -0.052352905 |

Both candidates exceed the absolute `0.55` floor and both target-topology and
random-backbone controls. Seed0 also clears the corrupted-source margin by
only `+0.000011559`. Seed1 misses that same pre-registered margin by
`0.001484756`. The high control AUCs show that the existing `198401`-parameter
classifier can recover much of this easy Dialga target even from corrupted or
random frozen representations. X3 therefore does not establish a stable
two-seed contribution from the topology learned on GIFT.

The correct claim is a protocol-valid, high-AUC but attribution-incomplete
local transfer diagnostic. It is not a passing cross-block-size shared-
backbone result. Stop X3 scaling and retain the completed GIFT-to-SKINNY X2 as
the current supported cross-cipher boundary.

## Evidence-Backed Next Action

Prepare a separately pre-registered X4 local linear-probe gate. Its question
is whether X3's large nonlinear target classifier absorbed the Dialga task and
masked source-topology attribution. X4 is a new capacity hypothesis, not a
threshold change or scale-up of X3.

Freeze the executable design as:

```text
research question = does a 385-parameter Linear(384, 1) target probe expose a
                    stable correct-GIFT-source advantage on Dialga?
same-budget anchor = completed X3 four-role rows and D1 contextual anchors
required controls = corrupted source, corrupted Dialga target, random frozen
                    backbone
one changed variable = target head only: existing 198401-parameter classifier
                       -> one 385-parameter linear probe
source checkpoints = exact X3 GIFT restored-best SHA-256 states
target data = exact D1 cache, Dialga prefix-r4, difference 0x40, 4 pairs/sample
scale = 2048/class train, 1024/class validation
seeds = 0, 1
epochs = 5
execution = local CPU
readiness gate = strict source loads, common probe SHA, only target_head
                 trainable, 385 trainable parameters, frozen extractor hash,
                 finite nonconstant 384-wide real-cache representations,
                 zero cache generation
advance gate = each seed candidate AUC >= 0.55 and candidate exceeds all three
               controls by >= +0.005
stop gate = any protocol miss, either candidate below 0.55, or any attribution
            margin below +0.005
```

Do not increase samples, epochs, head depth or trainable backbone parameters
to rescue X4. Do not launch remote scale. If X4 also fails source attribution,
close frozen GIFT-to-Dialga head adaptation and return to an end-to-end shared
backbone method rather than another probe-capacity sweep.

## Explicitly Blocked

- No Runtime-E5 reuse or another Dialga-only architecture tweak.
- No new data, difference, keys, negatives, pairs, epochs, metrics or source
  checkpoints.
- No target-backbone fine-tuning, tensor resizing or compatibility adapter.
- No remote GPU, medium/formal scale or parallel rescue matrix.
- No attack, SOTA, breakthrough or universal-SPN claim from this local gate.
