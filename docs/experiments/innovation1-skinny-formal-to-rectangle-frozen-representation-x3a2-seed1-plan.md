# Innovation 1 X3-A2: SKINNY To RECTANGLE Frozen Representation Seed1 Replication

Date: 2026-07-25

## Research Question

Does the positive X3-A frozen-representation result survive an independent
RECTANGLE target dataset and training seed, while the formal SKINNY source,
network, controls, budget and thresholds remain unchanged?

This local replication runs while RTG3 seed1 and the dependent RCT2 medium
anchor remain under their existing watcher workflow. It does not consume the
remote GPU lane and does not replace RCT2.

## Frozen Authorities

Source authority remains exactly the X3-A source:

```text
SKINNY-64/64 r7 RTG3-A formal seed0
train      = 1000000/class
validation = 500000/class
true AUC   = 0.653191631304
wrong AUC  = 0.607162432806
```

Target authority changes only from the RCT1 seed0 split to the independently
generated RCT1 seed1 split:

```text
cipher                  = RECTANGLE-80 r6
train cache seed        = 1
validation cache seed   = 10001
end-to-end true AUC     = 0.765672683716
X3-A seed0 candidate AUC = 0.784893989563
```

The exact existing RCT1 disk caches are reused. No samples, labels, keys,
negative definition or features are regenerated.

## One Variable

```text
changed   = RECTANGLE target data/training seed 0 -> 1
unchanged = formal SKINNY seed0 source checkpoints
unchanged = correct/corrupted/random source roles
unchanged = correct/corrupted target topology roles
unchanged = target-head initialization
unchanged = network, optimizer, loss, epochs and thresholds
```

The source seed intentionally remains zero. X3-A2 tests target-split and
training stability, not independent formal-source checkpoint stability; the
latter remains owned by the running RTG3 seed1 experiment.

## Four-Role Matrix

```text
candidate       = correct SKINNY source + correct RECTANGLE target
source control  = corrupted SKINNY source + correct RECTANGLE target
target control  = correct SKINNY source + corrupted RECTANGLE target
random control  = random extractor + correct RECTANGLE target
```

Only the independent `target_head.*` parameters may train. Every extractor and
its original source classifier must remain frozen and byte-identical.

## Fixed Budget

```text
run id            = i1_skinny_formal_to_rectangle_frozen_representation_x3a2_2048_seed1_20260725
train              = 4096 total = 2048/class
validation         = 2048 total = 1024/class
target/training seed = 1
validation seed    = 10001
pairs per sample   = 4 independent ciphertext pairs
input              = 512 raw ciphertext-pair bits
negative           = encrypted random plaintexts
epochs             = 5
batch              = 256
optimizer          = Adam, lr 1e-4, weight decay 1e-5
loss               = MSE on sigmoid output
checkpoint         = best validation AUC, strictly restored
execution          = local CPU replication
```

## Gates

Protocol failure occurs if any authority, cache, source checkpoint, topology,
head initialization, parameter ownership, frozen hash or checkpoint replay
check fails.

Research pass requires all of:

```text
candidate AUC >= 0.55
candidate - corrupted-source AUC >= 0.005
candidate - corrupted-target AUC >= 0.005
candidate - random-source AUC >= 0.005
abs(candidate - X3-A seed0 candidate) <= 0.05
```

## Executable Decision

- Pass: combine X3-A and X3-A2 as two-target-seed local readiness evidence,
  continue waiting for RCT2, and allow X3-B only if RCT2 also passes.
- Hold: do not launch X3-B. Treat the seed0 transfer result as split-sensitive
  and redesign the transfer protocol locally after RTG3/RCT2 finish.
- Fail: repair evidence only; do not change data, roles or thresholds after
  observing metrics.

Blocked regardless of outcome: medium transfer before RCT2 passes, extractor
unfreezing, formal/universal-SPN claims, attack/SOTA claims, or mechanical
`262144/class` transfer scaling.

## Completed Result

The exact preregistered local command completed successfully with the derived
output root for the frozen run id. Four result rows and four best-checkpoint
payloads were produced. Every source authority, target cache identity,
parameter ownership, checkpoint replay, frozen extractor hash, frozen source
classifier hash and explicit seed identity check passed.

```text
correct SKINNY source + correct RECTANGLE target AUC = 0.753169536591
corrupted SKINNY source + correct target AUC         = 0.645070075989
correct source + corrupted RECTANGLE target AUC      = 0.622686386108
random source + correct RECTANGLE target AUC          = 0.664473056793
RCT1 seed1 end-to-end target anchor AUC                = 0.765672683716
X3-A seed0 frozen-transfer candidate AUC               = 0.784893989563
```

Margins and replication stability:

```text
candidate - corrupted source = +0.108099460602
candidate - corrupted target = +0.130483150482
candidate - random source    = +0.088696479797
candidate - seed1 end-to-end anchor = -0.012503147125
candidate - X3-A seed0 candidate    = -0.031724452972
```

The candidate stayed within the preregistered `0.05` cross-seed drift bound
and retained large margins over all three controls. The weaker absolute AUC is
directionally consistent with the lower RCT1 seed1 end-to-end anchor.

Artifacts:

```text
outputs/local_diagnostic/i1_skinny_formal_to_rectangle_frozen_representation_x3a2_2048_seed1_20260725/results.jsonl
outputs/local_diagnostic/i1_skinny_formal_to_rectangle_frozen_representation_x3a2_2048_seed1_20260725/history.csv
outputs/local_diagnostic/i1_skinny_formal_to_rectangle_frozen_representation_x3a2_2048_seed1_20260725/gate.json
outputs/local_diagnostic/i1_skinny_formal_to_rectangle_frozen_representation_x3a2_2048_seed1_20260725/validation.json
outputs/local_diagnostic/i1_skinny_formal_to_rectangle_frozen_representation_x3a2_2048_seed1_20260725/curves.svg
outputs/local_diagnostic/i1_skinny_formal_to_rectangle_frozen_representation_x3a2_2048_seed1_20260725/checkpoints/
```

The final SVG passed `visual-qa-redraw` at `1896x1056`: Chinese glyphs,
seed-specific title, subtitle, conclusion, axes, bar values, curve separation,
opaque legend, footer and clipping checks all passed. The completed result was
then refreshed into `outputs/00_RECENT_RESULTS.md` as entry `001`.

## Adjudication And Next Action

```text
status   = pass
decision = innovation1_skinny_rectangle_frozen_representation_seed1_replication_supported
```

X3-A and X3-A2 now provide two-target-seed local readiness evidence that a
formal SKINNY RuntimeE4 representation can be rebound to RECTANGLE and used by
an independently trained target head. The source still comes from one formal
SKINNY seed, both transfer runs remain only `2048/class`, and only one
source-target cipher pair has been tested. This is not medium/formal transfer,
universal-SPN evidence, a paper reproduction, attack, SOTA or breakthrough.

Continue waiting for the already queued RCT2 same-protocol RECTANGLE end-to-end
medium anchor. If and only if RCT2 returns a complete, plan-aligned passing
gate locally, prepare X3-B with the same four roles and one changed variable:

```text
train      = 65536/class
validation = 32768/class
seed       = 0
epochs     = 5
pairs      = 4
execution  = remote GPU with parameter-matched disk cache and progress
```

X3-B retains the `0.55` candidate floor and all three `+0.005` control margins.
Do not launch it if RCT2 fails, and do not mechanically continue to
`262144/class`, additional target ciphers or formal transfer after one medium
seed.
