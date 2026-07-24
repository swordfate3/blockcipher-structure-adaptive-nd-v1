# Innovation 1 X3-A: SKINNY Formal To RECTANGLE Frozen Representation Readiness

Date: 2026-07-25

## Research Question

Does a RuntimeE4 representation learned from the project-formal SKINNY-64/64
r7 seed0 run remain useful after rebinding the same parameter geometry to the
non-contiguous RECTANGLE-80 runtime structure, when only an independent target
head is trained?

This is a local readiness experiment. It does not replace the queued
RECTANGLE RCT2 end-to-end medium anchor and cannot authorize medium transfer
before RCT2 itself passes.

## Frozen Evidence

Source authority:

```text
run       = i1_rtg3a_skinny64_general_gf2_formal_1000000_seed0_20260725
scale     = 1000000/class train, 500000/class validation
true AUC  = 0.653191631304
wrong AUC = 0.607162432806
```

Target authority:

```text
run       = i1_rct1_rectangle80_runtime_e4_noncontiguous_attribution_2048_seed0_seed1_20260725
cipher    = RECTANGLE-80 r6
seed0 end-to-end true-topology AUC = 0.791468620300293
```

The exact RCT1 seed0 train and validation disk caches are reused. No labels,
keys, negatives, difference, feature encoding or split identity are regenerated.

## Four-Role Matrix

```text
candidate       = correct SKINNY source + correct RECTANGLE target
source control  = corrupted SKINNY source + correct RECTANGLE target
target control  = correct SKINNY source + corrupted RECTANGLE target
random control  = random extractor + correct RECTANGLE target
```

All roles use the same deterministic independent target-head initialization.
The complete RuntimeE4 extractor and its original source classifier remain
frozen and in evaluation mode. Only `target_head.*` parameters may train.

## Fixed Budget

```text
train             = 4096 total = 2048/class
validation        = 2048 total = 1024/class
seed              = 0
pairs per sample  = 4 independent ciphertext pairs
input             = 512 raw ciphertext-pair bits
negative          = encrypted random plaintexts
epochs            = 5
batch             = 256
optimizer         = Adam, lr 1e-4, weight decay 1e-5
loss              = MSE on sigmoid output
checkpoint        = best validation AUC, strictly restored
execution         = local CPU readiness
```

## Gates

Protocol failure if any source/checkpoint/cache identity changes, source
extractor or classifier updates, roles use different target data or head
initialization, a checkpoint cannot be strictly replayed, or any parameter
outside `target_head.*` is trainable.

Research pass requires all of:

```text
candidate AUC >= 0.55
candidate - corrupted-source AUC >= 0.005
candidate - corrupted-target AUC >= 0.005
candidate - random-source AUC >= 0.005
```

The candidate is not required to beat the end-to-end RCT1 anchor. Its exact
gap to that anchor must be reported.

## Decision And Next Action

- Pass: retain the frozen-transfer route, wait for RCT2, and only if RCT2
  passes prepare one same-role `65536/class` seed0 medium confirmation.
- Hold: stop SKINNY-to-RECTANGLE frozen-transfer scaling. Keep the reusable
  adapter but use end-to-end target training.
- Fail: repair evidence only. Do not alter data, roles, epochs or thresholds
  after observing results.

Blocked regardless of outcome: universal-SPN, formal transfer, attack, SOTA or
breakthrough claims; unfreezing the extractor; and launching medium transfer
before a passing RCT2 result exists locally.

## Completed Result

Run id:

```text
i1_skinny_formal_to_rectangle_frozen_representation_x3a_2048_seed0_20260725
```

The run completed locally on CPU and reused the exact RCT1 seed0 disk caches.
All four source/target roles used the same deterministic target-head
initialization. Checkpoint replay, source checkpoint attribution, target cache
identity, parameter ownership, frozen feature-extractor hashes and frozen
source-classifier hashes all passed.

```text
correct SKINNY source + correct RECTANGLE target AUC = 0.784893989563
corrupted SKINNY source + correct target AUC         = 0.661293029785
correct source + corrupted RECTANGLE target AUC      = 0.611306190491
random source + correct RECTANGLE target AUC          = 0.675201892853
RCT1 seed0 end-to-end target anchor AUC                = 0.791468620300
```

Control margins:

```text
candidate - corrupted source = +0.123600959778
candidate - corrupted target = +0.173587799072
candidate - random source    = +0.109692096710
candidate - end-to-end anchor = -0.006574630737
```

The first fail-closed invocation stopped before training because the X3-A
metadata check had copied the RECTANGLE difference as `0x2100000020`. RCT1's
plan, result row and disk cache all authoritatively use `0x2100010020`. X3-A
was corrected to import the frozen RCT1 constant directly, its regression test
was updated, and the complete run then passed without changing data, labels,
roles, network, budget or thresholds.

Artifacts:

```text
outputs/local_diagnostic/i1_skinny_formal_to_rectangle_frozen_representation_x3a_2048_seed0_20260725/results.jsonl
outputs/local_diagnostic/i1_skinny_formal_to_rectangle_frozen_representation_x3a_2048_seed0_20260725/history.csv
outputs/local_diagnostic/i1_skinny_formal_to_rectangle_frozen_representation_x3a_2048_seed0_20260725/gate.json
outputs/local_diagnostic/i1_skinny_formal_to_rectangle_frozen_representation_x3a_2048_seed0_20260725/validation.json
outputs/local_diagnostic/i1_skinny_formal_to_rectangle_frozen_representation_x3a_2048_seed0_20260725/curves.svg
outputs/local_diagnostic/i1_skinny_formal_to_rectangle_frozen_representation_x3a_2048_seed0_20260725/checkpoints/
```

## Adjudication

```text
status   = pass
decision = innovation1_skinny_rectangle_frozen_representation_readiness_supported
```

At this small fixed budget, a head trained over the frozen formal SKINNY
representation nearly matches the same-data end-to-end RECTANGLE anchor while
beating corrupted-source, corrupted-target and random-source controls by large
margins. This supports retaining the cross-cipher representation-transfer
route. It is one local `2048/class` seed and therefore does not establish
medium or formal transfer, universal-SPN adaptation, an attack, SOTA or a
breakthrough.

## Executable Next Action

The next question is whether the strong small-budget RECTANGLE signal remains
at medium scale under the unchanged end-to-end RuntimeE4 protocol. The queued
RCT2 run is the required same-budget target anchor and retains the RCT1 cipher,
rounds, difference, four-pair input, encrypted-random-plaintext negatives,
models and five-epoch training; only the data scale and seed matrix change.

Do not launch transfer scaling while RCT2 is pending. If and only if RCT2
returns a plan-aligned passing gate locally, prepare X3-B as one remote seed0
matrix with the same four X3-A roles:

```text
train      = 65536/class
validation = 32768/class
seed       = 0
epochs     = 5
pairs      = 4
device     = remote GPU with disk-backed cache and progress
one change = target data scale only
```

X3-B may advance only if every frozen/protocol check passes, candidate AUC is
at least `0.55`, and candidate exceeds corrupted-source, corrupted-target and
random-source controls by at least `0.005` each. If RCT2 fails, or if any X3-B
control margin fails, stop frozen SKINNY-to-RECTANGLE scaling and return to
end-to-end target training. Do not proceed mechanically to `262144/class`,
multiple target ciphers or formal transfer before this medium attribution gate.
