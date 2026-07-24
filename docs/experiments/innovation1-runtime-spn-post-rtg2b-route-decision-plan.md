# Innovation 1 Runtime SPN Post-RTG2-B Route Decision Plan

Date: 2026-07-24

## Status

```text
stage      = completed decision audit; Route A selected
dependency = RTG2-B seed1 verified retrieval and two-seed joint gate
open slot  = exactly one next experiment route
status     = pass
decision   = select_route_a_formal_skinny_scale
selected   = RTG3-A SKINNY 1000000/class seed0, then conditional seed1
```

This audit prevents a passing `262144/class` result from automatically turning
into a costly `1000000/class` run. It compares the two remaining method-level
questions after the frozen RTG2-B protocol is adjudicated.

## Competing Routes

### Route A: Formal SKINNY Scale

```text
question       = does the within-cipher correct-topology advantage survive
                 at formal project scale?
anchor         = RTG2-B correct / corrupted / no-topology matrix
one variable   = samples_per_class 262144 -> 1000000
train          = 2000000 total/seed
validation     = at least 1000000 total/seed unless separately justified
seeds          = 0 then conditional seed1
epochs         = frozen 5/model
models         = correct / corrupted / no topology
execution      = remote A6000, disk-backed cache, verified result branches
advance gate   = both seeds AUC >= 0.55 and both control margins >= +0.005
```

Strength: the model, data path, controls and remote evidence pipeline are
already exercised at `262144/class`. This is the shortest path to stronger
within-SKINNY evidence.

Limitation: it mostly strengthens scale confidence. It does not answer whether
one shared set of learned weights can adapt across SPN ciphers, which is closer
to the runtime-parameterized method claim.

### Route B: Frozen Shared Backbone Plus Target Head

```text
question       = are X1's below-chance zero-step scores caused by a
                 cipher-specific output orientation rather than an unusable
                 shared representation?
anchor         = X1 zero-step audit plus SKINNY T2-C full-target anchors
one variable   = train only a freshly initialized target classifier while
                 the GIFT Runtime-E4 backbone remains frozen
train          = 4096 total/seed = 2048/class
validation     = 2048 total/seed = 1024/class
seeds          = 0 and 1
epochs         = 5
roles          = correct source/correct target, corrupted source,
                 corrupted target, random frozen backbone
execution      = local CPU using existing immutable caches
advance gate   = both seeds AUC >= 0.55 and all three margins >= +0.005
```

Strength: X2 is much cheaper and directly tests the unresolved cross-cipher
representation mechanism. It can falsify the proposed explanation before a
new remote scale commitment.

Limitation: X2 is a small diagnostic and reuses historical source/target
artifacts. Even a pass requires fresh medium confirmation before a general
cross-cipher claim.

## Frozen Decision Order

```text
RTG2-B joint protocol fail:
  repair evidence only; open neither route

RTG2-B joint hold:
  stop scale-up; audit seed variance/training dynamics; do not run X2 as a
  rescue for unstable within-cipher evidence

RTG2-B joint pass:
  run local X2 first because it changes the next decision at far lower cost
  and directly tests the method-level cross-cipher mechanism

X2 pass:
  rank a fresh medium Runtime-E4 target-head confirmation against formal
  SKINNY scale; open only one remote slot through a new plan

X2 hold:
  close frozen-backbone head-only scaling; Route A becomes the preferred next
  evidence step, subject to a separately frozen 1000000/class two-seed plan

X2 protocol fail:
  repair X2 only; do not reinterpret metrics or switch routes until valid
```

The preference for X2 after a joint pass is an information-per-cost decision,
not a claim that cross-cipher adaptation has already succeeded. X1 currently
shows only active topology sensitivity, not zero-step discrimination.

## Completed Post-X2 Adjudication

X2 completed with the following candidate and same-data full-target AUCs:

| Seed | Frozen GIFT backbone + SKINNY head | Full SKINNY training | Deficit |
| ---: | ---: | ---: | ---: |
| 0 | 0.552013397 | 0.612733364 | -0.060719967 |
| 1 | 0.598568439 | 0.614543915 | -0.015975475 |

Both candidates passed the frozen source-topology, target-topology and random-
backbone margins. This supports a reusable cross-cipher representation at
small scale. It does not yet make Route B the better remote experiment:

1. The source GIFT checkpoints were themselves trained at only `2048/class`.
2. Scaling only the SKINNY head would leave source-representation scale as an
   unresolved confound; scaling both would change more than one variable.
3. Seed0 is only `+0.002013` above the absolute AUC gate and remains `0.060720`
   below the same-data end-to-end target anchor.
4. RTG2-B already provides two stable `262144/class` seeds near `0.648`, with
   correct topology ahead of corrupted topology by about `0.045` and ahead of
   no topology by about `0.137`.

The single next remote slot is therefore Route A. RTG3-A changes only
`samples_per_class` from `262144` to `1000000`; seed0 runs first and seed1 is
conditional on the exact seed0 gate. Route B is held as a supported small
mechanism result, not discarded. It may reopen only after a separate plan
freezes a medium GIFT source-backbone anchor without mixing source training,
target-head training and model changes.

## Claim Boundaries

RTG2-B remains medium evidence. X2 remains a small local diagnostic. Route A
would be project-formal scale but still not an exact reproduction of a
published attack or paper protocol. No branch may be described as universal
SPN support, SOTA, breakthrough, an attack, or definitive failure without its
own complete multi-seed evidence and controls.

## Blocked Actions

- Do not launch `1000000/class` without the completed post-X2 decision above,
  a committed RTG3-A plan, remote disk-cache readiness and publication gate.
- Do not run X2 before the RTG2-B joint gate passes.
- Do not run Route A and a medium Route B simultaneously.
- Do not change the difference, keys, negatives, pair count, topology controls,
  optimizer, metric or checkpoint rule while choosing between routes.
- Do not reopen DDT, trail, partial-decryption or zero-step score inversion as
  a rescue for either route.
