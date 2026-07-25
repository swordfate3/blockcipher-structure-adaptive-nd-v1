# Innovation 1 X4: SKINNY To RECTANGLE RuntimeE4 Linear Probe Attribution

Date: 2026-07-25

## Research Question

Is the transferable RECTANGLE signal in the frozen formal SKINNY RuntimeE4
representation directly accessible to a `384 -> 1` linear probe, or does the
positive X3 result require the `198401`-parameter nonlinear target head to
relearn most of the task?

This is a local representation-attribution experiment. It runs while RTG3
seed1 and RCT2 remain under their existing remote watcher workflow and does
not replace or authorize medium transfer.

## Evidence Anchors

```text
X3-A target seed0 candidate AUC  = 0.784893989563
X3-A target seed0 random AUC     = 0.675201892853
X3-A2 target seed1 candidate AUC = 0.753169536591
X3-A2 target seed1 random AUC    = 0.664473056793
```

Both anchors use an independently initialized nonlinear target head with
`198401` trainable parameters. X4 reduces the trainable head to one affine
layer with `384 + 1 = 385` parameters.

## Frozen Source And Target Data

Source checkpoints remain the exact formal SKINNY RTG3-A seed0 best
checkpoints:

```text
correct source SHA   = edb4b37a74eb876164a14a8f4924607e6c31616b616810e9d43c32b13e816cc1
corrupted source SHA = 797217c85b84edd507a66c9675c1752a65dec13478fa3b574ef95ae99e325f42
```

Target data reuses the exact RCT1 RECTANGLE-80 r6 disk caches:

```text
panel seed0: train seed0, validation seed10000
panel seed1: train seed1, validation seed10001
train per panel      = 4096 total = 2048/class
validation per panel = 2048 total = 1024/class
```

No cipher samples, labels, keys, negative definition or raw features are
regenerated.

## Eight-Row Attribution Matrix

Each target seed uses four roles:

```text
candidate       = correct SKINNY source + correct RECTANGLE target
source control  = corrupted SKINNY source + correct RECTANGLE target
target control  = correct SKINNY source + corrupted RECTANGLE target
random control  = random extractor + correct RECTANGLE target
```

For each role, the complete RuntimeE4 extractor and original source classifier
remain frozen in evaluation mode. The exact pre-classifier RuntimeE4
representation is extracted once in deterministic batches and cached as
`float32 [rows, 384]`. Only a fresh deterministic `Linear(384, 1)` probe trains.

## Fixed Probe Protocol

```text
run id       = i1_skinny_rectangle_runtime_e4_linear_probe_x4_2048_seed0_seed1_20260725
probe        = Linear(384, 1), 385 trainable parameters
epochs       = 100
batch        = 256
optimizer    = Adam, lr 1e-3, weight decay 1e-5
loss         = MSE on sigmoid output
checkpoint   = best validation AUC, strictly restored
train eval   = epochs 1, 10, 20, ..., 100
execution    = local CPU
```

The longer probe schedule is not a same-compute comparison with X3. It is a
convergence-oriented test of linear accessibility. All four roles and both
seeds receive exactly the same extraction and probe optimization protocol.

## Protocol Gates

Fail closed if any source/target authority changes, a raw cache identity
changes, a representation cache is incomplete, the two seeds reuse the same
target data, target-head input width is not 384, the probe has other than 385
parameters, role initializations differ within a seed, an extractor or source
classifier hash changes, a checkpoint cannot be replayed, or any role changes
optimizer/data/epoch settings.

## Research Gates

Each seed must independently satisfy all of:

```text
candidate linear-probe AUC >= 0.55
candidate - corrupted-source AUC >= 0.005
candidate - corrupted-target AUC >= 0.005
candidate - random-source AUC >= 0.005
```

The two candidate AUCs must additionally differ by at most `0.05`.

## Decision And Next Action

- Pass: retain the claim that the frozen RuntimeE4 representation exposes
  cross-cipher signal even to a 385-parameter linear readout. Continue waiting
  for RCT2; X3-B remains conditional on RCT2, not on X4 alone.
- Hold: retain X3 nonlinear-head transfer evidence, but narrow the mechanism
  claim to nonlinear target adaptation. Do not scale the linear-probe route.
- Fail: repair evidence only and do not interpret AUC.

Blocked regardless of outcome: medium transfer before RCT2 passes, unfreezing
the extractor, changing target data after observing results, treating 100
linear-probe epochs as same-compute superiority, or claiming formal/universal
SPN transfer, attack, SOTA or breakthrough evidence.

## Completed Result

Run completed locally on 2026-07-25:

```text
run id = i1_skinny_rectangle_runtime_e4_linear_probe_x4_2048_seed0_seed1_20260725

seed0 candidate         AUC = 0.791651725769
seed0 corrupted source  AUC = 0.732803344727
seed0 corrupted target  AUC = 0.616181373596
seed0 random source     AUC = 0.732495307922

seed1 candidate         AUC = 0.761355400085
seed1 corrupted source  AUC = 0.714918136597
seed1 corrupted target  AUC = 0.629990577698
seed1 random source     AUC = 0.714735031128
```

Control margins:

```text
seed0 candidate - corrupted source = +0.058848381042
seed0 candidate - corrupted target = +0.175470352173
seed0 candidate - random source    = +0.059156417847

seed1 candidate - corrupted source = +0.046437263489
seed1 candidate - corrupted target = +0.131364822388
seed1 candidate - random source    = +0.046620368958

absolute candidate seed drift      = 0.030296325684
```

All preregistered research gates passed for both seeds. The seed drift also
passed the `<= 0.05` stability gate. All protocol checks passed, including
exactly eight result rows, 385 trainable parameters, source and target topology
attribution, frozen extractor and source-classifier hashes, distinct target
data across seeds, parameter-matched representation-cache replay, and best
checkpoint replay.

Independent artifact verification reopened all eight checkpoints and all 16
representation caches. It passed exact file-set, SHA-256, tensor-geometry,
history, final-metric and metadata checks with no errors. The final SVG was
rendered at `1920x1080` and passed `visual-qa-redraw` after correcting the left
axis so the declared `0.50` and `0.55` reference lines are visible.

```text
status   = pass
decision = innovation1_skinny_rectangle_linear_probe_accessibility_supported
```

## Interpretation And Recommended Next Action

The X3 transfer signal does not require the `198401`-parameter nonlinear head
to become accessible. A single affine `384 -> 1` readout reaches candidate AUC
`0.7917/0.7614` and retains material margins over all three controls on both
target seeds. This supports the narrower mechanism statement that the frozen
formal SKINNY RuntimeE4 representation exposes linearly accessible RECTANGLE
signal under this local RCT1 protocol.

This remains `2048/class` local representation attribution, not medium or
formal cross-cipher transfer. It does not establish universal SPN adaptation,
an attack, SOTA or a breakthrough. The 100-epoch linear schedule is also not a
same-compute superiority comparison against the five-epoch X3 nonlinear head.

Recommended next action: keep the linear readout as the preferred diagnostic
head and continue waiting for the already queued RECTANGLE RCT2 same-protocol
`65536/class` anchor. If and only if RCT2 passes its own topology controls,
prepare one seed0 medium frozen-transfer confirmation using the unchanged X3
roles, with the linear probe included as the low-capacity mechanism control.
Do not launch X3-B before RCT2, unfreeze the extractor, alter negatives or
differences, or mechanically enlarge X4 itself.

Evidence:

```text
outputs/local_diagnostic/i1_skinny_rectangle_runtime_e4_linear_probe_x4_2048_seed0_seed1_20260725/results.jsonl
outputs/local_diagnostic/i1_skinny_rectangle_runtime_e4_linear_probe_x4_2048_seed0_seed1_20260725/gate.json
outputs/local_diagnostic/i1_skinny_rectangle_runtime_e4_linear_probe_x4_2048_seed0_seed1_20260725/validation.json
outputs/local_diagnostic/i1_skinny_rectangle_runtime_e4_linear_probe_x4_2048_seed0_seed1_20260725/artifact-verification.json
outputs/local_diagnostic/i1_skinny_rectangle_runtime_e4_linear_probe_x4_2048_seed0_seed1_20260725/curves.svg
outputs/local_diagnostic/i1_skinny_rectangle_runtime_e4_linear_probe_x4_2048_seed0_seed1_20260725/visual-qa.json
```
