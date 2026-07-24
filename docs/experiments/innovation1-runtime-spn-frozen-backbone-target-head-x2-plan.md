# Innovation 1 Runtime SPN Frozen-Backbone Target-Head X2 Plan

Date: 2026-07-24

## Status

```text
stage       = completed local diagnostic
run_id      = i1_rtg1_gift_to_skinny_frozen_backbone_target_head_x2_seed0_seed1_20260724
execution   = local diagnostic after RTG2-B two-seed adjudication
dependency  = RTG2-B seed1 retrieved, validated, visually checked and jointly adjudicated
result      = pass / runtime_spn_frozen_backbone_target_head_supported
script      = configs/remote/generated/monitor_i1_runtime_spn_x2_after_rtg2b_20260724.sh
claim scope = small cross-cipher representation diagnostic only
started     = 2026-07-24 23:54:01 +08:00
completed   = 2026-07-24 23:55:23 +08:00
visual QA   = pass / 2304x1308 rendered-pixel inspection
index       = outputs/00_RECENT_RESULTS.md entry 001 at completion
```

X1 proved that a GIFT-trained Runtime-E4 checkpoint reacts to a new SKINNY
runtime topology, but its zero-step score direction is not discriminative. The
candidate AUCs were `0.460700` and `0.407617`, below all three controls. X2
tests the narrowest remaining explanation: the cipher-name-free backbone may
be reusable while the final binary output orientation remains target-specific.

X2 must not alter or supervise the running RTG2-B seed1 experiment. Its
performance run is closed until the RTG2-B two-seed gate exists locally and
the retrieved seed1 chart has a pixel-inspected `visual_qa_passed.marker`.

## Completed Result

| Seed | Correct source + correct target | Corrupted source | Corrupted target | Random frozen backbone | Full-target anchor |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.552013397 | 0.537303925 | 0.523563862 | 0.527775764 | 0.612733364 |
| 1 | 0.598568439 | 0.569044113 | 0.523379803 | 0.542519569 | 0.614543915 |

Candidate attribution margins:

```text
seed0 candidate-corrupted-source = +0.014709473
seed0 candidate-corrupted-target = +0.028449535
seed0 candidate-random-backbone  = +0.024237633

seed1 candidate-corrupted-source = +0.029524326
seed1 candidate-corrupted-target = +0.075188637
seed1 candidate-random-backbone  = +0.056048870
```

All eight rows, forty epoch histories, eight best checkpoints, data and source
hashes, strict state-dict loads, parameter ownership, frozen-backbone hashes,
classifier updates, and best-checkpoint replays passed validation. Both seeds
therefore pass the preregistered `AUC >= 0.55` and `+0.005` attribution gates.

The supported claim is narrow: a GIFT-trained Runtime-E4 backbone retains
SKINNY-useful information after only the SKINNY classifier is retrained, and
the result depends on both the source and target topology. It does not match
the same-data end-to-end SKINNY anchors: the deficits are `-0.060720` for
seed0 and `-0.015975` for seed1. Seed0 is also only `+0.002013` above the
absolute AUC gate, so this remains a small, fragile transfer diagnostic rather
than evidence for universal or scale-ready adaptation.

## Recommended Next Action

Run a document-only route-decision audit before assigning another training
slot. The exact unresolved comparison is whether the next evidence budget is
better spent on:

```text
A. formal SKINNY confirmation at >=1000000/class with multiple seeds,
   anchored to the completed RTG2-B 262144/class correct/corrupted/no-topology
   panel; or
B. medium frozen-backbone target-head adaptation, anchored to this X2 panel
   and its same-data end-to-end SKINNY controls.
```

The audit must compare evidence gain, compute cost, control completeness, and
thesis relevance. X2 alone does not authorize B: its seed0 absolute margin is
fragile and both seeds remain below end-to-end target training. RTG2-B has the
stronger present evidence for A, but the route decision must explicitly weigh
formal within-cipher confirmation against the cross-cipher adaptation claim.
Do not mechanically enlarge X2, unfreeze its backbone, or launch either route
until that decision is recorded in a separate experiment plan.

## Research Question

With the GIFT Runtime-E4 feature extractor frozen, can retraining only the
existing classifier head on SKINNY r7 recover a two-seed discriminative signal
that depends on both the correct source checkpoint and the correct target
runtime topology?

This is target-head adaptation, not zero-shot transfer, full-model fine-tuning,
formal scale, or a new data protocol.

## Same-Budget Anchor And One Variable

The historical same-data target anchor is the completed SKINNY RTG1-T2-C
correct-topology model trained end-to-end for five epochs:

```text
seed0 AUC = 0.612733364
seed1 AUC = 0.614543915
```

X2 reuses the exact T2-C train/validation arrays and five-epoch optimizer
protocol. The experimental change is initialization/training ownership:

```text
T2-C = initialize on SKINNY and train all 442466 parameters
X2   = initialize backbone from GIFT or a matched control, replace the
       classifier with one shared deterministic initialization, freeze the
       backbone, and train only the 198401-parameter classifier
```

The full-target anchor is contextual evidence, not one of the four X2 rows.

## Frozen Data And Training Protocol

For each seed independently:

```text
source checkpoint       = completed GIFT RTG1-R2F best checkpoint
target cipher           = SKINNY-64/64
target rounds           = 7
target difference       = 0x2000
train key               = 0x0000000000000000
validation key          = 0x1111111111111111
train                    = 4096 total = 2048/class
validation               = 2048 total = 1024/class
pairs/sample             = 4 independent ciphertext pairs = 512 input bits
negative                 = encrypted random plaintexts
model                    = Runtime-E4, 442466 total parameters
trainable                = classifier only, 198401 parameters
epochs                   = 5
batch                    = 256
optimizer                = Adam, learning rate 1e-4
loss                     = MSE
weight decay             = 1e-5
checkpoint               = best validation AUC
device                   = local CPU
```

All four roles within a seed must use byte-identical feature/label arrays, the
same batch order and the same optimizer hyperparameters. All eight roles across
both seeds must use the same deterministic classifier initialization. The two
seeds use their already independent T2-C datasets and corresponding R2F source
checkpoints.

## Four-Role Matrix

| Role | Frozen backbone | Target structure | Purpose |
| --- | --- | --- | --- |
| `true_source_true_target` | GIFT correct-topology best checkpoint | SKINNY correct GF(2) | candidate |
| `corrupted_source_true_target` | GIFT corrupted-topology best checkpoint | SKINNY correct GF(2) | source-topology attribution |
| `true_source_corrupted_target` | same GIFT correct checkpoint as candidate | deterministic corrupted SKINNY GF(2) | target-topology attribution |
| `random_source_true_target` | deterministic untrained Runtime-E4 state | SKINNY correct GF(2) | matched frozen-random representation control |

The candidate and target-corrupted role must share the exact source checkpoint
SHA-256. The candidate, source-corrupted and random controls must share the
exact target structure. Every role must replace the loaded classifier with the
same deterministic classifier state before training.

## Readiness Gate

Before performance training, require:

1. The RTG2-B joint gate exists locally and is protocol-valid; a running,
   fallback-only, or single-seed state does not open X2.
2. Both GIFT R2F source roots and both SKINNY T2-C cache roots match the frozen
   X1/T2-C provenance.
3. Every source state loads strictly into the SKINNY adapter.
4. Every model has `442466` total and `198401` trainable parameters.
5. Only `backbone.classifier.*` has `requires_grad=True`.
6. All eight roles begin with byte-identical classifier states.
7. A one-batch backward check produces gradients only for the classifier.
8. No new dataset is generated and no existing cache is modified.

Readiness is implementation evidence only. It does not interpret AUC or enter
the completed-results index.

## Result Gate

Protocol failure if row count, source/cache hashes, structure identities,
classifier initialization, parameter ownership, five-epoch histories,
best-checkpoint replay, or finite metrics fail.

For both seeds require:

```text
candidate AUC >= 0.55
candidate - corrupted-source AUC >= +0.005
candidate - corrupted-target AUC >= +0.005
candidate - random-frozen AUC >= +0.005
```

The gate also reports, without changing the support thresholds:

```text
candidate - full-target T2-C anchor AUC
```

Decisions:

```text
both seeds pass:
  decision    = runtime_spn_frozen_backbone_target_head_supported
  next_action = compare this route against formal SKINNY scale in a separate
                route audit; do not launch medium adaptation automatically

probabilities/train loss change but a research margin misses:
  decision    = runtime_spn_target_head_signal_unstable
  next_action = stop X2 scaling; retain cipher-specific full-target head need

protocol or gradient ownership fails:
  decision    = runtime_spn_target_head_protocol_invalid
  next_action = repair evidence only without changing data or thresholds
```

## Required Artifacts

```text
outputs/local_diagnostic/
  i1_rtg1_gift_to_skinny_frozen_backbone_target_head_x2_seed0_seed1_20260724/
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

After a completed performance run, validate all eight rows, replay the gate,
run `visual-qa-redraw` on rendered pixels, refresh both recent-result indexes,
and update this record with metrics and the evidence-backed next action.

## Blocked Routes

Do not run X2 before the RTG2-B joint adjudication. Do not unfreeze the
backbone, change the SKINNY difference/keys/negatives, add epochs/samples,
invert below-chance X1 scores, tune thresholds on validation labels, or launch
remote scale from X2 readiness. Do not call any positive result zero-shot,
formal, paper-scale, universal-SPN, SOTA, breakthrough, or an attack.
