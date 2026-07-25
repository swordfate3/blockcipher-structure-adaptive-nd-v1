# Innovation 1 H1-A2 Representation Gradient Equalization Plan

Date: 2026-07-26

```text
status = completed / partial hold
execution = local 2048/class/source diagnostic
remote_scale = no
```

## Research Question

Can equalizing each source task's per-step representation-gradient L2 norm
remove the Dialga-dominated optimization failure identified by H1-A1 and make
RECTANGLE zero-fine-tuning topology attribution stable across both seeds?

H1-A1 found that Dialga supplied `69.9%` and `85.8%` of the frozen shared
representation-gradient norm, with raw norm ratios of `6.21x` and `18.64x`
against the median of the other source tasks. This directly authorizes one
optimization-only candidate before any architecture change.

## One Changed Variable

Original H1 combines four task losses and performs one backward pass:

```text
g_rep = mean(g_GIFT, g_SKINNY, g_uKNIT, g_Dialga)
```

H1-A2 first measures each task's representation-gradient norm, rescales every
task to the four-task mean norm, and then averages:

```text
mean_norm = mean(||g_i||_2)
g_rep = mean(g_i * mean_norm / max(||g_i||_2, eps))
```

The shared classifier keeps the original raw arithmetic-mean gradient. Adam,
learning rate and weight decay remain unchanged. This avoids conflating source
gradient equalization with classifier calibration or a new architecture.

## Frozen Protocol

```text
sources = GIFT-64 r6, SKINNY-64/64 r7, uKNIT prefix-r5, Dialga prefix-r4
holdout = RECTANGLE-80 r6
train = 2048/class/source
source validation = 1024/class/source
target evaluation = 1024/class
pairs/sample = 4
negative = encrypted random plaintexts
seeds = 0, 1
epochs = 10
batch = 256
loss = MSE
optimizer = Adam, lr 1e-4, weight decay 1e-5
checkpoint = source-only validation macro AUC
model = unchanged Runtime-E4, 442466 parameters
target training rows = 0
```

The exact H1 caches, initialization seeds, loader seeds, source tasks, target
rows and H1 completed correct checkpoints are the anchor. RECTANGLE remains
absent from training, validation-based selection, threshold selection and all
optimizer steps.

## Lean Candidate Matrix

Train only one candidate per seed. Each candidate checkpoint is evaluated three
ways without changing weights:

```text
candidate_correct
candidate_corrupted_target
candidate_no_topology_target
```

The completed H1 correct checkpoint supplies the same-budget mean-loss anchor.
No new corrupted-source or no-topology-source model is trained because A2 tests
the optimizer mechanism, while same-checkpoint target counterfactuals retain
the necessary topology attribution control.

## Advance Gate

For each seed require:

```text
candidate_correct AUC >= 0.55
candidate_correct - corrupted_target >= +0.005
candidate_correct - no_topology_target >= +0.005
candidate source macro >= H1 source macro - 0.01
```

Also require:

```text
seed0 candidate AUC >= H1 seed0 candidate AUC - 0.02
seed1 candidate AUC >= H1 seed1 candidate AUC
all checkpoints, caches, source-only selection, same-checkpoint target
counterfactuals, gradient-scale observations and zero-target-step checks pass
```

A full pass supports the first stable dual-seed RECTANGLE whole-cipher holdout
for the unchanged Runtime-E4 architecture and opens a second independent
whole-cipher holdout design. It is still local diagnostic evidence, not formal
scale or universal adaptation.

If A2 improves the failing seed's worst topology margin by at least `0.005` but
does not fully pass, retain the mechanism as partial and use the A1 stable
SKINNY/Dialga conflict to rank one conflict-removal gate. If it does not improve
that margin, stop optimizer modification and audit representation geometry.

## Blocked Actions

Do not add MoE, Adapter, FiLM, typed GNN, cipher IDs, target heads, PCGrad,
samples, epochs or remote compute in A2. Do not change H1 labels, negatives,
task weights, sampling order, checkpoint metric or validation protocol.

## Evidence-Backed Next Action

Run the two local candidate seeds. A full pass opens only a preregistered second
holdout. A partial pass permits one conflict-removal plan but no scale-up. A
failure moves to a no-training representation alignment audit.

## Completed Result

The preregistered local run completed at:

```text
outputs/local_diagnostic/i1_runtime_spn_h1_representation_gradient_equalization_a2_2048_seed0_seed1_20260726/
status = hold
decision = innovation1_runtime_spn_h1_gradient_equalization_partial
result rows = 14
history rows = 20
gradient-scale rows = 8
checkpoints = 2
parameter count = 442466
target training rows = 0
target optimizer steps = 0
protocol validation = pass
```

All thirteen source authority, A1 authority, checkpoint, parameter, gradient
combination, gradient-scale, source-only selection, same-checkpoint target
counterfactual, strict-negative and zero-target-training checks passed.

RECTANGLE zero-fine-tuning target evidence:

| Seed | H1 correct | A2 correct | A2 corrupted | A2 no topology | Correct-corrupted | Correct-no topology |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.674040 | 0.684155 | 0.629177 | 0.600483 | +0.054978 | +0.083672 |
| 1 | 0.588227 | 0.659647 | 0.609788 | 0.604813 | +0.049858 | +0.054833 |

Both seeds now pass the absolute target floor and both same-checkpoint topology
counterfactual margins. Seed 1 improved by `+0.071420` AUC and its worst target
margin improved by `+0.067841`. Seed 0 also improved target AUC by `+0.010116`.

The full preregistered gate nevertheless did not pass because seed 0 source
macro AUC changed from `0.631157` to `0.616676`, a `-0.014482` delta beyond the
allowed `-0.01` tolerance. This was not a broad collapse: GIFT and uKNIT were
nearly unchanged and Dialga fell only `0.010142`, but SKINNY fell from
`0.535900` to `0.490007`. Seed 1 showed the same SKINNY-specific cost
(`0.534819 -> 0.474448`) while GIFT and uKNIT improved by `+0.025073` and
`+0.036065`.

The applied scales verify the intended mechanism across all 160 optimizer steps
per seed. Dialga averaged only `0.706/0.720`, while GIFT, SKINNY and uKNIT were
upscaled by approximately `1.98--2.78`.

## Final Next Action

Retain representation-gradient equalization and preregister one
parameter-matched stable-conflict removal gate. The only new variable may be a
deterministic projection that removes negative pairwise components among the
already equalized representation gradients before averaging; classifier
gradients remain the raw arithmetic mean. Compare against A2 and H1, with the
same two target counterfactuals and the same source-macro retention gate.

This follow-up is justified specifically by the A1 SKINNY/Dialga negative
cosines and the A2 SKINNY-only source degradation. Do not add architecture,
cipher routing, target supervision, more data, more epochs or remote compute.

## Visual QA Result

The final `curves.svg` was rendered and inspected at `1800x1115` and
`1280x793` under `visual-qa-redraw`. Exact labels were added to both A2 and H1
source bars so the SKINNY degradation and GIFT/uKNIT changes remain readable.
Both renders passed title, axis, bar-label, legend, reference-line, overlap,
clipping and readability checks.
