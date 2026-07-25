# Innovation 1 H1-A3 Equalized Fixed-Order PCGrad Plan

Date: 2026-07-26

```text
status = completed / partial hold
execution = local 2048/class/source diagnostic
remote_scale = no
```

## Research Question

Can deterministic negative-gradient projection preserve A2's dual-seed unseen
RECTANGLE topology attribution while repairing the SKINNY source degradation
caused by the stable SKINNY/Dialga conflict?

A1 found SKINNY/Dialga representation-gradient cosines of `-0.039/-0.148`.
A2 successfully made both RECTANGLE seeds topology-attributed, but SKINNY AUC
fell from `0.535900/0.534819` to `0.490007/0.474448`. This is a specific
conflict-removal hypothesis, not authorization for a broader optimizer search.

## One Changed Variable

A3 retains A2's per-task representation-gradient L2 equalization. For each
source task in the fixed order GIFT, SKINNY, uKNIT, Dialga, it then projects the
current equalized representation gradient against every other original
equalized task gradient when their dot product is negative:

```text
if dot(g_i, g_j) < 0:
    g_i = g_i - dot(g_i, g_j) / ||g_j||^2 * g_j
```

The fixed source order removes PCGrad's random-order variable. The shared
classifier still receives the original raw arithmetic-mean gradient.

## Frozen Protocol

Everything else is byte-for-byte or value-for-value aligned with H1/A2:

```text
model = Runtime-E4, 442466 parameters
sources = GIFT r6, SKINNY r7, uKNIT prefix-r5, Dialga prefix-r4
holdout = RECTANGLE r6
train/validation/target = 2048/1024/1024 per class
pairs = 4
strict negative = encrypted random plaintexts
seeds = 0, 1
epochs = 10
batch = 256
MSE + Adam, lr 1e-4, weight decay 1e-5
checkpoint = source-only macro AUC
target training rows and optimizer steps = 0
```

Train only the two A3 candidate checkpoints. Evaluate correct, corrupted-target
and no-topology-target with each identical checkpoint. H1 and A2 are frozen
same-budget anchors.

## Advance Gate

For both seeds require:

```text
A3 correct target AUC >= 0.55
A3 correct - corrupted target >= +0.005
A3 correct - no topology target >= +0.005
A3 correct target >= A2 correct target - 0.02
A3 source macro >= H1 source macro - 0.01
A3 SKINNY AUC >= A2 SKINNY AUC + 0.01
at least one actual conflict projection per seed
all zero-leakage and source-only checkpoint checks pass
```

A full pass opens a second independent whole-cipher holdout design. A SKINNY
improvement of at least `0.005` without a full pass is partial mechanism
evidence only. No improvement closes optimizer changes and moves to a
no-training representation alignment audit.

## Blocked Actions

Do not add architecture modules, learnable task weights, random projection
order, cipher IDs, target heads, samples, epochs or remote compute. Do not
change the H1/A2 gate after observing A3.

## Evidence-Backed Next Action

Run the two local candidate seeds. Advance only according to the frozen gate;
under no outcome may A3 itself be scaled remotely or called formal evidence.

## Completed Result

The preregistered run completed at:

```text
outputs/local_diagnostic/i1_runtime_spn_h1_equalized_pcgrad_a3_2048_seed0_seed1_20260726/
status = hold
decision = innovation1_runtime_spn_h1_equalized_pcgrad_partial
result rows = 14
history rows = 20
checkpoints = 2
parameter count = 442466
target training rows = 0
target optimizer steps = 0
protocol validation = pass
```

RECTANGLE zero-fine-tuning target evidence was:

| Seed | H1 correct | A2 correct | A3 correct | A3 corrupted | A3 no topology | Correct-corrupted | Correct-no topology |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.674040 | 0.684155 | 0.690377 | 0.630342 | 0.609710 | +0.060036 | +0.080667 |
| 1 | 0.588227 | 0.659647 | 0.660227 | 0.610850 | 0.616998 | +0.049376 | +0.043229 |

Both seeds retained the A2 unseen-target signal, passed the absolute target
floor and passed both same-checkpoint topology counterfactual margins. A3
improved target AUC over A2 by `+0.006222` for seed 0 and `+0.000580` for seed
1. The candidate checkpoint recorded `637/636` conflict projections, proving
that the changed mechanism was exercised in both seeds.

The source result did not pass the frozen full gate:

| Seed | A3 source macro | H1 source macro | Delta vs H1 | A3 SKINNY | A2 SKINNY | SKINNY delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.619280 | 0.631157 | -0.011878 | 0.498333 | 0.490007 | +0.008326 |
| 1 | 0.606625 | 0.609725 | -0.003100 | 0.483876 | 0.474448 | +0.009428 |

Seed 0 missed the source-macro retention floor by `0.001878`, and both seeds
missed the preregistered `+0.010` SKINNY recovery requirement. Both SKINNY
deltas exceed the frozen `+0.005` partial-mechanism threshold, so this is
partial evidence that negative-gradient projection repairs some of A2's
SKINNY-specific damage. The gate was not relaxed after observing the result.

This remains a local sub-medium optimizer diagnostic. It is not formal-scale
evidence, a universality result, an attack result or a SOTA comparison.

## Final Next Action

Stop optimizer enumeration and perform one no-training representation
accessibility audit using the frozen H1, A2 and A3 checkpoints. For each seed
and source cipher, measure class-centroid separation, within-class dispersion,
linear-probe accessibility and the fixed shared-classifier AUC on the same
parameter-matched validation rows. The one question is whether SKINNY signal is
absent from the shared representation or present but inaccessible to the
shared classifier.

If A3 restores SKINNY representation separation but not shared-classifier AUC,
the next architecture hypothesis may target a structure-conditioned readout
without cipher IDs or target supervision. If separation itself remains weak,
the next design must change the shared representation primitive. Do not add
another optimizer, increase samples or epochs, launch remote training, train on
RECTANGLE, or revive MoE/Adapter/FiLM/typed GNN before this audit resolves that
decision.

## Visual QA

The final `curves.svg` was rendered and inspected at `1800x1184` and
`1280x842` under `visual-qa-redraw`. Both views passed title, subtitle, axes,
bar-label, legend, reference-line, Chinese-glyph, overlap, clipping and
readability checks without redraw.
