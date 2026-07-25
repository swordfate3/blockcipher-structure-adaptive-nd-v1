# Innovation 1 H1-A4 Representation Accessibility Audit Plan

Date: 2026-07-26

```text
status = completed / hold
execution = local frozen-checkpoint audit
remote_scale = no
```

## Research Question

Does A3's weak SKINNY shared-classifier AUC come from absent class information
in the shared Runtime-E4 representation, or from a fixed shared classifier that
cannot access information already present in that representation?

H1-A1 identified Dialga-dominated and SKINNY-conflicting gradients. A2 and A3
stabilized zero-fine-tuning RECTANGLE topology attribution, while A3 recovered
only `+0.008326/+0.009428` SKINNY AUC over A2 and did not pass the frozen full
gate. A3 therefore closes optimizer enumeration and authorizes this audit.

## Frozen Evidence

The audit reads exactly six source-selected checkpoints:

```text
roles = H1 mean loss, A2 L2 equalized, A3 L2 equalized + fixed-order PCGrad
seeds = 0, 1
model = unchanged Runtime-E4, 442466 parameters
source ciphers = GIFT r6, SKINNY r7, uKNIT prefix-r5, Dialga prefix-r4
split = source validation only, 1024/class/cipher
RECTANGLE rows loaded = 0
```

The config freezes every checkpoint path and SHA256, every source gate decision
and the H1 config SHA256. Checkpoints remain immutable; no neural parameter is
updated and no target cipher is loaded.

## Measurements

For every checkpoint role, seed and source cipher, extract the exact 384-wide
pre-classifier representation and record:

1. fixed shared-classifier AUC;
2. standardized class-centroid distance;
3. within-class RMS dispersion;
4. centroid separation ratio;
5. deterministic two-fold sample-out-of-fold linear-probe AUC.

The probe is a closed-form ridge solve with fixed `lambda=0.01`. Each class is
split deterministically between two folds. Standardization statistics and
ridge weights are computed on the fit fold only, then evaluated on the unseen
fold; both directions are pooled before AUC. There are no epochs, optimizer
steps, checkpoint selection or probe hyperparameter search.

A matched label-shuffle control is run only for A3-SKINNY in both seeds. It
permutes fit labels with frozen seed `260726` while preserving the true
evaluation labels. This checks that the probe does not manufacture
accessibility through leakage.

## Frozen Decision Gate

Call a shared-classifier bottleneck only if both A3 SKINNY seeds satisfy:

```text
closed-form probe AUC >= 0.55
probe AUC - fixed shared-classifier AUC >= +0.02
abs(label-shuffle probe AUC - 0.5) <= 0.05
```

If both A3 SKINNY probe AUCs remain below `0.55`, call the shared
representation weak. Any mixed result is an accessibility split and permits
only a representation-mode audit before architecture training.

## Evidence-Backed Next Action

If the classifier-bottleneck gate passes, preregister one structure-conditioned
shared readout that has no cipher ID, target head or target supervision. If the
representation is weak, change the shared structure primitive rather than the
optimizer or classifier. If mixed, inspect deterministic representation modes
before choosing either architecture route.

Do not add another optimizer treatment, MoE, cipher-specific expert, Adapter,
FiLM, typed GNN, samples, epochs, RECTANGLE supervision or remote compute in
this audit.

## Completed Result

The frozen audit completed at:

```text
outputs/local_audit/i1_runtime_spn_h1_representation_accessibility_a4_seed0_seed1_20260726/
status = hold
decision = innovation1_runtime_spn_h1_shared_representation_weak
metric rows = 24
label-shuffle controls = 2
checkpoint hashes = 6/6 exact
source validation caches = 24/24 valid
neural optimizer steps = 0
RECTANGLE rows loaded = 0
protocol validation = pass
```

A3 SKINNY supplied the primary decision evidence:

| Seed | Frozen shared head | Closed-form probe | Probe gain | Label-shuffle probe | Centroid separation ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.498333 | 0.509273 | +0.010939 | 0.502095 | 0.046452 |
| 1 | 0.483876 | 0.545830 | +0.061954 | 0.506388 | 0.060498 |

Both label-shuffle controls remained inside the frozen `0.50 +/- 0.05` band,
so the probe path did not expose a fitted-label or fold leak. Seed 1 showed
some classifier-accessibility gap, but both true probe AUCs remained below the
preregistered `0.55` representation floor. The classifier-bottleneck gate
therefore did not pass.

The complete SKINNY checkpoint history was:

| Seed | Checkpoint | Shared head | Closed-form probe | Centroid separation ratio |
| --- | --- | ---: | ---: | ---: |
| 0 | H1 | 0.535900 | 0.496176 | 0.061184 |
| 0 | A2 | 0.490007 | 0.503647 | 0.045953 |
| 0 | A3 | 0.498333 | 0.509273 | 0.046452 |
| 1 | H1 | 0.534819 | 0.518745 | 0.083283 |
| 1 | A2 | 0.474448 | 0.518982 | 0.059788 |
| 1 | A3 | 0.483876 | 0.545830 | 0.060498 |

A3 recovered some sample-out-of-fold accessibility relative to A2, especially
for seed 1, but did not restore stable class separation. The same A3 probe did
not automatically improve every source task: GIFT, uKNIT and Dialga remained
close to their frozen shared-head AUCs. This supports the intended diagnostic
interpretation rather than a generic stronger-head explanation.

## Final Next Action

Close optimizer balancing and larger-classifier routes. The next candidate
must change one shared representation primitive before invariant pooling:
preserve GF(2)-relation-conditioned cell statistics that the current
mean/max/activity pooling erases. Use fixed runtime structure descriptors and
shared projection weights only; do not introduce cipher IDs, target heads or
cipher-specific experts.

The next local readiness and same-budget gate must retain Runtime-E4, H1 source
data, strict negatives, four-pair input, seeds, epochs and RECTANGLE zero-row
holdout. Compare one relation-conditioned pooling candidate against A3 and the
necessary corrupted/no-topology same-checkpoint controls. Do not launch remote
scale unless that local gate first demonstrates dual-seed source retention and
unseen-target topology attribution.

## Visual QA

The final `curves.svg` was rendered and inspected at `1800x1191` and
`1280x847` under `visual-qa-redraw`. Both views passed title, subtitle, axes,
bar labels, legends, 0.50/0.55 reference lines, Chinese glyphs, overlap,
clipping and readability checks without redraw.
