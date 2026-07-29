# Innovation 1 K1-BB Position-Preserving Operator Readiness

**Status:** completed / pass / K1-BC local training authorized
**Date:** 2026-07-29
**Run ID:** `i1_uknit_family_position_preserving_operator_k1bb_readiness_20260729`

## 1. Research Question

K1-BA constructed different invertible GF(2) operators whose matrix bits changed
by `4.59%-9.18%` while K1-AZ's 18-value linear summaries remained bitwise
identical. Consequently the edge gate, probability hashes and AUCs were also
identical in all twelve frozen panels.

K1-BB asks one narrower question before any new training:

> Can a shared encoder retain the actual directed GF(2) relation through
> sample-conditioned edge messaging, separate every K1-BA same-summary
> corruption, and leave the completed K1-AZ model exactly unchanged when the
> new path is disabled?

This is a local CPU readiness gate with zero epochs and zero optimizer steps.
It cannot improve AUC or support a family-transfer claim by itself.

## 2. One Representation Change

K1-AZ's edge gate currently consumes 18 invariant statistics. K1-BB introduces
one token for every nonzero entry of each runtime inverse linear matrix:

```text
normalized/sinusoidal transition position       3
normalized/sinusoidal source cell position      3
normalized/sinusoidal target cell position      3
source bit role, four-way one-hot                4
target bit role, four-way one-hot                4
actual nonzero GF(2) relation                    1
                                                --
token width                                     18
```

The same token MLP is used for every edge, transition, cipher and block width.
There is no cipher ID, learned position table, per-cipher head, adapter, router,
expert or MoE.

The important ordering constraint is:

```text
operator edge token
  -> interact with that edge's source/target sample states
  -> aggregate messages at the actual target bit
  -> update bit states in transition order
  -> only then pool to a fixed 384-value modulation
```

This differs from K1-B, K1-H and K1-K. Those routes either appended endpoint
numbers to tokens that were subsequently pooled, or transported operator
support before an invariant bit/cell pool. K1-BB does not first reduce the
operator to a scalar or globally pooled structure vector before it meets the
sample representation.

## 3. Frozen Authority

K1-BB binds the completed K1-BA audit and, transitively, K1-AZ's two epoch-9
checkpoints, eighteen disk-backed datasets and twelve fresh panels:

```text
ciphers          = uKNIT-BC r5, Midori64 r4, Dialga-128 r4
replicas         = 0/1
fresh splits     = same-key / cross-key
probe rows       = first 64 frozen rows per panel
underlying data  = 1024/class per fresh split
pairs per sample = 4
negative mode    = encrypted random plaintexts
training         = 0 epochs / 0 optimizer steps
device           = local CPU
```

The K1-AZ base encoder, exact GF(2) edge residual, S-box transition residual,
classifier, checkpoint tensors, datasets and correct encryption runtime remain
frozen. The only new learned geometry is the shared operator encoder.

## 4. Required Controls

For every replica, cipher and split, evaluate:

| Condition | Role |
|---|---|
| correct operator | native source-to-target GF(2) edges |
| same-summary corrupted operator | K1-BA column-permuted invertible operator |
| cross-cipher operator | proves one shared fixed-width structure embedding accepts another family member |
| disabled new path | must exactly replay K1-AZ logits |
| jointly relabeled cells | transport cell identities with data and conjugated operator; output must be equivariant |

The corrupted operator changes only the structure supplied to the new
modulation path. Ciphertext generation and the K1-AZ runtime structure remain
correct. Cross-width cross-cipher controls are applied to the fixed-width
operator embedding, not as an invalid 64-bit operator on 128-bit samples.

## 5. Frozen Readiness Gates

All source, config, checkpoint and dataset bindings must be exact. Both
replicas and every cipher/split panel must satisfy:

```text
correct vs same-summary operator embedding max delta >= 1e-4
correct vs cross-cipher operator embedding max delta  >= 1e-4
correct vs same-summary sample modulation max delta   >= 1e-6
correct vs same-summary enabled logit max delta        >= 1e-6
disabled K1-BB vs K1-AZ logit max delta                = 0.0
```

The trainable parameter names and shapes must be identical for uKNIT-BC,
Midori64 and Dialga-128. A joint complete-cell relabeling transports the
original native position IDs with the permuted data and conjugated operator;
the structure embedding must replay within `1e-6`, and modulation/logits within
`1e-5`. This distinguishes semantic position from an accidental tensor index.

Every state dictionary must remain immutable. Any optimizer construction,
training row or parameter update invalidates readiness.

## 6. Decisions

- **All gates pass:** authorize a separate K1-BC same-budget local training
  plan. Compare one K1-BB candidate against K1-AZ, same-summary corrupted
  operator and cross-cipher mismatch controls at the existing
  `2048/class/cipher`, four-pair, two-replica, ten-epoch protocol.
- **Operator embedding separates but sample modulation/logit does not:** repair
  only the consumer connection; do not train or change token semantics.
- **Same-summary operators still collide:** reject this encoder and inspect the
  endpoint relation map; do not add width or data.
- **Disabled replay or equivariance fails:** repair only compatibility or
  transported-position handling and rerun unchanged.
- **Protocol invalid:** repair the failed source/checkpoint/data binding.

No outcome authorizes remote execution, 16 pairs, more samples, epochs, seeds
or width, a benchmark change, loss balancing, per-cipher modules or MoE.

## 7. Required Artifacts

Write under `outputs/local_readiness/<run_id>/`:

```text
preflight.json
dataset_manifest.jsonl
operator_controls.json
geometry.json
results.jsonl
checkpoint_manifest.json
gate.json
validation.json
summary.json
progress.jsonl
curves.svg
visual_qa_render_report.json
visual_qa_passed.marker
```

After completion, append exact metrics, claim scope and the next executable
action to this document, refresh both recent-result indexes, run focused tests,
commit the scoped K1-BB files and push the exact commit through the configured
GitHub workflow.

## 8. Completed Readiness Result

K1-BB completed all twelve frozen probe panels with zero training and zero
optimizer steps:

```text
status   = pass
decision = innovation1_uknit_family_k1bb_position_preserving_operator_readiness_authorized
result rows              = 12/12
operator control rows    = 6/6
failed protocol checks   = []
failed panel checks      = []
trainable parameters     = 41088
```

The new shared parameter geometry is identical on uKNIT-BC and Midori64's
64-bit states and Dialga-128's 128-bit state. It processes `384` actual inverse
GF(2) edges for each 64-bit two-transition window and `768` edges for the
128-bit window. No cipher-specific parameter or identity input exists.

Every replica/cipher/split panel separated the K1-BA same-summary corrupted
operator:

```text
minimum correct-vs-corrupted operator embedding delta = 0.300442994
minimum correct-vs-cross-cipher embedding delta       = 0.258015692
minimum correct-vs-corrupted edge modulation delta    = 0.054098368
minimum correct-vs-corrupted enabled logit delta      = 0.0003859997
```

The compatibility and equivariance controls also passed:

```text
maximum disabled K1-AZ replay delta          = 0.0
maximum joint-relabel embedding delta        = 4.7683716e-7
maximum joint-relabel sample modulation delta = 1.4305115e-6
maximum joint-relabel logit delta             = 1.4305115e-6
```

This resolves K1-BA's representational readiness question. Unlike the old
18-value statistics, distinct source-to-target operators now reach different
sample-conditioned representations and different frozen logits. It does not
show that gradient training will prefer the correct topology or improve AUC.

## 9. Next Executable Action: K1-BC

Preregister one same-budget local training matrix before constructing an
optimizer:

```text
question        = can learned K1-BB modulation improve or retain K1-AZ while
                  preferring correct actual topology?
anchor          = completed K1-AZ, same replica/data/checkpoint protocol
one variable    = replace the 18-statistic edge gate input with K1-BB's
                  per-edge sample-conditioned operator modulation
ciphers/rounds  = uKNIT-BC r5, Midori64 r4, Dialga-128 r4
train           = 2048/class/cipher
fresh           = 1024/class/cipher per same-key/cross-key split
pairs           = 4
replicas        = 0/1 with the frozen K1-AZ dataset seeds
epochs/batch    = 10 / 64
execution       = local CPU diagnostic
controls        = correct, same-summary corrupted, cross-cipher mismatch,
                  disabled exact K1-AZ compatibility
```

K1-BC must require nonnegative cross-key macro improvement per replica, no
fresh panel worse than K1-AZ by more than `0.005`, and correct-topology margins
of at least `0.001` on a preregistered majority of panels. If training improves
macro AUC but wrong topology wins, hold the architecture and audit optimization
attribution. If the candidate harms a cipher or fails topology controls, do not
increase pairs, samples, epochs, width or launch remote training.

## 10. Artifacts And Visual QA

Artifacts are under:

```text
outputs/local_readiness/i1_uknit_family_position_preserving_operator_k1bb_readiness_20260729/
```

The Chinese `curves.svg` was rendered to `2700 x 1800` pixels and inspected
through `visual-qa-redraw`. Its title, four panels, threshold lines, bar labels,
legends, logarithmic axes and next-action caption have no overlap, clipping,
missing glyphs or ambiguous scales. The pass is recorded in
`visual_qa_render_report.json` and `visual_qa_passed.marker`.
