# Innovation 1 K1-AQ Fixed Inverse-Norm Shared Training

**Status:** completed / hold
**Date:** 2026-07-29
**Run ID:** `i1_uknit_family_multicipher_inverse_norm_k1aq_2048_replica0_replica1_20260729`

## 1. Research Question

K1-AP found no cross-replica stable gradient-direction conflict, but Dialga's
median all-parameter gradient norm was consistently largest and exceeded
Midori by `4.28x` and `6.02x`. K1-AQ tests the narrow causal question:

> If the three cipher losses are given fixed inverse-gradient-norm scales, can
> one shared model recover uKNIT/Midori signal without losing Dialga or the
> `12/12` correct-S-box preference?

This is not dynamic multitask optimization. It changes one frozen scalar per
cipher and replica while preserving the complete K1-AO protocol.

## 2. Same-Budget Baseline

The authoritative baseline is K1-AO:

```text
model                          = unchanged K1-AK runtime SPN
samples/class/cipher           = 2048
pairs/sample                   = 4
epochs                         = 10
batches/cipher/epoch           = 64
Adam steps/epoch               = 192
Adam steps/replica             = 1920
replicas and dataset seeds     = unchanged
batch permutations and order  = unchanged
checkpoint metric              = minimum cross-key AUC across ciphers
```

Bind and rehash the K1-AO gate and 36-row controls. Also bind and rehash the
K1-AP gate, validation and 72-row summaries that authorize normalization.

## 3. One Changed Variable

For replica `r` and cipher `c`, let `n[r,c]` be the K1-AP correct-runtime
all-parameter median gradient norm. Define:

```text
geometric_target[r] = geometric_mean_c(n[r,c])
loss_scale[r,c]     = geometric_target[r] / n[r,c]
```

Frozen scales are:

| Replica | uKNIT | Midori | Dialga | Geometric mean |
|---:|---:|---:|---:|---:|
| 0 | 0.878602 | 2.207727 | 0.515540 | 1.000000 |
| 1 | 0.901555 | 2.584651 | 0.429147 | 1.000000 |

For each unchanged sequential cipher batch, optimize:

```text
scaled_loss = loss_scale[r,c] * MSE(sigmoid(logit), label)
```

Keep one shared model, classifier, Adam optimizer, checkpoint and optimizer
state per replica. Do not combine three cipher batches into one step: doing so
would reduce the budget from 1920 to 640 steps and invalidate attribution.

## 4. Training And Evaluation

Train from the same initialization seeds `30/31` using the same per-epoch
permutations and fixed order `uKNIT -> Midori -> Dialga`. Select checkpoints by
minimum cross-key AUC, tie-broken by mean cross-key AUC.

Restore each selected checkpoint and run the unchanged 36-row panel:

```text
3 ciphers x 2 replicas x 2 fresh splits x {
  correct runtime,
  wrong S-box same checkpoint,
  transition branch off same checkpoint
}
```

All evaluation rows perform zero optimizer steps and preserve the state hash.

## 5. Frozen Gates

Compare candidate correct-runtime AUC to the matching K1-AO row.

### Advance Gate

All clauses are required:

```text
uKNIT/Midori panels with candidate-baseline >= +0.010  >= 6 of 8
every correct-runtime panel candidate-baseline         >= -0.010
correct - wrong S-box                                 >= +0.005 in 12/12
correct - branch off                                  >= +0.005 in at least 11/12
```

This gate shows that fixed normalization improves the weak tasks broadly
without buying the gain by damaging another cipher or erasing semantic use.

### Full Support Gate

In addition to the advance gate, all 12 correct-runtime panels must retain their
independent cipher anchor within `-0.010`, and the branch-off gate must pass
`12/12`. Only full support can qualify the method for a later remote-readiness
audit; K1-AQ itself always has `remote_scale = no`.

### Stop Gate

Stop fixed inverse-norm scaling when the advance gate fails. Return to the
transition representation if target gains are absent; inspect over-correction
only when Midori rises while uKNIT or Dialga falls below `-0.010`. Do not tune
the scalar values after seeing validation AUC.

## 6. Required Artifacts

Write under `outputs/local_diagnostic/<run_id>/`:

```text
preflight.json
dataset_manifest.jsonl
checkpoint_manifest.json
results.jsonl
controls.jsonl
history.csv
comparison.csv
gate.json
validation.json
summary.json
progress.jsonl
curves.svg
```

The Chinese figure must show matched K1-AO/K1-AQ deltas separately from
same-checkpoint semantic margins and pass `visual-qa-redraw`. Refresh both
recent-result indexes after completion.

## 7. Prohibited Changes

Do not add PCGrad, dynamic task weighting, MoE, cipher IDs, per-cipher heads,
adapters or experts. Do not change pairs, data, labels, negatives, differences,
keys, seeds, epochs, width, Adam, learning rate, batch order, optimizer steps,
checkpoint metric or evaluation controls. Do not launch remotely from this
local diagnostic.

## 8. Completed Result

The run completed both replicas, selected one checkpoint per replica, and
evaluated the frozen 36-row panel. All seven protocol checks passed:

```text
training rows                    = 2/2
selected checkpoints             = 2/2
Adam steps per replica           = 1920/1920
candidate control rows           = 36/36
matched K1-AO baseline rows      = 36/36
evaluation optimizer steps       = 0
state mutation during evaluation = none
```

Correct-runtime cross-key AUC changed as follows:

| Cipher | Replica | K1-AO | K1-AQ | Delta |
|---|---:|---:|---:|---:|
| uKNIT-BC r5 | 0 | 0.642729 | 0.623635 | -0.019094 |
| uKNIT-BC r5 | 1 | 0.688768 | 0.629476 | -0.059293 |
| Midori64 r4 | 0 | 0.599349 | 0.675409 | +0.076060 |
| Midori64 r4 | 1 | 0.600397 | 0.636922 | +0.036525 |
| Dialga-128 r4 | 0 | 0.967916 | 0.942913 | -0.025002 |
| Dialga-128 r4 | 1 | 0.971022 | 0.927503 | -0.043519 |

The same-key panels showed the same directional pattern. All four Midori
panels improved by `+0.036525` to `+0.076060`. uKNIT changed by `-0.005636`
to `-0.070288`, while Dialga changed by `-0.022522` to `-0.051594`.

Frozen gates:

```text
target improvement               = 4/8  (required >= 6/8)
per-panel no-harm                = 5/12 (required 12/12)
correct minus wrong S-box        = 12/12
correct minus transition-off     = 12/12
independent-anchor retention     = 2/12
advance_gate                     = false
full_support_gate                = false
status                           = hold
decision                         = innovation1_uknit_family_k1aq_inverse_norm_scaling_not_supported
remote_scale                     = no
```

The final Chinese figure was rendered to `2160 x 1320` pixels and inspected
through `visual-qa-redraw`. The second render had no text overlap, clipping,
ambiguous threshold labels, misleading axis range or incomplete legend. The
run directory records `visual_qa_render_report.json` and
`visual_qa_passed.marker`.

## 9. Interpretation

Fixed inverse-norm scaling is not a family-wide repair. It is not inert: it
substantially restores Midori and changes the branch-off result from K1-AO's
`11/12` to `12/12`. However, the same fixed intervention consistently damages
uKNIT and Dialga. The evidence supports an over-correction interpretation:
the three tasks do not need one globally shared transition strength.

This result rules out further post-hoc tuning of the K1-AP-derived scalar
weights under this protocol. It does not show that the shared forward
representation is incapable, because the per-cipher anchors already prove
that the same component class can carry useful signal. The unresolved question
is where the shared base and transition-semantic paths help or interfere for
each runtime structure.

## 10. Recommended Next Action

Run a zero-training, same-checkpoint representation-contribution audit on the
K1-AO and K1-AQ checkpoints. For every cipher, replica and fresh split, record:

```text
base-path logit
transition residual logit
fused correct-runtime logit
transition gate contribution
residual/base RMS ratio
AUC of base-only, residual-only and fused outputs
```

The only variable is the observed internal path; datasets, labels, negatives,
keys, pairs, checkpoints and runtime structures remain frozen. Compare the
same-budget K1-AO and K1-AQ checkpoints and require two-replica directional
agreement before changing the network.

If Midori alone has a consistently under-powered but label-aligned transition
residual while uKNIT/Dialga show destructive or already-saturated fusion, the
next model may test a structure-derived gate that uses runtime operator
statistics but no cipher ID. If the path decomposition is not stable across
replicas, stop that gate design and inspect projection geometry instead.

Do not continue fixed loss-scale tuning, PCGrad, `16 pairs`, larger samples,
more epochs, width, MoE, experts or remote GPU execution from this result.
