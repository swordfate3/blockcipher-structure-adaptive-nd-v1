# Innovation 1 K1-AT Structure-Derived Gate Shared Training

**Status:** completed / valid hold / close current gate formula
**Date:** 2026-07-29
**Run ID:** `i1_uknit_family_structure_derived_gate_k1at_2048_replica0_replica1_20260729`

## 1. Research Question

K1-AS established that one bounded `34 -> 12 -> 1` gate can read runtime S-box
and GF(2) summaries, change the compact transition path observably, preserve the
old K1-AK path when disabled, and share identical parameter geometry across
uKNIT-BC, Midori64 and Dialga-128. It did not train the gate or measure AUC.

K1-AT asks the first performance question:

> Under the exact K1-AO equal-batch protocol, does the structure-derived gate
> improve shared cross-key performance without sacrificing any cipher, and
> does the selected checkpoint prefer the correct descriptor over S-box,
> linear-layer and complete descriptor mismatches?

## 2. One Changed Variable

The K1-AO data, replicas, batch permutations, optimizer, loss, step count,
checkpoint rule and runtime cipher structures remain unchanged. Only the
transition gate changes:

```text
K1-AO: tanh(global transition bias)
K1-AT: tanh(global transition bias + shared MLP(runtime structure summary))
```

The candidate has one parameter state per replica. It receives no cipher ID,
key, label, difference, absolute cell ID, independent head, adapter, router,
MoE or expert.

## 3. Frozen Data And Training Protocol

| Cipher | Rounds | Difference | Train | Fresh/split | Input |
|---|---:|---|---:|---:|---|
| uKNIT-BC | 5 | cell11 role1, `0x0000400000000000` | 2048/class | 1024/class | 4 pairs, 512 bits |
| Midori64 | 4 | cell8 role1, `0x0000000400000000` | 2048/class | 1024/class | 4 pairs, 512 bits |
| Dialga-128 | 4 | `0x40` | 2048/class | 1024/class | 4 pairs, 1024 bits |

All negatives are encrypted random plaintexts. Replica 0 uses initialization
seed 30 and dataset seeds `3/6/0`; replica 1 uses initialization seed 31 and
dataset seeds `4/7/1` for uKNIT/Midori/Dialga respectively.

```text
epochs                         = 10
batch size                     = 64
batches/cipher/epoch           = 64
optimizer steps/epoch          = 192
optimizer steps/replica        = 1920
loss                           = MSE(sigmoid(logit), label)
optimizer                      = Adam, lr 1e-4, weight decay 1e-5
checkpoint                     = maximum minimum cross-key AUC
tie-break                      = maximum mean cross-key AUC
execution                      = local CPU diagnostic
```

The three per-cipher permutations and alternating batch order are byte-for-byte
the K1-AO formulas. Every structure summary is derived once from the runtime
descriptor before training and reused; it is not recomputed per batch.

## 4. Same-Budget Anchor

The only training anchor is K1-AO equal-loss shared training under the same
replicas and budget. Its selected cross-key AUCs are:

| Cipher | Replica 0 | Replica 1 |
|---|---:|---:|
| uKNIT-BC r5 | 0.642729 | 0.688768 |
| Midori64 r4 | 0.599349 | 0.600397 |
| Dialga-128 r4 | 0.967916 | 0.971022 |

The corresponding cross-key macro AUCs are `0.736664` and `0.753396`. Stronger
single-cipher checkpoints remain context only; they do not replace the
same-budget shared anchor in this attribution gate.

## 5. Same-Checkpoint Descriptor Controls

For each `replica x cipher x fresh split`, restore one selected K1-AT
checkpoint and keep the true encryption runtime structure fixed. Perform zero
updates and change only the 34-value gate summary:

```text
correct descriptor
full mismatch from the preregistered other cipher
S-box-only mismatch: other S-box summary + correct linear summary
linear-only mismatch: correct S-box summary + other linear summary
descriptor disabled: exact learned global-bias path of this checkpoint
```

This yields `2 x 3 x 2 x 5 = 60` rows. A mismatch is never implemented by
encrypting or evaluating with the wrong cipher structure. Descriptor-disabled
is a supporting causal control; the three explicit mismatch families carry
the preregistered descriptor-semantic gates.

## 6. Frozen Gates

Before viewing K1-AT metrics, advance requires all of the following:

```text
cross-key macro AUC improvement vs K1-AO >= +0.005 in each replica
candidate AUC - K1-AO AUC              >= -0.005 in every 12-panel comparison
correct - each mismatch AUC            >= +0.001 in at least 10/12 panels
```

For every mismatch family, its passing panels must collectively contain every
cipher, both replicas and both fresh splits. This tolerates at most two noisy
AUC panels without allowing one cipher, replica or split to disappear from the
semantic evidence.

Protocol validity additionally requires exact K1-AS/K1-AO digests, the 18
rebound datasets, two strict checkpoints, exactly 1920 optimizer steps per
replica, 60 immutable same-checkpoint rows, finite bounded gate values, onsite
runtime-summary derivation, and identical shared parameter geometry.

## 7. Decisions

- **Pass:** K1-AS gating improves both replicas, harms no panel beyond 0.005,
  and uses all three descriptor components. Next create a separate
  `65536/class/cipher` remote-readiness plan with exact disk cache/resume gates;
  this local result alone does not authorize launch or a formal claim.
- **Valid hold:** close this exact gate formula. Use its failure pattern to
  choose one bounded audit of learned gate dynamics or summary identifiability;
  do not increase pairs, data, epochs, width or seeds.
- **Protocol invalid:** repair only the failed binding and replay unchanged.

No 16-pair expansion, loss balancing, PCGrad, second candidate, cipher-specific
parameters, MoE, expert, remote job or mechanical scale-up is part of K1-AT.

## 8. Required Artifacts

Write under `outputs/local_diagnostic/<run_id>/`:

```text
preflight.json
dataset_manifest.jsonl
structure_summaries.json
results.jsonl
controls.jsonl
history.csv
comparison.csv
checkpoint_manifest.json
gate.json
validation.json
summary.json
progress.jsonl
curves.svg
visual_qa_render_report.json
visual_qa_passed.marker
```

After completion, record metrics, claim scope and an evidence-backed next
action here, refresh both recent-result indexes, run focused tests and commit
only K1-AT files.

## 9. Completed Result

The two replicas completed all ten epochs and exactly `1920` Adam updates each.
Both selected epoch 9 under the frozen minimum-cross-key-AUC checkpoint rule.
All twelve protocol checks passed: the K1-AS and K1-AO artifact digests,
eighteen cached datasets, candidate geometry, two checkpoints, 60 zero-update
control rows, correct runtime structures, immutable model state and finite
bounded gate values were exact.

Selected cross-key AUCs were:

| Cipher | K1-AT R0 | K1-AO R0 | Delta | K1-AT R1 | K1-AO R1 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| uKNIT-BC r5 | 0.635852 | 0.642729 | -0.006877 | 0.670087 | 0.688768 | -0.018682 |
| Midori64 r4 | 0.617740 | 0.599349 | +0.018391 | 0.612134 | 0.600397 | +0.011737 |
| Dialga-128 r4 | 0.971906 | 0.967916 | +0.003990 | 0.970460 | 0.971022 | -0.000562 |

The replica-level cross-key macro comparison was:

```text
replica0 K1-AT = 0.741832
replica0 K1-AO = 0.736664
improvement    = +0.005168  pass

replica1 K1-AT = 0.750894
replica1 K1-AO = 0.753396
improvement    = -0.002502  fail
```

The per-panel no-harm gate failed three uKNIT panels. Replica0 cross-key was
`-0.006877`; replica1 same-key and cross-key were `-0.016212` and `-0.018682`.
The other nine panels stayed above the frozen `-0.005` floor. This means the
new gate shifted shared capacity toward Midori but did not preserve uKNIT
consistently.

Correct-descriptor semantic margins were much weaker than K1-AS observability
suggested:

```text
correct - full mismatch   >= +0.001 in 2/12 panels
correct - S-box mismatch  >= +0.001 in 2/12 panels
correct - linear mismatch >= +0.001 in 0/12 panels
```

Only the two uKNIT replica1 panels passed the complete and S-box mismatch
thresholds. No mismatch family met the `10/12` count or all-axis coverage gate.
Most margins were at the `1e-5` to `1e-4` scale. Descriptor-disabled inference
was better than the correct descriptor in three panels, including Midori
replica1 cross-key by `0.011684`; it remains supporting evidence because it was
not a preregistered advance gate.

The learned correct-descriptor gate values also changed ordering across
replicas:

```text
replica0: uKNIT 0.157849 < Midori 0.165584 < Dialga 0.171085
replica1: Midori 0.186524 < Dialga 0.190007 < uKNIT 0.210137
```

Thus the two independent trainings did not recover one stable mapping from
runtime structure to the transition strength implied by K1-AR. The gate is
structure-sensitive in forward arithmetic, but its selected checkpoints do
not use that sensitivity as reliable descriptor semantics.

Final adjudication:

```text
status              = hold
decision            = innovation1_uknit_family_k1at_structure_gate_training_not_supported
failed protocol     = []
remote_scale        = no
16-pair expansion   = blocked
```

This is valid local `2048/class/cipher`, four-pair, two-replica diagnostic
evidence. It is not formal scale, an attack, unseen-cipher transfer,
arbitrary-SPN generalization or a SOTA comparison.

## 10. Visualization And Artifacts

Artifacts are under:

```text
outputs/local_diagnostic/
  i1_uknit_family_structure_derived_gate_k1at_2048_replica0_replica1_20260729/
```

The Chinese four-panel SVG was rendered to `2160 x 1380` pixels and inspected
through `visual-qa-redraw`. The first render exposed a legend over the Dialga
bars, a second legend over control points, and overlapping markers for close
gate values. The second render moved both legends, separated the five marker
tracks and passed overlap, clipping, title, glyph, scale, legend, export-bound
and claim-scope checks. The directory contains
`visual_qa_render_report.json` and `visual_qa_passed.marker`.

## 11. Evidence-Backed Next Action

Close the current `34 -> 12 -> 1` gate formula. K1-AT already supplied enough
same-budget evidence to reject mechanical scaling: one replica missed macro
improvement, three uKNIT panels exceeded the harm allowance, and every
descriptor mismatch family failed its semantic gate.

Run one local zero-update K1-AU summary-identifiability audit:

```text
question       = did the 34-value descriptor collapse into an unstable scalar?
anchor         = the two selected K1-AT checkpoints and correct summaries
controls       = full, S-box-only, linear-only and disabled summaries
changed item   = summary intervention only; runtime/data/checkpoint stay fixed
data           = the same 12 fresh panels, using deterministic leading rows
replicas       = 0/1
training       = 0 epochs, 0 optimizer steps
measurements   = gate-output distance, hidden-embedding rank/distance,
                 input-summary Jacobian by S-box/GF(2) component,
                 logit sensitivity and descriptor ordering agreement
execution      = local CPU audit
```

Readiness requires exact checkpoint/data/state binding, a non-collapsed hidden
representation, both S-box and GF(2) components having observable independent
sensitivity, and the learned descriptor ordering agreeing across replicas.
If the hidden representation is informative but the scalar projection
collapses it, the next candidate may replace only the scalar fusion with one
bounded multi-channel modulation. If the 34-value summary itself is not
identifiable, redesign the summary before any new training. In either outcome,
do not add pairs, samples, epochs, width, seeds, loss reweighting, PCGrad,
experts, MoE or remote scale.
