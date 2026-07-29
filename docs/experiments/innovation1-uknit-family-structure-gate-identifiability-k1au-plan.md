# Innovation 1 K1-AU Structure-Gate Identifiability Audit

**Status:** completed / pass / final scalar projection bottleneck supported
**Date:** 2026-07-29
**Run ID:** `i1_uknit_family_structure_gate_identifiability_k1au_20260729`

## 1. Research Question

K1-AT validly trained one shared `34 -> 12 -> 1` structure gate. It improved
the cross-key macro AUC of replica0 by `+0.005168` but reduced replica1 by
`-0.002502`, harmed three uKNIT panels, and passed the complete, S-box and
linear descriptor mismatch gates in only `2/12`, `2/12` and `0/12` panels.
The correct-descriptor gate ordering also reversed across replicas.

K1-AU asks where descriptor identifiability is lost:

> Are the three runtime structures already indistinguishable in the frozen
> 34-value summary, do they collapse in the shared 12-value hidden embedding,
> or does the final one-value projection discard an otherwise distinct hidden
> representation?

This is a mechanism audit, not another model candidate. It performs zero
training and cannot improve AUC.

## 2. Frozen Authority

Bind the completed K1-AT config, gate, validation, results, 60-row controls,
structure summaries and checkpoint manifest by SHA-256. Restore exactly its
two epoch-9 checkpoints and rebind the same eighteen K1-AO datasets and three
runtime descriptors through the K1-AT source loader.

```text
replicas                    = 0/1
ciphers                     = uKNIT-BC r5, Midori64 r4, Dialga-128 r4
fresh splits                = same-key and cross-key
deterministic rows/split    = first 32
training                    = 0 epochs
optimizer steps             = 0
device                      = local CPU
```

K1-AU keeps the correct encryption runtime fixed for every intervention and
changes only the precomputed gate summary.

## 3. Layerwise Measurements

For each checkpoint and true cipher descriptor, inspect the correct summary
and its preregistered complete, S-box-only and linear-only mismatches.

```text
raw layer:
  L2 distance between the two 34-value summaries

hidden layer:
  L2 distance between the two 12-value tanh embeddings
  centered rank of the three correct cipher embeddings per replica

scalar projection:
  absolute cosine between each hidden mismatch direction and the final weight
  pre-tanh scalar delta and effective-gate delta

local sensitivity:
  Jacobian of the effective gate with respect to all 34 inputs
  separate L2 norms for the 16 S-box and 18 GF(2) coordinates
  cosine between replica0 and replica1 Jacobians for each correct descriptor

output response:
  maximum and mean absolute logit delta on 32 fixed rows per fresh split
  tensor hashes before and after every zero-update evaluation
```

Across the three correct descriptors, also calculate the exact Spearman rank
correlation of effective-gate ordering between the two replicas.

## 4. Frozen Gates

The raw and hidden representation are considered preserved only if:

```text
every raw mismatch L2 distance       >= 1e-3
every hidden mismatch L2 distance    >= 1e-4
centered correct hidden rank         = 2 in both replicas
S-box and GF(2) Jacobian L2 norms    >= 1e-6 at all six correct descriptors
```

The final scalar mapping is considered stable only if:

```text
|cos(final weight, hidden mismatch)| >= 0.1 in at least 15/18 pairs
cross-replica correct gate rank rho  = 1.0
cross-replica Jacobian cosine        >= 0.5 for all three ciphers
```

These thresholds are frozen before hidden states or Jacobians are inspected.
K1-AT's already observed AUC mismatch failure remains bound evidence; K1-AU
does not retune an AUC gate.

## 5. Decisions

- **Hidden preserved, scalar unstable:** locate the bottleneck after the shared
  hidden embedding. Open one readiness-only K1-AV design that replaces the
  single scalar with bounded multi-channel modulation tied to the existing
  edge and S-box-transition paths. Do not train it until exact replay,
  geometry, equivariance and wrong-descriptor controls pass.
- **Hidden collapsed:** do not add output channels. Redesign the summary
  encoder or structure representation first, then repeat a zero-update
  identifiability gate.
- **Scalar stable despite K1-AT semantic failure:** the selected modulation
  target, rather than descriptor encoding, is the blocker. Audit which residual
  path should receive each component before implementing another candidate.
- **Protocol invalid:** repair only the failed artifact, checkpoint, runtime,
  row, state or zero-update binding and replay unchanged.

No pairs, samples, epochs, width, seeds, loss balancing, PCGrad, adapter,
router, expert, MoE or remote execution is authorized.

## 6. Required Artifacts

Write under `outputs/local_audit/<run_id>/`:

```text
preflight.json
results.jsonl
controls.jsonl
checkpoint_manifest.json
gate.json
validation.json
summary.json
progress.jsonl
curves.svg
visual_qa_render_report.json
visual_qa_passed.marker
```

After completion, record the layer at which information is lost and an
executable next action here, refresh both recent-result indexes, run focused
tests and commit only K1-AU files.

## 7. Completed Result

K1-AU completed locally with zero training and zero optimizer updates. Both
epoch-9 K1-AT checkpoints, all data/runtime bindings and every evaluated tensor
state remained immutable. Validation passed all ten protocol checks with six
result rows and thirty-six control rows.

```text
minimum raw-summary L2 distance       = 0.015625      (gate >= 0.001)
minimum hidden L2 distance            = 0.003871892   (gate >= 0.0001)
correct hidden centered rank          = 2 / 2 replicas
minimum S-box Jacobian L2             = 0.035951752
minimum GF(2) Jacobian L2             = 0.039492104
projection-aligned mismatch panels    = 18 / 18       (gate >= 15 / 18)
minimum projection |cosine|           = 0.167526022   (gate >= 0.1)
```

The cross-replica Jacobian cosine also passed for every cipher:

```text
uKNIT-BC = 0.550754428
Midori64 = 0.563403726
Dialga-128 = 0.563373685
```

The final scalar gate ordering did not pass. Replica0 ordered the correct
gates as `uKNIT < Midori < Dialga`, while replica1 ordered them as
`Midori < Dialga < uKNIT`; their Spearman rank correlation was `-0.5` against
the preregistered requirement of `1.0`.

```text
representation_preserved_through_hidden = true
final_scalar_mapping_stable             = false
status                                  = pass
decision = innovation1_uknit_family_k1au_final_scalar_projection_bottleneck_supported
remote_scale                            = no
```

This supports a narrow mechanism claim only: the frozen K1-AT gate retains
runtime structure information in its 34-value summary and 12-value hidden
embedding, but its one-value projection produces an unstable cross-replica
cipher ordering. It is not a new AUC result, formal-scale evidence, an attack,
unseen-cipher transfer evidence or a SOTA claim.

The final `2160 x 1440` rendered figure passed the `visual-qa-redraw` workflow
after two iterations. The first iteration exposed a legend/data collision and
a Jacobian annotation/bar collision; the second moved both annotations into
unused panel space. The delivered figure has no text overlap, clipping,
missing Chinese glyphs, ambiguous legend association or unreadable threshold.

## 8. Recommended Next Action: K1-AV Readiness

K1-AV will ask one question: can the same preserved runtime summary drive the
two already existing structural residual paths separately without introducing
cipher-specific parameters? The same frozen K1-AT checkpoint and six frozen
fresh-split panels are the anchor. The only mechanism change is replacing the
shared `34 -> 12 -> 1` mapping with a shared `34 -> 12 -> 2` bounded mapping:

```text
channel 1 -> existing GF(2) edge residual modulation
channel 2 -> existing S-box transition residual modulation
```

The readiness matrix is local CPU only, with zero epochs, zero optimizer steps
and deterministic fixed rows. It must include correct, full-mismatch,
S-box-only-mismatch and linear-only-mismatch descriptors while keeping the
encryption runtime correct. It must not add a cipher ID, per-cipher head,
adapter, router, expert or MoE, and parameter geometry must be identical for
uKNIT-BC, Midori64 and Dialga-128.

Readiness advances only if all source/data/checkpoint hashes are exact, disabled
modulation replays the frozen anchor logits within the existing numerical
tolerance, channel 1 responds to GF(2) descriptor changes, channel 2 responds
to S-box descriptor changes, all mismatch controls remain observable, cell
relabeling invariance holds, and all model/tensor states remain unchanged.
Failure of any binding, exact replay, geometry, sensitivity or equivariance
check blocks training and requires a readiness-only repair.

K1-AV readiness does not authorize 16 pairs, larger samples, more epochs,
additional seeds, remote GPU execution, loss reweighting, PCGrad or expert/MoE
routes. Only after readiness passes may a separate same-budget training plan
compare the two-channel candidate with K1-AT and the minimum required descriptor
controls.
