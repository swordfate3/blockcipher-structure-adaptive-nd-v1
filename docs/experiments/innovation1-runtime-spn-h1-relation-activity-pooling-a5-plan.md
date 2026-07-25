# Innovation 1 H1-A5 GF(2)-Relation Activity Pooling Plan

Date: 2026-07-26

```text
status = completed / protocol invalid after identifiability re-audit
execution = local readiness then 2048/class/source diagnostic
remote_scale = no
decision = innovation1_runtime_spn_h1_relation_activity_pooling_invalid
```

## Research Question

Can a parameter-free GF(2)-relation-conditioned activity pool preserve source
class information that Runtime-E4's current mean/max/plain-activity pooling
erases, while retaining A3's dual-seed unseen RECTANGLE topology attribution?

A4 showed that A3 SKINNY closed-form probe AUC remained `0.509273/0.545830`.
The failure therefore lies in the shared pooled representation, not only the
classifier. Earlier Adapter, True FiLM and typed-relation residuals changed
message updates but left the final cell pooling unchanged. A5 changes only
that information bottleneck.

## One Changed Primitive

The current third cell summary is:

```text
activity(cell) = mean(current_cell_bits)
active_pool = sum(hidden(cell) * activity(cell)) / sum(activity(cell))
```

A5 replaces only its activity weight. For every target bit, compute from the
runtime inverse GF(2) matrix:

```text
relation_mass(bit) = incoming_edge_count(bit)
                     * nonempty_source_bit_role_count(bit)
```

Average this mass over the loaded runtime window, normalize it to global mean
one, align it through the supplied `cell_membership` and `bit_role`, and use:

```text
structured_activity(cell)
  = mean(current_cell_bits * normalized_relation_mass(cell, bit_role))
```

One-to-one P layers have one incoming edge and one source role per bit, so A5
reduces exactly to the old activity pool. General GF(2) layers expose
multi-source composition. Local signatures are cell-relabeling equivariant and
contain no cipher name, ID, width, key or global fingerprint.

The wrong-signature control maps every distinct local relation signature to
the next signature type in a deterministic cycle. It is based on signature
type rather than cell index, so global cell relabeling does not change it.

## Frozen Model And Protocol

```text
model = Runtime-E4, 442466 parameters for all pooling roles
sources = GIFT r6, SKINNY r7, uKNIT prefix-r5, Dialga prefix-r4
holdout = RECTANGLE r6, zero target training rows
train/validation/target = 2048/1024/1024 per class
pairs/sample = 4 independent ciphertext pairs
negative = encrypted random plaintexts
seeds = 0, 1
epochs = 10
batch = 256
optimizer = fixed A3 L2 equalization + fixed-order PCGrad
loss = MSE
checkpoint = source-only macro AUC
execution = local sub-medium diagnostic only
```

The A3 optimizer is frozen rather than treated as another variable. A3 is the
same-budget uniform-pooling anchor. A5 trains only one correct-pooling
checkpoint per seed. Uniform and wrong-signature roles are same-checkpoint,
no-training counterfactuals.

## Readiness Gate

Before training, require all of:

1. correct pooling is bit-exact with uniform pooling for GIFT and RECTANGLE;
2. SKINNY and uKNIT expose more than one signature type and correct differs
   from wrong-signature pooling;
3. homogeneous Dialga is allowed to remain uniform rather than receive an
   artificial cipher-specific difference;
4. correct/uniform/wrong roles have identical `442466` parameters and state
   keys, and load the exact A3 checkpoint;
5. independent relation mode forces uniform pooling;
6. correct and wrong-signature logits preserve cell-relabeling invariance on
   all five structures;
7. correct/uniform/wrong produce finite outputs and distinct SKINNY logits;
8. existing Runtime-E4, A3 and A4 protocol tests remain green.

Any failure stops A5 and permits only a focused readiness repair.

## Frozen Advance Gate

For both seeds require:

```text
RECTANGLE correct AUC >= 0.55
RECTANGLE correct - corrupted/no-topology/uniform/wrong >= +0.005
RECTANGLE correct >= A3 correct - 0.02
A5 four-source macro >= A3 four-source macro - 0.005
A5 SKINNY >= H1 SKINNY - 0.01
A5 SKINNY correct - uniform/wrong >= +0.005
A5 SKINNY+uKNIT macro correct - uniform/wrong >= +0.005
actual conflict projections >= 1
all readiness, checkpoint, cache and zero-target-step checks pass
```

A full pass opens a second independent whole-cipher holdout design. If A5
retains target attribution and improves SKINNY over A3 by at least `0.005` but
does not pass the full gate, retain only partial pooling evidence and audit the
remaining representation mode before any new architecture. Otherwise close
this pooling primitive.

Do not change the gate after results, add parameters, train control roles,
increase data or epochs, launch remote scale, train on RECTANGLE or revive
MoE/Adapter/FiLM/typed residuals as a rescue.

## Completed Run And Protocol Correction

The original readiness implementation passed and the frozen two-seed training
run completed. A post-run identifiability audit then found that the
preregistered target pooling gate contradicted the readiness invariant:

```text
RECTANGLE linear layer = one-to-one
correct pooling        = uniform pooling = shuffled pooling, bit-exact
frozen target gate     = correct - uniform/shuffled >= +0.005
```

The required target margin was therefore mathematically impossible, independent
of training quality. The corrected readiness adds
`target_pooling_controls_identifiable`; it fails only this check while all
parameter, checkpoint, cell-relabeling, finite-output and zero-target-row checks
remain valid. The completed result was re-adjudicated without training as:

```text
decision            = innovation1_runtime_spn_h1_relation_activity_pooling_invalid
protocol_valid      = false
supersedes_decision = innovation1_runtime_spn_h1_relation_activity_pooling_not_supported
target train rows   = 0
target steps        = 0
```

This is not a post-result gate relaxation and does not convert the run into a
positive result. It removes an invalid supported/not-supported claim and keeps
the original metrics as diagnostic observations only.

## Diagnostic Observations

The complete run artifacts are under:

```text
outputs/local_diagnostic/i1_runtime_spn_h1_relation_activity_pooling_a5_2048_seed0_seed1_20260726/
```

RECTANGLE retained the A3 target signal and still depended on the broader
runtime topology interventions, but—as required by one-to-one equivalence—had
exactly zero pooling-control margins:

| Seed | A5 correct | Corrupted | No topology | Uniform | Wrong signature |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.684390 | 0.629147 | 0.613319 | 0.684390 | 0.684390 |
| 1 | 0.653708 | 0.609951 | 0.606460 | 0.653708 | 0.653708 |

The source-side observations did not support a stable pooling benefit:

| Seed | A5 SKINNY | A3 SKINNY | Delta | Correct-uniform | Correct-wrong | Heterogeneous macro correct-uniform |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.501082 | 0.498333 | +0.002748 | +0.003040 | +0.014464 | +0.001502 |
| 1 | 0.465971 | 0.483876 | -0.017905 | +0.000809 | +0.003421 | +0.000446 |

Both source macro AUCs stayed within the frozen A3 retention tolerance, but
neither seed retained the H1 SKINNY floor or passed both source pooling margins.
These rows cannot rescue the invalid target gate and are not evidence for or
against unseen-cipher relation-pooling attribution.

The re-rendered `curves.svg` passed `visual-qa-redraw` pixel inspection at
`1800x1176` and `1280x836`: titles, labels, numeric annotations, legends, axes
and the protocol-invalid verdict have no overlap, clipping, missing glyphs or
structural ambiguity. The exact target-control equality remains visibly
represented rather than hidden, and `visual_qa_passed.marker` records the gate.

## Evidence-Backed Next Action

Preregister a new heterogeneous-GF(2) whole-cipher holdout before any further
training. The preferred next target is uKNIT because its runtime window exposes
14 local relation-signature types, so correct, uniform and wrong-signature
pooling are functionally distinguishable on the unseen cipher.

The next experiment must train from scratch on GIFT, SKINNY, RECTANGLE and
Dialga only, with uKNIT contributing zero training or checkpoint-selection
rows. Train a same-budget uniform-pooling anchor as well as the one correct
pooling candidate; A3 is not a valid zero-shot anchor because A3 trained on
uKNIT. Freeze `2048/class/source`, `1024/class/source` validation, pair4, seeds
0/1, 10 epochs and A3's equalized fixed-order PCGrad. Require a pre-training
symbolic/logit identifiability check on the held-out structure and use
same-checkpoint correct/uniform/wrong plus corrupted/no-topology target controls.

Do not remotely scale A5, reinterpret its invalid target gate as a model
failure, hold out another one-to-one P-layer cipher for this pooling question,
or add MoE/Adapter/FiLM/typed residuals before the heterogeneous holdout is
adjudicated.
