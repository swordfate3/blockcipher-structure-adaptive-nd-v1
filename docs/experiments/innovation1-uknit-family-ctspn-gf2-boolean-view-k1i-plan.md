# Innovation 1 uKNIT-Family CT-SPN Exact GF(2) Boolean View K1-I

**Date:** 2026-07-28
**Status:** completed / held before scale
**Execution:** local CPU mechanism diagnostic only

## 1. Evidence And Research Question

K1-G proved that the learned cell/path hypergraph could bind to concrete uKNIT
ciphertext rows: training AUC reached `0.797085/0.701712`, but fresh plaintexts
under the same key fell to `0.493531/0.497610`. K1-H removed learned path identity
and fixed transport to the exact runtime inverse-operator support, but ordinary
real-valued source means left uKNIT fresh-same-key AUC at
`0.502423/0.500109`. More importantly, the same bottleneck collapsed Dialga's
frozen `0.96-0.97` signal to `0.52-0.57`.

K1-H therefore did not test an exact Boolean execution of the supplied operator.
It tested exact routing support followed by a learned real-valued aggregation.
K1-I asks one narrower question:

> If each runtime binary matrix deterministically executes its declared XORs on
> the ciphertext-pair bits before any learned layer, can one width-independent
> shared model retain Dialga's calibrated signal and establish correct-operator
> attribution on fresh uKNIT plaintexts?

## 2. Route Ranking

1. **Exact GF(2) Boolean operator views:** selected. It directly tests the
   cancellation semantics missing from K1-H while keeping learned capacity low.
2. **Add a raw ResNet/MLP bypass:** blocked. It could restore Dialga while ignoring
   every runtime operator and would weaken the same-checkpoint attribution test.
3. **Add uKNIT S-box semantics:** deferred. K1-I first isolates whether exact
   linear Boolean execution repairs the calibrated Dialga loss. S-box/operator
   composition becomes eligible only after this variable is adjudicated.
4. **More data, width, epochs, pairs, seeds, experts or remote compute:** blocked.
   K1-H failed a mechanism calibration, not a sample-size slope gate.

## 3. Frozen Boolean Representation

For each pair `(C, C')`, first form three exact bit channels:

```text
X = [C, C', C xor C']
```

Let the two loaded runtime inverse matrices in forward descriptor order be
`L0, L1`. K1-H consumed them from the ciphertext side in reverse order. K1-I
freezes the corresponding Boolean views as:

```text
raw       = X
single_0  = L0 * X mod 2
single_1  = L1 * X mod 2
composed  = L0 * (L1 * X mod 2) mod 2
```

The per-bit learned input therefore has exactly `4 views * 3 channels = 12`
Boolean values. The composition order is part of the protocol and must not be
selected after training. Every multiplication is over GF(2), including exact
XOR cancellation. In particular, each transformed difference channel must equal
the XOR of its transformed `C` and `C'` channels bit-for-bit.

The learned path is:

```text
12 Boolean channels per runtime bit
  -> one shared bit encoder, identical at every bit and state width
  -> invariant mean/max/RMS pooling over bits and runtime cells
  -> pair projection
  -> unchanged four-pair attention plus mean/max pair pooling
  -> binary distinguisher head
```

The raw view is not a separate bypass branch: it enters the same shared bit
encoder and pooled head as every operator-derived view. There is no learned
message passing or path token.

Constraints:

- hidden width `32`, pair embedding width `128`, dropout `0`;
- total learned parameters `<= 150000`;
- 64- and 128-bit instances have identical state-dict keys and tensor shapes;
- no bit/cell/round embedding, absolute position, cipher ID, key, learned router,
  path token, inverse S-box, S-box truth table, DDT, trail or partial decryption;
- operators are non-trainable runtime data and do not enter the state dict;
- joint bit/cell relabeling of input and operators preserves logits within
  `1e-6`;
- changing either loaded operator changes at least one deterministic Boolean
  view and a fixed-weight logit on a nondegenerate fixture.

## 4. Same-Budget Anchor And Frozen Data Protocol

Reuse the exact K1-H/K1-G caches and frozen Runtime-E4 checkpoints. No row,
sample order, key, label, negative definition, checkpoint or metric may be
regenerated or reselected.

| Field | uKNIT-BC | Dialga-128 |
|---|---:|---:|
| Prefix rounds | 5 | 4 |
| Runtime transitions | descriptor rounds 3-4 | descriptor rounds 2-3 |
| Train | `2048/class` = 4096 total | `2048/class` = 4096 total |
| Fresh-same-key holdout | `1024/class` = 2048 total | `1024/class` = 2048 total |
| Cross-key validation | `1024/class` = 2048 total | `1024/class` = 2048 total |
| Seeds | 0, 1 | 0, 1 |
| Pairs/sample | 4 | 4 |
| Epochs | 10 | 10 |
| Batch size | 64 | 64 |
| Loss / optimizer | MSE / Adam | MSE / Adam |
| Learning rate / weight decay | `1e-4` / `1e-5` | `1e-4` / `1e-5` |
| Checkpoint | best cross-key validation AUC | same |
| Positive | fixed input difference `0x40` | same |
| Negative | encrypted random plaintexts | same |

The same-budget neural anchor is the exact frozen Runtime-E4 checkpoint used by
K1-H, evaluated without retraining on all three cached splits.

## 5. Same-Checkpoint Operator Controls

Each trained K1-I checkpoint is strict-loaded without optimizer steps under:

```text
exact_ordered       = exact L0, L1 and exact L0(L1X) composition
operator_reversed   = exact operator multiset bound as L1, L0
operator_corrupted  = deterministic nonidentity source permutation per matrix
no_topology         = identity matrices in both slots
```

Controls change only non-trainable matrices and their derived Boolean views.
They retain identical learned state, data, labels, split order and classifier
parameters. The corrupted matrices remain binary, square, invertible and have
nonzero target degree.

## 6. Zero-Training Readiness Gate

Before the first optimizer step require:

1. exact K1-G and K1-H source decisions and artifact hashes;
2. four candidate tasks, four frozen Runtime-E4 anchors and all twelve caches
   match cipher, rounds, seed, keys, data and optimizer protocol;
3. every cache is reused by exact digest, with no regeneration;
4. every runtime matrix is binary, invertible and descriptor-bound;
5. scalar reference XOR and vectorized GF(2) views agree on zero, unit,
   multi-source and cancellation fixtures for both state widths;
6. transformed differences equal transformed-left XOR transformed-right for all
   four views;
7. composed views equal sequential `L1` then `L0` scalar execution and differ
   from at least one single-transition view on each real descriptor;
8. the earlier and later operators are both consumed: changing either changes
   deterministic views and fixed-weight logits;
9. correct and all controls strict-load one state dict while retaining distinct
   operator/view fingerprints and logits;
10. learned parameter geometry is identical for 64-/128-bit states and remains
    within the cap;
11. joint runtime relabeling preserves views and logits within `1e-6`;
12. no forbidden metadata or standalone raw bypass enters a learned layer;
13. readiness consumes zero dataset training rows and optimizer steps.

Any failure permits only repair of the failed invariant. It does not authorize
training, protocol changes or a fallback network.

## 7. Frozen Research Gate

For each uKNIT seed, require on both fresh-same-key and cross-key holdouts:

```text
candidate AUC >= 0.520
candidate - frozen Runtime-E4 anchor >= +0.005
candidate - every same-checkpoint control >= +0.005
```

For each Dialga seed, require on both holdouts:

```text
candidate >= frozen Runtime-E4 anchor - 0.005
candidate - every same-checkpoint control >= +0.005
```

The original training split must also beat every control by `>= +0.005`.
Training success cannot compensate for either holdout. No cipher, seed, split,
anchor or control may be hidden by averaging.

## 8. Decisions

- **All gates pass:** retain exact Boolean views and plan a separate remote
  `65536/class` diagnostic with parameter-matched disk caches before any larger
  scale.
- **Dialga anchor retained, uKNIT fails:** exact linear execution is viable but
  insufficient for heterogeneous uKNIT; next audit exact S-box/operator
  composition locally, without scaling K1-I.
- **Dialga anchor collapses:** reject this invariant Boolean-view bottleneck and
  audit which Runtime-E4 position/cell interactions carry the calibrated signal.
- **Absolute AUC passes but controls fail:** reject the structural interpretation.
- **Protocol invalid:** repair and rerun unchanged under the same plan.

This is a local mechanism diagnostic, not formal training, an attack, SOTA,
arbitrary-SPN transfer or uKNIT-ceiling evidence.

## 9. Run IDs And Required Artifacts

```text
readiness = i1_uknit_family_ctspn_gf2_boolean_view_k1i_readiness_20260728
diagnostic = i1_uknit_family_ctspn_gf2_boolean_view_k1i_2048_seed0_seed1_20260728
```

Readiness requires `preflight.json`, `results.jsonl`, `validation.json`,
`gate.json` and `progress.jsonl`. The diagnostic additionally requires
`controls.jsonl`, `split_attribution.csv`, `checkpoint_manifest.json`,
`history.csv`, `summary.json`, `curves.svg`, `plot_report.json` and a
`visual_qa_passed.marker` after rendered-pixel inspection.

Every completed run must refresh `outputs/00_RECENT_RESULTS.md` and JSON. The
completed experiment record must contain exact metrics, claim scope and the
evidence-backed next action.

## 10. Readiness Result

The frozen zero-training readiness completed locally on 2026-07-28:

```text
run_id   = i1_uknit_family_ctspn_gf2_boolean_view_k1i_readiness_20260728
status   = pass
decision = innovation1_uknit_family_ctspn_k1i_execution_authorized
training rows = 0
optimizer steps = 0
failed protocol checks = []
failed evidence checks = []
```

All twelve K1-G caches and four Runtime-E4 anchors were digest-bound. Both real
descriptors passed scalar/vectorized GF(2) agreement, transformed-difference
identity, frozen composition order, binary/invertible binding, cross-width
strict state loading and joint relabeling invariance. The learned parameter
count is `94754` for both state widths.

Replacing only the earlier or later operator changed fixed-weight logits by:

| Cipher | Earlier operator | Later operator | Relabel max error |
|---|---:|---:|---:|
| uKNIT-BC | `0.052836` | `0.067777` | `1.19e-7` |
| Dialga-128 | `0.030172` | `0.075081` | `1.19e-7` |

Every reversed, corrupted and identity control strict-loaded the same state,
retained a distinct Boolean-view fingerprint and changed fixed-weight logits.
The exact four-row local diagnostic is therefore authorized without changing
the preregistered protocol.

## 11. Completed Diagnostic Result

The frozen diagnostic completed locally on 2026-07-28. It produced four
training rows and sixty zero-step evaluation rows while reusing all twelve
parameter-matched caches without regeneration:

```text
run_id   = i1_uknit_family_ctspn_gf2_boolean_view_k1i_2048_seed0_seed1_20260728
status   = hold
decision = innovation1_uknit_family_ctspn_k1i_dialga_signal_recovered_operator_attribution_not_supported
failed protocol checks = []
```

The uKNIT rows were:

| Seed | Split | Candidate | Runtime-E4 anchor | Reversed | Corrupted | No topology | Weakest control margin |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | train seen | `0.529811` | `0.624762` | `0.505612` | `0.502591` | `0.518929` | `+0.010882` |
| 0 | same-key fresh | `0.508171` | `0.503880` | `0.514278` | `0.506203` | `0.502133` | `-0.006107` |
| 0 | cross-key | `0.513726` | `0.526651` | `0.494117` | `0.495021` | `0.476428` | `+0.018704` |
| 1 | train seen | `0.545691` | `0.556674` | `0.500835` | `0.499526` | `0.501297` | `+0.044394` |
| 1 | same-key fresh | `0.495883` | `0.514069` | `0.492994` | `0.477796` | `0.514840` | `-0.018957` |
| 1 | cross-key | `0.504108` | `0.528809` | `0.475343` | `0.481297` | `0.484936` | `+0.019172` |

Both cross-key rows exceeded every same-checkpoint control, but neither reached
the preregistered `0.520` floor or retained the Runtime-E4 anchor. Both
same-key-fresh rows remained near chance and failed at least one control. The
experiment therefore provides no uKNIT generalization evidence.

The Dialga calibration rows were:

| Seed | Split | Candidate | Runtime-E4 anchor | Reversed | Corrupted | No topology | Weakest control margin |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | train seen | `0.962183` | `0.971672` | `0.958871` | `0.956340` | `0.527543` | `+0.003312` |
| 0 | same-key fresh | `0.965361` | `0.969792` | `0.963014` | `0.963330` | `0.530279` | `+0.002030` |
| 0 | cross-key | `0.958006` | `0.961386` | `0.959939` | `0.958119` | `0.522653` | `-0.001933` |
| 1 | train seen | `0.961957` | `0.970026` | `0.954565` | `0.948838` | `0.520128` | `+0.007391` |
| 1 | same-key fresh | `0.958613` | `0.962512` | `0.954698` | `0.952209` | `0.531655` | `+0.003916` |
| 1 | cross-key | `0.954252` | `0.960392` | `0.951816` | `0.947887` | `0.518332` | `+0.002437` |

Exact GF(2) views restored Dialga from K1-H's roughly `0.52-0.57` to
`0.954-0.965`. Relative to no topology, the recovered advantage was
`+0.426958` to `+0.441828`. This is strong evidence that deterministic XOR
execution preserves the signal discarded by real-valued source means.

It is not evidence that the model attributed that signal to the correct
operator. Reversed or corrupted operators also reached approximately
`0.948-0.960`; five of six Dialga rows missed the frozen `+0.005` control
margin, and the seed-0 cross-key candidate was below the reversed control.
Absolute Dialga quality therefore cannot override the failed attribution gate.

The rendered chart passed `visual-qa-redraw` after separating Dialga's small
wrong-operator margins onto the left axis and its approximately `+0.43`
no-topology advantage onto an explicit right axis. No text overlap, clipping,
missing glyph, ambiguous title or unreadable curve compression remained.

## 12. Claim Scope And Decision

K1-I establishes one reusable primitive:

```text
exact runtime GF(2) Boolean views preserve a strong Dialga signal
```

It does not establish any of the following:

- uKNIT generalization;
- stable preference for the correct runtime matrices;
- arbitrary-SPN transfer;
- a formal-scale result, cryptanalytic attack or SOTA comparison;
- an uKNIT architecture ceiling.

This is only a two-seed local `2048/class` mechanism diagnostic with
`1024/class` fresh-same-key and cross-key evaluation splits.

## 13. Evidence-Backed Next Action: K1-J

The next question is whether K1-I's invariant bit/cell pooling erased the
position and cell interactions that make Runtime-E4's Dialga checkpoint
operator-sensitive. K1-J will change only the evaluation-time position/cell
interaction mask while keeping the trained checkpoints, Boolean views, data,
labels and metrics frozen.

Executable protocol:

- same-budget anchor: the four frozen Runtime-E4 Dialga checkpoints and the two
  trained K1-I Dialga checkpoints;
- required controls: exact matrices, reversed matrices, corrupted matrices and
  identity/no-topology, all strict-loaded with no optimizer step;
- one variable: deterministic bit/cell position masking, within-cell
  permutation, across-cell permutation and interaction-preserving joint
  relabeling;
- data: the existing Dialga seed-0/1 `2048/class` train,
  `1024/class` same-key-fresh and `1024/class` cross-key caches;
- scale: zero new rows, zero epochs and zero optimizer steps, local CPU only;
- readiness: checkpoint hashes, cache hashes, exact row counts and zero-step
  state equality must pass before the audit;
- advance gate: the same position/cell interaction family must explain at least
  `80%` of the candidate-minus-no-topology gap on both fresh splits and both
  seeds, while joint input/operator relabeling remains invariant within
  `1e-6`;
- stop gate: inconsistent interactions across seeds/splits, less than `80%`
  explained gap, changed model state or any hidden retraining closes this
  attribution route.

Do not remotely scale K1-I, add data, epochs, width, pairs, seeds or experts,
or mix in an S-box, DDT, trail, key, cipher ID or raw bypass before K1-J
identifies a reproducible missing interaction. A passing K1-J result will
authorize one position-preserving GF(2) network redesign; a failed result will
redirect the next local mechanism experiment to exact heterogeneous
S-box/operator composition rather than mechanical scaling.
