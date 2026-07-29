# Innovation 1 K1-BI Runtime-Cell Joint GF(2) Response Audit

**Status:** completed / hold
**Date:** 2026-07-29
**Run ID:** `i1_uknit_family_cell_joint_gf2_operator_response_k1bi_audit_replica0_replica1_20260729`

## 1. Research Question

K1-BH applied the exact supplied GF(2) operators but represented every bit
independently before averaging over four pairs. That representation retained
strong correct-operator signal for Midori-64 r4 (`0.603104-0.613360`) and
Dialga-128 r4 (`0.987278-0.993649`), while every uKNIT-BC r5 panel stayed near
chance (`0.470592-0.514756`). K1-Q/K1-S already found `0.806228-0.825591`
fresh AUC on the same uKNIT cell11 data with position-preserving four-bit cell
value histograms. The next unresolved bottleneck is therefore the loss of the
joint value inside each native cell, not the input-difference position.

K1-BI asks one question:

> If the same exact GF(2) response bits are reassembled into their runtime
> native four-bit cells before pair reduction, does the correct operator retain
> an exclusive label signal on uKNIT without destroying the Midori/Dialga
> anchors?

## 2. Single Representation Variable

K1-BH and K1-BI share the same three ciphertext channels and four ordered
operator views:

```text
channels = left, right, left XOR right
views    = raw, inverse_0, inverse_1, inverse_0(inverse_1(.))
```

Only the final representation changes:

```text
K1-BH: bit x view x channel -> mean over four pairs
K1-BI: runtime-cell x view x channel x value(0..15)
       -> one-hot each native cell value
       -> mean over the same four pairs
```

For each response channel, K1-BI uses the supplied runtime
`cell_membership` and `bit_role` tensors to reconstruct the native value

```text
cell_value = XOR/OR over bit_value << bit_role
```

before one-hot encoding. Cell position, view order and channel order remain
explicit. The frozen feature dimensions are:

```text
uKNIT-BC:  16 cells x 4 views x 3 channels x 16 values = 3072
Midori-64: 16 cells x 4 views x 3 channels x 16 values = 3072
Dialga:    32 cells x 4 views x 3 channels x 16 values = 6144
```

No S-box transformation, neural parameter, cipher ID, adapter, expert or
per-cipher component is introduced.

## 3. Same-Budget Anchor And Controls

K1-BH is the exact same-budget anchor. K1-BI rehashes its completed artifacts
and reuses the underlying 18 disk-backed datasets, structures, both wrong
operators and deterministic shuffle seeds:

```text
uKNIT-BC r5:   seed3 / seed4
Midori-64 r4:  seed6 / seed7
Dialga-128 r4: seed0 / seed1

4096 total train rows per replica/cipher
2048 total rows per fresh split
fresh splits = same_key_fresh + cross_key_validation
4 pairs/sample
negative = encrypted random plaintexts
device = local CPU
neural parameters = 0
optimizer steps = 0
```

The diagonal Fisher scorer is fitted once on the correct-operator
`train_seen` features. The same scorer evaluates correct, same-summary wrong,
cross-cipher wrong and identity features. Wrong operators must not be refitted.
A second scorer is fitted using the identical correct features with frozen
shuffled labels.

## 4. Frozen Gates

Every one of the twelve `replica x cipher x fresh split` panels must satisfy:

```text
correct AUC                         >= 0.55
correct - identity AUC             >= 0.01
correct - same-summary wrong AUC   >= 0.01
correct - cross-cipher wrong AUC   >= 0.01
correct - label-shuffle AUC        >= 0.03
0.47 <= label-shuffle AUC <= 0.53
```

The last condition fixes K1-BH's exposed one-sided gate and is not negotiable
after observing K1-BI. In addition, every Midori/Dialga correct-operator panel
must remain within `0.02` AUC of its matching K1-BH anchor. Every wrong and
identity response must differ from the correct response by finite positive
RMS.

Protocol gates require exact source/config digests, 72 feature manifests, 12
scorers, 60 fresh results, exact feature dimensions and row counts, identical
datasets across operator interventions, count-preserving shuffles, correct-fit
scorer reuse, finite metrics, zero neural updates and no data generation.

## 5. Decision Table

- **All gates pass:** keep the native-cell categorical primitive and
  preregister a shared position-preserving neural residual that consumes exact
  transported cell tokens. K1-BI passing does not itself authorize scale.
- **Symmetric shuffle gate fails:** freeze the representation and preregister
  a multi-permutation, orientation-invariant shuffled-label null. Do not tune
  data, difference or feature while repairing attribution.
- **uKNIT correct AUC remains below `0.55`:** stop linear-only response
  redesign. Bind the already-supported runtime S-box-aware five-stage cell
  statistic as the next family primitive.
- **Midori/Dialga anchor drops by more than `0.02`:** audit cell reconstruction
  and Fisher variance handling before architecture work.
- **Correct signal survives but topology margins fail:** audit wrong-operator
  equivalence; do not train a network on a non-identifying representation.
- **Protocol failure:** repair only the failed binding or implementation
  invariant and rerun unchanged.

No outcome authorizes 16 pairs, larger data, extra epochs/seeds/width, remote
GPU, MoE, cipher-specific heads, difference rescanning or benchmark changes.

## 6. Required Artifacts

Write under `outputs/local_audit/<run_id>/`:

```text
preflight.json
dataset_manifest.jsonl
feature_manifest.jsonl
scorers.jsonl
results.jsonl
gate.json
validation.json
summary.json
progress.jsonl
curves.svg
visual_qa_render_report.json
visual_qa_passed.marker
```

The Chinese SVG must explain the cipher rounds, four-pair cell-joint mechanism,
zero-neural scope, correct/wrong controls and symmetric shuffle diagnostic. It
must pass rendered-pixel inspection at `2700 x 1800` through
`visual-qa-redraw` before the result is complete.

## 7. Recommended Next Action

Execute this local audit unchanged. The next architecture decision is strictly
conditional on the frozen decision table above; do not start neural training
or remote work from a partial or protocol-invalid result.

## 8. Completed Result

K1-BI completed the frozen local audit with no protocol error and no neural
update:

```text
feature manifests = 72 / 72
scorer rows       = 12 / 12
fresh result rows = 60 / 60
protocol errors   = []
status            = hold
decision          = innovation1_uknit_family_k1bi_shuffle_attribution_not_supported
```

Correct-operator AUC on the four fresh panels per cipher:

| Cipher / rounds | replica0 same-key | replica0 cross-key | replica1 same-key | replica1 cross-key |
|---|---:|---:|---:|---:|
| uKNIT-BC r5 | `0.499398` | `0.492171` | `0.469312` | `0.501842` |
| Midori-64 r4 | `0.928709` | `0.930943` | `0.927871` | `0.918214` |
| Dialga-128 r4 | `0.997000` | `0.994392` | `0.995729` | `0.997225` |

Against the matching K1-BH bit-mean anchor, the cell-joint representation
changed AUC by:

```text
uKNIT:  +0.000029, -0.004195, -0.001280, -0.012915
Midori: +0.324931, +0.317582, +0.314674, +0.315110
Dialga: +0.003350, +0.007114, +0.005629, +0.005503
```

All eight Midori/Dialga anchor-retention checks passed. Their correct operators
also beat identity and both wrong-operator controls on every panel. Native
four-bit joint values are therefore a materially better structure-sensitive
surface for Midori and preserve Dialga's already saturated signal.

All four uKNIT correct-operator panels remained below `0.55`. Their own
shuffled-label AUCs were inside the symmetric band (`0.477516-0.502333`), so
the uKNIT failure is not explained by the exposed reverse-orientation problem.
Pure exact linear transport, even with native four-bit joint values, does not
recover the known uKNIT r5 cell11 signal.

The symmetric shuffled-label control failed on all four Dialga panels and both
replica1 Midori panels:

```text
Dialga replica0: 0.455143 / 0.445283
Dialga replica1: 0.355984 / 0.351163
Midori replica1: 0.433025 / 0.428197
```

The frozen decision table therefore holds architecture work before a
family-wide attribution claim. This does not erase the uKNIT-specific
linear-route failure or the large Midori/Dialga representation gain.

## 9. Evidence-Backed Next Action: K1-BJ Multi-Shuffle Null

K1-BJ must keep the completed K1-BI features and every benchmark variable
fixed. Change only the label-control statistic:

```text
K1-BI control: one frozen shuffled-label Fisher scorer
K1-BJ control: multiple preregistered shuffled-label Fisher scorers
               evaluated with abs(AUC - 0.5)
```

Use at least 31 deterministic permutation seeds per replica/cipher, evaluate
all scorers on both unchanged fresh splits, and compare the correct scorer's
orientation-invariant strength against the empirical shuffle-null
distribution. Do not reselect permutations after seeing their AUCs. The
same-budget anchors are the K1-BI correct and wrong-operator rows; no feature,
data, difference, pair, cipher, round, key or Fisher setting may change.

Advance to a shared neural cell-token residual only if the orientation-
invariant null supports correct-operator attribution and all original
topology/signal gates pass. Because uKNIT already fails with clean per-panel
shuffle controls, the expected route after attribution is to stop linear-only
transport and preregister the runtime S-box-aware five-stage cell primitive,
not to add data, pairs, capacity, MoE or remote training.
