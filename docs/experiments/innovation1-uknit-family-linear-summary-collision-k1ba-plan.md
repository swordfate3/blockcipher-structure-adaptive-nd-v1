# Innovation 1 K1-BA Linear-Summary Collision Audit

**Status:** completed / pass / invariant-summary collision supported
**Date:** 2026-07-29
**Run ID:** `i1_uknit_family_linear_summary_collision_k1ba_20260729`

## 1. Research Question

K1-AZ trained component-separated connectivity so that only the 18 linear
summary values can affect the GF(2) edge gate. A wrong cross-cipher linear
descriptor changed that gate by approximately `0.0010-0.0090`, but the correct
descriptor exceeded the mismatch by at most `0.000202` AUC and passed zero of
twelve frozen `+0.001` panels.

K1-BA asks where this remaining failure occurs:

> Does the 18-value invariant linear summary fail to identify the actual
> runtime linear operator, so that a structurally different GF(2) topology can
> produce exactly the same descriptor and therefore exactly the same model
> output?

This is a zero-training mechanism audit. It cannot improve AUC and is not a
new model candidate.

## 2. Frozen Authority

Bind K1-AZ's completed hold gate, passed validation, two training rows, sixty
same-checkpoint controls, two epoch-9 checkpoints, structure summaries and
eighteen disk-backed datasets by SHA-256.

```text
ciphers          = uKNIT-BC r5, Midori64 r4, Dialga-128 r4
replicas         = 0/1
fresh splits     = same-key / cross-key
rows per split   = 1024/class = 2048 total
pairs per sample = 4
negative mode    = encrypted random plaintexts
training         = 0 epochs / 0 optimizer steps
device           = local CPU
```

The true encryption runtime remains correct under every descriptor condition.

## 3. One Intervention

For each correct runtime structure, construct one valid corrupted GF(2)
operator using the runtime structure's deterministic column permutation with
seed `20260729`. The corruption must change the exact linear-matrix tensor and
remain invertible. It is used only to generate a gate descriptor; it never
changes ciphertext generation or the model's runtime operator.

Evaluate three summaries:

```text
correct descriptor
cross-cipher linear mismatch       = K1-AZ's existing frozen control
same-summary corrupted linear      = correct S boxes + corrupted GF(2) operator
```

The current `linear_structure_summary` retains distributions of row weights,
column weights, ranks, density and the number of unique matrices. A column
permutation preserves those quantities while changing the concrete operator.
K1-BA tests this expected collision rather than assuming it.

## 4. Measurements

For every replica/cipher/split/condition record:

```text
AUC and probability hash
effective edge and transition gates
maximum and mean absolute probability difference from correct
cross-cipher probability Spearman correlation
source K1-AZ AUC and probability-hash replay
checkpoint, state, dataset and summary hashes
```

For every cipher also record the exact matrix Hamming fraction, summary
maximum absolute difference and active cross-cipher linear-summary dimensions.

Expected output is:

```text
2 replicas x 3 ciphers x 2 splits x 3 conditions = 36 rows
```

## 5. Frozen Gates

Protocol passes only if all source hashes, two checkpoints, eighteen datasets,
three structures, 36 evaluation rows, source AUC/probability replays, correct
runtime bindings, zero optimizer steps and state immutability are exact.

The invariant-summary collision is supported only if all three ciphers and all
twelve panels satisfy:

```text
corrupted matrix Hamming fraction       >= 0.001
correct-vs-corrupted summary max delta  = 0.0
correct-vs-corrupted edge gate delta    = 0.0
correct-vs-corrupted probability hash   identical
correct-vs-corrupted AUC delta          = 0.0
```

The secondary scalar-rank-inertia mechanism is supported only if every panel
satisfies:

```text
cross-cipher edge gate delta            >= 0.0005
absolute cross-cipher AUC delta         <= 0.001
probability Spearman correlation        >= 0.999
```

## 6. Decisions

- **Invariant-summary collision passes:** reject the 18 invariant values as a
  topology-identifying input. Open K1-BB readiness for a shared,
  position-preserving linear-operator token encoder that consumes actual
  source/target connectivity. Keep the trained backbone and all data fixed.
- **No collision, scalar rank inertia passes:** retain the descriptor but
  replace scalar edge amplitude with bounded channelwise modulation after a
  zero-update readiness gate.
- **Neither passes:** hold both interpretations and inspect the failed metric;
  do not train another candidate.
- **Protocol invalid:** repair only the failed binding and replay unchanged.

No branch authorizes more pairs, data, epochs, seeds, width, loss balancing,
per-cipher modules, experts/MoE or remote execution.

## 7. Required Artifacts

Write under `outputs/local_audit/<run_id>/`:

```text
preflight.json
dataset_manifest.jsonl
structure_collisions.json
results.jsonl
panel_summary.csv
checkpoint_manifest.json
gate.json
validation.json
summary.json
progress.jsonl
curves.svg
visual_qa_render_report.json
visual_qa_passed.marker
```

After completion, record exact metrics and the selected next action here,
refresh both recent-result indexes, run focused regressions and commit/push
only K1-BA files.

## 8. Completed Result

K1-BA completed all 36 frozen rows. Every K1-AZ source probability hash and
AUC replayed exactly, both epoch-9 checkpoint hashes remained exact, all model
states were immutable and validation passed all eight protocol checks:

```text
result rows                       = 36/36
distinct operator collisions      = 3/3
source probability/AUC replays    = exact
training                          = false
optimizer steps                   = 0
failed protocol checks            = []
```

The constructive collision passed for every cipher:

| Cipher | Changed matrix bits | Active dimensions under the cross-cipher mismatch | Correct vs corrupted 18-value summary |
|---|---:|---:|---:|
| uKNIT-BC | 9.033203% | 1/18 | exact, max delta 0.0 |
| Midori64 | 9.179688% | 12/18 | exact, max delta 0.0 |
| Dialga-128 | 4.589844% | 11/18 | exact, max delta 0.0 |

The correct and corrupted operators have different matrix and runtime-window
SHA-256 values in all three cases. Nevertheless, their descriptors, effective
edge and transition gates, probability hashes and AUCs are bitwise identical
on all twelve replica/cipher/split panels.

```text
collision ciphers                 = 3/3
collision panels                  = 12/12
minimum matrix Hamming fraction   = 0.0458984375
maximum summary delta             = 0.0
```

The secondary cross-cipher control also passed its rank-inertia gate:

```text
minimum edge-gate change          = 0.000997319818
minimum probability Spearman      = 0.999936236053
maximum absolute AUC change       = 0.000202178955
passing panels                    = 12/12
```

This separates two facts. The trained gate does react to the coarse statistics
that happen to differ between ciphers, so the linear input is not disconnected.
But the descriptor itself is not topology-identifying: it discards the actual
source-to-target relations before the neural network sees them. No amount of
training, data or scalar-gate tuning can distinguish two operators that enter
the model as the same tensor.

Final decision:

```text
status = pass
decision = innovation1_uknit_family_k1ba_invariant_linear_summary_not_topology_identifying_supported
remote_scale = no
```

This is a zero-training local mechanism result on the existing
`2048/class/cipher`, four-pair evidence. It is not formal-scale accuracy, an
attack, unseen-cipher transfer or SOTA evidence.

## 9. Recommended Next Action: K1-BB Readiness

K1-BB should replace only the 18 invariant linear statistics with a shared,
position-preserving operator-token encoder. It must consume the actual
nonzero GF(2) relations rather than a cipher name or fixed-width flattened
matrix:

```text
one token = round position + source cell/bit role + target cell/bit role
token encoder = one shared MLP/message function for every cipher and edge
aggregation = source-to-target message passing with fixed output width
consumer = existing GF(2) edge residual modulation path
```

Deterministic normalized or sinusoidal position features should support both
64-bit and 128-bit states without learned cipher-specific embedding tables.
The K1-AZ backbone, edge residual, S-box transition residual, classifier,
checkpoints, data and encryption runtime stay frozen. The only readiness
variable is the linear-operator representation and its connection to the
existing edge path.

K1-BB is local CPU, zero epochs and zero optimizer steps. It must require:

```text
same parameter geometry for uKNIT-BC, Midori64 and Dialga-128;
no cipher ID, per-cipher head, adapter, router, expert or MoE;
correct versus same-summary corrupted operator embedding delta >= 1e-4;
correct versus same-summary corrupted edge modulation delta >= 1e-6;
disabled new path exactly replays each K1-AZ checkpoint;
joint source/target relabeling produces the preregistered equivariant response;
all source/checkpoint/data hashes and model states remain exact.
```

Passing readiness will authorize one same-budget local K1-BC training plan
against K1-AZ and the required correct/corrupted/cross-cipher controls. It will
not authorize 16 pairs, larger data, more epochs/seeds/width, loss balancing,
per-cipher modules, experts/MoE or remote execution.

## 10. Artifacts And Visual QA

Artifacts are under:

```text
outputs/local_audit/i1_uknit_family_linear_summary_collision_k1ba_20260729/
```

The Chinese `curves.svg` was rendered to `2700 x 1800` pixels and inspected
through `visual-qa-redraw`. The title, four panels, bar labels, legends,
thresholds, local AUC axis and next-action caption have no overlap, clipping,
missing glyphs, ambiguous association or misleading scale. The check is
recorded in `visual_qa_render_report.json` and `visual_qa_passed.marker`.
