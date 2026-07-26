# Innovation 1 Runtime-SPN Source Topology Diversity D1 Plan

```text
status = completed / hold / zero-training audit only
run_id = i1_runtime_spn_source_topology_diversity_d1_20260726
remote = no
training_rows = 0
optimizer_steps = 0
```

## Research Question

Can a deterministic family of invertible synthetic GF(2) linear layers expand
the structural coverage of the existing Runtime-E4 source topologies beyond an
equal-count cell-relabeling control, without using cipher identity or any held-
out cipher topology as a generator input?

C2 explicitly permits a preregistered source-topology-diversity mechanism on
the supported exact-GF(2) branch. It prohibits reopening S-box truth-table,
ANF, DDT, inverse-triplet, Adapter, FiLM, MoE, target-supervision and mechanical
C1/S2 scaling routes. D1 therefore changes no model, dataset, label, loss,
optimizer or C3 protocol. It is a zero-training feasibility audit that runs
locally while the independent C3 remote monitor owns PRESENT formal training.

## Real Structure Panel

The audit loads the existing runtime factories without cipher-name features:

| Structure | Width | Transition window |
| --- | ---: | ---: |
| PRESENT-80 | 64 | one homogeneous transition |
| GIFT-64 | 64 | one homogeneous transition |
| RECTANGLE-80 | 64 | one homogeneous transition |
| SKINNY-64/64 | 64 | one homogeneous transition |
| uKNIT-BC | 64 | all ten heterogeneous transitions |
| Dialga | 128 | one complete four-transition cycle |

Every transition is evaluated in a leave-one-cipher-out fold. A candidate for
one fold may use only transitions from other ciphers. A result is not an unseen-
cipher claim because the generator family and gate are evaluated against all
six known structures; a future empirical holdout requires a new cipher that did
not influence D1.

## Frozen Generator

For each allowed source transition, generate candidates with:

```text
generator = sparse_elementary_row_column_v1
seed material = SHA-256(source GF(2) matrix, cell membership, bit roles)
mutation counts = 4, 8, 16, 32
seeds = 0, 1, 2, 3
operations alternate row XOR and column XOR
source and target bit indices must differ
GF(2) rank must remain equal to block width
```

An elementary row or column XOR is invertible, so the generator changes the
linear operator without introducing a singular layer. The source cipher name
and transition id are provenance only and must not influence the operation
sequence. Candidate provenance must record source cipher, source transition,
topology-content seed hash, seed and mutation count.

For a 128-bit Dialga holdout, no Dialga transition may be used. Each allowed
64-bit source is first lifted to a 128-bit block-diagonal double copy. The same
frozen elementary-operation sequence then includes cross-half operations so
the candidate is not merely two disconnected 64-bit layers.

## Matched Control

For every synthetic candidate, generate one deterministic cell-relabeling
control from the same source transition and width. The control conjugates the
matrix and cell coordinates by a cell permutation while preserving bit roles.
Its topology-invariant feature vector must match its source exactly.

This equal-count control answers the key attribution question. More candidate
rows alone can trivially reduce nearest-neighbor distance; D1 advances only if
non-isomorphic elementary mutations beat an equally large pool of known
isomorphic relabelings.

## Frozen Topology Features

`normalized_gf2_topology_v1` contains only runtime linear-layer structure:

```text
normalized row-weight histogram
normalized column-weight histogram
target-cell source-count histogram
source-cell target-count histogram
cell-pair edge-count histogram
16 target-role x source-role edge fractions
same-cell and same-role edge fractions
permutation-cycle histogram and permutation indicator
rank((M^(2^k)) + I) / width for k = 0, 1, 2, 3
```

Features use no cipher name, key width, round count, task label, neural score or
target AUC. Distance is root-mean-square Euclidean distance between normalized
feature vectors at the same block width.

## Protocol Checks

D1 is valid only if all checks pass:

1. all six real structures and all 18 transition rows load;
2. every real and synthetic matrix has full GF(2) rank;
3. no fold uses its held-out cipher in synthetic or control provenance;
4. synthetic and relabel-control counts match in every fold;
5. every relabel-control feature vector exactly matches its source;
6. every 128-bit Dialga-fold candidate derives only from a non-Dialga 64-bit
   transition and contains at least one cross-half edge;
7. every synthetic candidate changes its source invariant signature;
8. the candidate manifest is deterministic under a repeated same-config build;
9. training rows, optimizer steps and remote execution remain zero.

A protocol failure is invalid evidence and permits repair only of the failed
invariant.

## Research Gate

Freeze these thresholds before reading distances:

```text
unique synthetic invariant signatures >= 90%
changed-from-source fraction = 100%
overall median relative distance improvement over relabel control >= 10%
ciphers with median relative improvement >= 10% >= 4 of 6
uKNIT median relative improvement >= 10%
Dialga median relative improvement >= 10%
synthetic invariant collisions with heldout transitions = 0
```

Relative improvement is:

```text
(nearest_relabel_control_distance - nearest_synthetic_distance)
----------------------------------------------------------------
max(nearest_relabel_control_distance, 1e-12)
```

If the control distance is zero, that holdout transition cannot count as
improved under this feature contract.

## Decisions

Pass:

```text
decision = innovation1_runtime_spn_source_topology_diversity_feasible
next = D2 synthetic-cipher signal and data-generation readiness audit
```

D2 must prove that generated topologies admit a fixed, nontrivial differential
task and disk-backed data path before any neural training. D1 does not authorize
joint training, C3 changes, remote compute or a whole-cipher transfer claim.

Hold:

```text
decision = innovation1_runtime_spn_source_topology_diversity_not_ready
next = stop synthetic scaling and inspect the failed coverage/invariant gate
```

Do not tune mutation counts, seeds, features or thresholds after seeing D1.

## Artifacts

```text
configs/experiment/innovation1/
  innovation1_runtime_spn_source_topology_diversity_d1_20260726.json
outputs/local_audit/
  i1_runtime_spn_source_topology_diversity_d1_20260726/
    candidates.jsonl
    results.jsonl
    validation.json
    gate.json
    summary.json
    progress.jsonl
```

No visualization is required. The primary evidence is a fold-level table and
exact protocol checks; a decorative two-dimensional projection would not prove
coverage and would introduce an avoidable interpretation surface.

## Completed Result

```text
completed = 2026-07-26
validation = pass
protocol checks = 10/10
candidate rows = 448
holdout result rows = 18
manifest SHA-256 = c3720c81cf7a5e00d0641b00eff7cb5c1c045c93d7bbe834fa1c0ff949c848a9
repeated manifest SHA-256 = c3720c81cf7a5e00d0641b00eff7cb5c1c045c93d7bbe834fa1c0ff949c848a9
status = hold
decision = innovation1_runtime_spn_source_topology_diversity_not_ready
```

The source-name protocol defect found before result reveal was repaired before
the completed run: deterministic operation sequences are now seeded from the
SHA-256 digest of the source GF(2) matrix, cell membership and bit roles. Cipher
name and transition id remain provenance only. The frozen research thresholds,
mutation counts and public seeds were not changed.

Exact research metrics:

| Metric | Result | Gate | Pass |
| --- | ---: | ---: | :---: |
| Unique synthetic signature fraction | 1.000000 | >= 0.900000 | yes |
| Changed-from-source fraction | 1.000000 | = 1.000000 | yes |
| Overall transition-row median improvement | 0.152854 | >= 0.100000 | yes |
| Ciphers with median improvement >= 0.10 | 2 / 6 | >= 4 / 6 | no |
| uKNIT median improvement | 0.203936 | >= 0.100000 | yes |
| Dialga median improvement | 0.080761 | >= 0.100000 | no |
| Held-out signature collisions | 0 | = 0 | yes |

Cipher-level medians show why the aggregate median is insufficient:

| Cipher | Median relative improvement |
| --- | ---: |
| PRESENT | -0.106806 |
| GIFT | -0.568678 |
| RECTANGLE | -0.291992 |
| SKINNY | +0.202949 |
| uKNIT | +0.203936 |
| Dialga | +0.080761 |

The generator therefore expands invariant signatures, but it does not provide
broad source-to-heldout coverage. It helps SKINNY and all ten uKNIT transition
rows, misses the preregistered Dialga threshold, and is worse than unchanged
source topology plus equal-count relabeling for PRESENT, GIFT and RECTANGLE.
The ten uKNIT rows also dominate the unweighted 18-row overall median, so that
passing aggregate must not override the cipher-balanced gate.

## Adjudicated Next Action

D1 does not permit D2. Do not increase candidate count, tune mutation counts or
seeds, change the frozen feature vector, or launch synthetic-topology neural
training. Those actions would optimize against the revealed six-cipher panel.

The only active training adjudication remains the separately preregistered C3
PRESENT-80 r7 formal gate:

```text
question = does the exact runtime GF(2) topology have neural value at formal scale?
same-budget anchor = correct topology vs corrupted topology vs no topology
one changed variable = runtime linear topology descriptor
train = 1,000,000/class
validation = 500,000/class
pairs/sample = 16
epochs = 5/model
seed = 0
execution = remote A6000, existing monitor-owned run
advance = correct AUC >= 0.520 and both control deltas >= +0.005
replication = identical seed1 only after seed0 passes
stop = no seed1 and no mechanical scale-up if seed0 fails
```

If C3 seed0 passes, its exact seed1 replication takes priority. If it fails,
combine C3 with the C2 method boundary and this D1 hold to redesign the method
at the representation/task level; do not reopen the closed synthetic generator
or descriptor-conditioning routes by changing compute alone.

Completed evidence:

```text
outputs/local_audit/i1_runtime_spn_source_topology_diversity_d1_20260726/
  candidates.jsonl
  results.jsonl
  validation.json
  gate.json
  summary.json
  progress.jsonl
```
