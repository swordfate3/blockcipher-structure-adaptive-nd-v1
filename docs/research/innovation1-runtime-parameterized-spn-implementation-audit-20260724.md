# Innovation 1 Runtime-Parameterized SPN Implementation Audit

Date: 2026-07-24

## Method Objective

Build one cipher-name-free SPN neural backbone that receives the cell partition,
bit roles, S-box descriptors and linear diffusion graph at runtime. The same
learnable parameter geometry must support permutation P-layers and invertible
general GF(2) layers. Correct topology must be judged against equal-capacity
corrupted-topology and no-linear-topology controls at the same data and training
budget.

## Current Implementation Evidence

| Requirement | Implementation | Current evidence | Status |
| --- | --- | --- | --- |
| External cell partition and bit roles | `RuntimeSpnStructure.cell_membership` and `bit_role` | invalid partitions rejected; cell relabeling tests pass | implemented |
| External S-box type | per-round, per-cell 4-bit truth descriptors | changing PRESENT/GIFT descriptors changes logits with fixed weights | implemented for 4-bit cells |
| External linear topology | runtime GF(2) matrices plus exact inverses | PRESENT/GIFT permutations and SKINNY sparse GF(2) pass exact inverse tests | implemented |
| Fixed backbone geometry | runtime tensors are not parameters or state entries | one model instance handles 64-bit and synthetic 128-bit structures without state-shape changes | implemented |
| External training descriptor | strict JSON loader plus cipher-name-free registry entries | production PRESENT permutation and SKINNY GF(2) descriptors match built-in structures exactly | implemented |
| Non-round-aligned real SPN | uKNIT-BC descriptor with cell/round-specific S-boxes and 11 distinct GF(2) transitions | four official vectors, 11 prefix states, 13 round keys and descriptor windows match | implemented |
| Cell relabeling invariance | cell-equivariant E4 mixer and invariant pooling | GIFT/SKINNY and heterogeneous-S-box relabeling tests pass | implemented |
| Correct versus controls | equal-geometry correct, corrupted and no-topology adapters | GIFT two-seed local attribution passed; SKINNY `65536/class` and `262144/class` two-seed attribution passed | replicated medium evidence supported |
| General-GF(2) scale replication | frozen SKINNY r7 RTG2-A/RTG2-B protocol | RTG2-B correct AUC `0.649229/0.647783`; corrupted `0.603562/0.602584`; no topology `0.510190/0.513038` | formal RTG3-A seed0 prepared |
| Frozen cross-cipher representation | GIFT Runtime-E4 backbone with only a SKINNY target head trained | X2 candidate `0.552013/0.598568` exceeds source, target and random controls but trails full-target `0.612733/0.614544` | small mechanism supported; scale held |

The claim boundary remains narrow. These facts prove a runtime-parameterized
4-bit-cell implementation and controlled diagnostic evidence. They do not prove
arbitrary cell widths, universal transfer, paper-scale performance, an attack,
SOTA or a breakthrough.

## External Descriptor Training Entry

The runtime backbone previously accepted `forward(features, structure)`, but the
ordinary training registry still selected PRESENT, GIFT or SKINNY through
cipher-specific Python model names and structure factories. A new generic entry
now loads the structure from `model_options.runtime_structure_path` without
changing the backbone parameter geometry:

```text
runtime_spn_e4_equivariant_true
runtime_spn_e4_equivariant_corrupted
runtime_spn_e4_equivariant_independent
```

The versioned JSON schema records the cell membership, bit role, 4-bit S-box
tables and one or more linear layers. Linear layers may be either a
`source_to_target` permutation or a sparse GF(2) `target_sources` relation. A
single round may be repeated only when `repeat_single_round=true`; every loaded
matrix must be invertible over GF(2). Unknown fields, non-integer arrays,
malformed permutations, duplicate or out-of-range GF(2) sources, round-count
mismatches and singular matrices are rejected before model construction.

Three production descriptors are available:

```text
configs/runtime/spn/present64.json  = one-to-one PRESENT P-layer
configs/runtime/spn/skinny64.json   = SKINNY ShiftRows + MixColumns GF(2) layer
configs/runtime/spn/uknit64.json    = 11 distinct uKNIT-BC transition layers
```

Tests compare all three descriptors against their Python factories for cell
membership, bit roles, S-box truth bits, forward linear matrices and exact
inverse matrices. The generic correct, corrupted and independent controls also
share identical state geometry, complete a forward pass and expose the
descriptor name, resolved path, raw-file SHA-256 and control mode through result
metadata. This closes the cipher-name-free training-entry gap for supported
4-bit-cell SPNs. It is an implementation result only: it adds no training run,
AUC evidence, cross-cipher generalization result or scale decision.

## Cell-Specific S-Box Ownership Gap

The empirically selected `late_pair` mode injects the S-box descriptor after
topology extraction, which preserves the GIFT E4 anchor. Its descriptor is the
mean over all cells, however. Two structures with the same S-box multiset but
different assignments of those S-boxes to cells therefore receive the same
late context, apart from floating-point reduction noise. That is sufficient for
PRESENT, GIFT and SKINNY because each tested round shares one S-box across all
cells, but it is not sufficient for a general SPN with cell-specific S-boxes.

The new opt-in `late_cell` mode injects each encoded S-box descriptor into its
own cell token after the topology mixer and before sequence normalization. It:

- preserves the successful late topology-extraction path;
- retains the S-box-to-cell assignment;
- adds no learnable parameters and does not change state geometry;
- remains invariant when the input, topology and S-box assignment are relabeled
  together;
- leaves the existing `early_add` and frozen `late_pair` numerical paths
  unchanged.

Deterministic tests use alternating PRESENT and GIFT S-boxes under one fixed
linear graph. Swapping the S-box assignment is invisible to `late_pair` within
`1e-6`, while `late_cell` changes the logits by more than both `1e-6` and 100
times the `late_pair` numerical delta. A simultaneous cell relabeling preserves
the `late_cell` output within `1e-6`.

This is an implementation capability check, not a neural distinguisher result.
No training, AUC, scale or route promotion follows from the deterministic test.

## Route Separation

RTG2-A seed1 remains pinned to:

```text
source commit = 9120a1ff96815975f31f1f461342bb7831e2d035
S-box mode    = late_pair
cipher        = SKINNY-64/64 r7
scale         = 65536/class train, 32768/class validation
```

Do not substitute `late_cell` into that run, its joint gate or a scale-only
`262144/class` successor. If the RTG2-A joint gate passes, the scale ladder must
keep `late_pair` so sample scale remains the only changed variable. If the joint
gate holds, `late_cell` is not a valid rescue because SKINNY uses a shared S-box
and the repaired capability targets a different structural question.

## Recommended Next Action

The implementation question and the empirical question are now separate. The
runtime object, generic registry entry, permutation/GF(2) operators, variable
cell count, S-box descriptors and invariant geometry are implemented. Do not
add another cipher-specific frontend while the supported general-GF(2) scale
replication is unresolved.

The current priority is the post-X2 selected RTG3-A seed0 matrix:

```text
question       = does the RTG2-B topology advantage survive project-formal scale?
anchor         = frozen RTG2-B correct/corrupted/no-topology protocol
changed field  = train samples_per_class 262144 -> 1000000 only
execution      = remote A6000 GPU0 with disk-backed cache
status         = plan, config, gate and watcher implementation in readiness
```

If seed0 passes all three frozen gates, prepare the identical conditional
`1000000/class` seed1 confirmation. If seed0 holds, stop mechanical scale-up
and inspect cache identity, complete five-epoch dynamics and the two topology
margins without changing the network, data protocol or control semantics. A
single-seed pass is project-formal evidence but is not a multi-seed formal
conclusion.

The uKNIT S-box-assignment route has already completed its bounded local loop.
U2-F and the U2-G same-checkpoint audit supported one prefix-r4 delta-U query
mechanism, but U2-H changed only to prefix-r5 and failed to reproduce it on
both seeds. Preserve U2-F/U2-G as narrow single-window mechanism evidence and
close the current cross-window, remote-scale and mechanical-sample extensions.
Do not run the cancelled U2-I or revive `late_cell`/edge-gate/triplet variants
without a new mechanism that directly explains the r4-to-r5 failure.

## Post-RTG2-A Evidence Update

RTG2-A seed1 subsequently completed and was retrieved from its verified result
branch. Correct topology reached `0.644612943` AUC versus `0.597460402`
corrupted and `0.513995145` no-topology controls. Together with seed0
`0.643590577`, the frozen two-seed joint gate passed. This upgrades the
general-GF(2) branch from local implementation evidence to repeatable
`65536/class` medium architecture/protocol evidence; it remains non-formal and
does not establish an attack, SOTA or universal-SPN claim.

The highest-priority remote action was therefore RTG2-B at `262144/class`
seed0, changing sample scale only while preserving the same three roles, r7
`0x2000` data protocol, four pairs, five epochs, strict encrypted-random-
plaintext negatives and `442466`-parameter geometry. Its fresh disk-cache,
readiness and exact-published-source gates passed, and the run is now active.
The separate uKNIT sequence has since reached a cross-window stop decision, as
recorded below; it is no longer an open local representation route.

## Post-uKNIT And RTG2-B Startup Update

The uKNIT diagnostic sequence is now complete. U2-F reached correct delta-U
query AUCs `0.543139/0.554935` on seeds `0/1` and exceeded both controls. U2-G
confirmed with the same candidate checkpoint that changing only the query
S-box ownership changes the logits in the expected direction. U2-H then tested
the same hypothesis at prefix-r5 and returned correct AUCs `0.490057/0.500100`,
below both required controls. Its decision is:

```text
innovation1_uknit_delta_u_cross_window_not_replicated
```

RTG2-B subsequently passed its exact-source launch gate and produced real
remote startup evidence at `2026-07-24 20:23:46 +08:00`. The synchronized logs
show source `061fd9a3c30cd1089a24e9df241f63964d147d6c`, clean detached status,
readiness `pass`, PyTorch `2.5.1+cu118`, CUDA `11.8` and one visible A6000. This
is launch evidence only. No RTG2-B AUC, result decision or scale conclusion
exists until the verified result branch is retrieved and re-adjudicated.

## Post-RTG2-B And X2 Route Update

RTG2-B later completed on both seeds. Correct runtime topology reached
`0.649229395/0.647782881`, exceeding deterministic corrupted topology by
`+0.045667696/+0.045198574` and no topology by
`+0.139039457/+0.134744390`. Both remote archives, checkpoint payloads and the
joint gate passed after immutable local re-adjudication.

The lower-cost X2 branch then froze the GIFT Runtime-E4 backbone and trained
only the SKINNY output head. Its candidate AUCs `0.552013397/0.598568439`
passed all three controls, supporting limited cross-cipher representation
reuse. They remained below same-data end-to-end SKINNY anchors by
`0.060719967/0.015975475`; the source GIFT checkpoints were also trained at
only `2048/class`. A medium X2 enlargement would therefore leave source scale
confounded or require changing source and target training together.

The completed route audit selects formal within-SKINNY confirmation as the
single next remote slot. RTG3-A freezes all RTG2-B fields and changes only
`samples_per_class` from `262144` to `1000000`, with seed0 first and seed1
conditional. X2 remains supported small mechanism evidence and is held, not
discarded. No DDT, trail, related-key, partial-decryption or broad-cipher
matrix route is reopened by this decision.

## Post-RTG3-Launch Multi-Round Processor Readiness

An opt-in Runtime-E4 recurrent-window path is now implemented without changing
the frozen RTG3 protocol. It separates loaded `runtime_rounds` from trainable
`processor_steps`, reuses one parameter geometry across runtime window lengths,
and consumes each round's inverse GF(2) map and per-cell S-box descriptors.
Earlier-linear and earlier-S-box fixed-weight counterfactuals, cell relabeling,
64/128-bit width, general-GF(2), supported cell views and finite-gradient tests
pass.

The implementation also records a `full|repeat_last` runtime-window control.
This exposed an important attribution boundary: PRESENT, GIFT and SKINNY use
homogeneous repeated structure tensors, so their full windows are exactly the
same as repeated-final. They can test recurrent structure-processing depth but
not distinct earlier-round topology use. uKNIT is the current real-cipher
heterogeneous anchor for that claim. No recurrent-window training result or AUC
exists yet, and every prior Runtime-E4 result remains last-transition evidence.

The transformed runtime object now contributes canonical per-transition
SHA256 values, an ordered window SHA256, a unique-transition count and a
homogeneous-window flag to model metadata. This makes the full, corrupted,
S-box-shuffled and repeated-final structures directly auditable from future
result rows instead of relying on model names or source-descriptor hashes.
