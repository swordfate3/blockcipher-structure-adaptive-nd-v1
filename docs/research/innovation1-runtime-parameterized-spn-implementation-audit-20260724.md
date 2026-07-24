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
| Correct versus controls | equal-geometry correct, corrupted and no-topology adapters | GIFT two-seed local attribution passed; SKINNY two-seed `65536/class` medium attribution passed | medium two-seed supported |
| General-GF(2) medium replication | frozen SKINNY r7 RTG2-A protocol | seed0 `0.643591`, seed1 `0.644613`; both control margins pass the joint gate | RTG2-B preparation authorized |

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

After the RTG2-A watcher returns, preserve its decision first. Independently,
uKNIT-BC is now implemented as the real 4-bit-cell SPN with genuinely
cell-dependent S-box assignments. Before training, create a lean local plan
with the same dataset and budget for:

```text
candidate = late_cell
anchor    = late_pair
control   = S-box assignment deterministically shuffled while linear topology stays exact
```

The mechanism reproduction, exact source boundary, prefix-round semantics and
the executable local gate are recorded in:

```text
docs/research/innovation1-uknit-runtime-spn-mechanism-reproduction-20260724.md
```

Change only the S-box injection/assignment variable. The readiness gate must
first prove identical parameters, identical initialization, identical data and
labels, exact runtime descriptors, nonzero assignment sensitivity and cell
relabeling invariance. Run only a sub-medium local diagnostic initially. Advance
to remote scale only if the correct assignment beats both the global-mean anchor
and shuffled-assignment control under the frozen gate. Do not add DDT/trail
features, change the negative definition or combine this attribution with the
RTG2-A sample-scale ladder.

## Post-RTG2-A Evidence Update

RTG2-A seed1 subsequently completed and was retrieved from its verified result
branch. Correct topology reached `0.644612943` AUC versus `0.597460402`
corrupted and `0.513995145` no-topology controls. Together with seed0
`0.643590577`, the frozen two-seed joint gate passed. This upgrades the
general-GF(2) branch from local implementation evidence to repeatable
`65536/class` medium architecture/protocol evidence; it remains non-formal and
does not establish an attack, SOTA or universal-SPN claim.

The highest-priority remote action is now an RTG2-B `262144/class` seed0 plan
that changes sample scale only and preserves the same three roles, r7
`0x2000` data protocol, four pairs, five epochs, strict encrypted-random-
plaintext negatives and `442466`-parameter geometry. It must pass a fresh
disk-cache/readiness gate before launch. The uKNIT S-box-assignment branch may
continue with bounded local representation diagnostics, but its successive
`2048/class` holds do not take priority over this already-supported scale
replication and do not authorize remote GPU use.
