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
| Cell relabeling invariance | cell-equivariant E4 mixer and invariant pooling | GIFT/SKINNY and heterogeneous-S-box relabeling tests pass | implemented |
| Correct versus controls | equal-geometry correct, corrupted and no-topology adapters | GIFT two-seed local attribution passed; SKINNY two-seed local attribution and seed0 `65536/class` passed | medium replication incomplete |
| General-GF(2) medium replication | frozen SKINNY r7 RTG2-A protocol | seed1 remote run is active under watcher control | running |

The claim boundary remains narrow. These facts prove a runtime-parameterized
4-bit-cell implementation and controlled diagnostic evidence. They do not prove
arbitrary cell widths, universal transfer, paper-scale performance, an attack,
SOTA or a breakthrough.

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
identify one implemented or implementable 4-bit-cell SPN with genuinely
cell-dependent S-box assignments. Before training, create a lean local plan with
the same dataset and budget for:

```text
candidate = late_cell
anchor    = late_pair
control   = S-box assignment deterministically shuffled while linear topology stays exact
```

Change only the S-box injection/assignment variable. The readiness gate must
first prove identical parameters, identical initialization, identical data and
labels, exact runtime descriptors, nonzero assignment sensitivity and cell
relabeling invariance. Run only a sub-medium local diagnostic initially. Advance
to remote scale only if the correct assignment beats both the global-mean anchor
and shuffled-assignment control under the frozen gate. Do not add DDT/trail
features, change the negative definition or combine this attribution with the
RTG2-A sample-scale ladder.
