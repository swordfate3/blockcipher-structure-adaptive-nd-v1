# Innovation 1 Runtime-Parameterized SPN Implementation Audit

Date: 2026-07-24; updated 2026-07-25

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
| External cell partition and bit roles | `RuntimeSpnStructure.cell_membership` and `bit_role` | invalid partitions rejected; cell relabeling tests pass; RECTANGLE column cells prove non-contiguous physical-bit membership | implemented |
| External S-box type | per-round, per-cell 4-bit truth descriptors | changing PRESENT/GIFT descriptors changes logits with fixed weights | implemented for 4-bit cells |
| External linear topology | runtime GF(2) matrices plus exact inverses | PRESENT/GIFT permutations and SKINNY sparse GF(2) pass exact inverse tests | implemented |
| Fixed backbone geometry | runtime tensors are not parameters or state entries | one model instance handles 64-bit and synthetic 128-bit structures without state-shape changes | implemented |
| External training descriptor | strict JSON loader plus cipher-name-free registry entries | production PRESENT/GIFT/RECTANGLE permutations and SKINNY GF(2) descriptors match built-in structures exactly | implemented |
| Non-round-aligned real SPN | uKNIT-BC descriptor with cell/round-specific S-boxes and 11 distinct GF(2) transitions | four official vectors, 11 prefix states, 13 round keys and descriptor windows match | implemented |
| Cell relabeling invariance | cell-equivariant E4 mixer and invariant pooling | GIFT/SKINNY and heterogeneous-S-box relabeling tests pass | implemented |
| Correct versus controls | equal-geometry correct, corrupted and no-topology adapters | GIFT two-seed local attribution passed; SKINNY `65536/class` and `262144/class` two-seed attribution passed; RECTANGLE RCT1 `2048/class` passed on both seeds | replicated multi-cipher diagnostic evidence supported |
| General-GF(2) scale replication | frozen SKINNY r7 RTG2-A/RTG2-B/RTG3-A protocol | RTG3-A seed0 at `1000000/class`: correct `0.653192`, corrupted `0.607162`, no topology `0.511826`; seed1 is active under the remote watcher | one-seed formal evidence supported; second seed pending |
| Frozen cross-cipher representation | formal SKINNY Runtime-E4 extractor rebound to RECTANGLE with only an independent target head trained | X3-A/X3-A2 candidate `0.784894/0.753170` exceeds corrupted-source, corrupted-target and random-source controls on both target seeds | local dual-seed mechanism supported; medium transfer blocked on RCT2 |
| Low-capacity cross-cipher readout | frozen formal SKINNY Runtime-E4 representation with `Linear(384,1)` target probe | X4 candidate `0.791652/0.761355`; all three control margins pass on both seeds with only `385` trainable parameters | direct linear accessibility supported locally |

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

Five production descriptors are available:

```text
configs/runtime/spn/present64.json  = one-to-one PRESENT P-layer
configs/runtime/spn/gift64.json     = one-to-one GIFT-64 P-layer
configs/runtime/spn/rectangle64.json = RECTANGLE non-contiguous column cells + row rotations
configs/runtime/spn/skinny64.json   = SKINNY ShiftRows + MixColumns GF(2) layer
configs/runtime/spn/uknit64.json    = 11 distinct uKNIT-BC transition layers
```

Tests compare all five descriptors against their Python factories for cell
membership, bit roles, S-box truth bits, forward linear matrices and exact
inverse matrices. GIFT and RECTANGLE additionally complete forward passes
through the generic `runtime_spn_e4_equivariant_true` entry without
cipher-specific model names. The generic correct, corrupted and independent controls also
share identical state geometry, complete a forward pass and expose the
descriptor name, resolved path, raw-file SHA-256 and control mode through result
metadata. This closes the cipher-name-free training-entry gap for supported
4-bit-cell SPNs. It is an implementation result only: it adds no training run,
AUC evidence, cross-cipher generalization result or scale decision.

## Non-Contiguous Real Cell Layout

PRESENT, GIFT, SKINNY and uKNIT store each four-bit S-box cell in four
consecutive physical bit positions. That evidence did not prove that the
runtime object could describe a bitsliced SPN whose S-box inputs are spread
across the state. RECTANGLE-80 closes this implementation gap without adding a
new model family.

For physical bit index `16 * row + column`, the production RECTANGLE descriptor
uses:

```text
cell_membership[index] = column
bit_role[index]        = 3 - row
cell column c          = [c, 16+c, 32+c, 48+c]
```

The descriptor therefore groups one bit from each 16-bit row into each S-box
cell. Its external S-box table is the actual `RECTANGLE_SBOX`; its one-to-one
linear topology is the actual row-rotation map `(0, 1, 12, 13)`. Deterministic
tests prove that runtime S-box application equals `rectangle_sub_columns` and
runtime GF(2) application equals `rectangle_shift_rows` on concrete 64-bit
states. The JSON descriptor matches the Python factory field by field, and the
same generic RuntimeE4 entry retains exactly the PRESENT parameter geometry.

RECTANGLE is also registered in the standard cipher factory, plan-name mapping,
structure profile and differential-data path. A strict encrypted-random-
plaintext diagnostic produces the expected 4-pair, 512-bit rows. These are
implementation and data-path checks only; no RECTANGLE neural training, AUC,
topology-attribution or transfer evidence is claimed.

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

## Strict Edge-Gate No-Topology Semantics

A post-readiness audit found that the E4 `independent` relation mode disabled
the exact inverse-linear state view but still passed the real inverse-linear
adjacency into `edge_gate`. That implementation was therefore not a strict
no-linear-topology control whenever `sbox_context_mode=edge_gate`.

The control now replaces the cell graph with identity adjacency while keeping
the S-box self-gate, parameter geometry and compute path. End-to-end tests cover
both `last_transition` and `recurrent_window`: independent logits are bit-exact
under a correct-versus-corrupted linear-graph swap, while true-relation logits
change. This change does not affect the completed GIFT/SKINNY attribution or
active RTG3-A evidence because those frozen protocols use `late_pair`, not
`edge_gate`. The only affected artifact was the untrained uKNIT recurrent
readiness probe; it was rerun before U3 training, passed all checks, and replaced
the no-topology output hash with
`eb3ff3369d716ed8c3aba0dfa6064889a4d05f8383dbacf990c8bbf1b8452397`.

A follow-up leakage audit now exercises all `40` legal combinations of window
mode, cell-input view and S-box-context mode. For every combination, changing
only the runtime linear graph leaves `independent` logits bit-exact. This
extends the regression contract beyond the repaired `state_triplet + edge_gate`
path to every currently supported Runtime-E4 view without changing model
behavior or training geometry.

## Formal-Scale Seed0 And Stable Representation API

RTG3-A seed0 subsequently completed at the project formal evidence floor of
`1000000/class` training rows and `500000/class` validation rows. The frozen
SKINNY-64/64 r7 RuntimeE4 protocol produced AUC `0.653191631304` for correct
GF(2) topology, `0.607162432806` for deterministic corrupted topology and
`0.511826118586` for no topology. The correct-topology margins were therefore
`+0.046029198498` and `+0.141365512718`. Result validation, strict best-
checkpoint replay and rendered-figure visual QA passed. This is one-seed
formal-scale structure-attribution evidence, not yet a multi-seed conclusion,
paper-scale reproduction, attack, SOTA result or universal-SPN claim. The
identical seed1 confirmation is active under the remote watcher.

Cross-cipher X2 previously accessed `backbone.classifier` directly to obtain a
frozen RuntimeE4 representation. Evaluation code now exposes
`extract_runtime_e4_representation`, which returns both the logits and the
exact tensor received by the classifier. The interface uses a temporary
pre-forward hook so it does not alter the frozen model path, learnable
parameters, state-dict keys or checkpoint compatibility. It fails closed for
non-E4 adapters and removes the hook on both successful and exceptional
forwards.

Deterministic tests prove that the returned representation has fixed width
`3 * pair_embedding_dim`, exactly replays the classifier logits, and retains
the same state dictionary across PRESENT, GIFT, SKINNY, RECTANGLE and a
synthetic 128-bit SPN while block width, pair count and recurrent window length
change. This is an implementation and evaluation-interface result only; it
does not add a new trained model or AUC. A native model `encode()` method
remains deferred until the frozen RECTANGLE RCT2 source handoff no longer
depends on an unchanged RuntimeE4 model file.

The public representation boundary now also provides
`FrozenRuntimeE4HeadAdapter`. It keeps the complete target-structure-bound
RuntimeE4 extractor, including its original source classifier, frozen and in
evaluation mode while an independent target head trains on the pooled
representation. The adapter therefore does not overwrite a source checkpoint
component or let dropout drift inside a nominally frozen extractor. Its state
dictionary loads strictly across supported runtime structures when the shared
RuntimeE4 specification and target-head geometry match; only target-head
parameters receive gradients. This is reusable infrastructure for a future
cross-cipher adaptation matrix, not evidence that such a matrix has already
passed.

The frozen-head adapter is compatible with the ordinary binary trainer rather
than requiring a second training loop. A real two-epoch integration test now
proves target-head-only optimization, best-validation-checkpoint restoration,
serialized state equality and strict reload onto a different runtime SPN
structure. The wrapper mirrors the bound target structure's descriptor,
transition-window and input-order metadata, so standard result construction
retains the runtime topology identity instead of reporting only aggregate
parameter counts. Adapter-specific ownership flags remain available to a
future experiment gate. No `engine/`, `training/` or frozen RuntimeE4 source
path changed while RECTANGLE RCT2 is queued.

## Cross-Cipher RECTANGLE Attribution And Linear Readout

RECTANGLE RCT1 subsequently supplied a non-contiguous-cell target with two
independent target-data seeds under the same four-pair, strict encrypted-
random-plaintext protocol. Rebinding the formal SKINNY seed0 RuntimeE4
extractor to the correct RECTANGLE descriptor while freezing the complete
extractor and original source classifier produced the following X3 target-head
results:

```text
target seed0 candidate AUC = 0.784893989563
target seed1 candidate AUC = 0.753169536591
```

Both candidates exceeded a corrupted SKINNY source checkpoint, a corrupted
RECTANGLE target topology and a deterministic random source extractor by at
least `0.088696`. They remained within `0.012504` of the same-data end-to-end
RECTANGLE anchors. The target head had `198401` trainable parameters, so X3
established frozen-representation usefulness but did not by itself show that
the source representation exposed a simple reusable decision direction.

X4 therefore froze the same eight source/target/seed roles and replaced the
nonlinear target head with one affine `Linear(384,1)` probe containing only
`385` trainable parameters. The candidate results were:

```text
target seed0 candidate AUC = 0.791651725769
target seed1 candidate AUC = 0.761355400085

minimum candidate-control margin = +0.046437263489
absolute candidate seed drift    = 0.030296325684
```

All preregistered source-topology, target-topology, random-source and seed-
stability gates passed. Independent artifact verification reopened all eight
best checkpoints and all 16 disk-backed representation caches, checking file
sets, SHA-256 identities, tensor geometry, final metrics, histories and
metadata. The result supports the narrow statement that the formal SKINNY
RuntimeE4 representation exposes linearly accessible RECTANGLE signal under
the local RCT1 protocol. It does not establish medium or formal cross-cipher
transfer, universal SPN adaptation, an attack, SOTA or a breakthrough. The
100-epoch linear-probe schedule is also not a same-compute superiority claim
over the five-epoch nonlinear X3 head.

## Remaining Completion Gaps

The implementation objective and the empirical method objective must remain
separate. Runtime structure loading, fixed parameter geometry, permutation and
general-GF(2) support, heterogeneous recurrent windows, frozen representation
extraction and low-capacity adaptation are implemented. The full method-level
goal remains incomplete until the following evidence is available:

1. **Formal general-GF(2) replication.** Retrieve and jointly adjudicate the
   active SKINNY RTG3-A `1000000/class` seed1 panel. One formal seed is not a
   multi-seed formal conclusion.
2. **Second-cipher medium topology anchor.** Complete RECTANGLE RCT2 at
   `65536/class` with correct, corrupted and no-topology controls. RCT1 is only
   a local diagnostic and cannot authorize a medium transfer claim.
3. **Heterogeneous multi-transition use.** Complete the queued local uKNIT U3
   dual-seed recurrent-window panel. The candidate must beat repeat-last,
   corrupted and no-topology controls; otherwise recurrent depth cannot be
   attributed to earlier runtime structure.
4. **Controlled medium cross-cipher confirmation.** Only if RCT2 passes,
   perform one unchanged SKINNY-to-RECTANGLE X3-B medium confirmation and keep
   the `385`-parameter probe as the low-capacity mechanism control.
5. **Breadth before universality.** A universal-SPN or broadly reusable-weight
   statement requires another preregistered source-target direction or real
   cipher beyond the current SKINNY-to-RECTANGLE pair, with the same control
   logic and multiple seeds. Implementation compatibility alone is not that
   evidence.

The current execution order is therefore fixed:

```text
RTG3-A seed1 retrieval and joint gate
    -> RECTANGLE RCT2 medium anchor
    -> conditional X3-B medium transfer

RTG3-A joint gate
    -> local uKNIT U3 heterogeneous-window attribution
```

Do not fill remote idle time with another architecture family, DDT/trail
features, related-key data, changed negatives or a broad cipher matrix. The
missing evidence is controlled replication and scale alignment of the same
runtime method.
