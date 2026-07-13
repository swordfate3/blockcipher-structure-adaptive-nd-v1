# Innovation 1 Cross-SPN Typed Cell Transfer Design

**Status:** design frozen under the user's standing autonomous-execution approval
**Date:** 2026-07-13
**Experiment label:** E4

## Decision Summary

E4 tests one method-level Innovation 1 hypothesis:

```text
A neural operator that assigns different semantic roles to current-round
Delta cells and cipher-spec inverse-permuted previous-round cells can share
all trainable weights across PRESENT-80 and GIFT-64, and PRESENT pretraining
can improve a GIFT target beyond scratch and topology controls.
```

This is not another PRESENT-only network variant. It contains no DDT, trail,
beam search, S-box-table score, deterministic trail statistic, or new negative
sample definition. The only cipher-specific component is a fixed, generated,
non-persistent 64-bit inverse-permutation index buffer.

E4 has two conditional research rungs:

1. `E4-R1`: establish that the shared typed operator retains attributable
   source signal on strict PRESENT Case2 data while remaining executable with
   an identical state-dict geometry on GIFT-64.
2. `E4-R2`: only if R1 passes, test full state-dict PRESENT-to-GIFT transfer
   against scratch, source-topology, target-topology, and prior GIFT anchors.

No remote run is authorized by this design. All `8192/class` work remains a
local diagnostic, not formal SPN evidence.

## Evidence Behind The Design

### Closed PRESENT-only architecture gates

H2 and E3-R1 used the same strict PRESENT-80 r7 Case2 data and both rejected
their new learners:

```text
H2 Case3 residual candidate - Token-Mixer anchor = -0.003814578056 AUC
E3-R1 DBitNet candidate      - Token-Mixer anchor = -0.233713239431 AUC
```

The E3-R1 DBitNet rows reached final train AUC near `1.0` while validation
remained near random. More epochs, a seed1 repeat, or mechanical scale is not
the next justified slot.

### Existing GIFT evidence

The old GIFT route used:

```text
feature = C || C' || DeltaC || InvP_cipher(DeltaC)
model   = generic spn_token_mixer_pairset
```

At `2048/class`, three seeds, that aligned row was weak but consistently best
with mean AUC `0.523893992106`. At `8192/class`, three seeds, its mean collapsed
to `0.505356142918` and it lost the XOR control on seed1. The exact old route is
therefore held and must not be rerun as E4.

The old model treats the four 64-bit words as one flat sequence of 64 generic
4-bit tokens. It does not encode `DeltaC` and `InvP(DeltaC)` as two typed views
of the same 16 SPN cells, does not share a checkpoint across ciphers, and does
not test transfer.

### Structural compatibility

PRESENT-80 and GIFT-64 both expose:

```text
block bits       = 64
cell bits        = 4
cells per state  = 16
SPN operations   = 4-bit substitution + bit permutation
```

Both repository cipher implementations provide callable
`inverse_permutation_layer` methods. Their inverse mappings differ, which is
the intended cipher-specific type adapter.

## Alternatives Considered

### A. Recommended: conditional typed-cell source gate then controlled transfer

Use one shared typed-cell architecture for both ciphers. First prove the
architecture preserves attributable PRESENT source signal. Then warm-start the
GIFT target from PRESENT true/shuffled source checkpoints and compare against
scratch and target-shuffled controls.

This changes one method hypothesis at a time and can distinguish operator
failure from transfer failure.

### B. Direct PRESENT-to-GIFT transfer without a source gate

This is faster but uninterpretable if transfer fails: the shared operator may
never have learned a valid source representation. Rejected as the primary
protocol.

### C. Joint PRESENT/GIFT multi-task training with cipher-specific heads

This is a plausible later method, but it changes data scheduling, objective
weighting, optimizer dynamics, and architecture at once. It also requires a
new joint-data runner before any evidence gate. Rejected for E4.

## Cipher-Spec Mapping Contract

The mapping generator receives `cipher_key` and `mapping_mode`:

```text
cipher_key   = present80 | gift64
mapping_mode = true | shuffled | raw
```

For `true`, derive the MSB-ordered tensor index map from the cipher object's
`inverse_permutation_layer`; do not duplicate PRESENT or GIFT permutation
tables in the model.

For `shuffled`, use one fixed deterministic 64-bit permutation generated from
a documented seed. The same shuffled mapping is used for both ciphers, every
split, and every run.

For `raw`, use identity indices `0..63`.

Required mapping tests:

- each map is a permutation of `0..63`;
- generated PRESENT true mapping equals the existing
  `present_inverse_p_indices("true")` behavior;
- generated GIFT true mapping equals bitwise application of
  `Gift64.inverse_permutation_layer`;
- true, shuffled, and raw views differ on a deterministic non-symmetric input;
- the mapping buffer is non-persistent and is absent from `state_dict()`.

## Shared Typed Operator

All E4 model variants consume raw ciphertext pairs:

```text
pair_bits = 128
input = [C0(64), C1(64)]
```

For each pair:

```text
[batch, pairs, 2, 64]
  -> DeltaC = C0 xor C1
  -> PreviousDelta = mapping_cipher(DeltaC)
  -> reshape both as [batch, pairs, 16 cells, 4 bits]
  -> current_cell_encoder(DeltaC cell)
  -> previous_cell_encoder(PreviousDelta cell)
  -> concatenate the two role-specific embeddings
  -> shared typed fusion projection
  -> shared 16-cell mixer blocks
  -> shared per-pair summary
  -> shared permutation-invariant pair evidence pooling
  -> shared binary classifier
```

The current and previous cell encoders do not share parameters with each
other, because they represent different semantic types. Each encoder is,
however, shared across all 16 cell positions, all pairs, PRESENT, and GIFT.

The model must not receive a cipher ID, cipher embedding, round number, S-box
table, DDT table, key, difference-profile ID, or hand-authored topology
metadata. This prevents a cipher shortcut and keeps the transfer claim tied to
the generated mapping.

The following must have identical trainable parameter counts and identical
initial tensors under a common seed:

```text
PRESENT true / shuffled / raw
GIFT    true / shuffled / raw
```

All six variants must have identical `state_dict` keys and tensor shapes. This
is the mechanical prerequisite for full cross-cipher checkpoint transfer.

## Fixed Model Roles

Use thin registry aliases around one shared implementation:

| Cipher | Role | Mapping | Purpose |
| --- | --- | --- | --- |
| PRESENT | `anchor` | existing true InvP Token-Mixer | strongest same-protocol source reference |
| PRESENT | `typed_true` | generated PRESENT InvP | shared-operator source candidate |
| PRESENT | `typed_shuffled` | fixed shuffled map | source topology control |
| PRESENT | `typed_raw` | identity | source mapping/typed-capacity control |
| GIFT | `anchor` | raw-input wrapper around old aligned-input Token-Mixer | strongest directly comparable prior GIFT route |
| GIFT | `typed_true` | generated GIFT InvP | scratch target candidate |
| GIFT | `typed_shuffled` | fixed shuffled map | target topology control |
| GIFT | `typed_raw` | identity | target mapping/typed-capacity control |

PRESENT and GIFT are separate four-row matrices so each comparison remains
lean and same-protocol. The two matrices share architecture settings and
training budget but preserve each cipher's existing benchmark construction.

The GIFT anchor wrapper has no new trainable behavior. It accepts the same raw
128-bit pair input as the typed rows, reconstructs
`C || C' || DeltaC || InvP_GIFT(DeltaC)` inside the model, and delegates to the
existing `SpnTokenMixerPairSetDistinguisher` with 256 bits/pair. A locked
equivalence test must show identical logits to the old external encoder plus
old model under the same weights. This wrapper lets all four GIFT rows reuse
the exact raw dataset cache rather than merely regenerating nominally similar
samples through different feature encodings.

## Frozen Data Protocols

### PRESENT source

```text
cipher                    = PRESENT-80
rounds                    = 7
sample_structure          = zhang_wang_case2_official_mcnd
pairs_per_sample          = 16
feature_encoding          = ciphertext_pair_bits
negative_mode             = encrypted_random_plaintexts
effective_key_schedule    = per_pair_random, verified from metadata
difference_profile        = present_zhang_wang2022_mcnd member 0
```

### GIFT target

```text
cipher                    = GIFT-64
rounds                    = 6
sample_structure          = independent_pairs
pairs_per_sample          = 4
feature_encoding          = ciphertext_pair_bits
negative_mode             = encrypted_random_plaintexts
difference_profile        = gift64_shen2024_spn_screen member 0
train_key                 = 0x00000000000000000000000000000000
validation_key            = 0x11111111111111111111111111111111
```

Do not modify labels, validation construction, negative sampling, difference
members, metric computation, or checkpoint selection to improve E4.

Both matrices use:

```text
loss                      = mse
optimizer                 = adam
learning_rate             = 0.0001
weight_decay              = 0.00001
checkpoint_metric         = val_auc
restore_best_checkpoint   = true
train_eval_interval       = 1
```

## Execution Ladder

### E4-R0: implementation readiness

Run both four-role matrices with:

```text
train samples_per_class      = 64
validation samples_per_class = 32
seed                         = 0
epochs                       = 1
batch_size                   = 32
device                       = cpu
```

R0 checks exact generated mappings, typed tensor views, registry construction,
cross-cipher state-dict compatibility, equal candidate/control initialization,
finite forward/backward, disk-cache create/reuse, exact plan alignment,
histories/checkpoints, SVG generation, and neutral gate replay. R0 metrics are
not interpreted.

### E4-R1: local within-cipher source/target diagnostic

Run both frozen four-role matrices with:

```text
train samples_per_class      = 8192
validation samples_per_class = 4096
seed                         = 0
epochs                       = 10
batch_size                   = 256
device                       = cpu
```

This is diagnostic-only. R1 does not authorize seed1 or remote scale.

#### PRESENT source gate

R1 authorizes R2 only when:

```text
typed_true AUC >= 0.65
typed_true - anchor        >= -0.01 AUC
typed_true - typed_shuffled >= +0.003 AUC
typed_true - typed_raw      >= +0.003 AUC
```

The small non-inferiority allowance recognizes that E4 tests transferability,
not a new PRESENT SOTA. The topology and raw margins remain mandatory.

#### GIFT scratch diagnostic

GIFT scratch results are recorded with the same four-role attribution, but
they do not independently block R2 when the PRESENT source gate passes. A weak
scratch target is precisely where transfer may add value. Protocol errors,
unequal candidate/control capacity, broken cache reuse, missing histories, or
incompatible state-dict shapes do block R2.

If PRESENT is ordered but misses one source margin by at most `0.002`, run one
same-budget PRESENT seed1 fragility check. Otherwise reject this shared
operator before transfer.

### E4-R2: conditional PRESENT-to-GIFT transfer gate

R2 uses the restored R1 source checkpoints. The true and shuffled PRESENT
source models must have equal source budgets and complete provenance.

Target rows use the same GIFT train/validation cache and the same fine-tuning
budget:

| Role | Initialization | Target mapping |
| --- | --- | --- |
| `gift_anchor` | scratch raw-input wrapper around old aligned Token-Mixer | internally generated true GIFT aligned view |
| `gift_typed_scratch` | scratch shared typed operator | true GIFT |
| `true_to_true` | full PRESENT true state dict | true GIFT |
| `shuffled_to_true` | full PRESENT shuffled state dict | true GIFT |
| `true_to_shuffled` | full PRESENT true state dict | shuffled GIFT |

Do not reset the classifier or silently drop incompatible tensors. Full load
must use strict state-dict matching. This tests whether the complete learned
binary structural representation transfers. All target rows then receive the
same number of GIFT fine-tuning epochs.

R2 seed0 promotes a same-budget seed1 repeat only when:

```text
true_to_true AUC >= 0.52
true_to_true - gift_anchor          >= +0.003 AUC
true_to_true - gift_typed_scratch   >= +0.005 AUC
true_to_true - shuffled_to_true     >= +0.003 AUC
true_to_true - true_to_shuffled     >= +0.003 AUC
```

Interpretation:

| Outcome | Decision |
| --- | --- |
| all R2 margins pass | `promote_e4_transfer_seed1` |
| true-to-true is best but misses a margin | `weak_transfer_no_scale` |
| source-shuffled matches true source | `generic_pretraining_not_typed_transfer` |
| target-shuffled matches target true | `target_topology_not_attributed` |
| scratch or old anchor matches/beats transfer | `reject_e4_transfer` |
| protocol/provenance/state mismatch | `invalid_e4_protocol` |

No `65536/class`, `262144/class`, remote GPU, or formal run is authorized until
both transfer seeds pass a separately frozen joint gate.

## Cache And Artifact Contract

Every executed rung must use disk-backed, parameter-matched train/validation
caches and produce:

```text
results.jsonl
progress.jsonl
validation.json
history.csv
curves.svg
strict gate JSON
best checkpoints
cache metadata and completion events
```

R2 additionally records:

```text
source checkpoint path and SHA-256
source cipher/model/mapping/seed/budget
strict state-dict load report
target initialization role
target fine-tuning budget
```

If a later remote plan is ever authorized, it must use the existing
`G:\lxy\blockcipher-structure-adaptive-nd-runs` boundary, disk-backed cache,
progress logging, pushed source commit, and watcher-managed retrieval.

## Stop Boundary

E4 does not authorize:

- another DDT, trail, beamstats, or residual-focus run;
- another flattened DBitNet or fixed PRESENT Conv2D adapter;
- changing PRESENT or GIFT data construction to help the candidate;
- calling `8192/class`, `65536/class`, or `262144/class` formal training;
- remote execution before the local transfer gate passes;
- claiming cross-SPN transfer from shared code or compatible tensor shapes
  alone.

If E4 fails its source attribution or transfer controls, Innovation 1 should
stop adding architecture variants and consolidate the verified contribution:

```text
a controlled SPN structural-representation methodology showing which
cipher-derived views survive true/shuffled/raw and cross-cipher transfer gates,
including rigorous negative results where apparent gains do not transfer.
```

## Immediate Next Action

Implement only E4-R0 and E4-R1 first:

1. generic cipher-spec mapping generator;
2. one shared typed-cell operator with six fixed mapping aliases;
3. two four-role matrices and one readiness/source gate;
4. local R0 readiness;
5. local R1 only after R0 passes;
6. implement the R2 transfer runner only if the PRESENT source gate authorizes
   it.
