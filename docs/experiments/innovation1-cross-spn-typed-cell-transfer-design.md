# Innovation 1 Cross-SPN Typed Cell Transfer Design

**Status:** E4-R2 two-seed transfer signal confirmed; E4-R3 medium diagnostic design required
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

E4-R0, E4-R1, and E4-R2 seed0 were implemented and completed on 2026-07-13.
R2 seed0 passed every frozen transfer margin. The only authorized next
training action is an identical local target-seed1 repeat at `8192/class`;
do not increase sample count or launch remotely from the seed0 result.

## 2026-07-13 E4-R0/R1 Execution Record

### Implementation and readiness

```text
implementation commit = 3a54f75
R0 status             = pass
R0 decision           = implementation_ready
typed parameter count = 187426 for all PRESENT/GIFT true/shuffled/raw aliases
focused tests          = 40 passed
project/focused tests  = 492 passed, 8 warnings
```

R0 ran both four-role matrices at `64/class` training and `32/class`
validation, seed0, one epoch, CPU. Both matrices passed exact result/plan
alignment, finite history/checkpoint checks, SVG parsing, and neutral gate
replay. A second execution reused the parameter-matched train and validation
caches for all rows. R0 AUC values were not interpreted.

### E4-R1 frozen protocol

```text
PRESENT source:
  cipher/rounds       = PRESENT-80 r7
  sample structure    = Zhang/Wang Case2 official MCND
  train/validation    = 8192/class / 4096/class
  pairs/sample        = 16
  effective keys      = per_pair_random

GIFT target scratch:
  cipher/rounds       = GIFT-64 r6
  sample structure    = independent pairs
  train/validation    = 8192/class / 4096/class
  pairs/sample        = 4
  effective keys      = fixed split keys

shared:
  seed/epochs/device  = seed0 / 10 / CPU
  negative mode       = encrypted_random_plaintexts
  loss/optimizer      = MSE / Adam
  learning rate       = 0.0001
  weight decay        = 0.00001
  checkpoint          = restored best val_auc
```

This is a local single-seed `8192/class` diagnostic, not formal training,
paper-scale evidence, a ceiling result, or a breakthrough claim.

### E4-R1 results

PRESENT source:

| Role | Validation AUC | Accuracy | Best epoch | Final train AUC |
| --- | ---: | ---: | ---: | ---: |
| InvP-only Token-Mixer anchor | `0.745933175087` | `0.670043945312` | 10 | `0.769022487104` |
| typed true InvP | `0.743810147047` | `0.662353515625` | 10 | `0.770136669278` |
| typed shuffled mapping | `0.575898259878` | `0.545043945312` | 9 | `0.659925296903` |
| typed raw identity | `0.586375117302` | `0.553955078125` | 10 | `0.633669070899` |

```text
typed true absolute AUC = 0.743810147047
true - anchor           = -0.002123028040  >= -0.010 gate
true - shuffled         = +0.167911887169  >= +0.003 gate
true - raw              = +0.157435029745  >= +0.003 gate
```

GIFT scratch:

| Role | Validation AUC | Accuracy | Best epoch | Final train AUC |
| --- | ---: | ---: | ---: | ---: |
| historical aligned Token-Mixer anchor | `0.506567180157` | `0.500610351562` | 9 | `0.576292179525` |
| typed true InvP | `0.551968932152` | `0.535766601562` | 6 | `0.640496343374` |
| typed shuffled mapping | `0.500088214874` | `0.499877929688` | 1 | `0.642125204206` |
| typed raw identity | `0.501148313284` | `0.500732421875` | 3 | `0.587141521275` |

```text
true - anchor   = +0.045401751995
true - shuffled = +0.051880717278
true - raw      = +0.050820618868
```

The GIFT scratch result was not required to authorize transfer, but its strong
same-input control separation is additional diagnostic evidence that the
generated typed mapping is learning cipher-specific structural signal rather
than generic capacity alone.

### Validation and artifacts

```text
PRESENT plan:
  configs/experiment/innovation1/innovation1_spn_present_cross_spn_typed_cell_8192_seed0.csv
PRESENT artifacts:
  outputs/local_smoke/i1_present_cross_spn_typed_cell_r1_seed0/results.jsonl
  outputs/local_smoke/i1_present_cross_spn_typed_cell_r1_seed0/progress.jsonl
  outputs/local_smoke/i1_present_cross_spn_typed_cell_r1_seed0/history.csv
  outputs/local_smoke/i1_present_cross_spn_typed_cell_r1_seed0/curves.svg
  outputs/local_smoke/i1_present_cross_spn_typed_cell_r1_seed0/checkpoints/

GIFT plan:
  configs/experiment/innovation1/innovation1_spn_gift64_cross_spn_typed_cell_8192_seed0.csv
GIFT artifacts:
  outputs/local_smoke/i1_gift64_cross_spn_typed_cell_r1_seed0/results.jsonl
  outputs/local_smoke/i1_gift64_cross_spn_typed_cell_r1_seed0/progress.jsonl
  outputs/local_smoke/i1_gift64_cross_spn_typed_cell_r1_seed0/history.csv
  outputs/local_smoke/i1_gift64_cross_spn_typed_cell_r1_seed0/curves.svg
  outputs/local_smoke/i1_gift64_cross_spn_typed_cell_r1_seed0/checkpoints/

joint gate:
  outputs/local_smoke/i1_cross_spn_typed_cell_r1_seed0/gate.json
  status   = pass
  decision = promote_e4_r2
  errors   = []
```

Both result validators returned `status=pass`, four expected rows, no missing
or duplicate keys, and no field mismatches. The strict joint gate verified
complete histories/checkpoints, disk-cache creation and six control reuse
events per cipher, effective key schedules, equal typed capacities, and
cross-cipher state geometry.

### Historical R2 seed0 action (completed)

The completed seed0 research question was whether PRESENT pretraining
transfers useful typed structural weights to GIFT beyond GIFT scratch and both
source/target mapping controls. Freeze a separate E4-R2 implementation plan
before running:

1. Same-budget target anchor: completed GIFT aligned wrapper and typed scratch
   rows under the exact R1 GIFT cache and 10-epoch budget.
2. Required controls: PRESENT-shuffled to GIFT-true and PRESENT-true to
   GIFT-shuffled, in addition to PRESENT-true to GIFT-true.
3. One variable: full strict source checkpoint initialization and mapping role;
   do not change GIFT data, optimizer, target epochs, or checkpoint selection.
4. Readiness: prove checkpoint SHA-256 provenance, strict full state-dict load,
   identical target initial logits for a reloaded source checkpoint, and
   parameter-matched cache reuse before R2 interpretation.
5. R2 scale: local seed0, `8192/class` train, `4096/class` validation,
   10 target epochs, CPU. Apply the already frozen five-margin transfer gate.
6. Advance only if all five R2 margins pass; otherwise classify the exact
   failed attribution control and stop transfer scale.

Explicitly stopped now: PRESENT/GIFT R1 repeats, PRESENT seed1 fragility,
`65536/class`, `262144/class`, remote GPU, DDT/trail reopening, and ad hoc R2
training without checkpoint provenance.

The original implementation sequence was:

1. generic cipher-spec mapping generator;
2. one shared typed-cell operator with six fixed mapping aliases;
3. two four-role matrices and one readiness/source gate;
4. local R0 readiness;
5. local R1 only after R0 passes;
6. implement the R2 transfer runner only if the PRESENT source gate authorizes
   it.

## 2026-07-13 E4-R2 Seed0 Execution Record

### Implementation readiness and provenance

```text
implementation commit = cc8fff3
orchestration fix     = 6f3a904
R0 status             = pass
R0 decision           = implementation_ready
R0 protocol           = 64/class, seed0, 1 epoch, CPU
typed parameter count = 187426
focused tests          = 21 passed
full test suite        = 1260 passed, 14 warnings
ruff / diff check      = pass / pass
```

The readiness gate did not interpret its tiny-run AUC values. It verified the
five target roles, strict state-dict loading, source result/checkpoint
identity, equal typed capacity, shared target cache, and the frozen source
checkpoint hashes:

```text
PRESENT true source SHA-256:
  eae5ef9175fea3abeff7a78bc1608ac1922200dc341e7872c793eaba880a71c1
PRESENT shuffled source SHA-256:
  fff2e23d55c0daa3c8b3a346d2a3e5b66a3bbf2848e7f59d8aae87f7118e7c22
```

### Frozen target protocol

```text
cipher/rounds       = GIFT-64 r6
sample structure    = independent pairs
train/validation    = 8192/class / 4096/class
pairs/sample        = 4
target seed         = 0
target key schedule = fixed split keys
negative mode       = encrypted_random_plaintexts
epochs/device       = 10 / CPU
loss/optimizer      = MSE / Adam
learning rate       = 0.0001
weight decay        = 0.00001
checkpoint          = restored best val_auc
dataset cache       = outputs/local_cache/i1_gift64_cross_spn_typed_cell_r1_seed0
```

All five rows reused the exact R1 GIFT target train/validation cache. The only
experimental variable was checkpoint initialization and target mapping role.

### Results and gate

| Role | Validation AUC | Accuracy | Best epoch |
| --- | ---: | ---: | ---: |
| GIFT aligned anchor | `0.506567180157` | `0.500610351562` | 9 |
| GIFT typed scratch | `0.551968932152` | `0.535766601562` | 6 |
| PRESENT true to GIFT true | `0.569627493620` | `0.541015625000` | 10 |
| PRESENT shuffled to GIFT true | `0.544660240412` | `0.533325195312` | 6 |
| PRESENT true to GIFT shuffled | `0.508949667215` | `0.498901367188` | 3 |

```text
true_to_true absolute AUC = 0.569627493620  pass >= 0.52
true_to_true - anchor     = +0.063060313463 pass >= +0.003
true_to_true - scratch    = +0.017658561468 pass >= +0.005
true - source-shuffled    = +0.024967253208 pass >= +0.003
true - target-shuffled    = +0.060677826405 pass >= +0.003

status      = pass
decision    = promote_e4_transfer_seed1
next_action = freeze_identical_e4_r2_seed1_repeat
```

The source-shuffled control also exceeded target scratch, so generic
pretraining may contribute. It does not explain the full candidate gain:
true-source transfer remains `+0.024967253208` AUC above source-shuffled under
the same GIFT-true target mapping, and the target-shuffled control separately
loses by `+0.060677826405`. At this seed and budget, both learned source
topology and correct target topology are therefore attributable contributors.

This is a single-seed local `8192/class` diagnostic. It is not formal
training, paper-scale evidence, a SOTA result, or a breakthrough claim.

### Validation and artifacts

```text
plan:
  configs/experiment/innovation1/innovation1_spn_gift64_cross_spn_typed_transfer_8192_seed0.csv
source manifest:
  configs/experiment/innovation1/innovation1_spn_cross_spn_typed_transfer_seed0_sources.json
artifacts:
  outputs/local_smoke/i1_gift64_cross_spn_typed_transfer_r2_seed0/results.jsonl
  outputs/local_smoke/i1_gift64_cross_spn_typed_transfer_r2_seed0/progress.jsonl
  outputs/local_smoke/i1_gift64_cross_spn_typed_transfer_r2_seed0/validation.json
  outputs/local_smoke/i1_gift64_cross_spn_typed_transfer_r2_seed0/history.csv
  outputs/local_smoke/i1_gift64_cross_spn_typed_transfer_r2_seed0/curves.svg
  outputs/local_smoke/i1_gift64_cross_spn_typed_transfer_r2_seed0/gate.json
  outputs/local_smoke/i1_gift64_cross_spn_typed_transfer_r2_seed0/checkpoints/
```

The result validator returned five plan-aligned rows with no missing,
duplicate, unexpected, or mismatched keys. The SVG parsed successfully and
the history CSV contains 50 epoch rows. The strict gate returned no errors.

### Executable next action

The next research question is whether the attributed transfer margin repeats
under an independent target seed. Freeze one E4-R2 seed1 repeat as follows:

1. Same-budget anchor and controls: the same five roles, architectures,
   optimizer, source checkpoints, and 10-epoch target budget.
2. One variable: change only the GIFT target seed from 0 to 1 and use a new
   parameter-matched disk cache; keep both PRESENT source SHA-256 values fixed.
3. Scale and path: local CPU, `8192/class` train, `4096/class` validation,
   four pairs/sample, with five restored-best checkpoints and full progress.
4. Readiness gate: validate the seed1 CSV, strict source loads, shared target
   cache, complete histories/checkpoints, and SVG/CSV artifacts before
   interpreting metrics.
5. Advance gate: apply the identical absolute and four-margin thresholds to
   seed1, then freeze a joint two-seed adjudication. If seed1 fails any
   attribution control, stop transfer scale and retain R2 seed0 as provisional
   single-seed evidence only.
6. Explicitly stopped until the two-seed gate passes: `65536/class`,
   `262144/class`, formal-scale training, remote GPU, DDT/trail reopening, and
   architecture changes.

## 2026-07-14 E4-R2 Seed1 And Joint Adjudication

### Frozen seed1 protocol

Seed1 changed only the GIFT target seed and its parameter-matched disk cache.
The five roles, source checkpoints, architecture, data definition, optimizer,
loss, target epochs, validation policy, and gate thresholds remained identical
to seed0:

```text
cipher/rounds       = GIFT-64 r6
train/validation    = 8192/class / 4096/class
train/validation    = 16384 total / 8192 total
pairs/sample        = 4 independent pairs
target seed         = 1
negative mode       = encrypted_random_plaintexts
epochs/device       = 10 / CPU
loss/optimizer      = MSE / Adam
checkpoint          = restored best val_auc
typed parameters    = 187426
dataset cache       = outputs/local_cache/i1_gift64_cross_spn_typed_cell_r1_seed1
```

The frozen source checkpoint identities remained:

```text
PRESENT true source SHA-256:
  eae5ef9175fea3abeff7a78bc1608ac1922200dc341e7872c793eaba880a71c1
PRESENT shuffled source SHA-256:
  fff2e23d55c0daa3c8b3a346d2a3e5b66a3bbf2848e7f59d8aae87f7118e7c22
```

### Seed1 results

| Role | Validation AUC | Accuracy | Calibrated accuracy | Best epoch |
| --- | ---: | ---: | ---: | ---: |
| GIFT aligned anchor | `0.551836639643` | `0.535888671875` | `0.541259765625` | 9 |
| GIFT typed scratch | `0.563941299915` | `0.542114257812` | `0.546630859375` | 10 |
| PRESENT true to GIFT true | `0.575072139502` | `0.551513671875` | `0.557128906250` | 7 |
| PRESENT shuffled to GIFT true | `0.559742510319` | `0.539428710938` | `0.540649414062` | 7 |
| PRESENT true to GIFT shuffled | `0.518017381430` | `0.509399414062` | `0.516601562500` | 10 |

```text
true_to_true absolute AUC = 0.575072139502  pass >= 0.52
true_to_true - anchor     = +0.023235499859 pass >= +0.003
true_to_true - scratch    = +0.011130839586 pass >= +0.005
true - source-shuffled    = +0.015329629183 pass >= +0.003
true - target-shuffled    = +0.057054758072 pass >= +0.003

status      = pass
decision    = promote_e4_transfer_joint_gate
next_action = run_frozen_e4_r2_joint_gate
```

### Joint two-seed gate

| Target seed | True-to-true AUC | vs anchor | vs scratch | vs source-shuffled | vs target-shuffled |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `0.569627493620` | `+0.063060313463` | `+0.017658561468` | `+0.024967253208` | `+0.060677826405` |
| 1 | `0.575072139502` | `+0.023235499859` | `+0.011130839586` | `+0.015329629183` | `+0.057054758072` |

Both target seeds pass the same absolute threshold and all four attribution
margins. The evidence supports a repeatable local transfer signal in which the
correct PRESENT source topology and correct GIFT target topology both matter
beyond scratch and shuffled controls.

```text
status      = pass
decision    = two_seed_transfer_signal_confirmed
next_action = design_e4_r3_same_protocol_medium_diagnostic
```

This remains a two-seed local `8192/class` diagnostic. It is not formal
training, paper-scale evidence, remote evidence, SOTA, or a breakthrough.
Remote launch, formal claims, DDT/trail/E1/H2 reopening, and unplanned
architecture changes remain stopped.

### Validation and artifacts

```text
plan:
  configs/experiment/innovation1/innovation1_spn_gift64_cross_spn_typed_transfer_8192_seed1.csv
seed1 artifacts:
  outputs/local_smoke/i1_gift64_cross_spn_typed_transfer_r2_seed1/results.jsonl
  outputs/local_smoke/i1_gift64_cross_spn_typed_transfer_r2_seed1/progress.jsonl
  outputs/local_smoke/i1_gift64_cross_spn_typed_transfer_r2_seed1/validation.json
  outputs/local_smoke/i1_gift64_cross_spn_typed_transfer_r2_seed1/history.csv
  outputs/local_smoke/i1_gift64_cross_spn_typed_transfer_r2_seed1/curves.svg
  outputs/local_smoke/i1_gift64_cross_spn_typed_transfer_r2_seed1/gate.json
joint gate:
  outputs/local_smoke/i1_gift64_cross_spn_typed_transfer_r2_joint_seed0_seed1/gate.json
```

The seed1 result validator returned five plan-aligned rows with no errors. The
SVG parsed successfully, the history CSV contains 50 epoch rows, and the
single-seed and joint gates both returned `status=pass` with empty error lists.

### Executable next action

Freeze E4-R3 before execution. Its single research question is whether the
attributable PRESENT-to-GIFT transfer advantage survives a larger, otherwise
identical local budget. Use GIFT-64 r6, target seeds 0 and 1, the same five
roles and source hashes, `65536/class` training, `32768/class` validation,
four independent pairs/sample, strict negatives, 10 epochs, restored-best
`val_auc`, and separate disk-backed caches. Freeze exact per-seed margins and
cache-reuse/readiness gates in the E4-R3 plan before launching. Do not advance
to `262144/class`, remote GPU, or formal scale from the E4-R2 result alone.
