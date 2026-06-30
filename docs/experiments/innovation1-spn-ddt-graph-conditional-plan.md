# Innovation 1 SPN DDT-Graph Conditional Plan

**Date:** 2026-06-29

**Status:** conditional / implementation ready / waiting for attribution-control gate

**Scope:** PRESENT-80 r7, Zhang/Wang 2022 Case2 `m=16`, strict encrypted-random-plaintext negatives. This plan is a next-route guardrail, not a launched run.

## Why This Exists

The current active remote run is:

```text
run_id = i1_invp_only_r7_1m_seed0_gpu1_20260629
model = present_nibble_invp_only_spn_only
scale = 1000000/class
monitor = tmux: monitor_i1_invp_only_1m_20260629
```

While that run is active, do not launch another GPU job. This document defines
the next design route if the 1M InvP-only gate shows that the medium-scale
`InvP(DeltaC)` signal is tied with or below the Zhang/Wang paper-scale anchor.

## Trigger

Use this route only if the InvP-only 1M result is tied or underperforming:

```text
InvP-only 1M AUC - Zhang/Wang 1M anchor AUC < +0.001
```

If InvP-only beats the Zhang/Wang 1M anchor by `>= +0.003` AUC, do not start
this route immediately. Instead, run a 1M seed1 confirmation for InvP-only.
If the result is weakly positive from `+0.001` to `+0.003` AUC, still run seed1
before making a route-strength claim.

## Hypothesis

The existing InvP-only route may capture a useful SPN representation but lacks
explicit local S-box transition priors. A DDT-aware cell graph can test whether
real SPN topology and local differential legality/probability add information
beyond:

```text
DeltaC-only
InvP(DeltaC)-only
DeltaC + InvP(DeltaC)
pair-consistency pooling
generic token mixing
```

## Proposed N3 Matrix

Scale for first non-smoke run:

```text
262144/class
seed = 0
```

Keep the matrix lean, 3 to 4 rows:

| Row | Model/route | Role |
|---:|---|---|
| 0 | `present_nibble_invp_only_spn_only` | current simple InvP anchor |
| 1 | `present_nibble_paligned_transition_residual` or successor | true-P topology/residual anchor |
| 2 | new `present_nibble_ddt_graph` | DDT-aware true-P cell graph candidate |
| 3 | new `present_nibble_shuffled_ddt_graph` | shuffled-P topology control |

Do not include Zhang/Wang in this matrix unless the preceding 1M result suggests
baseline drift. The already completed 262k/1M Zhang/Wang anchors are sufficient
for context; this matrix is an attribution test among SPN-structured routes.

## Model Sketch

Input from raw ciphertext pairs:

```text
For each pair:
  DeltaC bits
  InvP(DeltaC) bits
  16 PRESENT nibble cells
```

Per-cell features:

```text
delta_c_nibble_bits      4 bits
invp_delta_nibble_bits   4 bits
active_delta             1 bit
active_invp              1 bit
hw_delta                 scalar or 4-bit bucket
hw_invp                  scalar or 4-bit bucket
ddt_best_input_diff      4 bits
ddt_best_count           count bucket
ddt_margin/top2          optional bucket
```

Graph:

```text
nodes = 16 PRESENT S-box cells
edges = fixed PRESENT P-layer bit-to-cell adjacency
control_edges = deterministic shuffled adjacency
```

Network:

```text
shared cell encoder
DDT/active gate
2-3 fixed topology message-passing blocks
pair-level evidence pooling across 16 pairs
binary classifier head
```

## Required Controls

No topology claim is allowed without:

```text
true-P graph vs shuffled-P graph
same input, same budget
same seed
same negative_mode = encrypted_random_plaintexts
same validation key and metric computation
```

No DDT-prior claim is allowed without:

```text
DDT-aware graph vs same graph without DDT/active features
or an explicit no-DDT ablation in the next matrix
```

## Decision Gates

Primary metric:

```text
val_auc
```

Continue only if:

```text
DDT-graph AUC >= InvP-only anchor AUC + 0.001
and DDT-graph AUC >= shuffled-DDT-graph AUC + 0.001
and calibrated_accuracy is not worse than InvP-only
```

Weak continue:

```text
DDT-graph is best but margin < 0.001
```

Action:

```text
repeat 262144/class seed1 before any 1M run
```

Stop:

```text
DDT-graph <= InvP-only
or true-P topology <= shuffled topology
```

Action:

```text
record as negative evidence; do not scale N3 to 1M.
```

## Implementation Notes

Existing useful code:

```text
src/blockcipher_nd/models/structure/spn/present_p_layer_mixer.py
  - _present_nibble_adjacency_indices()
  - PresentPLayerMixerBlock

src/blockcipher_nd/models/structure/spn/present_nibble_paligned_mcnd.py
  - _PresentNibblePAlignedSpnEncoder
  - _PresentNibbleTransitionResidualEncoder
  - _present_inverse_p_index()
  - existing InvP/Delta transition model and shuffled controls

src/blockcipher_nd/features/encoders/present_sbox_ddt.py
  - PRESENT_SBOX_DDT
  - present_sbox_ddt_words()
  - present_sbox_ddt_topk_words()

src/blockcipher_nd/features/encoders/present_matrix.py
  - integer reference implementation for pair-xor, InvP, and SBox-DDT feature words
```

Likely smallest implementation:

```text
1. Add a compact DDT cell-feature builder inside present_nibble_paligned_mcnd.py first.
2. Reuse or promote the existing PRESENT P-layer adjacency helper; do not create
   a second incompatible graph convention.
3. Register only two new keys initially:
   - present_nibble_ddt_graph
   - present_nibble_shuffled_ddt_graph
4. Add forward/build tests before any matrix launch.
5. Add smoke CSV and CPU smoke.
6. Add 262144/class CSV only after smoke passes.
```

Do not make this generic for SKINNY/GIFT in the first implementation. Generic
SPN graph abstractions are allowed only after PRESENT shows a positive route.

### Code Audit Before Branch B

Audit date: 2026-06-29.

Current source state:

```text
Transition and InvP-only classes are currently in:
  src/blockcipher_nd/models/structure/spn/present_nibble_paligned_mcnd.py

There is no separate present_nibble_transition.py module.
```

Branch B should therefore start in the existing PRESENT nibble module, unless
the file is split first as a narrow mechanical refactor. Do not create a new
module and duplicate transition helpers without moving tests at the same time.

The first DDT builder should be tensor-native and should consume the same raw
`ciphertext_pair_bits` cache as InvP-only:

```text
raw pair bits -> DeltaC bits -> InvP(DeltaC) bits
              -> DDT best input-difference bits
              -> DDT confidence/count bits
              -> active DeltaC / active InvP flags
```

Use `PRESENT_SBOX_DDT` as the source of truth for best input-difference and
count/confidence values. Avoid the slower integer feature-encoding path for the
training model; keep the integer `present_matrix.py` functions as a reference
for tests and feature sanity checks.

The shuffled control should alter only topology/alignment, not inputs, hidden
size, pooling mode, training protocol, or negative-sample definition.

## Gate-To-Execution Branches

When the active InvP-only 1M run is retrieved, make exactly one branch decision
from the validated local JSONL. Do not start both branches.

### Branch A: InvP-only Survives Paper Scale Enough For Seed1

Strong condition:

```text
InvP-only 1M AUC - 0.793897025948 >= +0.003
```

Weak-positive condition:

```text
+0.001 <= InvP-only 1M AUC - 0.793897025948 < +0.003
```

Action:

```text
1. Update docs/experiments/innovation1-invp-only-1m-scale-plan.md with the
   retrieved metric, gate result, artifacts, and claim scope.
2. Run scripts/check-remote-readiness on the prepared seed1 remote config.
3. Launch a seed1 confirmation for present_nibble_invp_only_spn_only at
   1000000/class using the same protocol, strict negatives, and val_auc
   checkpoint metric, only if the readiness gate passes.
4. Do not implement DDT graph before seed1 unless seed0/seed1 disagree enough
   to require an attribution study.
```

Reason:

```text
If the simplest InvP-only route is at least weakly positive at paper scale,
the next evidence gap is stability, not architectural complexity.
```

### Branch B: InvP-only Is Tied Or Underperforms At Paper Scale

Condition:

```text
InvP-only 1M AUC - 0.793897025948 < +0.001
```

Action:

```text
1. Update docs/experiments/innovation1-invp-only-1m-scale-plan.md with the
   retrieved metric, gate result, artifacts, and decision to enter this route.
2. Implement the minimal DDT graph route below.
3. Smoke locally with tiny samples.
4. Commit/push implementation, smoke config, and tests.
5. Launch the 262144/class attribution matrix only after smoke passes.
```

Reason:

```text
If InvP-only is tied with or below the Zhang/Wang anchor at 1M/class, the next
useful question is whether explicit S-box differential priors and true P
topology add information beyond the current learned InvP view.
```

## Minimal DDT Graph Ready Pack

This is the implementation checklist for Branch B. Keep it intentionally small
so the first DDT result is attributable.

Implementation update, 2026-06-30:

```text
commit = ef8ee17 feat: add present ddt graph candidate
model aliases implemented:
  - present_nibble_ddt_graph
  - present_nibble_shuffled_ddt_graph
smoke plan:
  configs/experiment/innovation1/innovation1_spn_present_ddt_graph_smoke.csv
first non-smoke conditional matrix:
  configs/experiment/innovation1/innovation1_spn_present_ddt_graph_r7_262k.csv
```

The implementation was aligned to the Branch B v1 refinement below: each node
uses `InvP(DeltaC)` nibble bits plus the full PRESENT S-box DDT output column
normalized by `count / 16.0`. This keeps the first DDT route focused on local
S-box transition priors and P-layer topology, not hand-picked top-k trail
engineering.

### Source Changes

```text
src/blockcipher_nd/models/structure/spn/present_nibble_paligned_mcnd.py
  - add _PresentNibbleDdtGraphEncoder
  - add PresentNibbleDdtGraphDistinguisher
  - add PresentNibbleShuffledDdtGraphDistinguisher

src/blockcipher_nd/models/structure/spn/__init__.py
  - export the two new model classes

src/blockcipher_nd/models/structure/__init__.py
  - export the two new model classes because the SPN registry imports from
    blockcipher_nd.models.structure

src/blockcipher_nd/registry/model_families/spn.py
  - register present_nibble_ddt_graph
  - register present_nibble_shuffled_ddt_graph

tests/test_project_structure.py
  - add forward/build coverage for both aliases
  - add a view/feature sanity test that DDT features are deterministic and
    differ from the no-DDT transition residual view
```

### Model Inputs

Use raw `ciphertext_pair_bits`, not a precomputed DDT feature encoding, so this
route stays comparable to the current InvP-only models and can reuse the same
disk-backed dataset cache.

Earlier sketch, superseded by the full-column v1 refinement below:

```text
DeltaC nibble bits                 4
InvP(DeltaC) nibble bits           4
best DDT input-difference bits     4
DDT confidence/count bucket bits   4
active DeltaC flag                 1
active InvP flag                   1
```

Do not add beam-search trail statistics in the first graph route. Those are a
separate hypothesis and should not be mixed into the first topology/DDT test.

Implementation audit refinement, 2026-06-29:

```text
Prefer the simpler full-column DDT input for Branch B v1:

aligned InvP(DeltaC) nibble bits       4
DDT column counts by input difference 16
total                                 20 features per nibble node
```

Reason:

```text
The full DDT column is tensor-native, avoids Python top-k/sort tie-breaking in
the model forward pass, preserves all local S-box transition counts, and keeps
the first DDT route attributable to topology + local transition priors rather
than beam-search trail engineering.
```

Axis convention:

```text
PRESENT_SBOX_DDT[input_difference][output_difference]

Model lookup should observe an aligned output difference and retrieve all
candidate input-difference counts. Register or compute a transposed buffer:

ddt_by_output[output_difference][input_difference]
```

Normalize DDT counts to probability-like features with `count / 16.0` before
concatenating them with bit features.

Expected Branch B v1 tensor flow:

```text
features                      (B, 128 * pairs_per_sample)
raw pairs                     (B, P, 2, 64)
DeltaC bits                   (B, P, 64)
InvP-aligned DeltaC bits      (B, P, 64)
aligned output nibble bits    (B * P, 16, 4)
aligned output nibble ids     (B * P, 16)
DDT column counts             (B * P, 16, 16)
DDT graph token input         (B * P, 16, 20)
token hidden                  (B * P, 16, token_dim)
P-layer graph mixer output    (B * P, 16, token_dim)
pair embedding                (B * P, embedding_bits)
pair-set embeddings           (B, P, embedding_bits)
EvidencePooling output        (B, embedding_bits)
logits                        (B, 1)
```

Use `PresentPLayerMixerBlock(words_per_pair=1)` for the initial graph route.
Do not pass 32 tokens to a mixer configured with `words_per_pair=1`.

### Graph/Mixer

Use one deterministic message-passing block family:

```text
true route:
  fixed PRESENT P-layer nibble adjacency from present_p_layer_mixer.py

control route:
  deterministic shuffled adjacency with a fixed seed

shared:
  same cell encoder
  same hidden size
  same mixer depth
  same evidence pooling
  same classifier head
```

The shuffled route is required in the first matrix. Without it, any positive
result is only "more features helped", not "true SPN topology helped".

### Smoke Plan

Add a tiny smoke CSV before any remote launch, but only after the model aliases
exist and pass build/forward tests. Do not commit a CSV that references
unregistered model keys, because that creates a broken experiment entrypoint
that cannot be smoke-tested honestly.

```text
configs/experiment/innovation1/innovation1_spn_present_ddt_graph_smoke.csv
```

Rows:

```text
present_nibble_ddt_graph
present_nibble_shuffled_ddt_graph
```

Use:

```text
samples_per_class = 8
pairs_per_sample = 16
rounds = 7
seed = 0
feature_encoding = ciphertext_pair_bits
negative_mode = encrypted_random_plaintexts
checkpoint_metric = val_auc
```

### First Non-Smoke Plan

Only after smoke passes, add:

```text
configs/experiment/innovation1/innovation1_spn_present_ddt_graph_r7_262k.csv
```

Rows:

```text
present_nibble_invp_only_spn_only
present_nibble_paligned_transition_residual
present_nibble_ddt_graph
present_nibble_shuffled_ddt_graph
```

Keep `present_nibble_paligned_transition_residual` as the no-DDT topology/residual
anchor. Do not add Zhang/Wang to this matrix unless the retrieved 1M result shows
baseline drift or protocol mismatch.

Launch guardrail:

```text
Do not launch this 262144/class matrix until the active 1M attribution-control
run is retrieved, validated, and the gate selects the DDT/topology branch.
Prepared matrix status is planned/ready, not launched/running evidence.
```

### Implementation Order Guardrail

For Branch B, use this exact order:

```text
1. Implement encoder/model classes.
2. Register and export aliases.
3. Add unit tests for alias build/forward and deterministic DDT features.
4. Add smoke CSV.
5. Run CPU smoke.
6. Commit/push code + tests + smoke config.
7. Add 262144/class CSV and remote config.
8. Launch remote 262144/class only from the pushed commit.
```

Do not create remote config, launch script, or monitor handoff before step 6
passes. The DDT route is conditional evidence, so a broken or partially defined
config would add bookkeeping noise without advancing the research question.

### Branch B Test Guardrails

Add tests before the first smoke CSV is committed:

```text
1. DDT source table:
   - PRESENT_SBOX_DDT has shape 16 x 16
   - each input-difference row sums to 16
   - transposed lookup matches the original table convention

2. Tensor DDT features:
   - known raw ciphertext pair bits produce deterministic InvP-aligned nibble ids
   - ddt_by_output[output][input] equals PRESENT_SBOX_DDT[input][output]
   - count normalization is exactly count / 16.0

3. Alias build/forward:
   - present_nibble_ddt_graph builds from the registry
   - present_nibble_shuffled_ddt_graph builds from the registry
   - input_bits = 2048 and pair_bits = 128 produce logits shaped (batch, 1)

4. Control route:
   - true and shuffled routes differ only in alignment/topology
   - both routes keep the same feature encoding, pooling, hidden sizes, negative
     mode, validation protocol, and metric computation

5. Shape failures:
   - non-2D inputs or mismatched input_bits/pair_bits raise ValueError in the
     same style as existing PRESENT nibble models
```

Do not add a test that requires `present_nibble_ddt_graph` to exist before
Branch B is selected. Until the InvP-only 1M gate chooses Branch B, the DDT
aliases remain planned implementation aliases, not registered model keys.

## Current Waiting Condition

Before implementing or launching this route, inspect the completed artifacts for:

```text
outputs/remote_results/i1_invp_only_r7_1m_seed0_gpu1_20260629/
```

Required evidence:

```text
done.marker or failed.marker retrieved
results JSONL present
result_lines = 1
validate-results status = pass
curves/history regenerated locally
docs/experiments/innovation1-invp-only-1m-scale-plan.md updated and committed
```

Only then choose whether to launch InvP-only seed1 or implement the DDT-graph route.
