# Innovation 1 SPN DDT-Graph Conditional Plan

**Date:** 2026-06-30

**Status:** seed0 remote running / watcher-managed / attribution-control gate passed

**Scope:** PRESENT-80 r7, Zhang/Wang 2022 Case2 `m=16`, strict encrypted-random-plaintext negatives. This plan is a method-extension diagnostic route after the InvP-only attribution-control gate passed.

## Why This Exists

The completed attribution-control run is:

```text
run_id = i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630
models = present_nibble_delta_only_spn_only,
         present_nibble_shuffled_paligned_spn_only
scale = 1000000/class
decision = support_invp_structural_attribution
```

The InvP-only 1M seed0/seed1 confirmation has already shown a stable positive
signal over the local Zhang/Wang 1M anchor, and the 1M attribution controls
support true InvP/P-layer alignment as the useful SPN structure signal. This
document now launches DDT/topology as a stronger method-extension branch, not as
a rescue branch for a failed InvP explanation.

## Trigger

Use this route as the next experimental branch because the active
attribution-control run has been retrieved, validated, postprocessed, and
documented. The trigger is:

```text
decision = support_invp_structural_attribution
next-stage choice = optional DDT/topology method-extension branch
```

This route must be interpreted as medium diagnostic evidence. It tests whether
explicit S-box DDT priors and true P-layer graph topology can improve beyond
the now-supported InvP-only anchor. The first 262144/class seed0 run has been
launched and is currently watcher-managed; no result claim is allowed until the
watcher retrieves five plan-aligned rows and postprocess validation passes.

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

Keep the matrix lean but attribution-complete, 5 rows:

| Row | Model/route | Role |
|---:|---|---|
| 0 | `present_nibble_invp_only_spn_only` | current simple InvP anchor |
| 1 | `present_nibble_paligned_transition_residual` or successor | true-P topology/residual anchor |
| 2 | new `present_nibble_no_ddt_graph` | same graph mixer/pooling without DDT priors |
| 3 | new `present_nibble_ddt_graph` | DDT-aware true-P cell graph candidate |
| 4 | new `present_nibble_shuffled_ddt_graph` | shuffled-P topology control |

Do not include Zhang/Wang in this matrix unless a later audit suggests baseline
drift. The already completed 262k/1M Zhang/Wang anchors and the two-seed
InvP-only confirmation are sufficient for context; this matrix is an attribution
test among SPN-structured routes.

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
run the prepared 262144/class seed1 package as a variance check before any 1M run
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
3. Register three graph keys initially:
   - present_nibble_no_ddt_graph
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

When the active 1M attribution-control run is retrieved, make exactly one branch
decision from the validated local JSONL and the postprocess gate. Do not start
both branches.

### Branch A: Attribution Supports InvP

Condition:

```text
decision = support_invp_structural_attribution
```

Action:

```text
1. Update the InvP route-level evidence summary with the retrieved attribution
   metrics, gate result, artifacts, and claim scope.
2. Treat InvP-only as two-seed 1M positive confirmation plus paper-scale
   attribution-control support.
3. Decide whether the next useful work is formal multi-seed InvP evidence,
   Zhang/Wang baseline variance, or an optional DDT/topology method-extension
   branch.
4. Do not launch this DDT matrix merely because it is prepared.
```

Reason:

```text
Controls support true InvP/P-layer attribution. The selected next action for
this run is stronger-method exploration: test whether DDT/topology adds signal
beyond the supported InvP-only anchor.
```

### Branch B: Attribution Is Weak Or Negative

Condition:

```text
decision = weak_attribution_support
or
decision = weaken_invp_structural_attribution
```

Action:

```text
1. Update the InvP attribution plan and route-level summary with retrieved
   metrics, gate result, artifacts, and decision to enter this route.
2. Rerun the DDT remote readiness gate from the latest pushed commit.
3. Launch the prepared 262144/class DDT/topology attribution matrix only if
   readiness passes and current GPU availability supports it.
4. Hand off monitoring/retrieval to the local watcher; do not main-thread poll.
```

Reason:

```text
If the InvP gain is not clearly attributable to true alignment, the next useful
question is whether explicit S-box differential priors and true P topology add
information beyond generic DeltaC, false alignment, and the current InvP view.
```

## Minimal DDT Graph Ready Pack

This is the implementation checklist for Branch B. Keep it intentionally small
so the first DDT result is attributable.

Implementation update, 2026-06-30:

```text
commit = ef8ee17 feat: add present ddt graph candidate
model aliases implemented:
  - present_nibble_no_ddt_graph
  - present_nibble_ddt_graph
  - present_nibble_shuffled_ddt_graph
smoke plan:
  configs/experiment/innovation1/innovation1_spn_present_ddt_graph_smoke.csv
first non-smoke conditional matrix:
  configs/experiment/innovation1/innovation1_spn_present_ddt_graph_r7_262k.csv
prepared remote config:
  configs/remote/innovation1_spn_present_ddt_graph_r7_262k_gpu0_20260630.json
gate command:
  scripts/gate-ddt-graph-result
postprocess command:
  scripts/postprocess-ddt-graph-result
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
  - add PresentNibbleNoDDTGraphDistinguisher
  - add PresentNibbleDdtGraphDistinguisher
  - add PresentNibbleShuffledDdtGraphDistinguisher

src/blockcipher_nd/models/structure/spn/__init__.py
  - export the three new graph model classes

src/blockcipher_nd/models/structure/__init__.py
  - export the three new graph model classes because the SPN registry imports from
    blockcipher_nd.models.structure

src/blockcipher_nd/registry/model_families/spn.py
  - register present_nibble_no_ddt_graph
  - register present_nibble_ddt_graph
  - register present_nibble_shuffled_ddt_graph

tests/test_project_structure.py
  - add forward/build coverage for all three graph aliases
  - add a view/feature sanity test that DDT features are deterministic and the
    same-graph no-DDT control keeps only InvP-aligned nibble bits
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
present_nibble_no_ddt_graph
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
present_nibble_no_ddt_graph
present_nibble_ddt_graph
present_nibble_shuffled_ddt_graph
```

Keep `present_nibble_paligned_transition_residual` as the older no-DDT
topology/residual anchor, and add `present_nibble_no_ddt_graph` as the stricter
same-graph no-DDT control. This separates "the graph mixer/topology helped" from
"the S-box DDT prior helped". Do not add Zhang/Wang to this matrix unless the
retrieved 1M result shows baseline drift or protocol mismatch.

Launch guardrail:

```text
Launch this 262144/class matrix only after:
  - the 1M attribution-control run is retrieved and validated,
  - the attribution gate supports InvP alignment,
  - this route is explicitly selected as a method-extension diagnostic,
  - readiness passes from the latest pushed commit.
```

Prepared remote readiness command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/check-remote-readiness \
  --config configs/remote/innovation1_spn_present_ddt_graph_r7_262k_gpu0_20260630.json \
  --output /tmp/i1_spn_ddt_graph_r7_262k_gpu0_20260630_readiness.json
```

This readiness command was used before the 2026-06-30 seed0 launch. For any
future seed1 or rerun, rerun readiness from the latest pushed commit and choose
the GPU according to current remote availability.

Gate command after retrieval:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/gate-ddt-graph-result \
  --results outputs/remote_results/<run_id>/results/<run_id>.jsonl \
  --output outputs/remote_results/<run_id>/<run_id>_ddt_graph_gate.json \
  --expected-rows 5
```

The gate compares `present_nibble_ddt_graph` against four same-budget controls:

```text
present_nibble_invp_only_spn_only
present_nibble_paligned_transition_residual
present_nibble_no_ddt_graph
present_nibble_shuffled_ddt_graph
```

Decision meanings:

```text
support_ddt_graph_route -> run 262144/class seed1 confirmation before any 1M scale
weak_ddt_graph_signal   -> run 262144/class seed1 variance check before scaling
stop_ddt_graph_route    -> record tied/negative evidence and switch hypothesis
```

Prepared conditional seed1 assets:

```text
seed1_plan = configs/experiment/innovation1/innovation1_spn_present_ddt_graph_r7_262k_seed1.csv
seed1_remote_config = configs/remote/innovation1_spn_present_ddt_graph_r7_262k_seed1_gpu1_20260630.json
seed1_launcher = configs/remote/generated/run_i1_spn_ddt_graph_r7_262k_seed1_gpu1_20260630.cmd
seed1_monitor = configs/remote/generated/monitor_i1_spn_ddt_graph_r7_262k_seed1_gpu1_20260630.sh
claim_scope = conditional 262144/class medium seed1 only
```

These assets are intentionally prepared but not launched. They become actionable
only if seed0 is retrieved, validated, plan-aligned, and the gate decision is
`support_ddt_graph_route` or `weak_ddt_graph_signal`. Interpret the same seed1
matrix differently by branch:

```text
support_ddt_graph_route -> seed1 confirmation
weak_ddt_graph_signal   -> seed1 variance check
stop_ddt_graph_route    -> do not launch this seed1 package; switch to pair-set aggregation control
```

Bounded monitor-health command after launch:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/monitor-health \
  --run-id i1_spn_ddt_graph_r7_262k_seed0_gpu0_20260630 \
  --plan configs/experiment/innovation1/innovation1_spn_present_ddt_graph_r7_262k.csv \
  --plan-doc docs/experiments/innovation1-spn-ddt-graph-conditional-plan.md \
  --expected-rows 5 \
  --postprocess-kind ddt_graph
```

When `status` is `result_ready`, execute the emitted `postprocess_command`.
Do not postprocess `completed_missing_results`, `results_empty`, or
`results_incomplete`.

Full postprocess command after retrieval:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/postprocess-ddt-graph-result \
  --plan configs/experiment/innovation1/innovation1_spn_present_ddt_graph_r7_262k.csv \
  --results outputs/remote_results/<run_id>/results/<run_id>.jsonl \
  --output-dir outputs/remote_results/<run_id> \
  --run-id <run_id> \
  --expected-rows 5 \
  --update-plan-doc docs/experiments/innovation1-spn-ddt-graph-conditional-plan.md
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
   - present_nibble_no_ddt_graph builds from the registry
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

## Satisfied Launch Preconditions

Before the seed0 DDT launch, the completed attribution-control artifacts were
inspected under:

```text
outputs/remote_results/i1_invp_attribution_controls_r7_1m_seed0_gpu0_20260630/
```

Required evidence:

```text
done.marker or failed.marker retrieved
results JSONL present
result_lines = 2
validate-results status = pass
attribution gate JSON present
postprocess summary present
docs/experiments/innovation1-invp-only-formal-attribution-plan.md updated
docs/experiments/innovation1-invp-route-level-evidence-summary.md updated
docs updates verified, committed, and pushed
```

This evidence is satisfied as of commit `cc05d2a`. The selected action is to
launch the prepared DDT/topology matrix as a method-extension diagnostic from a
pushed commit, then let the local watcher retrieve and postprocess the result.

## Launch Record

Planned launch:

```text
run_id = i1_spn_ddt_graph_r7_262k_seed0_gpu0_20260630
scale = 262144/class
expected_rows = 5
device = cuda:0
claim_scope = medium diagnostic method-extension, not formal evidence
```

Readiness/smoke gates:

```text
check-remote-readiness status = pass
CPU smoke status = pass, 3/3 rows
```

Actual launch handoff:

```text
launch_commit = 26e9f26 docs: prepare ddt graph method extension launch
launch_time = 2026-06-30 14:53 +08:00
remote_launcher = G:\lxy\run_i1_spn_ddt_graph_r7_262k_seed0_gpu0_20260630.cmd
remote_run_root = G:\lxy\blockcipher-structure-adaptive-nd-runs\i1_spn_ddt_graph_r7_262k_seed0_gpu0_20260630
local_watcher = tmux monitor_i1_spn_ddt_graph_262k_20260630
local_result_root = outputs/remote_results/i1_spn_ddt_graph_r7_262k_seed0_gpu0_20260630
monitor_health = running, heartbeat not stale, no failed marker, no done marker
result_status = running / not yet retrieved
```

This launch record is not result evidence. The run remains medium diagnostic
only until the watcher retrieves five plan-aligned rows and
`postprocess-ddt-graph-result` validates the gate.
