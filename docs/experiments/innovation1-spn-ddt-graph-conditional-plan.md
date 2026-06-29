# Innovation 1 SPN DDT-Graph Conditional Plan

**Date:** 2026-06-29

**Status:** conditional / waiting for InvP-only 1M gate

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
`InvP(DeltaC)` signal is weak or unstable at paper scale.

## Trigger

Use this route only if the InvP-only 1M result is not strong enough:

```text
InvP-only 1M AUC - Zhang/Wang 1M anchor AUC < +0.003
```

If InvP-only beats the Zhang/Wang 1M anchor by `>= +0.003` AUC, do not start
this route immediately. Instead, run a 1M seed1 confirmation for InvP-only.

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
src/blockcipher_nd/models/structure/spn/present_nibble_paligned_mcnd.py
src/blockcipher_nd/features/encoders/present_sbox_ddt.py
src/blockcipher_nd/features/encoders/present_matrix.py
```

Likely smallest implementation:

```text
1. Add a compact DDT cell-feature builder inside the SPN model module first.
2. Reuse existing PRESENT P-layer adjacency logic.
3. Register only two new keys initially:
   - present_nibble_ddt_graph
   - present_nibble_shuffled_ddt_graph
4. Add forward/build tests before any matrix launch.
5. Add smoke CSV and CPU smoke.
6. Add 262144/class CSV only after smoke passes.
```

Do not make this generic for SKINNY/GIFT in the first implementation. Generic
SPN graph abstractions are allowed only after PRESENT shows a positive route.

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
