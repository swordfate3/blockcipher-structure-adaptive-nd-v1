# Innovation 1 Pair-Set Aggregation Control Plan

**Date:** 2026-06-30

**Status:** planned / gated / do not launch while DDT-topology run is active

**Scope:** PRESENT-80 r7, Zhang/Wang 2022 Case2 `m=16`, strict encrypted-random-plaintext negatives.

## Research Question

Does a learned SPN pair-set model extract useful cross-pair structure, or is the
observed multi-pair signal mostly explained by independent single-pair scores
combined with a simple aggregation rule?

This is a required control before claiming that pair-set evidence pooling is a
structure-adaptive data/model contribution.

## Why This Exists

The current strongest supported Innovation 1 route is:

```text
present_nibble_invp_only_spn_only
evidence = two-seed 1000000/class positive confirmation
attribution = paper-scale controls support true InvP/P-layer alignment
```

The project also contains a joint pair-set candidate:

```text
present_nibble_invp_pair_consistency_spn_only
```

That model pools 16 pair embeddings with mean/max/std and learned evidence
pooling. Medium-scale results previously showed only a small advantage over
InvP-only. Before expanding this direction, we need to separate two effects:

```text
Effect A: more independent pair evidence improves the distinguisher.
Effect B: the neural model learns non-trivial consistency across the 16 pairs.
```

Only Effect B supports a pair-set structure innovation claim.

## Trigger

Do not launch this run while:

```text
i1_spn_ddt_graph_r7_262k_seed0_gpu0_20260630
```

is running. This plan becomes actionable after one of these conditions:

```text
1. DDT/topology result is retrieved and negative/tied:
   switch from graph topology extension to pair-set attribution control.

2. DDT/topology result is positive:
   keep this as a later attribution control before making any pair-set claim.

3. User explicitly chooses pair-set data representation as the next route.
```

## Single Hypothesis

Holding cipher, rounds, sample structure, negative mode, train/validation keys,
optimizer, scheduler, checkpoint metric, and scale fixed:

```text
Learned pair-set consistency should beat frozen single-pair score aggregation
by at least +0.001 AUC if it captures real cross-pair structure.
```

If it does not beat frozen aggregation, pair-set pooling should be treated as
statistical aggregation or diagnostic-only, not as a primary structural
innovation.

## Required Implementation

This plan needs one small evaluation route before launch:

```text
frozen_single_pair_invp_logodds_aggregate
```

Expected behavior:

```text
1. Train or load a single-pair InvP-only scorer with pairs_per_sample = 1.
2. For a Zhang/Wang Case2 m=16 sample, slice the 2048-bit input into 16
   independent 128-bit ciphertext-pair views.
3. Apply the frozen single-pair scorer to each pair.
4. Aggregate logits or calibrated log-odds with fixed rules:
   - mean logit
   - sum log-odds
   - top-k mean or top-k logsumexp
5. Report the best predeclared aggregation rule on validation only.
6. Evaluate the chosen rule on the same held-out validation/test protocol as
   the joint pair-set model.
```

The frozen aggregator must not update weights on 16-pair samples. Otherwise it
becomes another learned pair-set model and no longer controls for independent
score aggregation.

## Implementation Status

Completed local foundation:

```text
module = src/blockcipher_nd/evaluation/pairset_aggregation.py
tests = tests/test_pairset_aggregation.py
capability = split a 16-pair sample into pair views, apply a frozen scorer to
             each pair, aggregate logits/log-odds, and compute binary metrics
```

Not implemented yet:

```text
1. Persisting best training checkpoints as run artifacts.
2. CLI for loading a saved single-pair InvP scorer checkpoint.
3. Smoke CSV and remote matrix rows for frozen aggregation controls.
```

Therefore this plan is still not launch-ready. The current code only makes the
core aggregation math testable and reusable.

## Candidate Matrix

First non-smoke scale:

```text
262144/class
seed = 0
expected_rows = 3 or 4
claim_scope = medium diagnostic only
```

Lean matrix:

| Row | Route | Role |
|---:|---|---|
| 0 | `present_nibble_invp_only_spn_only` | current supported InvP 16-pair anchor |
| 1 | `present_nibble_invp_pair_consistency_spn_only` | learned joint pair-set candidate |
| 2 | `frozen_single_pair_invp_logodds_aggregate` | independent score aggregation control |
| 3 | optional `frozen_single_pair_delta_only_logodds_aggregate` | checks whether aggregation alone works without InvP |

Do not include unrelated graph/DDT models in this matrix. This plan asks only
whether pair-set learning beats independent score aggregation.

## Protocol

| Field | Value |
|---|---|
| Cipher | `PRESENT-80` |
| Rounds | `7` |
| Difference profile | `present_zhang_wang2022_mcnd` |
| Sample structure | `zhang_wang_case2_official_mcnd` |
| Pairs per sample | `16` for joint/evaluation samples |
| Single-pair scorer | `pairs_per_sample = 1` checkpoint |
| Negative mode | `encrypted_random_plaintexts` |
| Train key | `0x00000000000000000000` |
| Validation key | `0x11111111111111111111` |
| Scheduler | `official_cyclic` |
| Checkpoint metric | `val_auc` |
| Primary metric | `val_auc` |

## Decision Gates

Continue:

```text
pair_consistency_auc >= frozen_aggregate_auc + 0.001
and pair_consistency_auc >= invp_only_anchor_auc + 0.001
```

Weak continue:

```text
pair_consistency_auc is best, but margin < 0.001
```

Action:

```text
repeat 262144/class seed1 before any 1M run
```

Stop:

```text
pair_consistency_auc <= frozen_aggregate_auc
or pair_consistency_auc <= invp_only_anchor_auc
```

Action:

```text
Do not scale pair-consistency to 1M as a main route. Treat it as aggregation
or diagnostic context, and return to topology/DDT, active-pattern, or new SPN
feature construction.
```

## Evidence Language

Allowed if the gate passes:

```text
At 262144/class diagnostic scale, learned pair-set consistency outperformed
the frozen single-pair aggregation control under the same PRESENT r7 protocol.
```

Not allowed:

```text
formal route evidence
proof of multi-pair structural learning
breakthrough
SOTA
```

## Remote Launch Requirements

Before any meaningful remote launch:

```text
1. Implement the frozen aggregation route and tests.
2. Add smoke CSV with tiny samples.
3. Run local CPU smoke.
4. Add 262144/class plan CSV and remote config.
5. Verify disk-backed cache/progress paths under G:\lxy.
6. Run scripts/check-remote-readiness.
7. Commit and push.
8. Launch from the pushed commit.
9. Hand off retrieval/postprocess to local tmux watcher.
```

## Relationship To Current DDT/Topology Run

Current running run:

```text
run_id = i1_spn_ddt_graph_r7_262k_seed0_gpu0_20260630
status = running / watcher-managed
```

This pair-set aggregation plan should not preempt that result. It is the next
clean attribution plan for the data-representation branch once the current
DDT/topology branch either returns a result or explicitly yields the GPU.
