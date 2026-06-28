# Innovation 1 InvP-Centered Next Plan

**Date:** 2026-06-28

**Status:** planned / implementation next

**Scope:** PRESENT-80 r7, Zhang/Wang 2022 Case2 `m=16`, strict encrypted-random-plaintext negatives.

## Motivation

The SPN-only attribution experiment found that `InvP(DeltaC)` is the strongest current Innovation 1 signal.

Completed diagnostic:

```text
run_id = i1_spn_only_attr_r7_262k_seed0_gpu1_20260628
scale = 262144/class
status = completed remotely, fallback-retrieved locally, plan-aligned
```

Key AUC:

| Model | View | AUC |
|---|---|---:|
| `present_zhang_wang_keras_mcnd` | raw `C0 || C1` baseline | 0.783228 |
| `present_nibble_delta_only_spn_only` | `DeltaC` | 0.782918 |
| `present_nibble_shuffled_paligned_spn_only` | `DeltaC || shuffled(DeltaC)` | 0.784487 |
| `present_nibble_paligned_spn_only` | `DeltaC || InvP(DeltaC)` | 0.790665 |
| `present_nibble_invp_only_spn_only` | `InvP(DeltaC)` | **0.792536** |

Interpretation:

```text
The useful PRESENT/SPN signal is concentrated in inverse-P aligned ciphertext
difference tokens. Generic DeltaC and shuffled alignment are much weaker.
```

This upgrades the Innovation 1 direction from generic SPN-only to:

```text
InvP-centered SPN neural distinguisher
```

## Research Question

Can an InvP-centered pair-set consistency model improve over the current InvP-only SPN-only anchor without changing the benchmark protocol?

Current InvP-only aggregation:

```text
pair_embeddings -> mean pooling + max pooling -> classifier
```

Hypothesis:

```text
For MCND samples, the 16 pair embeddings may carry useful consistency patterns.
Adding low-cost pair-set consistency statistics and evidence pooling can improve
the distinguisher while preserving the same InvP-centered representation.
```

## Fixed Protocol

All rows must preserve:

| Field | Value |
|---|---|
| Cipher | `PRESENT-80` |
| Rounds | `7` |
| Difference profile | `present_zhang_wang2022_mcnd` |
| Sample structure | `zhang_wang_case2_official_mcnd` |
| Pairs per sample | `16` |
| Feature encoding on disk | `ciphertext_pair_bits` |
| Negative mode | `encrypted_random_plaintexts` |
| Train key | `0x00000000000000000000` |
| Validation key | `0x11111111111111111111` |
| Scheduler | `official_cyclic` |
| Checkpoint metric | `val_auc` |
| Scale | `262144/class` medium diagnostic |

Do not change validation data, labels, negative mode, metric computation, keying protocol, or Zhang/Wang Case2 sample construction.

## Proposed Model

Model key:

```text
present_nibble_invp_pair_consistency_spn_only
```

Input view:

```text
InvP(DeltaC)
```

Minimal architecture:

```text
C0 || C1
  -> DeltaC = C0 xor C1
  -> InvP(DeltaC)
  -> 16 nibble tokens per pair
  -> shared SPN token mixer
  -> 16 pair embeddings
  -> concat(
       mean(pair_embeddings),
       max(pair_embeddings),
       std(pair_embeddings),
       top-k/logsumexp evidence pooled embedding
     )
  -> classifier
  -> binary score
```

This intentionally changes only pair aggregation, not the dataset or per-pair InvP encoder.

Default options:

```text
spn_mixer_depth = 2
pooling = topk_logsumexp
top_k = 4
lse_temperature = 1.0
activation = relu
norm = layernorm
```

## Experiment Matrix

New smoke config:

```text
configs/experiment/innovation1/innovation1_spn_present_invp_centered_smoke.csv
```

New medium diagnostic config:

```text
configs/experiment/innovation1/innovation1_spn_present_invp_centered_r7_262k.csv
```

Rows:

| Rank | Model key | Role |
|---:|---|---|
| 0 | `present_zhang_wang_keras_mcnd` | same-protocol baseline |
| 1 | `present_nibble_invp_only_spn_only` | current strongest InvP-only anchor |
| 2 | `present_nibble_invp_pair_consistency_spn_only` | proposed InvP-centered consistency model |
| 3 | `present_nibble_paligned_spn_only` | old `DeltaC + InvP` anchor |
| 4 | `present_nibble_delta_only_spn_only` | DeltaC-only attribution control |
| 5 | `present_nibble_shuffled_paligned_spn_only` | shuffled-P attribution control |

## Decision Gates

Primary metric:

```text
validation AUC
```

Supporting metrics:

```text
calibrated accuracy, validation loss, fixed-threshold accuracy
```

Keep/continue conditions:

| Condition | Interpretation | Action |
|---|---|---|
| Pair-consistency AUC >= InvP-only AUC + 0.002 | aggregation improvement | run `262144/class` seed1 |
| Pair-consistency AUC within +/- 0.001 of InvP-only and lower loss | possible simplification/stability | run one more diagnostic seed before deciding |
| Pair-consistency below InvP-only by > 0.002 | aggregation not useful | keep InvP-only anchor, discard this model |
| InvP-only again beats DeltaC-only and shuffled-P | attribution stable | plan 1M/class seed0 for strongest InvP route |
| InvP-only fails against controls | attribution unstable | pause scaling and inspect seed/protocol/cache |

No formal claim is allowed from this single `262144/class` run.

## Execution Plan

1. Implement `present_nibble_invp_pair_consistency_spn_only` by reusing `_PresentNibblePAlignedSpnEncoder(view_mode="inv_p")`.
2. Add registry export and model factory branch.
3. Add tests for model construction/forward and plan protocol invariants.
4. Add smoke and `262144/class` configs.
5. Run local targeted tests and smoke.
6. If smoke passes, commit/push and launch remote `262144/class` with disk-backed shared dataset cache and tmux monitor retrieval.
7. When results are retrieved, update this document with gate, artifacts, metrics, deltas, decision, and next action.

## Claim Scope

This route can support:

```text
medium diagnostic evidence for InvP-centered architecture selection
```

It cannot yet support:

```text
formal reproduction
breakthrough claim
paper-scale multi-seed conclusion
```
