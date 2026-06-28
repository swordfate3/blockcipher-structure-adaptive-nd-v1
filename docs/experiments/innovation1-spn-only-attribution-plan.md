# Innovation 1 SPN-only Attribution Plan

**Date:** 2026-06-28

**Status:** planned / implementation ready

**Scope:** PRESENT-80 r7, Zhang/Wang 2022 Case2 `m=16`, strict encrypted-random-plaintext negatives.

## Question

The current strongest Innovation 1 signal is `present_nibble_paligned_spn_only`.

At `262144/class`, this SPN-only anchor reached:

| Run | Model | Accuracy | Calibrated Accuracy | AUC |
|---|---|---:|---:|---:|
| `i1_spn_n1v2_ablation_r7_262k_seed0_gpu0_20260627` | `present_nibble_paligned_spn_only` | 0.716358 | 0.716434 | 0.791488 |
| `i1_spn_n2_transition_r7_262k_seed0_gpu1_20260627` | `present_nibble_paligned_spn_only` | 0.715187 | 0.715763 | 0.790665 |

The question is:

```text
Is the SPN-only gain caused by true PRESENT P-layer alignment, or mostly by generic DeltaC/XOR information?
```

This plan is an attribution experiment, not a formal reproduction or breakthrough claim.

## Fixed Protocol

All rows preserve the same data and validation protocol:

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

The cached dataset still stores raw ciphertext pair bits:

```text
C0 || C1
```

The attribution variants differ only in the model-internal SPN view derived from each ciphertext pair.

## Model Variants

| Rank | Model key | Internal view | Purpose |
|---:|---|---|---|
| 0 | `present_zhang_wang_keras_mcnd` | raw `C0 || C1` | same-protocol Zhang/Wang-style baseline |
| 1 | `present_nibble_paligned_spn_only` | `DeltaC || InvP(DeltaC)` | current Innovation 1 SPN-only anchor |
| 2 | `present_nibble_delta_only_spn_only` | `DeltaC` only | tests whether XOR difference alone explains the gain |
| 3 | `present_nibble_invp_only_spn_only` | `InvP(DeltaC)` only | tests whether inverse-P aligned view alone carries signal |
| 4 | `present_nibble_shuffled_paligned_spn_only` | `DeltaC || shuffled(DeltaC)` | controls for extra tokens/parameters without true P-layer alignment |

For each pair:

```text
DeltaC = C0 xor C1
InvP(DeltaC) = DeltaC indexed by the PRESENT inverse P-layer
```

The SPN-only encoder tokenizes the selected view into 4-bit nibble tokens, applies a token mixer, projects each pair to an embedding, then aggregates the 16 pair embeddings by mean and max pooling.

## Configs

Smoke config:

```text
configs/experiment/innovation1/innovation1_spn_present_spn_only_attribution_smoke.csv
```

Medium diagnostic config:

```text
configs/experiment/innovation1/innovation1_spn_present_spn_only_attribution_r7_262k.csv
```

## Evidence Gate

This experiment can support an attribution decision only at medium diagnostic scale:

```text
262144/class -> diagnostic attribution evidence
```

It must not be described as formal evidence. Formal claims still require:

```text
1000000/class + multiple seeds + completed/retrieved/plan-aligned artifacts
```

## Decision Rules

Primary metric:

```text
validation AUC
```

Secondary metrics:

```text
calibrated accuracy, validation loss, fixed-threshold accuracy
```

Interpretation:

| Outcome | Interpretation | Next Action |
|---|---|---|
| Anchor > Delta-only and Anchor > shuffled | true P-layer alignment likely contributes useful signal | scale SPN-only anchor to `1M/class` and design pair-consistency model |
| Delta-only ~= Anchor | gain likely comes mostly from ciphertext XOR signal | focus on stronger structure beyond simple `DeltaC` |
| Shuffled ~= Anchor | current P-layer alignment attribution is weak | do not claim P-layer-specific contribution |
| InvP-only strong | inverse-P alignment alone is a compact signal | consider an InvP-centered SPN token model |
| All SPN-only variants below baseline | current SPN-only signal is unstable | stop scaling and inspect seed/protocol/cache issues |

## Planned Run Command

Local smoke:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_spn_only_attribution_smoke.csv \
  --epochs 1 \
  --batch-size 8 \
  --hidden-bits 8 \
  --device cpu \
  --output outputs/smoke/innovation1_spn_only_attribution_smoke.jsonl
```

Remote medium diagnostic:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_spn_only_attribution_r7_262k.csv \
  --epochs 20 \
  --batch-size 1024 \
  --hidden-bits 32 \
  --device cuda \
  --output <remote-run-root>/innovation1_spn_present_spn_only_attribution_r7_262k.jsonl
```

Before remote launch, verify the run uses disk-backed dataset cache/progress artifacts under `G:\lxy`, starts from a pushed GitHub commit, and is handed off to a local tmux monitor for retrieval.

## Current Next Step

1. Verify model construction and view attribution tests locally.
2. Run the smoke matrix locally.
3. Commit and push code/config/docs.
4. Launch the `262144/class` attribution matrix remotely.
5. On result retrieval, update this document with metrics, artifacts, gate status, and keep/discard decision.
