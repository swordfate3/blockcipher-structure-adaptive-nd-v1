# Innovation 1 PRESENT r8 Bridge Protocol Attribution Plan

**Date:** 2026-07-08

**Status:** local 512/class diagnostic completed

## Purpose

The current evidence has two endpoints:

```text
matched-negative / integral rows:
  trail-position is very strong and keeps residual value over the
  pair_xor_column_sum_variance deterministic baseline.

fully independent pairs:
  trail-position is near chance at the 2048/class local gate.
```

This plan tests the bridge between those endpoints. The goal is to find which
piece of structure makes the signal appear or collapse. This is not formal
PRESENT r8 evidence.

## Matrix

```text
configs/experiment/innovation1/innovation1_spn_present_r8_bridge_protocol_attribution_512_seed0_seed1.csv
```

Fixed scope:

```text
cipher = PRESENT-80
rounds = 8
samples_per_class = 512
pairs_per_sample = 16
seeds = 0, 1
negative_mode = encrypted_random_plaintexts
difference_profile = present_zhang_wang2022_mcnd
train_key = 0x00000000000000000000
validation_key = 0x11111111111111111111
```

## Bridge Controls

| Control | Sample structure | Question |
|---|---|---|
| matched anchor | `plaintext_integral_nibble_difference_matched_negative` | Does the known integral/matched signal reproduce at this budget? |
| pair-shuffled | `plaintext_integral_nibble_difference_matched_negative_pair_shuffled` | Does trail-position depend on fixed pair index inside the 16-pair row? |
| random-active | `plaintext_integral_nibble_difference_matched_negative_random_active` | Does fixed active-nibble alignment drive the signal? |
| partial8 | `plaintext_integral_nibble_difference_matched_negative_partial8` | Is half of the integral row enough, or does signal require all 16 structured pairs? |
| same-difference | `plaintext_integral_nibble_same_difference_random_negative` | Does the easier random-negative family remain too separable? |
| independent endpoint | `independent_pairs` | Does the signal collapse when integral row structure is removed? |

Each bridge control runs both:

```text
present_pairset_global_stats
present_trail_position_stats_pairset
```

Feature-drop rows under the matched-negative anchor:

```text
ciphertext_xor_bits + MLP
ciphertext_xor_spn_paligned_bits + global stats
present_pair_xor_paligned_sinv_cell_matrix_bits + global stats
full beamstats + global stats
full beamstats + trail-position
```

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_bridge_protocol_attribution_512_seed0_seed1.csv \
  --epochs 3 \
  --batch-size 64 \
  --hidden-bits 16 \
  --device cpu \
  --learning-rate 0.0001 \
  --optimizer adam \
  --weight-decay 0.00001 \
  --loss mse \
  --checkpoint-metric val_auc \
  --restore-best-checkpoint \
  --train-eval-interval 1 \
  --dataset-cache-root outputs/local_cache/i1_present_r8_bridge_protocol_attribution_512_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_bridge_protocol_attribution_512_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_bridge_protocol_attribution_512_seed0_seed1/progress.jsonl
```

Validation:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_bridge_protocol_attribution_512_seed0_seed1.csv \
  --results outputs/local_smoke/i1_present_r8_bridge_protocol_attribution_512_seed0_seed1/results.jsonl \
  --expected-rows 30
```

Plot:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-results \
  --results outputs/local_smoke/i1_present_r8_bridge_protocol_attribution_512_seed0_seed1/results.jsonl \
  --output outputs/local_smoke/i1_present_r8_bridge_protocol_attribution_512_seed0_seed1/curves.svg \
  --history-csv outputs/local_smoke/i1_present_r8_bridge_protocol_attribution_512_seed0_seed1/history.csv \
  --title i1_present_r8_bridge_protocol_attribution_512_seed0_seed1
```

## Gate

Interpretation rules:

```text
If pair-shuffled trail collapses while matched trail stays high:
  fixed pair index is a major part of the trail-position signal.

If random-active collapses:
  fixed active-nibble alignment is a major part of the signal.

If partial8 remains high:
  the model needs less than the full 16-pair integral sweep.

If partial8 collapses:
  the full integral row is likely essential.

If same-difference remains near perfect:
  it is still an easy protocol and should not be scaled.

If independent remains near chance:
  the signal is row-structure dependent, not ordinary independent-pair evidence.

If weak feature-drop rows are already high:
  the result is not isolated to full DDT/beamstats/trail-position.
```

Claim scope:

```text
This is a 512/class local diagnostic. It can guide the next protocol or model
edit, but it is not a medium scale-up gate by itself and not a breakthrough
claim.
```

## Local 512/Class Result

Run artifacts:

```text
results = outputs/local_smoke/i1_present_r8_bridge_protocol_attribution_512_seed0_seed1/results.jsonl
progress = outputs/local_smoke/i1_present_r8_bridge_protocol_attribution_512_seed0_seed1/progress.jsonl
curves = outputs/local_smoke/i1_present_r8_bridge_protocol_attribution_512_seed0_seed1/curves.svg
history = outputs/local_smoke/i1_present_r8_bridge_protocol_attribution_512_seed0_seed1/history.csv
cache = outputs/local_cache/i1_present_r8_bridge_protocol_attribution_512_seed0_seed1
```

Validation:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_bridge_protocol_attribution_512_seed0_seed1.csv \
  --results outputs/local_smoke/i1_present_r8_bridge_protocol_attribution_512_seed0_seed1/results.jsonl \
  --expected-rows 30

status = pass
field_mismatches = []
```

Metrics:

| Seed | Protocol | Model | Feature | AUC | Best accuracy |
|---:|---|---|---|---:|---:|
| 0 | matched | global | full | `0.839035034` | `0.779297` |
| 0 | matched | trail | full | `0.973907471` | `0.917969` |
| 0 | pair-shuffle | global | full | `0.698509216` | `0.648438` |
| 0 | pair-shuffle | trail | full | `0.951980591` | `0.886719` |
| 0 | random-active | global | full | `0.496200562` | `0.523438` |
| 0 | random-active | trail | full | `0.541259766` | `0.550781` |
| 0 | partial8 | global | full | `0.513381958` | `0.535156` |
| 0 | partial8 | trail | full | `0.498107910` | `0.515625` |
| 0 | same-diff | global | full | `0.869972229` | `0.800781` |
| 0 | same-diff | trail | full | `0.999969482` | `0.998047` |
| 0 | independent | global | full | `0.494293213` | `0.521484` |
| 0 | independent | trail | full | `0.527923584` | `0.556641` |
| 0 | matched | MLP | xor | `0.501892090` | `0.529297` |
| 0 | matched | global | paligned | `0.510559082` | `0.527344` |
| 0 | matched | global | sinv | `0.576507568` | `0.574219` |
| 1 | matched | global | full | `0.742767334` | `0.695312` |
| 1 | matched | trail | full | `0.994628906` | `0.966797` |
| 1 | pair-shuffle | global | full | `0.697776794` | `0.644531` |
| 1 | pair-shuffle | trail | full | `0.992156982` | `0.966797` |
| 1 | random-active | global | full | `0.530517578` | `0.542969` |
| 1 | random-active | trail | full | `0.529830933` | `0.537109` |
| 1 | partial8 | global | full | `0.490631104` | `0.517578` |
| 1 | partial8 | trail | full | `0.497482300` | `0.521484` |
| 1 | same-diff | global | full | `0.891494751` | `0.816406` |
| 1 | same-diff | trail | full | `0.999984741` | `0.998047` |
| 1 | independent | global | full | `0.502098083` | `0.525391` |
| 1 | independent | trail | full | `0.543220520` | `0.552734` |
| 1 | matched | MLP | xor | `0.502578735` | `0.529297` |
| 1 | matched | global | paligned | `0.559188843` | `0.568359` |
| 1 | matched | global | sinv | `0.497726440` | `0.525391` |

Interpretation:

```text
pair-shuffle:
  Trail-position stays high on both seeds: 0.951980591 and 0.992156982.
  The signal is therefore not mainly a fixed pair-index shortcut.

random-active:
  Full trail collapses to 0.541259766 and 0.529830933.
  This strongly points to fixed active-nibble alignment as a core ingredient
  of the matched/integral trail-position signal.

partial8:
  Full trail collapses to 0.498107910 and 0.497482300.
  Keeping only half of the 16-pair integral sweep is not enough at this budget.

same-difference:
  Full trail remains almost perfect: 0.999969482 and 0.999984741.
  This protocol remains too easy and should not be scaled as candidate evidence.

independent:
  Full trail remains near chance: 0.527923584 and 0.543220520.
  This confirms the strict independent-pair endpoint is not the current
  trail-position route.

feature drop:
  Matched-negative xor MLP and PAligned rows are near chance; SInv is unstable
  and weak. The high matched trail signal appears only after the full
  DDT/beamstats/trail-position representation.
```

Decision:

```text
active_nibble_alignment_is_primary_bridge_signal
```

Next action:

```text
Do not scale same-difference or independent-pair variants.

Do not treat pair order as the leading explanation; pair-shuffled trail remains
strong.

Focus the next iteration on active-nibble generalization:
  1. train with randomized active nibble but expose active-nibble metadata or
     equivariant nibble coordinates to the model;
  2. train on a subset of active nibbles and validate on held-out active
     nibbles;
  3. compare fixed-active, random-active, and heldout-active under the same
     matched-negative protocol.
```
