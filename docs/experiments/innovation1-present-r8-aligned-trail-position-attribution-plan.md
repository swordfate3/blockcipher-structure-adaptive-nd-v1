# Innovation 1 PRESENT r8 Aligned Trail-Position Attribution Gate

## Status

status = completed local diagnostic gate
claim_scope = not formal PRESENT r8 evidence
plan =

- configs/experiment/innovation1/innovation1_spn_present_r8_aligned_trail_position_attribution_512_seed0_seed1.csv
- configs/experiment/innovation1/innovation1_spn_present_r8_aligned_trail_position_permute_512_seed0_seed1.csv
- configs/experiment/innovation1/innovation1_spn_present_r8_aligned_trail_only_512_seed0_seed1.csv

## Question

The aligned-active-difference full gate showed that the aligned random16
trail-position route is high at 512/class on seed0+seed1, while the same full
feature under ordinary global statistics is much weaker. This gate asks a more
specific attribution question:

Does the aligned random16 score depend on the trail-position semantics, or does
it survive after the model destroys those semantics internally?

## Rows

All rows use:

- cipher = PRESENT-80
- rounds = 8
- samples_per_class = 512
- pairs_per_sample = 16
- negative_mode = encrypted_random_plaintexts
- sample_structure = plaintext_integral_nibble_aligned_difference_matched_negative_random_active
- integral_active_nibbles = `{0..15}`
- feature_encoding = present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits
- model = present_trail_position_stats_pairset
- seeds = 0, 1

Routes per seed:

| route | control | purpose |
| --- | --- | --- |
| full | none | same-protocol anchor for aligned random16 |
| prefix_only | `trail_position_control = prefix_only` | zero trail words inside the model while preserving protocol and input shape |
| trail_only | `trail_position_control = trail_only` | zero prefix words inside the model while preserving trail words |
| reverse_trail_positions | `trail_position_control = reverse_trail_positions` | preserve trail values but reverse trail-position order before statistics |
| permute_trail_positions | `trail_position_control = permute_trail_positions` | preserve trail values but apply a deterministic non-monotonic trail-word permutation |
| per_sample_key_full | key_rotation_interval = 1 | check whether the full route depends on fixed train/validation keys |

## Gate

- If `prefix_only` stays high, the prefix words are enough and the trail words
  are not necessary.
- If `reverse_trail_positions` stays high, the model may rely on value
  distribution more than trail-position semantics.
- If `reverse_trail_positions` drops while full stays high, the ordered
  trail-position layout is important.
- If `per_sample_key_full` drops sharply, the full route may depend on fixed
  key artifacts and should not be scaled remotely.
- If full and per-sample-key are high while prefix/reverse controls drop, the
  next step can be a larger local/remote ladder for this trail-position route.

## Command

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_aligned_trail_position_attribution_512_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_aligned_trail_position_attribution_512_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_aligned_trail_position_attribution_512_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_aligned_trail_position_attribution_512_seed0_seed1/progress.jsonl
```

## Decision Rule

Do not remote-scale from this plan alone unless the attribution controls support
the full trail-position route. This is a local diagnostic gate to decide
whether remote scale is scientifically worth spending.

## Results

Artifacts:

- Main attribution results: `outputs/local_smoke/i1_present_r8_aligned_trail_position_attribution_512_seed0_seed1/results.jsonl`
- Main attribution plot: `outputs/local_smoke/i1_present_r8_aligned_trail_position_attribution_512_seed0_seed1/curves.svg`
- Main attribution history CSV: `outputs/local_smoke/i1_present_r8_aligned_trail_position_attribution_512_seed0_seed1/history.csv`
- Permute control results: `outputs/local_smoke/i1_present_r8_aligned_trail_position_permute_512_seed0_seed1/results.jsonl`
- Permute control plot: `outputs/local_smoke/i1_present_r8_aligned_trail_position_permute_512_seed0_seed1/curves.svg`
- Permute control history CSV: `outputs/local_smoke/i1_present_r8_aligned_trail_position_permute_512_seed0_seed1/history.csv`
- Trail-only control results: `outputs/local_smoke/i1_present_r8_aligned_trail_only_512_seed0_seed1/results.jsonl`
- Trail-only control plot: `outputs/local_smoke/i1_present_r8_aligned_trail_only_512_seed0_seed1/curves.svg`
- Trail-only control history CSV: `outputs/local_smoke/i1_present_r8_aligned_trail_only_512_seed0_seed1/history.csv`

Validation:

- Main attribution matrix: `validate-results --expected-rows 8` passed.
- Permute control matrix: `validate-results --expected-rows 2` passed.
- Trail-only control matrix: `validate-results --expected-rows 2` passed.

Metrics:

| route | seed0 AUC | seed1 AUC | interpretation |
| --- | ---: | ---: | --- |
| full | 0.971763611 | 0.972396851 | aligned random16 anchor remains high |
| prefix_only | 0.600173950 | 0.601318359 | prefix words alone are not enough |
| trail_only | 0.973968506 | 0.973541260 | trail-derived words alone are enough |
| reverse_trail_positions | 0.960464478 | 0.971984863 | simple trail order reversal does not destroy the signal |
| permute_trail_positions | 0.985290527 | 0.979141235 | fixed non-monotonic trail-word permutation does not destroy the signal |
| per_sample_key_full | 0.970565796 | 0.980911255 | per-sample key rotation does not destroy the signal |

## Interpretation

This gate sharpens the attribution:

- The high aligned random16 score is not carried by the prefix words alone.
- The trail-derived words are sufficient to recover the high score.
- Exact trail-word order is not required at this local scale; both reversal and
  fixed non-monotonic permutation remain high.
- The signal also survives per-sample key rotation at 512/class seed0+seed1.

The current best interpretation is therefore not "the model needs exact
trail-position semantics." It is more precise to say:

The aligned random16 signal is concentrated in the public trail-derived
statistics themselves, and the trail-position model can extract that signal
even when the ordered trail-word layout is scrambled.

This is still useful SPN-structure evidence, but it changes the next question.
The next control should destroy or mismatch the trail-derived values, not only
their order. Good next controls are:

- random-trail-value control: keep input shape but replace trail words with a
  deterministic unrelated trail source;
- mismatched-source control: compute trail words from the wrong aligned
  difference while preserving the real prefix;
- label-preserving protocol control: keep labels and encryption protocol but
  break the relation between each sample's ciphertext pair and its trail words.

Do not remote-scale yet as a formal claim. If value-mismatch controls drop
while full/trail-only remain high, then a 65k/class local-to-remote ladder
becomes scientifically cleaner.
