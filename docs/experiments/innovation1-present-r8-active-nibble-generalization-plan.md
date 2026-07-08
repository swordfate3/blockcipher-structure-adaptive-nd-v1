# Innovation 1 PRESENT r8 Active-Nibble Generalization Diagnostic

## Status

status = completed local diagnostic
claim_scope = not formal PRESENT r8 evidence
scale = 512/class seed0+seed1
plan = configs/experiment/innovation1/innovation1_spn_present_r8_active_nibble_generalization_512_seed0_seed1.csv

## Question

The bridge attribution run showed that the matched-negative integral
trail-position route is strong when the active plaintext nibble is fixed, but
collapses when the active nibble is randomized. This diagnostic asks a narrower
question:

Does the model learn a fixed active-nibble position template, or can it learn a
more general SPN rule once active-nibble coordinates are held out, explicitly
conditioned, or aligned to a relative coordinate system?

## Same-Budget Rows

All rows use:

- cipher = PRESENT-80
- rounds = 8
- samples_per_class = 512
- pairs_per_sample = 16
- negative_mode = encrypted_random_plaintexts
- difference_profile = present_zhang_wang2022_mcnd member 0
- feature_encoding = present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits
- model family = present_trail_position_stats_pairset
- checkpoint_metric = val_auc
- seeds = 0, 1

Rows per seed:

| Row | Route | Train active nibbles | Validation active nibbles | Purpose |
| --- | --- | --- | --- | --- |
| fixed-active trail | fixed active nibble 0 | 0 | 0 | Same-budget anchor from the strong bridge route. |
| random-active trail | random active nibble | 0..15 | 0..15 | Checks whether unconditioned random coordinates collapse. |
| heldout-active trail | random active subset | 0..3 | 4..7 | Tests position-template memorization versus generalization. |
| random-active + metadata | random active with 16-bit one-hot active metadata | 0..15 | 0..15 | Tests whether explicit coordinate conditioning restores signal. |
| random-active + relative-coordinate | random active with feature-cell rotation to active-zero coordinates | 0..15 | 0..15 | Tests a first nibble-equivariant/coordinate-normalized representation. |

## Implementation Notes

New task fields:

- `integral_active_nibbles`: active nibble set used by training rows.
- `validation_integral_active_nibbles`: validation override for held-out active
  tests.

New sample structures:

- `plaintext_integral_nibble_difference_matched_negative_random_active_metadata`
  appends a 16-bit one-hot active-nibble vector to each row.
- `plaintext_integral_nibble_difference_matched_negative_random_active_relative`
  rotates the feature-cell coordinate so the sampled active nibble is aligned to
  a common active-zero coordinate.

The metadata route uses the same trail-position model with
`model_options.metadata_bits = 16`; the trail-statistics branch consumes only
the base pair features, then concatenates active metadata to the statistics
vector. This avoids treating metadata bits as extra pair bits.

## Commands

Training:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_active_nibble_generalization_512_seed0_seed1.csv \
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
  --dataset-cache-root outputs/local_cache/i1_present_r8_active_nibble_generalization_512_seed0_seed1 \
  --dataset-cache-chunk-size 256 \
  --dataset-cache-workers 4 \
  --output outputs/local_smoke/i1_present_r8_active_nibble_generalization_512_seed0_seed1/results.jsonl \
  --progress-output outputs/local_smoke/i1_present_r8_active_nibble_generalization_512_seed0_seed1/progress.jsonl
```

Validation:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/validate-results \
  --plan configs/experiment/innovation1/innovation1_spn_present_r8_active_nibble_generalization_512_seed0_seed1.csv \
  --results outputs/local_smoke/i1_present_r8_active_nibble_generalization_512_seed0_seed1/results.jsonl \
  --expected-rows 10
```

Plot:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/plot-results \
  --results outputs/local_smoke/i1_present_r8_active_nibble_generalization_512_seed0_seed1/results.jsonl \
  --output outputs/local_smoke/i1_present_r8_active_nibble_generalization_512_seed0_seed1/curves.svg \
  --history-csv outputs/local_smoke/i1_present_r8_active_nibble_generalization_512_seed0_seed1/history.csv \
  --title i1_present_r8_active_nibble_generalization_512_seed0_seed1
```

## Gate

This is a local diagnostic gate, not a publication claim.

Support for the active-generalization route requires at least one of:

- heldout-active AUC remains meaningfully above the unconditioned random-active
  row on both seeds;
- random-active + metadata restores a large fraction of fixed-active AUC,
  showing coordinate conditioning is the missing ingredient;
- random-active + relative-coordinate restores signal without explicit metadata,
  supporting the SPN-adaptive representation route.

If all random/heldout/metadata/relative rows stay near chance while fixed-active
is high, do not remote-scale the existing fixed-active protocol by default.
Redesign the representation locally first.

## Results

Completed on 2026-07-08 local CPU with disk-backed dataset cache.

Artifacts:

- results = outputs/local_smoke/i1_present_r8_active_nibble_generalization_512_seed0_seed1/results.jsonl
- progress = outputs/local_smoke/i1_present_r8_active_nibble_generalization_512_seed0_seed1/progress.jsonl
- curves = outputs/local_smoke/i1_present_r8_active_nibble_generalization_512_seed0_seed1/curves.svg
- history = outputs/local_smoke/i1_present_r8_active_nibble_generalization_512_seed0_seed1/history.csv
- cache = outputs/local_cache/i1_present_r8_active_nibble_generalization_512_seed0_seed1

Validation:

```text
validate-results status = pass
expected_rows = 10
field_mismatches = []
duplicate_plan_keys = []
duplicate_result_keys = []
```

Metrics:

| Route | seed0 AUC | seed1 AUC | Interpretation |
| --- | ---: | ---: | --- |
| fixed-active trail | 0.989562988 | 0.992324829 | Fixed active-nibble anchor remains very strong at this small diagnostic budget. |
| random-active trail | 0.541259766 | 0.529830933 | Random active coordinates remain near chance. |
| heldout-active trail | 0.539070129 | 0.473754883 | Training on active {0,1,2,3} does not transfer to validation active {4,5,6,7}. |
| random-active + metadata | 0.473663330 | 0.471611023 | Explicit one-hot active metadata did not restore signal in this 3-epoch local gate. |
| random-active + relative-coordinate | 0.534194946 | 0.483474731 | First feature-cell relative-coordinate alignment did not restore signal. |

Decision:

```text
decision = hold_remote_scale_existing_fixed_active_protocol
reason = active-nibble generalization gate failed locally
```

This does not prove the route is impossible; 512/class is only a local
diagnostic. But it is strong triage evidence against spending remote GPU on a
larger version of the same fixed-active protocol right now. The next useful
work is local representation redesign or a more faithful active-conditioned
SPN model, not 65k/262k scale-up of the current fixed-active high-AUC setup.
