## [LRN-20260709-012] best_practice

**Logged**: 2026-07-09T16:52:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
PRESENT r8 raw-prefix relative summary statistics remain weak, and
P-layer-relative statistics do not beat metadata-only controls at 512/class.

### Details
The local diagnostic gate used `present_trail_position_stats_pairset` with
`trail_depth = 0` and `trail_words_per_depth = 0`, so the model summarized only
raw prefix words and consumed no DDT trail-value block.

It compared:

- `p_layer_relative_stats`
- `relative_stats`
- `none` with `metadata_bits = 16` as a metadata-only control

All rows used aligned random16 PRESENT r8, 512/class, seed0+seed1, strict
encrypted random-plaintext negatives.

Results:

- p-layer-relative AUC = 0.523223877 seed0, 0.560836792 seed1
- simple-relative AUC = 0.510566711 seed0, 0.543212891 seed1
- metadata-only AUC = 0.549835205 seed0, 0.576095581 seed1

P-layer-relative is slightly above simple-relative, but metadata-only is higher
on both seeds. This blocks any claim that the summary-stat route has learned a
useful PRESENT P-layer relative coordinate system.

### Suggested Action
Do not scale raw-prefix relative summary statistics. The next local route should
preserve cell/edge-local evidence as tokens, for example a dynamic
active-conditioned SPN cell graph with true-vs-shuffled topology and
metadata-only controls.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-aligned-raw-prefix-relative-stats-plan.md, configs/experiment/innovation1/innovation1_spn_present_r8_aligned_raw_prefix_relative_stats_512_seed0_seed1.csv, src/blockcipher_nd/models/structure/spn/present_trail_position_stats.py
- Tags: innovation1, present, spn, raw-prefix, relative-stats, active-conditioning, local-gate
- See Also: LRN-20260709-011, LRN-20260709-010, LRN-20260709-009
- Pattern-Key: innovation1.spn_present.raw_prefix_relative_stats_gate
- Recurrence-Count: 1
- First-Seen: 2026-07-09
- Last-Seen: 2026-07-09

---

## [LRN-20260709-011] best_practice

**Logged**: 2026-07-09T16:32:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
PRESENT r8 active-conditioned raw-prefix P-layer topology remains weak and does
not consistently beat shuffled topology at the 512/class aligned gate.

### Details
The local diagnostic gate removed DDT trail values and tested raw prefix
features with explicit active-nibble metadata:

- `present_pair_xor_paligned_sinv_cell_matrix_bits`
- `present_pair_xor_paligned_cell_matrix_bits`
- `metadata_bits = 16`
- `active_conditioning = p_layer_active_token_bias`

It compared true PRESENT P-layer adjacency, shuffled P-layer adjacency, and a
no-`sinv` control under aligned random16 PRESENT r8, 512/class, seed0+seed1,
strict encrypted random-plaintext negatives.

Results:

- true-sinv AUC = 0.550659180 seed0, 0.510643005 seed1
- shuffled-sinv AUC = 0.506835938 seed0, 0.530029297 seed1
- true-no-sinv AUC = 0.502883911 seed0, 0.512893677 seed1

The active marker did not rescue the raw-prefix topology route. True topology
won seed0 but lost seed1, and all routes stayed near chance.

### Suggested Action
Do not scale active-conditioned raw-prefix topology. The next local route should
use a stronger relative-coordinate or cell-equivariant representation, not just
active one-hot metadata or token bias. Keep true-vs-shuffled topology controls
and wrong-source/fixed-source mismatch controls as local gates before any
medium or remote PRESENT r8 run.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-aligned-active-conditioned-raw-prefix-topology-plan.md, configs/experiment/innovation1/innovation1_spn_present_r8_aligned_active_conditioned_raw_prefix_topology_512_seed0_seed1.csv, src/blockcipher_nd/models/structure/spn/present_p_layer_mixer.py
- Tags: innovation1, present, spn, raw-prefix, p-layer-topology, active-conditioning, local-gate
- See Also: LRN-20260709-010, LRN-20260709-009, LRN-20260709-006
- Pattern-Key: innovation1.spn_present.active_conditioned_raw_prefix_topology_gate
- Recurrence-Count: 1
- First-Seen: 2026-07-09
- Last-Seen: 2026-07-09

---

## [LRN-20260709-010] best_practice

**Logged**: 2026-07-09T16:03:18+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Unconditioned PRESENT r8 raw-prefix P-layer topology mixer is near chance and
does not beat shuffled topology at the 512/class aligned random-active gate.

### Details
The local diagnostic gate removed DDT trail values and tested raw prefix
features only:

- `present_pair_xor_paligned_sinv_cell_matrix_bits`
- `present_pair_xor_paligned_cell_matrix_bits`

It compared `present_p_layer_mixer_pairset` with true PRESENT P-layer topology,
a shuffled P-layer topology control, a no-`sinv` control, and a
`present_pairset_global_stats` baseline under aligned random16 PRESENT r8,
512/class, seed0+seed1, strict encrypted random-plaintext negatives.

Results:

- true-sinv AUC = 0.540695190 seed0, 0.520507812 seed1
- shuffled-sinv AUC = 0.567588806 seed0, 0.537811279 seed1
- true-no-sinv AUC = 0.535354614 seed0, 0.524757385 seed1
- global-sinv AUC = 0.528747559 seed0, 0.485214233 seed1

The true P-layer topology did not beat the shuffled topology, and all raw-prefix
routes stayed near chance. This suggests that the strong prior DDT trail routes
were not reproduced by simply giving a token mixer the PRESENT P-layer topology.

### Suggested Action
Do not scale the unconditioned raw-prefix topology route. The next local
diagnostic should test active-conditioned raw-prefix topology, because the
aligned random-active protocol changes the active coordinate per sample. If
active-conditioned true topology remains weak or tied with shuffled topology,
the architecture needs a stronger SPN-coordinate mechanism than basic P-layer
message passing.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-aligned-raw-prefix-topology-plan.md, configs/experiment/innovation1/innovation1_spn_present_r8_aligned_raw_prefix_topology_512_seed0_seed1.csv, src/blockcipher_nd/models/structure/spn/present_p_layer_mixer.py
- Tags: innovation1, present, spn, raw-prefix, p-layer-topology, active-nibble, local-gate
- See Also: LRN-20260709-009, LRN-20260709-008, LRN-20260709-007
- Pattern-Key: innovation1.spn_present.raw_prefix_topology_gate
- Recurrence-Count: 1
- First-Seen: 2026-07-09
- Last-Seen: 2026-07-09

---

## [LRN-20260709-009] best_practice

**Logged**: 2026-07-09T15:51:43+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
PRESENT r8 weak trail summaries `beamstats4deep2` and `beamstats2deep1` do not
solve the aligned wrong-source/fixed-source mismatch gate.

### Details
The local diagnostic gate tested shallower/local parameterized S-box-DDT
statistics under aligned random16 PRESENT r8, 512/class, seed0+seed1, strict
encrypted random-plaintext negatives:

- `present_delta_paligned_sinv_sboxddt_beamstats4deep2_*`
- `present_delta_paligned_sinv_sboxddt_beamstats2deep1_*`

Each family compared full, `maskedsource`, and `constantsource` using the same
`present_trail_position_stats_pairset` model with matching `trail_depth`.

Results:

- beamstats4deep2 full AUC = 0.959899902 seed0, 0.952911377 seed1
- beamstats4deep2 maskedsource AUC = 0.959884644 seed0, 0.974182129 seed1
- beamstats4deep2 constantsource AUC = 0.929473877 seed0, 0.934555054 seed1
- beamstats2deep1 full AUC = 0.904388428 seed0, 0.901809692 seed1
- beamstats2deep1 maskedsource AUC = 0.945175171 seed0, 0.916732788 seed1
- beamstats2deep1 constantsource AUC = 0.909851074 seed0, 0.942443848 seed1

The desired pattern was full high with wrong-source/fixed-source lower. Instead,
wrong-source and fixed-source controls remain close to, or above, full. This
means simply reducing beam width/depth under the same concatenated
DDT-trail-value representation is not enough.

### Suggested Action
Do not remote-scale weak trail summaries in this representation. The next local
route should move away from feeding DDT trail values as a dense input block.
Prefer a native SPN cell/coordinate model over raw prefix signals, or add a
control-aware objective that penalizes full and wrong-source trail embeddings
being equally useful.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-aligned-weak-trail-summary-plan.md, src/blockcipher_nd/features/encoders/present_sbox_ddt.py, src/blockcipher_nd/features/encoders/present_matrix.py, configs/experiment/innovation1/innovation1_spn_present_r8_aligned_weak_trail_summary_512_seed0_seed1.csv
- Tags: innovation1, present, spn, trail-position, weak-trail, mismatch-control, local-gate
- See Also: LRN-20260709-008, LRN-20260709-007, LRN-20260709-006
- Pattern-Key: innovation1.spn_present.weak_trail_summary_mismatch_gate
- Recurrence-Count: 1
- First-Seen: 2026-07-09
- Last-Seen: 2026-07-09

---

## [LRN-20260709-008] best_practice

**Logged**: 2026-07-09T15:35:10+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
PRESENT r8 gated auxiliary trail fusion reduces fixed-source scaffold but does
not solve wrong-source beamstats8deep4 mismatch at the 512/class local gate.

### Details
Two local diagnostic gates tested `present_trail_position_stats_pairset` with a
split prefix/trail architecture:

- vector gate: `trail_fusion = gated_auxiliary`, `trail_auxiliary_scale = 0.25`
- scalar gate: same but `trail_gate = scalar`

Both used aligned random16 PRESENT r8, 512/class, seed0+seed1, strict encrypted
random-plaintext negatives, and compared full route against `maskedsource` and
`constantsource` mismatch controls.

Vector-gated result:

- full AUC = 0.959564209 seed0, 0.958099365 seed1
- maskedsource AUC = 0.899963379 seed0, 0.956344604 seed1
- constantsource AUC = 0.737449646 seed0, 0.787750244 seed1

Scalar-gated result:

- full AUC = 0.896041870 seed0, 0.959915161 seed1
- maskedsource AUC = 0.927520752 seed0, 0.977416992 seed1
- constantsource AUC = 0.787994385 seed0, 0.757537842 seed1

Compared with the earlier concatenated/full beamstats8deep4 controls, gated
fusion meaningfully reduces the fixed-source scaffold. However, wrong-source
`maskedsource` remains close to full under the vector gate and exceeds full
under the scalar gate. This means fusion control alone is not enough.

### Suggested Action
Do not remote-scale gated auxiliary full beamstats8deep4. Keep the branch/gate
code as a useful diagnostic architecture, but the next route should weaken or
localize the trail feature family itself, using the same
full/maskedsource/constantsource gate before any medium or remote run.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-aligned-gated-aux-trail-plan.md, docs/experiments/innovation1-present-r8-aligned-gated-scalar-aux-trail-plan.md, src/blockcipher_nd/models/structure/spn/present_trail_position_stats.py
- Tags: innovation1, present, spn, trail-position, gated-auxiliary, mismatch-control, local-gate
- See Also: LRN-20260709-007, LRN-20260709-006, LRN-20260709-005
- Pattern-Key: innovation1.spn_present.gated_aux_trail_mismatch_gate
- Recurrence-Count: 1
- First-Seen: 2026-07-09
- Last-Seen: 2026-07-09

---

## [LRN-20260709-007] best_practice

**Logged**: 2026-07-09T15:12:41+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
PRESENT r8 aligned trail-stat centering/z-score normalization does not remove
the beamstats8deep4 mismatch shortcut at the 512/class local gate.

### Details
The local diagnostic gate tested `present_trail_position_stats_pairset` with
model-side trail normalization applied only to the trail block:

- `trail_center`: subtract each sample-pair trail-block mean.
- `trail_zscore`: subtract that mean and divide by trail-block standard
  deviation.

It used aligned random16 PRESENT r8, 512/class, seed0+seed1, strict encrypted
random-plaintext negatives, and compared full route against `maskedsource` and
`constantsource` mismatch controls.

Results:

- trail_center full AUC = 0.973510742 seed0, 0.979187012 seed1
- trail_center maskedsource AUC = 0.992431641 seed0, 0.991027832 seed1
- trail_center constantsource AUC = 0.905258179 seed0, 0.946670532 seed1
- trail_zscore full AUC = 0.987579346 seed0, 0.984100342 seed1
- trail_zscore maskedsource AUC = 0.991653442 seed0, 0.990798950 seed1
- trail_zscore constantsource AUC = 0.921203613 seed0, 0.948593140 seed1

The desired pattern was full high with maskedsource/constantsource dropping.
That did not happen. Wrong-source trail remains about as high as the true full
route, and fixed-source trail remains far above the earlier prefix-only anchor
around 0.60 AUC.

### Suggested Action
Do not remote-scale the current full beamstats8deep4 route with simple
trail-stat normalization. Treat normalization as a diagnostic option only. The
next local redesign should change the information path: split prefix/trail
branches or gated auxiliary-trail input, with mismatch controls required to drop
before any medium or remote run.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-aligned-trail-normalization-plan.md, src/blockcipher_nd/models/structure/spn/present_trail_position_stats.py, configs/experiment/innovation1/innovation1_spn_present_r8_aligned_trail_normalization_512_seed0_seed1.csv
- Tags: innovation1, present, spn, trail-position, normalization, mismatch-control, local-gate
- See Also: LRN-20260709-006, LRN-20260709-005, LRN-20260709-004
- Pattern-Key: innovation1.spn_present.aligned_trail_normalization_gate
- Recurrence-Count: 1
- First-Seen: 2026-07-09
- Last-Seen: 2026-07-09

---

## [LRN-20260709-006] best_practice

**Logged**: 2026-07-09T14:55:20+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
PRESENT r8 aligned trail-value mismatch gate blocks remote scale for the
current beamstats8deep4 trail route.

### Details
The intended final local gate before remote scale tested whether trail-derived
values must correspond to the current ciphertext pair. The 512/class seed0+seed1
aligned random16 control used the same full 2496-bit pair shape but changed the
trail-stat source:

- full anchor AUC = 0.971763611 seed0, 0.972396851 seed1
- trail_only anchor AUC = 0.973968506 seed0, 0.973541260 seed1
- prefix_only anchor AUC = 0.600173950 seed0, 0.601318359 seed1
- maskedsource AUC = 0.981689453 seed0, 0.988342285 seed1
- constantsource AUC = 0.902770996 seed0, 0.938705444 seed1

`maskedsource` keeps the real prefix but computes beamstats8deep4 trail words
from `sinv_delta xor 0x111...`. `constantsource` keeps the real prefix but
computes trail words from a fixed source `0x9`. Both preserve the full
trail-stat input shape.

The mismatch controls did not drop as required. In particular, masked wrong
source is as high as the full route, and fixed-source trail remains much higher
than prefix-only. This suggests that the current beamstats8deep4 trail-value
block can act as a strong representation scaffold or shortcut even when trail
values are not correctly sample-specific.

### Suggested Action
Do not remote-scale the current aligned full/trail-only route as a formal
readiness run. Redesign locally instead: normalize trail-stat blocks, separate
prefix/trail branches with same-budget controls, reduce beam depth/width, or
move trail information into a gated auxiliary graph/token representation rather
than a large dense hand-crafted trail-stat block.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-aligned-trail-value-mismatch-plan.md, src/blockcipher_nd/features/encoders/present_matrix.py, configs/experiment/innovation1/innovation1_spn_present_r8_aligned_trail_value_mismatch_512_seed0_seed1.csv
- Tags: innovation1, present, spn, trail-position, trail-value, mismatch-control, local-gate
- See Also: LRN-20260709-005, LRN-20260709-004, LRN-20260709-003
- Pattern-Key: innovation1.spn_present.aligned_trail_value_mismatch_gate
- Recurrence-Count: 1
- First-Seen: 2026-07-09
- Last-Seen: 2026-07-09

---

## [LRN-20260709-005] best_practice

**Logged**: 2026-07-09T13:36:34+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
PRESENT r8 aligned random16 trail signal is carried by trail-derived values,
not by prefix-only features, exact trail-word order, or fixed-key artifacts at
the local 512/class gate.

### Details
The PRESENT r8 aligned trail-position attribution gate at 512/class seed0+seed1
tested `present_trail_position_stats_pairset` with internal model controls:

- full anchor AUC = 0.971763611 seed0, 0.972396851 seed1
- prefix_only AUC = 0.600173950 seed0, 0.601318359 seed1
- trail_only AUC = 0.973968506 seed0, 0.973541260 seed1
- reverse_trail_positions AUC = 0.960464478 seed0, 0.971984863 seed1
- permute_trail_positions AUC = 0.985290527 seed0, 0.979141235 seed1
- per_sample_key_full AUC = 0.970565796 seed0, 0.980911255 seed1

This sharpens the previous full-gate interpretation. The prefix words are not
sufficient, but the trail-derived words are sufficient. Exact ordered
trail-position semantics are not necessary at this small scale, because both
simple reversal and a deterministic non-monotonic trail-word permutation remain
high. Per-sample key rotation also remains high, so the result is not explained
by fixed train/validation key artifacts at this gate.

The best current wording is: the aligned random16 signal is concentrated in the
public trail-derived statistics themselves, and the trail-position statistics
model can extract that signal even after trail-word order is scrambled.

### Suggested Action
Do not remote-scale as a formal claim yet. The next local control should
destroy or mismatch trail-derived values, not only their order. Prefer
random-trail-value, mismatched-source, or label-preserving protocol controls
that keep input shape and encryption protocol but break the relation between
each sample's ciphertext pair and its trail words.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-aligned-trail-position-attribution-plan.md, src/blockcipher_nd/models/structure/spn/present_trail_position_stats.py, configs/experiment/innovation1/innovation1_spn_present_r8_aligned_trail_position_attribution_512_seed0_seed1.csv, configs/experiment/innovation1/innovation1_spn_present_r8_aligned_trail_position_permute_512_seed0_seed1.csv, configs/experiment/innovation1/innovation1_spn_present_r8_aligned_trail_only_512_seed0_seed1.csv
- Tags: innovation1, present, spn, trail-position, attribution, active-nibble, key-rotation
- See Also: LRN-20260709-004, LRN-20260709-003, LRN-20260709-002
- Pattern-Key: innovation1.spn_present.aligned_trail_position_attribution
- Recurrence-Count: 1
- First-Seen: 2026-07-09
- Last-Seen: 2026-07-09

---

## [LRN-20260709-004] best_practice

**Logged**: 2026-07-09T12:49:23+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
PRESENT r8 aligned active-difference full local gate supports the
trail-position route diagnostically, but next evidence should attribute
trail-position semantics before remote scale.

### Details
The PRESENT r8 aligned-active-difference full gate completed three local
matrices:

- Matrix A full single-active sweep, 256/class, seed0+seed1:
  - seed0 min/mean/max AUC = 0.946472168 / 0.969291687 / 0.987121582
  - seed1 min/mean/max AUC = 0.922180176 / 0.957317352 / 0.978576660
  - All 16 fixed active nibbles stayed high on both seeds once the input
    difference moved with the active coordinate.
- Matrix B aligned random16, 512/class, seed0+seed1:
  - unconditioned seed0 AUC = 0.958282471
  - unconditioned seed1 AUC = 0.972396851
  - p-layer-relative seed0 AUC = 0.755661011
  - p-layer-relative seed1 AUC = 0.895446777
  - Current p-layer-relative statistics still underperform unconditioned
    trail-position statistics.
- Matrix C aligned feature ablation, 256/class, seed0, using compatible
  `present_pairset_global_stats`:
  - ciphertext xor AUC = 0.000000000
  - simple P-aligned and pair P-aligned features stayed near chance
  - S-inverse/DDT/beam tiers stayed around 0.52-0.55
  - full beamstats8deep4 feature reached only AUC = 0.609191895 under global
    stats

This means the old active0-only failure was mostly a protocol-alignment issue,
not a fundamental active-coordinate limitation. It also means the 0.95+ local
aligned-random16 score is not explained by weak ciphertext-xor or ordinary
global-statistics features. The best current interpretation is that the
combination of full SPN-aware feature layout plus the trail-position statistics
model is carrying the signal.

### Suggested Action
Do not remote-scale this local gate as a formal claim yet. The next local
evidence should separate architecture contribution from feature contribution
with compatible trail-position variants and stricter controls, such as
trail-compatible prefix ablations, trail-position permutation controls, and
label/protocol controls that preserve aligned input generation while destroying
trail-position semantics.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-aligned-active-difference-full-gate-plan.md, configs/experiment/innovation1/innovation1_spn_present_r8_aligned_full_single_active_256_seed0_seed1.csv, configs/experiment/innovation1/innovation1_spn_present_r8_aligned_random16_512_seed0_seed1.csv, configs/experiment/innovation1/innovation1_spn_present_r8_aligned_feature_ablation_256_seed0.csv
- Tags: innovation1, present, spn, active-nibble, trail-position, feature-ablation, local-gate
- See Also: LRN-20260709-003, LRN-20260709-002, LRN-20260709-001
- Pattern-Key: innovation1.spn_present.aligned_active_difference_full_gate
- Recurrence-Count: 1
- First-Seen: 2026-07-09
- Last-Seen: 2026-07-09

---

## [LRN-20260709-003] best_practice

**Logged**: 2026-07-09T11:48:13+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Aligned active-difference protocol resolves the active0-only PRESENT r8
artifact, but p-layer-relative stats may hurt mixed active-set performance.

### Details
The PRESENT r8 aligned-active-difference screen at 256/class seed0 added sample
structures where sampled active nibble `k` also moves the one-nibble input
difference to `0x9 << (4*k)`.

Result:

- aligned active0 p-layer-relative AUC = 0.971069336
- aligned active1 p-layer-relative AUC = 0.951477051
- aligned active5 p-layer-relative AUC = 0.983154297
- aligned active15 p-layer-relative AUC = 0.987121582
- aligned random16 unconditioned AUC = 0.958923340
- aligned random16 p-layer-relative AUC = 0.657165527
- aligned active4 p-layer-relative AUC = 0.712707520
- aligned heldout4to4 p-layer-relative AUC = 0.758972168

This confirms that the earlier single-active sweep's active0-only pattern was
largely a protocol-alignment issue: the active coordinate moved, but the input
difference stayed fixed at low nibble 0. Once the difference follows the active
coordinate, representative active positions all become high.

The unconditioned aligned random16 row being high shows that the current strong
feature route can distinguish the aligned random-active protocol without active
metadata at this small scale. The p-layer-relative stats route is weaker on
mixed active sets, suggesting that its coordinate reordering may discard useful
absolute-coordinate or feature-order information.

### Suggested Action
Do not move directly to remote or active-token-bias based only on this screen.
Next run full aligned single-active seed0+seed1 and aligned random16
unconditioned versus p-layer-relative at 512/class seed0+seed1, plus aligned
feature ablation to identify which feature tier creates the high score.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-aligned-active-difference-screen-plan.md, configs/experiment/innovation1/innovation1_spn_present_r8_aligned_active_difference_screen_256_seed0.csv, src/blockcipher_nd/data/differential/rows.py
- Tags: innovation1, present, spn, active-nibble, input-difference, protocol-audit
- See Also: LRN-20260709-002, LRN-20260709-001, LRN-20260708-008
- Pattern-Key: innovation1.spn_present.aligned_active_difference_screen
- Recurrence-Count: 1
- First-Seen: 2026-07-09
- Last-Seen: 2026-07-09

---

## [LRN-20260709-002] best_practice

**Logged**: 2026-07-09T10:51:06+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
PRESENT r8 active-nibble generalization must align the input difference with
the sampled active nibble before interpreting random-active or heldout-active
failures as architecture failures.

### Details
The PRESENT r8 single-active sweep at 256/class seed0 tested
`present_trail_position_stats_pairset` with `active_conditioning =
p_layer_relative_stats` for active nibbles `{0}` through `{15}`. Only active
nibble 0 was high:

- active 0 AUC = 0.949279785
- active 1..15 AUC range = 0.475769043 to 0.547973633
- mean over all active nibbles = 0.529594421

Source inspection shows the likely protocol issue. The random-active integral
sample structures sample `integral_active_nibble`, but the Zhang/Wang
`present_zhang_wang2022_mcnd` input difference remains fixed at
`0x0000000000000009`, i.e. low nibble 0. Thus the protocol moves the integral
active coordinate while leaving the differential anchor at nibble 0. Previous
active1/fixed-active high scores should be interpreted as "active nibble 0 plus
low-nibble difference is learnable", not as evidence that any fixed active
coordinate is equally learnable.

### Suggested Action
Before testing more active-conditioned architectures or remote scale, add an
aligned-active-difference protocol: when active nibble `k` is sampled, use
input difference `0x9 << (4*k)` for that row. Then rerun aligned single-active,
aligned random-active, conditioned random-active, and heldout-active gates.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-single-active-sweep-plan.md, configs/experiment/innovation1/innovation1_spn_present_r8_single_active_sweep_256_seed0.csv, src/blockcipher_nd/data/differential/rows.py, src/blockcipher_nd/registry/difference_profiles.py
- Tags: innovation1, present, spn, active-nibble, input-difference, protocol-audit
- See Also: LRN-20260709-001, LRN-20260708-008, LRN-20260708-007
- Pattern-Key: innovation1.spn_present.active_nibble_requires_aligned_difference
- Recurrence-Count: 1
- First-Seen: 2026-07-09
- Last-Seen: 2026-07-09

---

## [LRN-20260709-001] best_practice

**Logged**: 2026-07-09T00:08:48+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
P-layer-relative statistics improve narrow active-set stability but still do
not unlock PRESENT r8 random-active or heldout-active transfer.

### Details
The PRESENT r8 P-layer-relative active curriculum local diagnostic at
512/class seed0+seed1 added `active_conditioning = p_layer_relative_stats` to
`present_trail_position_stats_pairset`. This maps the active nibble metadata
into PRESENT feature-cell coordinates, places the active S-box cell first, and
places its direct P-layer target cells next before computing trail-position
statistics.

The local gate result:

- Unconditioned random16 baseline stayed near chance: seed0 AUC 0.500534058,
  seed1 AUC 0.525375366.
- Previous simple `relative_stats` random16 stayed unstable: seed0 AUC
  0.526580811, seed1 AUC 0.481582642.
- P-layer-relative active1 stayed high: seed0 AUC 0.975738525, seed1 AUC
  0.993362427.
- P-layer-relative active2 became stable and stronger than the previous
  narrow-set route: seed0 AUC 0.754165649, seed1 AUC 0.761306763.
- P-layer-relative active4/8/16 still collapsed near chance, and heldout train
  {0,1,2,3} -> validation {4,5,6,7} did not transfer: heldout seed0 AUC
  0.492782593, seed1 AUC 0.492889404.

This blocks remote scale for the current statistics-only P-layer-relative
route. The result is useful because it shows PRESENT-aware coordinate ordering
helps the two-active curriculum, but it still does not provide broad
active-coordinate generalization.

### Suggested Action
Do not move `p_layer_relative_stats` to 65k/262k by default. The next local
step should be a real active-conditioned PRESENT graph/token model, where
active metadata modulates token embeddings or P-layer message passing directly,
then rerun the same random-active/heldout gate against the unconditioned
baseline on seed0+seed1.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-p-layer-relative-active-curriculum-plan.md, configs/experiment/innovation1/innovation1_spn_present_r8_p_layer_relative_active_curriculum_512_seed0_seed1.csv, src/blockcipher_nd/models/structure/spn/present_trail_position_stats.py
- Tags: innovation1, present, spn, active-conditioning, p-layer, local-gate
- See Also: LRN-20260708-008, LRN-20260708-007, LRN-20260708-006
- Pattern-Key: innovation1.spn_present.p_layer_relative_stats_gate
- Recurrence-Count: 1
- First-Seen: 2026-07-09
- Last-Seen: 2026-07-09

---

## [LRN-20260708-008] best_practice

**Logged**: 2026-07-08T23:32:48+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Simple stats-level active-relative shifting is not enough to unlock PRESENT r8
random-active or heldout-active transfer.

### Details
The PRESENT r8 active-conditioned curriculum local diagnostic at 512/class
seed0+seed1 added `active_conditioning = relative_stats` to
`present_trail_position_stats_pairset`. Instead of only appending active one-hot
metadata to the final statistics vector, it rotates the per-word cell activity
before computing trail-position statistics, so the active nibble becomes
relative cell 0.

The local gate result:

- Unconditioned random16 baseline stayed near chance: seed0 AUC 0.498641968,
  seed1 AUC 0.525375366.
- Shallow metadata random16 stayed below chance-ish: seed0 AUC 0.473663330,
  seed1 AUC 0.471611023.
- Relative-stats active1 stayed high: seed0 AUC 0.979125977, seed1 AUC
  0.994842529.
- Relative-stats active2 retained some signal: seed0 AUC 0.691108704, seed1
  AUC 0.768592834.
- Relative-stats active4/8/16 mostly collapsed to near chance, and heldout
  train {0,1,2,3} -> validation {4,5,6,7} did not transfer: heldout seed0 AUC
  0.496734619, seed1 AUC 0.460891724.

This blocks remote scale for the current active-conditioned stats route. The
route is locally useful as a diagnostic: it shows the model can use active
conditioning for fixed or two-coordinate cases, but the representation still
does not carry the PRESENT coordinate relationship needed for broad
random-active or heldout-active generalization.

### Suggested Action
Do not move `relative_stats` to 65k/262k by default. The next local step should
use a stricter PRESENT-aware relative-coordinate representation based on the
S-box/P-layer graph, then rerun the same active-set curriculum and heldout gate
against the unconditioned random-active baseline on seed0+seed1.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-active-conditioned-curriculum-plan.md, configs/experiment/innovation1/innovation1_spn_present_r8_active_conditioned_curriculum_512_seed0_seed1.csv, src/blockcipher_nd/models/structure/spn/present_trail_position_stats.py
- Tags: innovation1, present, spn, active-conditioning, active-nibble, local-gate
- See Also: LRN-20260708-007, LRN-20260708-006, LRN-20260708-005
- Pattern-Key: innovation1.spn_present.active_conditioned_relative_stats_gate
- Recurrence-Count: 1
- First-Seen: 2026-07-08
- Last-Seen: 2026-07-08

---

## [LRN-20260708-007] best_practice

**Logged**: 2026-07-08T23:55:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Do not remote-scale the current PRESENT r8 fixed-active matched-negative trail
protocol until active-nibble generalization is redesigned locally.

### Details
The PRESENT r8 active-nibble generalization local diagnostic at 512/class
seed0+seed1 compared the same-budget fixed-active anchor against random-active,
heldout-active, explicit active-nibble metadata, and first-pass relative
coordinate alignment:

- Fixed-active trail stayed high: seed0 AUC 0.989562988, seed1 AUC
  0.992324829.
- Random-active trail stayed near chance: seed0 AUC 0.541259766, seed1 AUC
  0.529830933.
- Heldout-active train {0,1,2,3} -> validation {4,5,6,7} did not transfer:
  seed0 AUC 0.539070129, seed1 AUC 0.473754883.
- Random-active + 16-bit active metadata did not restore signal at this local
  budget: seed0 AUC 0.473663330, seed1 AUC 0.471611023.
- Random-active + first-pass feature-cell relative-coordinate alignment also
  did not restore signal: seed0 AUC 0.534194946, seed1 AUC 0.483474731.

This supports the interpretation that the current high fixed-active score is
mostly tied to fixed active-nibble coordinates/templates, not yet a robust
SPN-general active-nibble adaptive distinguisher. The result is diagnostic only
because 512/class is small, but it is enough to block default remote scale-up
of the unchanged fixed-active protocol under the local-gate rule.

### Suggested Action
For Innovation 1 PRESENT/SPN, redesign the active-conditioned or
nibble-equivariant representation locally before requesting 65k/262k remote
scale. A future remote run should require a local gate where heldout-active,
metadata-conditioned, or relative-coordinate rows beat the unconditioned
random-active control on both seeds.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-active-nibble-generalization-plan.md, configs/experiment/innovation1/innovation1_spn_present_r8_active_nibble_generalization_512_seed0_seed1.csv, src/blockcipher_nd/data/differential/rows.py, src/blockcipher_nd/models/structure/spn/present_trail_position_stats.py
- Tags: innovation1, present, spn, active-nibble, generalization, local-gate
- See Also: LRN-20260708-006, LRN-20260708-005, LRN-20260708-004
- Pattern-Key: innovation1.spn_present.active_nibble_generalization_gate
- Recurrence-Count: 1
- First-Seen: 2026-07-08
- Last-Seen: 2026-07-08

---

## [LRN-20260708-006] best_practice

**Logged**: 2026-07-08T22:45:00+08:00
**Priority**: high
**Status**: promoted
**Area**: research
**Promoted**: AGENTS.md

### Summary
Use local small-scale evidence as the default gate before remote large-scale
SPN/PRESENT runs.

### Details
The user clarified the desired remote-scaling discipline: for Innovation 1
SPN/PRESENT experiments, first use a small local diagnostic to check whether a
route has a real positive signal and whether controls invalidate the protocol.
Only when the local gate is promising should the route move to remote
medium/large scale to test stability. If the local small-scale result is near
chance, unstable, or already invalidated by an easier control/protocol shortcut,
do not spend remote GPU on a larger version of the same route by default.

This rule complements the existing scale language: small runs are not formal
training and are not definitive failures, but they are still valuable
triage/readiness gates. A remote run after a bad local gate should require an
explicit diagnostic reason, such as proving a suspected data-scarcity effect,
not just "try bigger anyway."

### Suggested Action
Before launching SPN/PRESENT remote medium/large runs, require a documented
local readiness gate with same-protocol baseline/control comparison. If the
local gate is weak or invalidated, stop or redesign the protocol/model locally
instead of scaling. Record any exception as an explicit diagnostic exception in
the experiment plan.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, docs/experiments/innovation1-present-r8-bridge-protocol-attribution-plan.md
- Tags: innovation1, present, spn, remote-scale, local-gate, gpu-budget
- See Also: LRN-20260708-005, LRN-20260708-004, LRN-20260621-001
- Pattern-Key: innovation1.remote_scale.requires_promising_local_gate
- Recurrence-Count: 1
- First-Seen: 2026-07-08
- Last-Seen: 2026-07-08

---

## [LRN-20260708-005] best_practice

**Logged**: 2026-07-08T22:15:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Treat fixed active-nibble alignment as the leading explanation for PRESENT r8
matched/integral trail-position strength until active-nibble generalization is
tested.

### Details
The PRESENT r8 bridge protocol attribution diagnostic at 512/class seed0+seed1
showed a clear split:

- Matched-negative full trail stayed high: seed0 AUC 0.973907471, seed1 AUC
  0.994628906.
- Pair-shuffled full trail also stayed high: seed0 AUC 0.951980591, seed1 AUC
  0.992156982. Fixed pair index is therefore not the main explanation.
- Random-active full trail collapsed: seed0 AUC 0.541259766, seed1 AUC
  0.529830933.
- Partial8 full trail collapsed: seed0 AUC 0.498107910, seed1 AUC 0.497482300.
- Same-difference full trail remained near perfect, confirming that protocol is
  too easy to scale as candidate evidence.
- Independent-pair full trail remained near chance, confirming it is not the
  current route.

The useful next research step is active-nibble generalization rather than
larger same-difference or independent-pair scale. Candidate next controls:
train with randomized active nibble plus explicit/equivariant nibble metadata,
or train on some active nibbles and validate on held-out active nibbles.

### Suggested Action
For future Innovation 1 PRESENT/SPN planning, prioritize active-nibble
generalization controls before medium/large matched-negative scale-up. Do not
describe pair-order dependence as the leading hypothesis unless new evidence
contradicts the pair-shuffle result. Do not scale same-difference or
independent-pair variants based on the current bridge diagnostic.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-bridge-protocol-attribution-plan.md, configs/experiment/innovation1/innovation1_spn_present_r8_bridge_protocol_attribution_512_seed0_seed1.csv, src/blockcipher_nd/data/differential/rows.py
- Tags: innovation1, present, spn, bridge-protocol, active-nibble, trail-position
- See Also: LRN-20260708-004, LRN-20260708-003, LRN-20260708-002
- Pattern-Key: innovation1.spn_present.active_nibble_alignment_bridge_signal
- Recurrence-Count: 1
- First-Seen: 2026-07-08
- Last-Seen: 2026-07-08

---

## [LRN-20260708-004] best_practice

**Logged**: 2026-07-08T18:08:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Separate PRESENT r8 multiset shortcuts from matched-negative trail-position
residual evidence.

### Details
The PRESENT r8 weak-XOR deterministic audit found two different effects that
must not be collapsed into one interpretation:

- Strict random-negative and same-difference random-negative protocols are
  perfectly separable by `left_right_column_sum_l1_mean`: positives have
  identical left/right ciphertext multisets across the 16 integral pairs while
  negatives do not. These protocols are too easy and should not be scaled.
- The matched-negative protocol removes that exact multiset shortcut
  (`left_right_column_sum_l1_mean` AUC = 0.5), but still has a strong
  deterministic `pair_xor_column_sum_variance` baseline around 0.88-0.89 AUC at
  2048/class.

In the matched residual neural audit, the global full-beamstats row did not
beat the deterministic pair-XOR variance baseline, but the trail-position row
did by about +0.11 AUC on seed0 and seed1, including per-row key rotation.
That supports local residual value for trail-position, but remains diagnostic
evidence rather than formal PRESENT r8 evidence.

### Suggested Action
For future Innovation 1 PRESENT/SPN reports, say explicitly which level is
being discussed:

1. strict/same-difference random-negative protocol ease from left/right
   multiset mismatch;
2. matched-negative deterministic pair-XOR variance baseline;
3. trail-position residual over that deterministic baseline.

Do not recommend remote scale-up for strict or same-difference random-negative
protocols. For matched-negative scale-up, first require a frozen-score or
residualized comparison against `pair_xor_column_sum_variance`.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-leakage-audit-plan.md, src/blockcipher_nd/cli/audit_integral_parity_signal.py, configs/experiment/innovation1/innovation1_spn_present_r8_matched_residual_audit_2048_seed0_seed1.csv
- Tags: innovation1, present, spn, leakage-audit, matched-negative, deterministic-baseline, trail-position
- See Also: LRN-20260708-003, LRN-20260708-002, LRN-20260621-001
- Pattern-Key: innovation1.spn_present.matched_residual_vs_deterministic_baseline
- Recurrence-Count: 1
- First-Seen: 2026-07-08
- Last-Seen: 2026-07-08

---

## [LRN-20260708-003] best_practice

**Logged**: 2026-07-08T17:28:35+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
For PRESENT r8 integral/trail-position audits, explain weak XOR sign-flipped
separation before scaling a high-AUC protocol.

### Details
The PRESENT r8 leakage audit showed that strict random-negative and
same-difference random-negative variants can both keep full beamstats and
trail-position AUC near 1.0 at 2048/class across seed0 and seed1. Per-row key
rotation also did not collapse those rows.

However, the weak `ciphertext_xor_bits` global-stat row produced AUC 0.0 with
oriented AUC 1.0 in both audits, while P-layer aligned and InvS-only views were
also above chance. This means a simple public XOR-derived view already carries
class structure in a sign-flipped direction. A larger remote run of the exact
full-feature protocol would mostly confirm that the protocol is easy, not that
the trail-position network has isolated a publication-quality PRESENT r8
neural distinguisher.

### Suggested Action
Before scaling PRESENT/SPN integral or trail-position protocols that score near
1.0 locally, add a deterministic or non-neural audit that identifies the simple
statistic behind the `ciphertext_xor_bits` reversed ordering. Only scale a
candidate protocol after weak XOR-derived controls are explained, removed, or
clearly separated from the claimed trail-position residual signal.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-leakage-audit-plan.md, src/blockcipher_nd/data/differential/rows.py, configs/experiment/innovation1/innovation1_spn_present_r8_same_difference_audit_2048_seed0_seed1.csv
- Tags: innovation1, present, spn, leakage-audit, weak-xor, protocol-design
- See Also: LRN-20260708-002, LRN-20260621-001
- Pattern-Key: innovation1.spn_present.weak_xor_sign_flip_before_scaling
- Recurrence-Count: 1
- First-Seen: 2026-07-08
- Last-Seen: 2026-07-08

---

## [LRN-20260708-002] correction

**Logged**: 2026-07-08T16:58:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Do not describe `plaintext_integral_nibble_difference_matched_negative` as strict random-plaintext negatives.

### Details
While auditing why the PRESENT r8 trail-position route scores unusually high,
the source path in `src/blockcipher_nd/data/differential/rows.py` showed that
`_generate_integral_difference_matched_negative_row` constructs negative rows
from the same integral base and shifted active-nibble variants, then encrypts
both plaintexts. In this branch, the `negative_mode` metadata value
`encrypted_random_plaintexts` does not mean the second plaintext is an
independent random plaintext as in the ordinary independent-pair negative path.

This matters for interpretation: the route is a controlled matched-negative
integral diagnostic, not a strict random-plaintext-negative PRESENT r8
benchmark. Strong AUC may reflect a real active-nibble/input-difference
alignment signal in this structured protocol, but it should not be reported as
a standard raw 8-round distinguisher result without qualification.

### Suggested Action
When reporting trail-position beamstats results, explicitly call the protocol
`matched-negative integral` and separate it from strict encrypted-random-
plaintext-negative evidence. Before publication-style claims, add or require a
strict independent random-plaintext negative control and a random-key/generalized
key split audit.

### Metadata
- Source: source_audit
- Related Files: src/blockcipher_nd/data/differential/rows.py, docs/experiments/innovation1-present-r8-trail-position-beamstats-smoke-plan.md
- Tags: innovation1, present, spn, negative-samples, matched-negative, interpretation
- See Also: LRN-20260621-001
- Pattern-Key: innovation1.spn_present.matched_negative_not_strict_random_negative
- Recurrence-Count: 1
- First-Seen: 2026-07-08
- Last-Seen: 2026-07-08

---

## [LRN-20260708-001] correction

**Logged**: 2026-07-08T12:00:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Verify every arXiv identifier against its current title and authors before
using it in Innovation 1 route arbitration.

### Details
During a follow-up source check for the SPN/PRESENT route arbitration, the
project notes correctly treated arXiv:2511.06336 as relevant RX-neural
cryptanalysis evidence, but cited it under an incorrect title and author line.
The verified arXiv page identifies the paper as "Enhancing Deep Learning-Based
Rotational-XOR Attacks on Lightweight Block Ciphers Simon32/64 and
Simeck32/64" by Liu, Chen, Xiang, Zhang, and Zeng.

This is related to the earlier arXiv:2505.10792 correction: an arXiv ID can be
present in a local note while the title, topic, or authors are wrong. A route
decision should not rely on memory or copied citation metadata alone.

### Suggested Action
Before citing a paper as route evidence, open the arXiv/ePrint/source page and
verify the identifier, title, authors, year/date, and topic. If only the broad
takeaway remains valid after correction, update the note to separate the
verified bibliographic fact from the inferred route implication.

### Metadata
- Source: source_recheck
- Related Files: docs/research/innovation1-spn-route-arbitration-20260707.md, docs/research/innovation1-spn-bit-sensitivity-nonneighbor-route-20260707.md
- Tags: innovation1, literature, citation-hygiene, arxiv, spn
- See Also: LRN-20260707-002
- Pattern-Key: research.citation_id_must_match_title_topic
- Recurrence-Count: 2
- First-Seen: 2026-07-07
- Last-Seen: 2026-07-08

---

## [LRN-20260707-001] best_practice

**Logged**: 2026-07-07T07:05:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Do not overclaim raw117 stacking diagnostics when the train and validation score artifacts use different compressed feature scopes.

### Details
The PRESENT r8 trail-position plus raw117 candidate stacking diagnostic exposed
an interpretation risk. The validation-side compact structural expert was the
intended raw117 artifact:

```text
feature_view = compressed_span_summary
feature_count = 117
validation artifacts =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_span_raw_combo_anchor_plus_aux-depth-word_aux-word-global_scores
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed1_span_raw_combo_anchor_plus_aux-depth-word_aux-word-global_scores
```

At the time of the first stacking stability audit, no matching train raw117
score artifact was found. The train-side structural score input used for fitted
stacking was the wider trail-position-stat logistic artifact:

```text
feature_view = trail_position_stats
feature_count = 3708
train artifacts =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_train_trail_stats_logistic_scores
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed1_train_trail_stats_logistic_scores
```

Since `scripts/evaluate-stacked-ensemble` fits only on frozen score columns, this
is still a useful two-score calibration diagnostic if the model keys align. It
is not strict raw117 train-fitted calibration and should not be promoted as
evidence that raw117 stacking is solved. The clean evidence remains the
candidate-only fixed fusion on aligned validation artifacts; fitted stacking
should be rerun with matching train raw117 score artifacts before stronger
claims.

### Suggested Action
Before reporting future frozen-score stacking results as model-specific
calibration, inspect both train and validation `models.json` files for matching
feature view, feature count, feature selection, negative mode, sample structure,
and score split. If only score columns are matched but feature scopes differ,
label the result as a limited score-level calibration diagnostic and generate
matching train score artifacts before making stronger ensemble claims.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-bit-sensitivity-projection-expert-plan.md, scripts/evaluate-stacked-ensemble
- Tags: innovation1, spn, present, raw117, stacking, feature-scope, ensemble
- See Also: LRN-20260706-027, LRN-20260621-001
- Pattern-Key: innovation1.spn_present.stacking_feature_scope_must_match
- Recurrence-Count: 1
- First-Seen: 2026-07-07
- Last-Seen: 2026-07-07

### Resolution
- **Resolved**: 2026-07-07T07:18:00+08:00
- **Commit/PR**: pending
- **Notes**: Matching train/validation raw117 score artifacts were generated under `outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed{0,1}_*_span_raw117_matched_scores`, and a matched stacking stability follow-up was recorded in V14 of the experiment plan. The durable lesson remains: verify feature scope before interpreting stacking.

---

## [LRN-20260707-002] correction

**Logged**: 2026-07-07T09:55:00+08:00
**Priority**: medium
**Status**: pending
**Area**: research

### Summary
Do not trust a stored arXiv identifier in research notes until the title and
topic are re-opened and verified.

### Details
During the Innovation 1 SPN route refresh, the local research note
`docs/research/innovation1-spn-bit-sensitivity-nonneighbor-route-20260707.md`
listed arXiv:2505.10792 as "More Efficient Deep Learning-Based
Distinguishing Attacks with Multiple Ciphertext Pairs." Re-opening
`https://arxiv.org/abs/2505.10792` showed that the identifier belongs to an
unrelated retrieval-augmented generation / LLM paper, not a block-cipher
neural-cryptanalysis paper.

This kind of citation mismatch can distort route arbitration by making a
weakly verified literature thread look stronger than it is.

### Suggested Action
Before using a paper as evidence for route ranking, verify at least the title,
venue/source page, year, and topic from the actual URL or PDF. If an arXiv ID
does not match the claimed title/topic, remove it from the evidence list and
record the correction in the relevant research note.

### Metadata
- Source: research_audit
- Related Files: docs/research/innovation1-spn-bit-sensitivity-nonneighbor-route-20260707.md, docs/research/innovation1-spn-route-arbitration-20260707.md
- Tags: innovation1, literature, citation-hygiene, arxiv, spn
- See Also: LRN-20260707-001
- Pattern-Key: research.citation_id_must_match_title_topic
- Recurrence-Count: 1
- First-Seen: 2026-07-07
- Last-Seen: 2026-07-07

---

## [LRN-20260621-001] correction

**Logged**: 2026-06-21T20:45:00+08:00
**Priority**: critical
**Status**: promoted
**Area**: docs

### Summary
Do not treat SPN/PRESENT small or medium sample runs as formal evidence that a model or route has failed.

### Details
The user corrected an important experimental interpretation error: prior SPN/PRESENT work was discussed too much like formal training, but local completed metrics show PRESENT/SPN results had only reached about `65536` samples per class. Several logs mentioning `131072` or `262144` rows were often total rows across both classes, cache/progress rows, queue plans, or incomplete runs, not completed `>100000/class` formal training.

Correct project distinction:

- Smoke/screen: below `65536/class`.
- Medium diagnostic: `65536/class` through about `524288/class`.
- Formal training: at least `1000000/class`, preferably multi-seed, fixed protocol, completed, retrieved, and plan-aligned.

For SPN/PRESENT, before claiming a route has hit its ceiling or failed, require completed and retrieved scale evidence. A `32k/class` or `65k/class` result may reject only obviously dead variants; it must not be used as a definitive conclusion that the overall route cannot improve.

Current factual baseline from the 2026-06-21 audit:

- ARX/SPECK has completed results above `100000/class`, including `131072/class` and `262144/class`.
- SPN/PRESENT completed metric rows found locally maxed out around `65536/class`.
- Therefore, prior SPN/PRESENT accuracy bottleneck claims were under-supported by large-scale evidence.

### Suggested Action
For future SPN/PRESENT experiments, always state the scale class in reports and labels. Use small runs only as screens. Before making negative claims about accuracy ceilings, run and retrieve at least a medium scale ladder such as `65536/class -> 262144/class`, and reserve "formal result" language for `>=1000000/class` multi-seed completed runs.

### Metadata
- Source: user_feedback
- Related Files: outputs/, experiments/innovation1/plans/, experiments/innovation1/configs/remote/
- Tags: spn, present, experiment-scale, formal-training, accuracy-interpretation
- Pattern-Key: innovation1.spn_present.formal_scale_required
- Recurrence-Count: 1
- First-Seen: 2026-06-21
- Last-Seen: 2026-06-21
- Promoted: AGENTS.md

---

## [LRN-20260706-024] best_practice

**Logged**: 2026-07-06T12:08:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Trail-position r8 4096/class has a completed local residual gate; 65k remote launch remains blocked by unpushed source.

### Details
The PRESENT-80 r8 trail-position route was extended from a 4096/class neural
bridge to a completed same-scale residual diagnostic gate:

```text
plan = configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_4096_local.csv
results = outputs/local_smoke/i1_present_r8_trail_position_beamstats_4096/results.jsonl
control_audits =
  outputs/local_audits/i1_present_r8_trail_position_control_baseline_seed0_4096.json
  outputs/local_audits/i1_present_r8_trail_position_control_baseline_seed1_4096.json
gate = outputs/local_audits/i1_present_r8_trail_position_residual_gate_4096.json
decision = support_trail_position_neural_residual_local
pair_order_assessment = pair_order_not_bottleneck
```

Gate margins:

```text
min_candidate_margin_vs_deterministic_auc = 0.16052186489105225
min_candidate_margin_vs_global_auc = 0.09591805934906006
min_deterministic_margin_vs_mismatch_auc = 0.26223671436309814
```

Key metrics:

| Seed | Candidate AUC | Global control AUC | Deterministic baseline AUC | Max mismatch control AUC |
|---:|---:|---:|---:|---:|
| 0 | `0.9999396800994873` | `0.8999881744384766` | `0.7746385335922241` | `0.512401819229126` |
| 1 | `0.9999489784240723` | `0.9040309190750122` | `0.83942711353302` | `0.5200802683830261` |

Correct interpretation:

- The 4096/class result is now a full local residual diagnostic gate, not only
  a neural bridge over the global-stat control.
- It remains local diagnostic evidence only: not remote evidence, not formal
  SPN/PRESENT evidence, not a Zhang/Wang r7 Case2 result, not a breakthrough
  claim, and not a diverse multi-network ensemble result.
- The cached/progress-enabled `audit-spn-features` path is required for
  practical same-scale deterministic/mismatch controls.
- The prepared 65k/class remote diagnostic is still not launched. The local
  source publication gate remains blocked because `main` is ahead of
  `origin/main` and external `git push origin main` was rejected by safety
  review without explicit user approval for pushing all local commits.

### Suggested Action
When reporting current Innovation 1 trail-position status, cite 4096/class as
completed local residual diagnostic evidence and 65k/class as prepared only /
not launched. Before remote launch, obtain explicit approval for the exact
external push or otherwise get the source publication gate to pass through an
approved channel; do not use dirty-overlay or unpushed-code launches.

### Metadata
- Source: experiment_audit, implementation, safety_review
- Related Files: src/blockcipher_nd/tasks/innovation1/spn_feature_audit.py, tests/test_project_structure.py, docs/experiments/innovation1-present-r8-trail-position-beamstats-smoke-plan.md
- Tags: innovation1, spn, present, trail-position, residual-gate, local-diagnostic, source-publication
- See Also: LRN-20260706-022, LRN-20260706-021, LRN-20260706-020
- Pattern-Key: innovation1.spn_present.trail_position_4096_residual_gate_complete
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-025] best_practice

**Logged**: 2026-07-06T12:45:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Trail-position r8 8192/class local residual gate completed and still supports the route.

### Details
The PRESENT-80 r8 trail-position route now has a completed `8192/class` local
residual diagnostic gate:

```text
plan = configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_8192_local.csv
results = outputs/local_smoke/i1_present_r8_trail_position_beamstats_8192/results.jsonl
control_audits =
  outputs/local_audits/i1_present_r8_trail_position_control_baseline_seed0_8192.json
  outputs/local_audits/i1_present_r8_trail_position_control_baseline_seed1_8192.json
gate = outputs/local_audits/i1_present_r8_trail_position_residual_gate_8192.json
decision = support_trail_position_neural_residual_local
pair_order_assessment = pair_order_not_bottleneck
```

Gate margins:

```text
min_candidate_margin_vs_deterministic_auc = 0.16092461347579956
min_candidate_margin_vs_global_auc = 0.06049758195877075
min_deterministic_margin_vs_mismatch_auc = 0.274130642414093
```

Key metrics:

| Seed | Candidate AUC | Global control AUC | Deterministic baseline AUC | Max mismatch control AUC |
|---:|---:|---:|---:|---:|
| 0 | `0.9999940395355225` | `0.9394964575767517` | `0.7937679886817932` | `0.5196373462677002` |
| 1 | `0.9999982118606567` | `0.9360240697860718` | `0.8390735983848572` | `0.5226998701691628` |

Correct interpretation:

- The 8192/class result is a local diagnostic gate, not remote evidence and not
  formal SPN/PRESENT evidence.
- The global-stat neural control is very strong at about `0.936-0.939` AUC, so
  this benchmark contains strong non-position global structure.
- The trail-position candidate still clears the global control by at least
  `+0.0605` AUC and clears the deterministic baseline by at least `+0.1609`
  AUC, so the position-aware neural residual remains positive.
- Active-nibble/input-difference mismatch controls remain near random at about
  `0.52` AUC, while pair-order reverse tracks the deterministic baseline; treat
  this as `pair_order_not_bottleneck`, not as route failure.
- This is not evidence for a diverse multi-network ensemble. Diverse ensemble
  work still requires frozen scores, explicit expert-family metadata, and
  diversity/error-overlap gates.

### Suggested Action
Keep trail-position residual as the current strongest controlled local
SPN/integral candidate. The next meaningful scale step remains a cache-ready
medium diagnostic once source-publication/remote launch policy allows it, or a
structurally different non-neighbor expert with compatible frozen scores for
future diversity evaluation.

### Metadata
- Source: experiment_audit
- Related Files: configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_8192_local.csv, docs/experiments/innovation1-present-r8-trail-position-beamstats-smoke-plan.md
- Tags: innovation1, spn, present, trail-position, residual-gate, local-diagnostic
- See Also: LRN-20260706-024, LRN-20260706-023, LRN-20260706-021
- Pattern-Key: innovation1.spn_present.trail_position_8192_residual_gate_complete
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-027] best_practice

**Logged**: 2026-07-06T14:30:46+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
PRESENT r8 cell-value histogram screen completed and should not be promoted to the diverse expert pool in its current form.

### Details
The PRESENT-80 r8 cell-value histogram local diagnostic completed the planned
`2048/class`, seeds `0,1` screen:

```text
plan = configs/experiment/innovation1/innovation1_spn_present_r8_cell_value_histogram_2048_local.csv
results = outputs/local_smoke/i1_present_r8_cell_value_histogram_2048/results.jsonl
gate = outputs/local_smoke/i1_present_r8_cell_value_histogram_2048/gate.json
decision = hold_cell_value_histogram_local_screen
action = do_not_promote_to_diverse_expert_pool
```

Gate metrics:

| Seed | Global-stat control AUC | Histogram candidate AUC | Candidate margin |
|---:|---:|---:|---:|
| 0 | `0.8532476425170898` | `0.5871686935424805` | `-0.2660789489746094` |
| 1 | `0.874751091003418` | `0.6144542694091797` | `-0.2602968215942383` |

The histogram candidate is weak-positive above random on both seeds, but it
loses badly to the same-input global-statistics control. This means it is not a
useful diverse/non-neighbor expert for current multi-network aggregation. Do
not treat the gate script's `"status": "pass"` as candidate promotion; the
research decision is the explicit `hold_cell_value_histogram_local_screen`.

Correct interpretation:

- Completed local diagnostic only, not remote evidence and not formal
  SPN/PRESENT evidence.
- Not a breakthrough claim and not a diverse expert pool result.
- The current bottleneck for multi-network work is finding a structurally
  different expert that also clears strong same-budget controls, not simply
  adding a weak different-looking model.

### Suggested Action
Keep trail-position residual as the current strongest local controlled SPN
route. For ensemble work, require future non-neighbor experts to first clear
same-input global-stat controls and then produce compatible frozen scores for
diversity/error-overlap gates. Redesign or replace the cell-value histogram
route before spending more ensemble effort on it.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-cell-value-histogram-screen-plan.md, configs/experiment/innovation1/innovation1_spn_present_r8_cell_value_histogram_2048_local.csv, outputs/local_smoke/i1_present_r8_cell_value_histogram_2048/gate.json
- Tags: innovation1, spn, present, cell-value-histogram, diverse-ensemble, local-diagnostic
- See Also: LRN-20260706-026, LRN-20260706-025, LRN-20260706-021
- Pattern-Key: innovation1.spn_present.cell_value_histogram_2048_hold
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-028] best_practice

**Logged**: 2026-07-06T14:38:35+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Trail-position frozen-score recovery completed, but near-neighbor global/trail aggregation does not improve the best single expert.

### Details
The r8 trail-position `2048/class` local checkpoint recovery completed and
exported aligned frozen score artifacts for seed-local global-stat controls and
trail-position candidates:

```text
results = outputs/local_smoke/i1_present_r8_trail_position_beamstats_2048_score_artifacts/results.jsonl
seed0_ensemble = outputs/local_smoke/i1_present_r8_trail_position_beamstats_2048_score_artifacts/seed0_global_vs_trail_ensemble.json
seed1_ensemble = outputs/local_smoke/i1_present_r8_trail_position_beamstats_2048_score_artifacts/seed1_global_vs_trail_ensemble.json
```

Recovered training metrics:

| Seed | Global-stat AUC | Trail-position AUC | Trail margin |
|---:|---:|---:|---:|
| 0 | `0.8490915298461914` | `0.9990301132202148` | `+0.14993858337402344` |
| 1 | `0.874751091003418` | `0.9994010925292969` | `+0.1246500015258789` |

Frozen-score ensemble diagnostics:

| Seed | Best single AUC | Best ensemble AUC | Delta |
|---:|---:|---:|---:|
| 0 | `0.9985876083374023` | `0.9983005523681641` | `-0.00028705596923828125` |
| 1 | `0.9982948303222656` | `0.9974737167358398` | `-0.0008211135864257812` |

The diverse-expert gate reported:

```text
decision = diverse_expert_pool_not_ready
errors = ["too_few_eligible_families"]
eligible_family_count = 2
min_family_count = 3
```

Correct interpretation:

- Score artifact compatibility is now demonstrated for the global/trail pair.
- Simple near-neighbor aggregation of the global-stat control and
  trail-position candidate does not improve AUC over the best single
  trail-position expert.
- This is local application-level aggregation diagnostic evidence only, not
  remote evidence, not formal SPN/PRESENT evidence, and not raw single-sample
  SOTA evidence.
- Multi-network work remains valuable, but only after adding a genuinely
  different expert family that clears same-budget controls.

### Suggested Action
Do not average more near-neighbor global/trail variants expecting a real
Innovation 1 gain. Prioritize a third non-neighbor expert family that first
beats the same-input global-stat control, then export compatible frozen scores
and re-run diversity/error-overlap gates.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-trail-position-beamstats-smoke-plan.md, outputs/local_smoke/i1_present_r8_trail_position_beamstats_2048_score_artifacts/seed0_global_vs_trail_ensemble.json, outputs/local_smoke/i1_present_r8_trail_position_beamstats_2048_score_artifacts/seed1_global_vs_trail_ensemble.json
- Tags: innovation1, spn, present, trail-position, frozen-scores, neural-ensemble, diverse-expert
- See Also: LRN-20260706-027, LRN-20260706-026, LRN-20260706-021
- Pattern-Key: innovation1.spn_present.near_neighbor_ensemble_not_enough
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-026] best_practice

**Logged**: 2026-07-06T14:08:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Trail-position r8 16384/class local residual gate completed and remains the strongest controlled local SPN candidate.

### Details
The PRESENT-80 r8 trail-position route now has a completed `16384/class` local
residual diagnostic gate:

```text
plan = configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_16384_local.csv
results = outputs/local_smoke/i1_present_r8_trail_position_beamstats_16384/results.jsonl
control_audits =
  outputs/local_audits/i1_present_r8_trail_position_control_baseline_seed0_16384.json
  outputs/local_audits/i1_present_r8_trail_position_control_baseline_seed1_16384.json
gate = outputs/local_audits/i1_present_r8_trail_position_residual_gate_16384.json
decision = support_trail_position_neural_residual_local
pair_order_assessment = pair_order_not_bottleneck
```

Gate margins:

```text
min_candidate_margin_vs_deterministic_auc = 0.2178511768579483
min_candidate_margin_vs_global_auc = 0.04398629069328308
min_deterministic_margin_vs_mismatch_auc = 0.25441595166921616
```

Key metrics:

| Seed | Candidate AUC | Global control AUC | Deterministic baseline AUC | Max mismatch control AUC |
|---:|---:|---:|---:|---:|
| 0 | `0.9999993294477463` | `0.9545977339148521` | `0.7787629086524248` | `0.5243469569832087` |
| 1 | `0.9999985545873642` | `0.9560122638940811` | `0.7821473777294159` | `0.5265582799911499` |

Correct interpretation:

- The 16384/class result is a completed local residual diagnostic gate, not
  remote evidence and not formal SPN/PRESENT evidence.
- The same-input global-stat neural control is very strong at about `0.955`
  AUC, so the setting contains substantial global integral/statistical signal.
- The trail-position candidate still clears the global control by at least
  `+0.04398629069328308` AUC and clears the deterministic baseline by at least
  `+0.2178511768579483` AUC, so the position-aware neural residual remains
  positive under stronger local diagnostics.
- Active-nibble/input-difference mismatch controls remain near random at about
  `0.52` AUC, while pair-order reverse tracks the deterministic baseline; treat
  this as `pair_order_not_bottleneck`, not as route failure.
- This is not evidence for a diverse multi-network ensemble. Diverse ensemble
  work still requires compatible frozen scores, explicit `expert_family`
  metadata, and diversity/error-overlap gates.

### Suggested Action
Treat trail-position residual as the current strongest controlled local
SPN/integral candidate. The next route work should either wait for a
source-published cache-ready medium diagnostic, or continue screening a
structurally different non-neighbor expert such as cell-value histogram before
attempting a diverse ensemble.

### Metadata
- Source: experiment_audit
- Related Files: configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_16384_local.csv, docs/experiments/innovation1-present-r8-trail-position-beamstats-smoke-plan.md
- Tags: innovation1, spn, present, trail-position, residual-gate, local-diagnostic, diverse-ensemble-gate
- See Also: LRN-20260706-025, LRN-20260706-024, LRN-20260706-021, LRN-20260706-022
- Pattern-Key: innovation1.spn_present.trail_position_16384_residual_gate_complete
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-022] best_practice

**Logged**: 2026-07-06T09:20:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Prepare r8 trail-position 65k/class as a cache-ready medium diagnostic, not as an ensemble or formal claim.

### Details
After the user emphasized that route selection must not simply follow the
"multiple neural networks" idea, the current trail-position residual evidence
was rechecked. The strongest next SPN-adaptive route remains:

```text
active/difference-aligned SPN trail-position statistics
+ deterministic split/control baselines
+ neural residual gate
```

Prepared assets:

```text
plan = configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_65k_seed0.csv
remote_readiness_config = configs/remote/innovation1_spn_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706.json
scale = 65536/class
rows = present_pairset_global_stats, present_trail_position_stats_pairset
status = prepared only / not launched
```

The readiness config passed local static checking, including
`medium_scale_dataset_cache`, `cmd.exe /c`, `G:\lxy` artifact policy, and
training-protocol consistency. It was later hardened with
`trail_position_score_artifact_lock`, requiring `checkpoint_output_dir`,
`score_artifacts_root`, and per-row score export metadata. Generated
launch/monitor artifacts were then prepared and audited for the two-row 65k
matrix. This is still only a readiness asset. It is not remote-launch evidence,
not formal SPN/PRESENT evidence, not a breakthrough claim, and not a
multi-network aggregation result.

Correct route arbitration:

- Keep trail-position residual as the current best controlled local
  SPN/integral candidate.
- Keep multi-network aggregation as a later diversity-gated validator.
- Do not spend the next main slot on near-neighbor aggregation unless a
  structurally different expert has compatible frozen scores, weak-positive
  same-scale AUC, low error overlap, and no deterministic-control explanation.

### Suggested Action
Before any future trail-position remote launch, require a scoped commit, push,
generated launch-artifact audit, GPU/readiness gate, local monitor handoff, and
one bounded remote artifact confirmation. After retrieval, rerun the residual
gate at `65536/class` before making any stronger claim. If the route may enter
future diverse-ensemble work, export frozen scores with `expert_family` and
`candidate_status` metadata before running the diversity/error-overlap gate.

### Metadata
- Source: user_feedback, experiment_audit
- Related Files: configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_65k_seed0.csv, configs/remote/innovation1_spn_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706.json, docs/experiments/innovation1-present-r8-trail-position-beamstats-smoke-plan.md, docs/research/innovation1-spn-independent-route-recheck-20260706.md
- Tags: innovation1, spn, present, trail-position, medium-readiness, neural-ensemble, route-selection
- See Also: LRN-20260706-021, LRN-20260706-020, LRN-20260706-014
- Pattern-Key: innovation1.spn_present.trail_position_65k_readiness_not_ensemble_claim
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-023] best_practice

**Logged**: 2026-07-06T09:55:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Require explicit expert-family metadata before treating neural ensembles as diverse SPN expert pools.

### Details
The user asked whether multiple neural networks should combine diverse
directions, not merely similar nearby variants. The code now distinguishes:

```text
near-neighbor score aggregation
vs
diverse multi-network expert pooling
```

Implemented behavior:

```text
scripts/export-checkpoint-scores --expert-family <family> --candidate-status <status>
blockcipher_nd.evaluation.neural_ensemble.assess_diverse_expert_pool
neural_ensemble_summary.json["diverse_expert_pool"]
postprocess decision keep_near_neighbor_ensemble_control_not_diverse_pool
```

Default near-neighbor families:

```text
invp_cell
ddt_graph
p_layer_graph
```

Non-neighbor candidates should use labels such as:

```text
raw_mcnd
pair_evidence
inverse_round_matrix
trail_position
projection_feature
```

Correct interpretation:

- A positive ensemble delta alone is not enough to call a route a diverse
  multi-network result.
- If `diverse_expert_pool.status == fail`, postprocess must treat the result as
  a near-neighbor control even when AUC improves.
- Trail-position residual can become a future non-neighbor expert only after
  compatible frozen score artifacts exist and the deterministic/control gates
  still pass.

### Suggested Action
When exporting future score artifacts for ensemble work, always include
`--expert-family` and `--candidate-status`. Before preparing larger ensemble
confirmations, require the diverse-family gate to pass in addition to AUC delta
and error-overlap checks.

### Metadata
- Source: user_feedback, implementation
- Related Files: src/blockcipher_nd/evaluation/neural_ensemble.py, src/blockcipher_nd/cli/export_checkpoint_scores.py, src/blockcipher_nd/planning/neural_ensemble_postprocess.py, docs/experiments/innovation1-present-diverse-expert-pool-plan.md
- Tags: innovation1, spn, present, neural-ensemble, diverse-experts, frozen-scores
- See Also: LRN-20260706-022, LRN-20260706-014
- Pattern-Key: innovation1.spn_present.neural_ensemble_requires_expert_family_gate
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-013] best_practice

**Logged**: 2026-07-06T07:25:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Treat the first GIFT-64 generic SPN-aligned representation screen as weak-positive local evidence only.

### Details
After the user emphasized independent route selection over simply following a
larger multi-network ensemble idea, a controlled local GIFT-64 cross-SPN
representation diagnostic was run:

```text
plan = configs/experiment/innovation1/innovation1_spn_gift64_cross_spn_cell_repr_local.csv
cipher = GIFT-64
rounds = 6
samples_per_class = 2048
pairs_per_sample = 4
negative_mode = encrypted_random_plaintexts
sample_structure = independent_pairs
difference_profile = gift64_shen2024_spn_screen
```

Results:

| Model | Feature encoding | AUC |
|---|---|---:|
| `mlp` | `ciphertext_pair_bits` | `0.5167593955993652` |
| `spn_token_mixer_pairset` | `ciphertext_pair_bits` | `0.5172939300537109` |
| `spn_token_mixer_pairset` | `ciphertext_pair_xor_bits` | `0.4951457977294922` |
| `spn_token_mixer_pairset` | `ciphertext_pair_xor_spn_aligned_bits` | `0.5226421356201172` |

Deltas:

```text
SPN-aligned - C||C'||DeltaC token mixer AUC = +0.027496337890625
SPN-aligned - raw token mixer AUC = +0.00534820556640625
SPN-aligned - raw MLP AUC = +0.005882740020751953
```

Correct interpretation:

- The generic inverse-permutation SPN-aligned representation is the best row and
  clears the local gate against the `C||C'||DeltaC` control.
- The absolute AUC is only `0.5226421356201172`, so this is a weak local signal,
  not a GIFT result or remote-launch basis.
- This route is a candidate non-neighbor expert source for a future diverse
  pool only after local repeat or medium-scale confirmation.

The same local protocol was repeated on seeds `1` and `2`. In both repeat seeds,
the aligned row remained the best row:

```text
seed1 aligned AUC = 0.5273561477661133
seed2 aligned AUC = 0.5216836929321289
```

Three-seed aggregate:

| Model | Feature encoding | Mean AUC | Min AUC | Max AUC |
|---|---|---:|---:|---:|
| `mlp` | `ciphertext_pair_bits` | `0.5123786926269531` | `0.5064530372619629` | `0.5167593955993652` |
| `spn_token_mixer_pairset` | `ciphertext_pair_bits` | `0.507353941599528` | `0.5001511573791504` | `0.5172939300537109` |
| `spn_token_mixer_pairset` | `ciphertext_pair_xor_bits` | `0.4998714129130046` | `0.4951457977294922` | `0.5029439926147461` |
| `spn_token_mixer_pairset` | `ciphertext_pair_xor_spn_aligned_bits` | `0.5238939921061198` | `0.5216836929321289` | `0.5273561477661133` |

Updated interpretation:

- The result is now a stable weak local positive across three seeds at
  `2048/class`.
- It still does not justify remote launch or inclusion as a qualified ensemble
  expert, because the absolute AUC is low and the scale is diagnostic only.
- It does justify a medium diagnostic design that keeps the same four-row
  attribution structure.

The medium diagnostic was then run at `8192/class` across seeds `0`, `1`, and
`2`, using the same four-row attribution matrix. The aligned row still had the
best three-seed mean AUC, but the advantage collapsed toward chance and
seed-level ordering became mixed:

| Model | Feature encoding | Mean AUC | Min AUC | Max AUC |
|---|---|---:|---:|---:|
| `mlp` | `ciphertext_pair_bits` | `0.497802476088206` | `0.4951976239681244` | `0.49981915950775146` |
| `spn_token_mixer_pairset` | `ciphertext_pair_bits` | `0.5016828974088033` | `0.4986715614795685` | `0.5061184763908386` |
| `spn_token_mixer_pairset` | `ciphertext_pair_xor_bits` | `0.5035406351089478` | `0.5009808540344238` | `0.5053452849388123` |
| `spn_token_mixer_pairset` | `ciphertext_pair_xor_spn_aligned_bits` | `0.5053561429182688` | `0.5010229349136353` | `0.5075488686561584` |

Updated medium interpretation:

- The 2048/class positive was not cleanly amplified at 8192/class.
- The aligned row beat `C||C'||DeltaC` on seed0 and seed2 but lost on seed1.
- Mean aligned advantage over `C||C'||DeltaC` was only about `+0.0018` AUC.
- This is not enough for remote launch, GIFT route promotion, or qualified
  diverse-ensemble expert status.

### Suggested Action
Hold this exact GIFT-64 cross-SPN aligned route after the `8192/class` medium
diagnostic unless a new difference profile, stronger architecture/input
hypothesis, or deterministic audit shows a cleaner source of signal. Do not
launch remote training or include it as a qualified ensemble expert. Report it
as `gift64_cross_spn_aligned_medium_weak_unstable_hold`.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-gift64-cross-spn-cell-representation-local-plan.md, configs/experiment/innovation1/innovation1_spn_gift64_cross_spn_cell_repr_local.csv
- Tags: innovation1, spn, gift64, cross-spn, representation, inverse-permutation, diverse-experts
- See Also: LRN-20260706-011, LRN-20260705-002, LRN-20260705-003
- Pattern-Key: innovation1.spn_gift64.cross_spn_aligned_weak_positive_repeat_before_remote
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-011] best_practice

**Logged**: 2026-07-06T06:20:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
After independent SPN route recheck, stop current SGP, deterministic InvP aggregation, r8 pair-set scale, and r9 difference-screen branches.

### Details
The user explicitly asked for independent route judgment instead of simply
following the latest suggested multi-network direction. A local evidence
recheck plus external literature refresh showed:

```text
strongest supported anchor = present_nibble_invp_only_spn_only
evidence = two_seed_1000000_class_positive_with_attribution_control
seed0 AUC = 0.797470988906
seed1 AUC = 0.797347588554
```

Recently tested follow-up branches do not justify the next main experiment
slot:

```text
SGP raw/grouped audits = hold
InvP global stats = hold
InvP group distribution stats = hold
r8 pair-set 1M seed0 = stop_or_rethink_r8_pairset_scale
r9 curriculum = stop_or_rethink_r9_curriculum_route
r9 difference screen = all_candidates_near_random_stop_difference_screen
near-neighbor neural ensemble = weak positive but below gate
```

Correct current route framing:

- Do not continue SGP projection from the tested sources.
- Do not add more deterministic InvP(delta) aggregate statistics.
- Do not expand the near-neighbor ensemble until a non-neighbor weak-positive
  expert has compatible frozen scores and low error overlap.
- Do not repeat the current r9 difference screen or r8 pair-set scale without a
  new representation/data hypothesis.
- Prefer a small controlled learned pair/group-interaction diagnostic or a
  cross-SPN representation sanity check before any remote launch.

### Suggested Action
Before launching another meaningful Innovation 1 SPN experiment, update the
relevant `docs/experiments/` plan with a single narrow question and a same-budget
baseline. The next experiment should be local and explanatory unless it passes a
predeclared smoke gate. Keep the r7 InvP-only route as the evidence anchor and
treat wider ensembles only as a later diversity-gated validator.

### Metadata
- Source: conversation, experiment_audit, literature_recheck
- Related Files: docs/research/innovation1-spn-independent-route-recheck-20260706.md, docs/research/innovation1-spn-adaptation-literature-refresh-20260705.md, scripts/summarize-spn-evidence
- Tags: innovation1, spn, present, route-selection, sgp, invp, difference-screen, ensemble
- See Also: LRN-20260705-003, LRN-20260706-010, LRN-20260705-002
- Pattern-Key: innovation1.spn_present.stop_closed_branches_after_independent_recheck
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-012] best_practice

**Logged**: 2026-07-06T06:45:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Do not launch the prepared r8 pair-mixer 262k remote package from current local evidence.

### Details
After SGP and deterministic InvP aggregate-stat routes held, a controlled local
diagnostic tested learned pair/group interaction on the same r8 PRESENT
protocol:

```text
plan = configs/experiment/innovation1/innovation1_spn_present_learned_pair_group_interaction_r8_local.csv
scale = 512/class
negative_mode = encrypted_random_plaintexts
sample_structure = zhang_wang_case2_official_mcnd
feature_encoding = ciphertext_pair_bits
```

Results:

| Model | AUC |
|---|---:|
| `present_nibble_invp_only_spn_only` | `0.51275634765625` |
| `present_nibble_invp_pair_consistency_spn_only` | `0.5184173583984375` |
| `present_nibble_invp_pair_mixer_consistency_spn_only` | `0.5105438232421875` |

Deltas:

```text
pair-consistency - InvP-only AUC = +0.0056610107421875
pair-mixer - InvP-only AUC = -0.0022125244140625
pair-mixer - pair-consistency AUC = -0.00787353515625
```

Correct interpretation:

- The pair-consistency row is weak-positive locally but below the `+0.01` keep
  gate.
- The learned cross-pair mixer did not beat either InvP-only or pair-consistency.
- Existing prepared 262144/class pair-mixer remote assets are not launch
  evidence; prepared assets must be superseded by current gates.

### Suggested Action
Do not launch or expand the current r8 pair-mixer package unless a new local
representation/data hypothesis changes the evidence. Keep pair-consistency only
as weak diagnostic context. Move the next experiment slot toward a different
SPN representation/data route or cross-SPN cell-representation sanity check.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-learned-pair-group-interaction-local-plan.md, docs/experiments/innovation1-present-pair-mixer-consistency-plan.md, configs/experiment/innovation1/innovation1_spn_present_learned_pair_group_interaction_r8_local.csv
- Tags: innovation1, spn, present, pair-mixer, pair-consistency, learned-interaction, route-selection
- See Also: LRN-20260706-011, LRN-20260706-010, LRN-20260705-002
- Pattern-Key: innovation1.spn_present.pair_mixer_local_hold_no_remote_launch
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260705-002] correction

**Logged**: 2026-07-05T23:14:37+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Do not treat a small set of similar SPN/PRESENT neural models as sufficient evidence for the "multiple neural networks combined" route.

### Details
The user corrected the interpretation of the neural ensemble direction: combining several nearby SPN-aware variants, such as Zhang/Wang-style raw MCND, InvP-only, and DDT graph, is only a near-neighbor score aggregation diagnostic. It is not the full "multiple diverse neural networks" idea.

Correct framing:

- Near-neighbor ensemble rows may show whether closely related SPN views have any error complementarity.
- A real diverse expert pool should intentionally combine different representation and architecture families: raw bit/MCND, InvP cell tokens, DDT/P-layer graph priors, pair-evidence pooling, inverse-round/integral matrix features, and projection/truncated weak features when they are weak-positive and low-correlation.
- The selection rule should depend on both per-expert quality and pairwise diversity/error overlap, not just model count.
- Do not mechanically add more models from the same family when the current pool is weak positive but below gate.

Current evidence motivating this rule:

```text
run_id = i1_present_neural_ensemble_r7_65k_seed0_gpu0_20260705
best_single = present_nibble_ddt_graph, AUC 0.789112608414
best_ensemble = probability_mean, AUC 0.790061685257
delta = +0.000949076843, below the +0.001 gate
decision = weak_neural_ensemble_positive_below_gate
```

This shows mild complementarity but does not prove that the broader diverse-neural-network route is exhausted.

### Suggested Action
Create and use a `diverse expert pool` plan for future SPN/PRESENT ensemble work. Require candidate expert-family metadata and diversity gates, for example max error Jaccard and at least one low-overlap non-neighbor expert, before deciding to scale an ensemble route.

### Metadata
- Source: user_feedback
- Related Files: docs/experiments/innovation1-present-neural-ensemble-aggregation-plan.md, docs/experiments/innovation1-present-diverse-expert-pool-plan.md
- Tags: innovation1, spn, present, neural-ensemble, diverse-experts, evidence-gates
- See Also: LRN-20260628-004, LRN-20260630-001
- Pattern-Key: innovation1.spn_present.diverse_expert_pool_not_near_neighbor_ensemble
- Recurrence-Count: 1
- First-Seen: 2026-07-05
- Last-Seen: 2026-07-05

---

## [LRN-20260705-003] correction

**Logged**: 2026-07-05T23:55:00+08:00
**Priority**: high
**Status**: promoted
**Area**: research

### Summary
Do not automatically promote the user's proposed SPN route; independently rank it against current literature and local evidence.

### Details
The user corrected the collaboration style for Innovation 1: the agent should not merely follow the latest suggested direction, such as diverse neural-network aggregation. It should actively inspect local evidence, re-check relevant literature, and decide whether that route deserves the next experiment slot.

Correct framing:

- Treat user-proposed routes as hypotheses, not commands to promote them into the main branch.
- Compare candidate routes against the strongest current evidence and recent literature before spending remote GPU time.
- Keep diverse neural aggregation as a valid route, but only as a secondary validator until compatible weak-positive, low-correlation non-neighbor experts exist.
- Prefer SPN-aware data/feature representation when literature and local evidence point there more strongly than model-family aggregation.

Current route correction after the 2026-07-05 literature refresh:

```text
SPN feature/input search > structure-aware architecture > diverse ensemble
```

The same turn also retrieved a completed r8 integral/inverse screen:

```text
run_id = i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705
raw integral anchor AUC = 0.999995831400
InvP matrix AUC = 0.513465017546
InvP+Sinv matrix AUC = 0.505787684582
decision = stop_integral_inverse_feature_screen_for_now
interpretation = integral/multiset data structure signal, not inverse-round architecture gain
```

### Suggested Action
Before launching the next Innovation 1 SPN/PRESENT experiment, write or update the route document with a ranked decision that includes: local result status, same-budget baseline, literature support, expected claim scope, and why the chosen route beats the alternatives. Do not spend the next remote slot on a wider ensemble unless a non-neighbor expert family already has compatible weak-positive score artifacts.

### Metadata
- Source: user_feedback
- Related Files: docs/research/innovation1-spn-adaptation-literature-refresh-20260705.md, sources/research_spn_adaptation_20260705.md, docs/experiments/innovation1-present-diverse-expert-pool-plan.md
- Tags: innovation1, spn, route-selection, literature-refresh, independent-judgment, neural-ensemble
- See Also: LRN-20260705-002, LRN-20260630-001
- Pattern-Key: innovation1.spn_present.independent_literature_ranked_route_selection
- Recurrence-Count: 3
- First-Seen: 2026-07-05
- Last-Seen: 2026-07-06
- Promoted: AGENTS.md

### Resolution
- **Resolved**: 2026-07-06T00:00:00+08:00
- **Commit/PR**: pending
- **Notes**: Promoted a concise rule to `AGENTS.md`: user-proposed routes are hypotheses that must be ranked against literature, local evidence, same-budget baselines, and controls before consuming meaningful experiment slots. Reaffirmed on 2026-07-06 with an independent SPN route re-rank and a baseline-gated neural follow-up smoke plan.

---

## [LRN-20260706-001] best_practice

**Logged**: 2026-07-06T00:00:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Interpret the r8 matched-negative raw-pair integral signal as a deterministic SPN/multiset feature candidate, not a neural architecture win.

### Details
After the explicit pair-xor parity leak was removed with `plaintext_integral_nibble_matched_negative`, two local tiny smoke probes still showed raw-pair signal:

```text
seed0 raw-pair AUC = 0.805480957031
seed1 raw-pair AUC = 0.877990722656
```

Simple pair-alignment statistics did not explain the residual, but the follow-up deterministic feature-bank audit did. At `2048/class`, the same scalar statistic dominated both checked audit seeds:

```text
seed0 best statistic = pair_xor_column_sum_variance
seed0 best threshold accuracy = 0.979248046875
seed1 best statistic = pair_xor_column_sum_variance
seed1 best threshold accuracy = 0.982421875
```

Correct interpretation:

- The raw matched-negative neural smoke likely learned a simple pair-xor column-distribution variance statistic.
- This is useful SPN-aware input/feature evidence, but not evidence of an architecture breakthrough.
- The route should not consume a remote neural training slot until additional controls pass.

Pair-order scramble control then tested whether the statistic came only from same-index fixed-difference pairing:

```text
matched anchor pair_xor_column_sum_variance accuracy = 0.979248046875
scrambled-positive pair_xor_column_sum_variance accuracy = 0.8818359375
```

This weakens but does not remove the statistic. Same-index fixed-difference pairing appears to be a major amplifier, while the active-nibble integral multiset construction still leaves a strong residual column-distribution signal.

Clean active/difference variation with `plaintext_integral_nibble_difference_matched_negative` then removed the left/right column-sum mismatch for off-active differences. At `2048/class`, audit seed `17`:

```text
active0 + Zhang/Wang diff 0x9 accuracy = 0.81494140625
active1/7/15 + Zhang/Wang diff 0x9 accuracy ~= 0.52
active0 + AutoND / entropy / Wang-Jain differences accuracy ~= 0.51-0.53
```

This narrows the route further: the deterministic statistic is not a generic integral-multiset signal. It is strongest when the active integral nibble is aligned with the fixed input-difference support.

Aligned active-difference controls then showed that the effect is not unique to Zhang/Wang `0x9`; it also appears for other single-nibble differences when active nibble is aligned:

```text
Zhang/Wang diff 0x9 active0 accuracy = 0.81494140625
AutoND diff 0x0d000000 active6 accuracy = 0.804443359375
Entropy diff 0x00d00000 active5 accuracy = 0.804443359375
Wang/Jain two-nibble diff active2/14 accuracy ~= 0.518
```

Current narrower interpretation: the route is a single-nibble aligned active-difference deterministic feature candidate. It does not currently support a two-nibble difference under the one-active-nibble construction.

A second aligned active-difference audit seed preserved the split:

```text
Zhang/Wang diff 0x9 active0 accuracy = 0.805908203125
AutoND diff 0x0d000000 active6 accuracy = 0.79296875
Entropy diff 0x00d00000 active5 accuracy = 0.8056640625
Wang/Jain two-nibble diff active2/14 accuracy ~= 0.518
```

This strengthens the local deterministic route decision: first make `pair_xor_column_sum_variance` an explicit baseline and design a multi-active-cell control for multi-nibble differences; do not spend the next meaningful slot on a wider neural ensemble.

The fixed deterministic baseline evaluator has now been implemented:

```text
script = scripts/evaluate-integral-deterministic-baseline
api = integral_deterministic_baseline_from_task
default statistic = pair_xor_column_sum_variance
```

Future neural follow-ups should compare against this fixed statistic and should not treat a best-of-feature-bank result as neural architecture evidence.

The fixed baseline now reports AUC. At `2048/class`, audit seed `23`, the
single-nibble aligned controls have strong deterministic AUC:

```text
Zhang/Wang aligned active0 AUC = 0.8878759145736694
AutoND aligned active6 AUC = 0.8747416734695435
Entropy aligned active5 AUC = 0.8852955102920532
```

Future neural follow-ups on this route must beat or explain this fixed baseline
instead of merely showing a raw-pair neural signal.

The multi-active-cell control for Wang/Jain two-nibble differences was then
implemented and tested locally:

```text
sample_structure = plaintext_integral_multi_nibble_difference_matched_negative
pairs_per_sample = 256
seed29 pair_xor_column_sum_variance accuracy = 0.58203125
seed31 pair_xor_column_sum_variance accuracy = 0.59765625
seed29 feature-bank best accuracy = 0.58203125
```

This weak result means the two-nibble Wang/Jain integral route should not take
the next meaningful remote slot; keep the single-nibble aligned active-
difference route primary unless a different multi-cell statistic emerges.

### Suggested Action
Before scaling the r8 matched-negative integral route, use `pair_xor_column_sum_variance` AUC as an explicit deterministic baseline. Keep single-nibble aligned active-difference as the primary deterministic feature route; do not spend the next meaningful slot on the tested Wang/Jain two-nibble integral route unless a new multi-cell statistic or representation changes the evidence.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-integral-parity-control-plan.md, docs/research/innovation1-spn-adaptation-literature-refresh-20260705.md, src/blockcipher_nd/cli/audit_integral_parity_signal.py
- Tags: innovation1, spn, present, integral, deterministic-feature, matched-negative
- See Also: LRN-20260705-003, LRN-20260705-002
- Pattern-Key: innovation1.spn_present.matched_negative_raw_pair_feature_bank_explains_signal
- Recurrence-Count: 8
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-002] best_practice

**Logged**: 2026-07-06T01:30:57+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Do not re-add PRESENT single-step structural inverse-S as a new GPD feature; it already exists as the Sinv encoding.

### Details
During the independent SPN route re-check, the next plausible literature-backed
direction was Generic Partial Decryption / partial inverse feature engineering.
Code inspection showed that the repository already has the key single-step
zero-key inverse route:

```text
present_pair_xor_paligned_sinv_cell_matrix_bits
```

Its helper computes:

```text
S^{-1}(P^{-1}(C)) xor S^{-1}(P^{-1}(C'))
```

Therefore adding a new feature under a GPD name with the same semantics would
duplicate existing evidence and risk falsely presenting an old route as a new
SPN adaptation. The better local screen is to use existing multi-round DDT /
partial-inverse candidate-path encodings, such as:

```text
present_delta_paligned_sinv_sboxddt_beam4deep3_cell_matrix_bits
present_delta_paligned_sinv_sboxddt_beamstats4deep3_cell_matrix_bits
```

A first seed0 local smoke at `128/class` found only a tiny weak-positive beam
candidate:

```text
InvP control AUC = 0.496337890625
Sinv control AUC = 0.44287109375
DDT beam AUC = 0.5145263671875
DDT beamstats AUC = 0.462158203125
```

This is not remote-launch evidence. It only suggests that expanded DDT beam
paths are a more plausible next local repeat than compressed beamstats.

The seed1 repeat used the same protocol and changed only `seed = 1`:

```text
InvP control AUC = 0.54296875
Sinv control AUC = 0.5361328125
DDT beam AUC = 0.527587890625
DDT beamstats AUC = 0.606689453125
```

This keeps the GPD-style branch alive as a local representation candidate, but
it also shows the `128/class` validation setting is high variance: beamstats
was the best seed1 row but below random on seed0, while the expanded DDT beam
row was weak-positive in both seeds but did not consistently beat controls.
The route should be held for a larger local diagnostic, not remote-launched.

A `512/class` local diagnostic then reduced the variance and changed the route
reading:

```text
InvP control AUC = 0.540496826171875
Sinv control AUC = 0.5286407470703125
DDT beam AUC = 0.562957763671875
DDT beamstats AUC = 0.5418472290039062
```

The expanded DDT beam now beats all controls in this local diagnostic, while
the compressed beamstats seed1 spike did not reproduce. This is still local
diagnostic evidence only, but the next GPD-style step should be a `512/class`
seed1 repeat of the same matrix rather than dropping the branch or jumping
straight to remote scale.

The `512/class` seed1 repeat then corrected that narrow reading:

```text
InvP control AUC = 0.5263595581054688
Sinv control AUC = 0.56329345703125
DDT beam AUC = 0.51806640625
DDT beamstats AUC = 0.5724639892578125
```

Across the two `512/class` diagnostics, expanded DDT beam is not stable: it
wins seed0 but loses to both Sinv and beamstats on seed1. The compressed
beamstats row now has the best two-seed mean AUC (`0.5571556091308594`), but
its 128/class behavior was highly volatile and its 512 seed0 margin over InvP
was only about `+0.00135` AUC. Therefore the branch remains local diagnostic
only. Beamstats may be kept as a lightweight local candidate or future
non-neighbor expert source, but not as remote scale-up evidence yet.

### Suggested Action
When continuing the GPD-style branch, compare against the existing Sinv control
and prefer multi-round DDT/partial-inverse path statistics only when they beat
controls across repeated local diagnostics. After the `512/class` seed1 repeat,
do not launch this GPD-style branch remotely. If continuing it, run a lean local
confirmation or attribution check for the compressed beamstats row; demote the
expanded DDT beam unless a new attribution explains the seed1 failure.

### Metadata
- Source: conversation
- Related Files: src/blockcipher_nd/features/encoders/present_matrix.py, src/blockcipher_nd/features/encoders/present_sbox_ddt.py, configs/experiment/innovation1/innovation1_spn_present_r8_gpd_style_beamstats_smoke.csv, configs/experiment/innovation1/innovation1_spn_present_r8_gpd_style_beamstats_smoke_seed1.csv, configs/experiment/innovation1/innovation1_spn_present_r8_gpd_style_beamstats_512_seed0.csv, configs/experiment/innovation1/innovation1_spn_present_r8_gpd_style_beamstats_512_seed1.csv, docs/experiments/innovation1-present-r8-gpd-style-beamstats-plan.md
- Tags: innovation1, spn, present, gpd, partial-inverse, sinv, ddt-beam
- See Also: LRN-20260705-003, LRN-20260706-001
- Pattern-Key: innovation1.spn_present.gpd_do_not_duplicate_existing_sinv
- Recurrence-Count: 4
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06T02:35:00+08:00

---

## [LRN-20260706-003] best_practice

**Logged**: 2026-07-06T01:10:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Treat the r8 GPD-style beamstats row as an unstable local weak-expert candidate, not as a remote scale-up route.

### Details
The GPD-style beamstats diagnostic produced a weak two-seed local hint, but
semantic attribution did not identify a stable simple driver:

```text
512/class seed0 beamstats AUC = 0.5418472290039062
512/class seed1 beamstats AUC = 0.5724639892578125
2048/class attribution seed0 best scalar = confidence_std, AUC 0.5216715335845947
2048/class attribution seed1 best scalar = cumulative_mean, AUC 0.523328423500061
```

The best semantic scalar changes across seeds and stays near chance. This means
beamstats may still be useful as a future non-neighbor score artifact, but it is
not interpretable or stable enough to justify a `65536/class` remote launch
from the current branch.

Correct framing:

- Keep beamstats as a possible future diverse-expert family only if compatible
  weak-positive score artifacts and low-overlap/error-correlation evidence are
  produced.
- Do not use the current beamstats result to justify mechanically wider
  near-neighbor ensemble work.
- Prefer controlled SPN feature/input attribution before remote scaling.

### Suggested Action
If continuing this branch, run a lean local composite probe or diversity-score
check first. Otherwise return to the stronger controlled SPN feature/input
route and require any neural follow-up to beat or explain its deterministic
baseline.

### Metadata
- Source: conversation
- Related Files: docs/experiments/innovation1-present-r8-gpd-style-beamstats-plan.md, src/blockcipher_nd/tasks/innovation1/spn_feature_audit.py
- Tags: innovation1, spn, present, gpd, beamstats, attribution, diverse-experts
- See Also: LRN-20260705-002, LRN-20260705-003, LRN-20260706-001, LRN-20260706-002
- Pattern-Key: innovation1.spn_present.beamstats_local_candidate_not_scale_gate
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260630-001] correction

**Logged**: 2026-06-30T00:00:00+08:00
**Priority**: high
**Status**: promoted
**Area**: research

### Summary
Long-term Innovation 1 goals should be framed at the SPN method level, not as an overly detailed experiment SOP.

### Details
The user corrected the goal framing: the real objective is to develop SPN-structure-adaptive neural networks or SPN-adaptive data/feature representations, mainly for PRESENT/SPN, not to encode one specific run protocol, seed sequence, metric gate, remote launcher, or postprocess checklist as the long-term goal.

Correct framing:

- Long-term goal: iterate toward neural distinguishers that genuinely exploit SPN structure.
- Innovation can come from SPN-aware network architecture or SPN-aware data/feature construction.
- Concrete experiment details such as run id, seed, scale, remote config, checkpoint metric, and branch gate belong in `docs/experiments/`.
- Broader research route, method hypothesis, and literature synthesis belong in `docs/research/`.
- Positive results should move the project from exploration into confirmation, attribution, ablation, and formal evidence, rather than ending the research prematurely.

The long-term goal should answer "what kind of method are we trying to create?" The experiment plan should answer "which exact run tests the next hypothesis?"

### Suggested Action
Promote a concise rule to `AGENTS.md`: frame long-term Innovation 1 goals around SPN-structure-adaptive neural distinguishers and keep concrete run protocol details in experiment plans. Use `docs/自动化目标规则.md` as the high-level goal document, not as a replacement for per-run plans.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, docs/自动化目标规则.md, docs/experiments/, docs/research/
- Tags: innovation1, spn, goal-framing, research-workflow, experiment-plans
- Pattern-Key: innovation1.goal_framing.method_level_not_run_sop
- Recurrence-Count: 1
- First-Seen: 2026-06-30
- Last-Seen: 2026-06-30
- Promoted: AGENTS.md

---

## [LRN-20260621-003] best_practice

**Logged**: 2026-06-21T20:55:00+08:00
**Priority**: high
**Status**: promoted
**Area**: docs

### Summary
Promote only durable rules from `memory/` into `AGENTS.md`; keep experiment history and run-specific details in `memory/`.

### Details
The user approved the memory cleanup approach: do not merge all `memory/` content into `AGENTS.md`. Instead, extract compact, stable rules that affect future agent behavior. During the 2026-06-21 cleanup, repeated memory/task-plan rules were promoted for remote Windows GPU hygiene, monitor/retrieval workflow, evidence claim gates, and verification/workspace hygiene.

Specific promoted rule groups:

- Remote artifacts and generated project files must stay under `G:\lxy`.
- Windows remote schedule commands must use `cmd.exe /c`, not `cmd.exe /k`.
- After remote launch/handoff, main thread should not SSH-poll; use tmux/watchers/monitors and controlled gates.
- Result reports must distinguish planned, running, completed remotely, fallback-retrieved, verified-branch retrieved, and plan-aligned.
- Strict SPN/PRESENT claims require claim gates, encrypted-random-plaintext negatives, and explicit qualification for multi-query/application-level evidence.
- Use `uv run pytest ...`; keep project-root `tmp_*` clean.

### Suggested Action
When future memory files grow, periodically scan for repeated durable rules and promote only those concise rules to `AGENTS.md`. Leave detailed run ids, timestamps, and transient experiment states in `memory/` or `progress.md`.

### Metadata
- Source: user_feedback
- Related Files: memory/, task_plan.md, progress.md, AGENTS.md
- Tags: memory, agents, remote-workflow, evidence-gates, workspace-hygiene
- See Also: LRN-20260621-002
- Pattern-Key: workflow.memory_to_agents.promote_only_durable_rules
- Recurrence-Count: 1
- First-Seen: 2026-06-21
- Last-Seen: 2026-06-21
- Promoted: AGENTS.md

---

## [LRN-20260621-002] best_practice

**Logged**: 2026-06-21T20:43:13+08:00
**Priority**: high
**Status**: promoted
**Area**: docs

### Summary
When reading old conversation, handoff, progress, or memory documents, persist important corrections with the self-improvement workflow.

### Details
The user clarified that previous memory-document reading should use the `self-improvement` skill to store durable memory. Important findings from old dialogue, `memory/`, handoff summaries, `progress.md`, or result audits must not remain only in transient model context. If a finding changes future experimental interpretation, remote workflow, reporting language, or agent behavior, log it to `.learnings/LEARNINGS.md` using the skill format and promote concise operational rules to `AGENTS.md` when broadly applicable.

This is especially important for long-running Innovation 1 work because context-window loss and thread restarts have already caused confusion about what had actually completed, what scale counts as formal, and whether remote results were retrieved.

### Suggested Action
Before and after reading historical memory files for a major task, check whether any conclusion should be persisted. Use `.learnings/LEARNINGS.md` for detailed context and `AGENTS.md` for short rules that future agents should obey immediately.

### Metadata
- Source: user_feedback
- Related Files: memory/, progress.md, task_plan.md, .learnings/LEARNINGS.md, AGENTS.md
- Tags: memory, handoff, self-improvement, context-window, project-rules
- See Also: LRN-20260621-001
- Pattern-Key: workflow.memory_reading.persist_with_self_improvement
- Recurrence-Count: 1
- First-Seen: 2026-06-21
- Last-Seen: 2026-06-21
- Promoted: AGENTS.md

---

## [LRN-20260622-001] correction

**Logged**: 2026-06-22T11:40:00+08:00
**Priority**: critical
**Status**: promoted
**Area**: infra

### Summary
After every completed repository modification, make a scoped git commit and push when a remote exists so the workspace does not accumulate dirty state.

### Details
The user corrected a workflow failure: remote experiments are intended to pull code from GitHub, but the workspace had accumulated many uncommitted changes. To avoid committing unrelated dirty files, a remote run was started with `scp` overlays into `G:\lxy`, which made the run less reproducible than a clean GitHub commit-based launch.

Correct workflow:

- Complete any repository edit, including code, config, scripts, tests, README/docs, `.learnings/`, `AGENTS.md`, generated project files, or memory-rule updates.
- Run appropriate verification.
- Commit only the scoped files for the completed task.
- Push the branch to the remote repository when a remote is configured; if no remote exists, report that push is not possible.
- Keep the workspace clean for agent-authored changes before starting new work or launching remote experiments.
- Remote experiments should default to a GitHub-pushed commit. Dirty/scp overlay launches are emergency-only and must be explicitly labeled as such in status reports and handoff notes.

This rule does not authorize reverting or committing unrelated user changes. If unrelated dirty files already exist, isolate the task's files in a scoped commit and report the remaining unrelated dirty state separately.

### Suggested Action
Promote this to `AGENTS.md` under workspace hygiene and remote workflow. After any file modification, run the relevant verification, make a scoped commit for the files just changed, and push if a remote is configured. Before future remote launches, run `git status --short`, ensure required files are committed and pushed, and avoid relying on scp overlays for normal experiment reproducibility.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, .learnings/LEARNINGS.md, experiments/innovation1/configs/remote/, scripts/generated/remote/
- Tags: git, commit, push, workspace-hygiene, remote-reproducibility
- See Also: LRN-20260621-003
- Pattern-Key: workflow.git_commit_push_after_code_changes
- Recurrence-Count: 2
- First-Seen: 2026-06-22
- Last-Seen: 2026-06-24
- Promoted: AGENTS.md

### Recurrence Update
- **Updated**: 2026-06-24T15:08:01+08:00
- **Source**: user_feedback
- **Notes**: User explicitly reminded that every modification should be committed, not only code/config/script edits. This includes README/docs, `.learnings/`, `AGENTS.md`, and other memory or documentation updates. Push remains required when a remote is configured; if no remote exists, commit locally and report that push cannot be performed.

---

## [LRN-20260622-002] best_practice

**Logged**: 2026-06-22T13:05:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Treat SPN/PRESENT active-nibble accuracy as an auxiliary trail-activity signal, not as real-vs-random accuracy; promote it into an explicit active-pattern distinguisher route.

### Details
The user identified that the high `val_active_nibble_bit_accuracy` from the multitask active-nibble run should not be left as a side metric. It measures per-position correctness for 16 active/inactive nibble labels per sample, where "active" means a 4-bit differential/trail cell is non-zero. It is not ciphertext-value nonzero-ness, not whole-sample trail accuracy, and not real-vs-random classification accuracy.

Correct interpretation:

- `active nibble` means a nibble of a differential or candidate trail state is non-zero.
- The metric is averaged over all sample-position binary labels, e.g. validation rows times 16 positions.
- It can be inflated by inactive-class imbalance; always compare against all-inactive baseline and report active precision, recall, F1, and per-position rates.
- The research opportunity is to convert this auxiliary structure recognition into explicit real-vs-random evidence: active count, position frequency, candidate-trail disagreement, confidence, margin, pair-set consistency, and trail-family match scores.
- Strict evidence still requires `encrypted_random_plaintexts` negatives and separate reporting of single-sample raw accuracy, AUC, and multi-query aggregation.

### Suggested Action
Implement the active-pattern route as a staged Innovation 1 SPN plan:

1. Add deterministic active-pattern/statistics extraction from existing PRESENT beamstats/candidate-trail feature encodings.
2. Add diagnostics for active-label imbalance and real-vs-random distribution separation.
3. Train active-only and active-plus-candidate-statistics baselines before any large neural model.
4. If small/medium evidence is positive, run the route at 262144/class and then >=1000000/class multi-seed before making formal claims.

### Metadata
- Source: user_feedback
- Related Files: outputs/remote_results_incomplete/innovation1-spn-present-multitask-active-nibble-fast-gate-r7-gpu1-20260618/, src/blockcipher_ai_eval/features/pair_features.py, docs/superpowers/plans/
- Tags: innovation1, spn, present, active-nibble, trail-activity, distinguisher, evidence-gates
- See Also: LRN-20260621-001, LRN-20260621-002
- Pattern-Key: innovation1.spn_present.active_pattern_distinguisher
- Recurrence-Count: 2
- First-Seen: 2026-06-22
- Last-Seen: 2026-06-23

---

## [LRN-20260623-001] research

**Logged**: 2026-06-23T10:23:47+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
SPN/PRESENT active-only 24-dim active-pattern screen failed as a real-vs-random distinguisher despite the earlier high auxiliary active-nibble signal.

### Details
The active-pattern route completed and was retrieved from a verified result branch:

- Run ID: `innovation1-spn-present-active-pattern-r7-screen-gpu1-20260622`
- Result branch: `results/innovation1-spn-present-active-pattern-r7-screen-gpu1-20260622`
- Result commit: `6c3243137b821d5fc39d266d8aa5f39622ad4fdd`
- Local archive: `outputs/remote_results/innovation1-spn-present-active-pattern-r7-screen-gpu1-20260622/`
- Gate: `result_lines=2`, `expected_rows=2`, `runner_exit_code=0`, `archive_integrity=pass`
- Config: `rounds=7`, `samples_per_class=65536`, `pairs_per_sample=16`, `negative_mode=encrypted_random_plaintexts`, `sample_structure=zhang_wang_case2_mcnd`
- Mean result: `accuracy=0.500000`, `AUC=0.491578`, `feature_dim=24`
- Seed 0: `val_accuracy=0.5`, `val_auc=0.4894024282693863`
- Seed 1: `val_accuracy=0.5`, `val_auc=0.4937528520822525`

Failure interpretation:

- The auxiliary target and the distinguisher target are different. Active-nibble prediction asks whether each 4-bit trail/difference cell is non-zero; real-vs-random classification asks whether a pair set came from the target differential encryption process or encrypted-random-plaintext negatives.
- The 24-dim active summary is too coarse. It keeps only active position frequencies and aggregate density statistics, but discards value-level S-box/DDT evidence, candidate trail scores, top-k margins, confidence, per-pair ordering, and cross-pair consistency.
- Under this representation, real and random samples can share almost the same active-density and position-frequency distribution, i.e. `P(active-summary | real) ~= P(active-summary | random)`. The retrieved AUC below `0.5` is direct evidence that the current active-only summary gives little or no separation.
- Earlier high `active_nibble_bit_accuracy` may be partly inflated by inactive-class imbalance. Per-position bit accuracy can be high when many positions are inactive, so active precision, recall, F1, balanced accuracy, and all-inactive baselines are required before treating it as strong structural evidence.
- Therefore, active-pattern should be used as auxiliary supervision or one component of richer candidate-trail evidence, not as a standalone final feature family.

This was only a screen/medium diagnostic at `65536/class`, not formal `>=1000000/class` evidence. However, because the active-only baseline is already at chance with strict `encrypted_random_plaintexts` negatives, do not scale this exact 24-dim linear route as the main next experiment.

### Suggested Action
Retire the active-only 24-dim linear baseline as a main scaling route. Keep active-nibble information, but combine it with features that measure whether candidate trails actually support the observed sample:

- Top-1/top-2 trail score and top-k margin
- Candidate score entropy and confidence
- Candidate disagreement across pairs
- Active-pattern-to-top-trail match
- Pair-set trail-family consistency
- Transition-spectrum features and multi-query score aggregation

For future reports, state plainly: high active-nibble auxiliary accuracy proves the model can learn trail-activity propagation patterns; it does not prove real-vs-random distinguishability.

### Metadata
- Source: conversation
- Related Files: src/blockcipher_ai_eval/features/spn_active_pattern.py, experiments/innovation1/run_spn_active_pattern_baseline.py, outputs/remote_results/innovation1-spn-present-active-pattern-r7-screen-gpu1-20260622/
- Tags: innovation1, spn, present, active-pattern, active-nibble, real-vs-random, failure-analysis, evidence-gates
- See Also: LRN-20260622-002
- Pattern-Key: innovation1.spn_present.active_pattern_distinguisher
- Recurrence-Count: 1
- First-Seen: 2026-06-23
- Last-Seen: 2026-06-23

---

## [LRN-20260706-006] best_practice

**Logged**: 2026-07-06T02:40:04+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Before widening SPN/PRESENT ensembles, mine stable SPN-sensitive axes and score-distribution features.

### Details
The user corrected the collaboration style again: the agent should not simply
follow the user's current hypothesis, including the broader diverse-neural-
network aggregation idea. Re-checking local evidence and the literature notes
shows that the next main bottleneck is not model count. It is the absence of a
compatible, weak-positive, low-correlation non-neighbor expert family.

Current route reading:

```text
near-neighbor r7 ensemble = weak-positive below gate
projection v2 = unstable local priors
GPD/beamstats = weak local candidate only
candidate-trail = stopped at medium scale, weak local axes only
aligned integral route = explained by deterministic baseline
```

The better next route is a score-guided / sensitivity-guided SPN projection
audit: use candidate-evidence, beamstats, InvP(delta), and trail-family score
axes to select stable masks only if they survive seed/key stability and false-
family controls. Only then should the route train a small projection probe or
enter the diverse expert pool.

### Suggested Action
Create an SGP local audit before any new remote launch or wider ensemble. Gate
candidate masks by top-k composite AUC, seed/key stability, and shuffled/false-
family controls. Keep diverse expert aggregation as the secondary validator
after a genuine non-neighbor weak-positive score artifact exists.

### Metadata
- Source: user_feedback
- Related Files: docs/research/innovation1-spn-independent-route-recheck-20260706.md, docs/experiments/innovation1-present-diverse-expert-pool-plan.md, docs/experiments/innovation1-present-truncated-projection-feature-plan.md
- Tags: innovation1, spn, present, route-selection, projection, score-distribution, sensitivity, diverse-experts
- See Also: LRN-20260705-002, LRN-20260705-003, LRN-20260706-002
- Pattern-Key: innovation1.spn_present.stable_axis_before_diverse_ensemble
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-007] best_practice

**Logged**: 2026-07-06T03:40:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Do not advance raw-axis SGP projection after the r8 source sweep; try orbit/grouped InvP stability first.

### Details
The first PRESENT r8 SGP stable-axis source sweep ran at `2048/class`, seeds
`0` and `1`, strict `encrypted_random_plaintexts` negatives, and
`zhang_wang_case2_official_mcnd` sample structure. It tested three source
families:

```text
candidate_cell_structured
candidate_aggregate
invp_delta_bits
```

All three source reports returned `sgp_stable_axis_hold`:

| Source | Min composite AUC | Top-k Jaccard | Control delta |
|---|---:|---:|---:|
| `candidate_cell_structured` | `0.5262441635131836` | `0.14285714285714285` | `-0.0010590553283691406` |
| `candidate_aggregate` | `0.5145435333251953` | `0.16363636363636364` | `0.014543533325195312` |
| `invp_delta_bits` | `0.5609222650527954` | `0.0` | `0.06092226505279541` |

Correct interpretation:

- Candidate-cell evidence is weak and not better than the shuffled-cell control.
- Candidate-aggregate evidence is too weak to justify a projection smoke.
- InvP(delta) has weak-positive composite evidence, but exact flat bit-axis
  identity is unstable across seeds. This should not be forced through by
  weakening the raw-axis Jaccard gate.
- The next more plausible route is orbit/grouped stability over InvP(delta)
  axes, e.g. grouping by pair slot, SPN cell, P-layer orbit, or bit role.
- Multi-source SGP currently regenerates candidate evidence in memory without
  progress output; do not scale this audit path until SGP cache/progress
  support exists.

### Suggested Action
Hold raw-axis `sgp_top32_stable` projection and do not launch remote SGP. Add a
small orbit/grouped stability audit for InvP(delta) before any SGP projection
smoke. If grouped stability also fails, retire SGP as a projection route and
return to stronger representation priors.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-sgp-stable-axis-audit-plan.md, configs/experiment/innovation1/innovation1_spn_present_sgp_stable_axis_audit_r8_local.json, src/blockcipher_nd/tasks/innovation1/spn_feature_audit.py
- Tags: innovation1, spn, present, sgp, stable-axis, projection, invp, route-selection
- See Also: LRN-20260706-006, LRN-20260705-003, LRN-20260705-002
- Pattern-Key: innovation1.spn_present.sgp_raw_axis_hold_orbit_group_next
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-008] best_practice

**Logged**: 2026-07-06T04:45:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Do not promote grouped SGP after the InvP(delta) grouped/orbit follow-up; convert broad weak signal into intentional statistics instead.

### Details
The PRESENT r8 grouped/orbit SGP follow-up tested `invp_delta_bits` at
`2048/class`, seeds `0` and `1`, strict `encrypted_random_plaintexts`
negatives, and `zhang_wang_case2_official_mcnd` sample structure.

Final non-degenerate top4 artifact:

```text
outputs/local_audits/i1_present_r8_sgp_grouped_axis_audit_2048_top4.json
decision = sgp_grouped_axis_hold
best_group_scheme = word_bit_role
```

Summary table:

| Group scheme | Min composite AUC | Top-k Jaccard | Mask fraction |
|---|---:|---:|---:|
| `pair_word_cell` | `0.5344549417495728` | `0.0` | `0.0078125` |
| `word_cell` | `0.6075923442840576` | `0.14285714285714285` | `0.125` |
| `cell` | `0.6401443481445312` | `0.14285714285714285` | `0.25` |
| `word_bit_role` | `0.685741662979126` | `0.14285714285714285` | `0.5` |
| `p_layer_orbit` | `0.5724446773529053` | `0.0` | `0.09375` |

Correct interpretation:

- InvP(delta) contains broad weak separation, especially under coarse cell or
  bit-role aggregation.
- Exact pair-slot/cell and P-layer orbit groups are not stable across seeds.
- Coarse `word_bit_role` looks strongest but is too broad to be a projection
  expert; a degenerate full-width mask initially looked candidate-like when
  all 8 role groups were selected, so grouped SGP now has a
  `max_selected_axis_fraction` guard.
- This is not evidence for a remote SGP projection run and not a valid diverse
  expert for ensemble aggregation yet.

### Suggested Action
Retire SGP as the next immediate projection route. Use the broad weak
InvP(delta) signal as a hint for explicit pair/global statistics or
bit-role/cell distribution features, then compare that representation against
existing pairset/global-stat anchors before any remote launch. Keep diverse
neural aggregation secondary until a genuinely non-neighbor, weak-positive,
low-overlap expert exists.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-sgp-stable-axis-audit-plan.md, configs/experiment/innovation1/innovation1_spn_present_sgp_grouped_axis_audit_r8_local.json, src/blockcipher_nd/tasks/innovation1/spn_feature_audit.py
- Tags: innovation1, spn, present, sgp, grouped-axis, invp, route-selection, degeneracy-gate
- See Also: LRN-20260706-007, LRN-20260706-006, LRN-20260705-003, LRN-20260705-002
- Pattern-Key: innovation1.spn_present.sgp_grouped_hold_statistics_next
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-009] best_practice

**Logged**: 2026-07-06T05:20:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Do not promote existing InvP global activity statistics after the r8 local audit; they average away the grouped SGP signal.

### Details
After raw/grouped SGP held, the next local diagnostic tested explicit
`present_global_pairset_statistics` over `ciphertext_xor_spn_paligned_bits`.
Protocol:

```text
cipher = PRESENT-80
rounds = 8
samples_per_class = 2048
seeds = 0, 1
pairs_per_sample = 16
negative_mode = encrypted_random_plaintexts
sample_structure = zhang_wang_case2_official_mcnd
```

Artifact:

```text
outputs/local_audits/i1_present_r8_invp_global_stats_audit_2048.json
decision = invp_global_stats_hold
```

Result:

| Metric | Value |
|---|---:|
| Stat feature dim | `148` |
| Best stat AUC min | `0.5180071592330933` |
| Composite AUC min | `0.5185081958770752` |
| Composite AUC mean | `0.5251665115356445` |
| Top-k Jaccard min | `0.06666666666666667` |

Correct interpretation:

- The existing global activity statistics are too coarse for the signal found
  by grouped SGP.
- They should not trigger a neural smoke, remote launch, or diverse expert
  inclusion.
- If continuing the statistics route, use a targeted group-distribution feature
  bank over cell/word-cell/bit-role/orbit group activities, including variance,
  span, top-k means, and pair-slot consistency.

### Suggested Action
Keep `present_pairset_global_stats` as an existing model/control, not as a
validated next route from this audit. Next local work should test group-level
distribution statistics that preserve the structure where SGP showed broad weak
signal.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-invp-global-stats-audit-plan.md, configs/experiment/innovation1/innovation1_spn_present_invp_global_stats_audit_r8_local.json, src/blockcipher_nd/tasks/innovation1/spn_feature_audit.py
- Tags: innovation1, spn, present, invp, global-stats, distribution-statistics, route-selection
- See Also: LRN-20260706-008, LRN-20260706-007, LRN-20260705-003
- Pattern-Key: innovation1.spn_present.invp_global_stats_hold_group_distribution_next
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-010] best_practice

**Logged**: 2026-07-06T05:45:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Do not keep adding deterministic InvP(delta) aggregate statistics after group-distribution audit held.

### Details
After generic InvP global statistics held, a more targeted deterministic
group-distribution bank was tested over:

```text
pair_word_cell
word_cell
cell
word_bit_role
p_layer_orbit
```

For each scheme it audited:

```text
activity_mean
activity_std
activity_max
top2_activity_mean
top4_activity_mean
bottom2_activity_mean
bottom4_activity_mean
activity_span
```

Protocol:

```text
cipher = PRESENT-80
rounds = 8
samples_per_class = 2048
seeds = 0, 1
pairs_per_sample = 16
negative_mode = encrypted_random_plaintexts
sample_structure = zhang_wang_case2_official_mcnd
```

Artifact:

```text
outputs/local_audits/i1_present_r8_invp_group_distribution_audit_2048.json
decision = invp_group_distribution_hold
```

Result:

| Metric | Value |
|---|---:|
| Stat feature dim | `40` |
| Best stat AUC min | `0.514545202255249` |
| Composite AUC min | `0.5135400295257568` |
| Composite AUC mean | `0.5136241912841797` |
| Top-k Jaccard min | `0.18518518518518517` |

Correct interpretation:

- Grouped SGP's higher composite AUC does not translate into stable simple
  unsupervised distribution statistics.
- The broad InvP(delta) weak signal is too weak/unstable for another hand-built
  aggregate-stat pass.
- Do not create a deterministic group-distribution representation smoke, remote
  launch, or ensemble expert from this evidence.

### Suggested Action
Stop deterministic InvP(delta) aggregation for now. If continuing this family,
use a learned pair/group-interaction representation that directly consumes group
activities, or shift the next local slot to data/difference search. Do not add
more handwritten aggregate statistics around the same failed evidence without a
new reason.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-invp-group-distribution-audit-plan.md, configs/experiment/innovation1/innovation1_spn_present_invp_group_distribution_audit_r8_local.json, src/blockcipher_nd/tasks/innovation1/spn_feature_audit.py
- Tags: innovation1, spn, present, invp, group-distribution, deterministic-statistics, route-selection
- See Also: LRN-20260706-009, LRN-20260706-008, LRN-20260706-007, LRN-20260705-003
- Pattern-Key: innovation1.spn_present.invp_group_distribution_hold_stop_hand_stats
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260623-002] correction

**Logged**: 2026-06-23T21:56:37+08:00
**Priority**: critical
**Status**: promoted
**Area**: infra

### Summary
Remote training that generates datasets or derived features must use disk-backed cache/progress/reuse before launch; do not run large remote jobs with pure in-memory one-shot generation.

### Details
The user corrected a recurring workflow mistake: prior project work had already established that dataset generation should write reusable artifacts such as `features.npy`, `labels.npy`, metadata, CSV/JSONL summaries, and progress logs. The candidate-evidence route violated this principle by launching `65536/class` remotely through a new prototype runner that built `features: list[np.ndarray]` in memory and only wrote a final result after all feature generation, training, and evaluation finished.

Correct rule:

- Any remote training or medium/large screen that generates datasets, feature matrices, candidate-evidence features, trail statistics, or other derived training inputs must have disk-backed cache before launch.
- The cache must be under `G:\lxy` on the remote and normally inside the run directory or approved run cache root.
- Required artifacts include cache metadata, feature/label arrays or equivalent chunked files, progress JSONL/logging, and reuse/resume behavior when parameters match.
- New runners and new feature routes are not exempt. If they bypass `run_innovation_one_matrix.py`, they must implement an equivalent route-specific cache before remote scale-up.
- Smoke-only local experiments may use in-memory generation, but remote launches at `65536/class` or above must not.
- Do not call a remote experiment ready to launch until this cache/progress gate has been checked explicitly.

The immediate failure mode was the candidate-evidence baseline: positive local fast screens led to a remote `65536/class` run, but its feature generation was pure Python/in-memory and produced no progress or reusable cache, making the remote appear stalled and wasting time.

### Suggested Action
Promote this rule to `AGENTS.md` under Remote Windows GPU Rules / Verification. Add cache/progress support to `experiments/innovation1/run_spn_candidate_evidence_baseline.py` before relaunching scaled candidate-evidence experiments:

- `--feature-cache-root`
- `--feature-cache-chunk-size`
- `--progress-output`
- disk-backed `features.npy` / `labels.npy` / `metadata.json`
- cache identity including rounds, seeds, samples_per_class, pairs_per_sample, negative mode, sample structure, difference profile, key rotation, beam width, depth, source, and feature dimension
- chunk progress events and cache reuse

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, experiments/innovation1/run_spn_candidate_evidence_baseline.py, experiments/run_innovation_one_matrix.py, src/blockcipher_ai_eval/data/cache/disk.py
- Tags: remote-training, dataset-cache, feature-cache, progress-logging, innovation1, spn, present, candidate-evidence
- See Also: LRN-20260621-003
- Pattern-Key: remote_training.must_use_disk_cache_for_generated_data
- Recurrence-Count: 1
- First-Seen: 2026-06-23
- Last-Seen: 2026-06-23
- Promoted: AGENTS.md

---

## [LRN-20260624-001] correction

**Logged**: 2026-06-24T16:03:00+08:00
**Priority**: high
**Status**: pending
**Area**: docs

### Summary
Generated result plots must include human-readable coordinate axes, tick labels, and grid lines; bare lines without coordinate values are not acceptable.

### Details
The user corrected the first SVG output from `scripts/plot-results`: it drew train/validation curves, but the plot did not have enough coordinate values to interpret the metric scale and epoch positions like a normal deep learning training curve. A result figure should support human inspection without reading the raw JSON/CSV.

Correct plotting expectations:

- X axis should show epoch tick values, not only first/last labels.
- Y axis should show metric tick values, including intermediate values such as `0.25`, `0.5`, `0.75` for accuracy/AUC.
- Light grid lines should align with tick labels so the reader can estimate values.
- Axis labels such as `epoch`, `accuracy`, `auc`, and `loss` should be visible.
- Train and validation curves should remain visually distinct and annotated.

### Suggested Action
Keep visualization tests that assert generated SVG contains readable axis labels and intermediate tick values. When adding new plots, inspect the rendered artifact or its SVG text for axes, tick labels, and grid lines before calling the visualization complete.

### Metadata
- Source: user_feedback
- Related Files: src/blockcipher_nd/evaluation/plots.py, scripts/plot-results, tests/test_project_structure.py
- Tags: visualization, svg, training-curves, docs, result-reporting
- Pattern-Key: visualization.training_curves.require_readable_axes
- Recurrence-Count: 1
- First-Seen: 2026-06-24
- Last-Seen: 2026-06-24

---

## [LRN-20260624-002] correction

**Logged**: 2026-06-24T20:30:00+08:00
**Priority**: high
**Status**: promoted
**Area**: infra

### Summary
Remote experiment launches should not be personally supervised from the main thread; use tmux monitors to watch and retrieve results automatically.

### Details
The user clarified the intended remote workflow: after a remote GPU experiment is launched or handed off, the main agent should not keep personally supervising it through repeated SSH checks or manual polling. The correct pattern is to start a local tmux monitor/watcher that waits for completion artifacts, retrieves the result archive or raw fallback outputs, writes local logs/markers, and then lets the main thread continue or report only from local artifacts.

This strengthens the existing "do not SSH-poll from the main thread" rule. Manual remote contact should be reserved for controlled exceptions such as a local monitor health failure, a dry-run gate that explicitly allows SSH, or a user-requested repair. Normal long-running training should be monitored by tmux and retrieval scripts, not by interactive supervision.

### Suggested Action
For future remote runs, always launch or verify a supported local tmux monitor immediately after the remote command is handed off. Report the monitor session/log/marker paths to the user. Inspect results only after the monitor has pulled them back locally, unless the controlled gate says remote inspection is allowed.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, scripts/monitor_remote_results.py, outputs/remote_results/
- Tags: remote-training, tmux, monitoring, retrieval, ssh, workflow
- See Also: LRN-20260621-003, LRN-20260622-001
- Pattern-Key: remote_training.tmux_monitor_retrieves_results
- Recurrence-Count: 1
- First-Seen: 2026-06-24
- Last-Seen: 2026-06-24
- Promoted: AGENTS.md

---

## [LRN-20260625-001] best_practice

**Logged**: 2026-06-25T15:55:00+08:00
**Priority**: high
**Status**: promoted
**Area**: docs

### Summary
Use `blockcipher-auto-research` as the project research workflow and Karpathy-style guidelines as the default implementation discipline for this repository.

### Details
The user explicitly requested that future project work remember and follow the behavior methods from two skills:

- `skills/blockcipher-auto-research/SKILL.md` should drive the project-local research loop: define the question, identify the same-budget baseline, change one hypothesis at a time, run fixed-budget experiments, generate JSONL/CSV/SVG/gate artifacts, apply evidence-scale language, document keep/discard/crash/diagnostic status, and commit/push scoped repository changes after verification.
- `karpathy-guidelines` should guide coding behavior: read relevant code before editing, state uncertainty when evidence is incomplete, prefer simple boring implementations, avoid unnecessary abstractions/dependencies/frameworks, edit precisely, protect user work, validate against observable success criteria, and debug actual failures rather than guessing.

Correct default going forward:

- Treat `blockcipher-auto-research` as the workflow skeleton for Innovation 1 experiments and related reproduction work.
- Treat Karpathy-style guidelines as the execution style for code/config/docs changes inside that workflow.
- Use the combination to avoid untraceable experiment drift: do not change benchmark protocol, validation data, labels, negative-sample definition, metric computation, or plan-alignment logic while also changing model/feature hypotheses unless the user explicitly requests benchmark redesign.
- Keep changes attributable to one research hypothesis whenever possible, then verify, commit, and push.

### Suggested Action
Promote a concise version to `AGENTS.md` under a project research execution section so future agents default to this combined method. Continue using `paper-code-reproducer` for honest paper-reproduction boundaries and `remote-windows-gpu-conda-ssh` for remote A6000 launch/monitor/retrieval rules when those tasks apply.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, skills/blockcipher-auto-research/SKILL.md, /home/fate/.agents/skills/karpathy-guidelines/SKILL.md
- Tags: workflow, research, coding-discipline, innovation1, experiments, karpathy, auto-research
- See Also: LRN-20260622-001, LRN-20260624-002
- Pattern-Key: workflow.blockcipher_auto_research_plus_karpathy_guidelines
- Recurrence-Count: 1
- First-Seen: 2026-06-25
- Last-Seen: 2026-06-25
- Promoted: AGENTS.md

---

## [LRN-20260625-002] best_practice

**Logged**: 2026-06-25T16:10:00+08:00
**Priority**: high
**Status**: promoted
**Area**: docs

### Summary
Use clear documentation destinations: formal experiment plans in `docs/experiments/`, research blueprints in `docs/research/`, and historical agent plans only in archive-style locations.

### Details
The user clarified the intended documentation organization after discussing whether `$using-superpowers` creates Markdown plans under `docs/superpowers/`. The correct project convention is:

- New formal experiment plans, reproduction records, result analyses, and next-step execution plans belong under `docs/experiments/`.
- New research blueprints, literature syntheses, theory notes, and broad method proposals belong under `docs/research/`.
- Historical agent execution plans may remain under `docs/superpowers/plans/`, or be migrated to `docs/archive/agent-plans/` if the archive is reorganized.

This separates current project-facing research artifacts from historical superpowers/planning workflow outputs. In particular, do not create new current experiment plans under `docs/superpowers/`; that directory should be treated as historical plan archive unless explicitly reorganized.

### Suggested Action
Promote this convention to `AGENTS.md` under documentation organization. When writing new project docs, choose the destination before creating the file:

- `docs/experiments/` for executable experiment plans and result records.
- `docs/research/` for broad research strategy and literature-backed blueprints.
- `docs/superpowers/plans/` or `docs/archive/agent-plans/` only for historical agent execution plans.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, docs/experiments/, docs/research/, docs/superpowers/plans/
- Tags: docs, documentation-organization, experiments, research, archive, superpowers
- See Also: LRN-20260625-001
- Pattern-Key: docs.destination.experiments_research_agent_archive
- Recurrence-Count: 1
- First-Seen: 2026-06-25
- Last-Seen: 2026-06-25
- Promoted: AGENTS.md

---

## [LRN-20260625-003] correction

**Logged**: 2026-06-25T20:30:07+08:00
**Priority**: critical
**Status**: promoted
**Area**: docs

### Summary
Do not answer project implementation or tooling facts from memory; inspect source, config, logs, or artifacts before reporting them.

### Details
The user corrected a hallucinated implementation claim: the assistant previously stated that project plotting already used Matplotlib, but the actual implementation at that time was hand-written SVG in `src/blockcipher_nd/evaluation/plots.py`. The later visualization change did convert plotting to Matplotlib, but the earlier answer was still wrong because it was made without first checking the code.

Correct behavior:

- Before saying "the project currently uses X" or "this is implemented by Y", inspect relevant files with `rg`, `sed`, config/lockfile reads, tests, logs, or result artifacts.
- This applies especially to dependencies, plotting/rendering libraries, training protocols, remote launch scripts, artifact paths, experiment status, metrics, checkpoint selection, and result gates.
- If the evidence has not been checked in the current turn or handoff, state that the answer is an assumption or check before reporting.
- When a prior statement is found to be wrong, correct it explicitly and separate the old false claim from the newly verified state.
- Do not rely on a recent memory of a change unless the repository state or command output confirms it.

### Suggested Action
Promote a concise factual-reporting rule to `AGENTS.md`. For future implementation/status answers, first cite or consult the relevant source/config/artifact. If verification is not possible, qualify the uncertainty instead of making a definitive claim.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, src/blockcipher_nd/evaluation/plots.py, pyproject.toml, uv.lock, scripts/plot-results
- Tags: hallucination, factual-reporting, source-verification, implementation-facts, visualization
- See Also: LRN-20260624-001, LRN-20260625-001
- Pattern-Key: workflow.factual_reporting.verify_before_claim
- Recurrence-Count: 1
- First-Seen: 2026-06-25
- Last-Seen: 2026-06-25
- Promoted: AGENTS.md

---

## [LRN-20260626-001] best_practice

**Logged**: 2026-06-26T11:07:22+08:00
**Priority**: high
**Status**: promoted
**Area**: docs

### Summary
Update `docs/experiments/` for meaningful experiment lifecycle events, and update `docs/research/` only for research-direction changes.

### Details
The user clarified that future agents should not mechanically write both `docs/experiments/` and `docs/research/` after every action. The correct documentation workflow is:

- New formal or medium-scale experiment plans should be recorded under `docs/experiments/`.
- Completed meaningful experiments should update the relevant `docs/experiments/` record with run id, configuration, evidence gate, metrics, result status, and next action.
- Smoke tests, temporary debug runs, and local implementation checks do not require experiment documentation unless they change an evidence judgment, expose an important failure mode, or become part of the research record.
- `docs/research/` is for research blueprints, theory notes, literature syntheses, method proposals, and major route changes. It should not become a run-by-run log.
- When unsure, choose the smallest durable documentation update that preserves the evidence chain without creating noise.

### Suggested Action
Promote this as an execution rule in `AGENTS.md` under Documentation Organization. Future experiment launches and completions should first decide whether the event is documentation-worthy, then update only the appropriate destination.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, docs/experiments/, docs/research/
- Tags: docs, experiments, research, documentation-workflow, evidence-chain
- See Also: LRN-20260625-002
- Pattern-Key: docs.update_lifecycle.experiments_vs_research
- Recurrence-Count: 1
- First-Seen: 2026-06-26
- Last-Seen: 2026-06-26
- Promoted: AGENTS.md

---

## [LRN-20260627-001] correction

**Logged**: 2026-06-27T11:35:00+08:00
**Priority**: high
**Status**: promoted
**Area**: docs

### Summary
When a meaningful remote experiment result is completed and retrieved, automatically update `docs/experiments/` before reporting; do not ask whether to document it.

### Details
The user corrected the reporting workflow after the `I1-SPN-001` 1M/class single-seed result completed. The assistant initially suggested documenting the result as a next step instead of doing it immediately. For this project, meaningful experiment results are part of the evidence chain and should be written to the relevant experiment document as soon as the result is retrieved and parsed.

Correct behavior:

- If a meaningful experiment reaches a valid result gate and artifacts are retrieved, update the relevant `docs/experiments/` record in the same task turn.
- Include run id, protocol scale, gate status, local/remote artifact paths, metrics, deltas versus baseline, claim scope, and next action.
- Do not ask the user whether to write the result document; ask only if there is genuine ambiguity about which experiment record should own the result.
- Smoke tests, temporary debug checks, and local implementation checks still do not need docs unless they change the evidence judgment or expose an important failure mode.

### Suggested Action
Promote this rule to `AGENTS.md` under Documentation Organization. Future remote-result handling should follow this sequence: retrieve artifacts, parse metrics, generate missing local plots/history if needed, update `docs/experiments/`, verify, commit, push, then report.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, docs/experiments/, outputs/remote_results/
- Tags: docs, experiments, remote-results, evidence-chain, workflow
- See Also: LRN-20260626-001, LRN-20260624-002
- Pattern-Key: docs.results.auto_update_experiment_record
- Recurrence-Count: 1
- First-Seen: 2026-06-27
- Last-Seen: 2026-06-27
- Promoted: AGENTS.md

---

## [LRN-20260627-002] research

**Logged**: 2026-06-27T19:41:43+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
After the 262k N1-v2 structure ablation, do not scale the current gated-MCND route as the main Innovation 1 path; prioritize SPN-only and transition-aware SPN backbones.

### Details
The completed and locally retrieved run `i1_spn_n1v2_ablation_r7_262k_seed0_gpu0_20260627` tested N0 Zhang/Wang MCND, N1-v1 late fusion, SPN-only, true-P gated MCND, and shuffled-P gated MCND at `262144/class` with strict `encrypted_random_plaintexts` negatives and Zhang/Wang Case2 official MCND sample structure.

Observed AUC:

- `present_zhang_wang_keras_mcnd`: `0.784541`
- `present_nibble_paligned_mcnd`: `0.784299`
- `present_nibble_paligned_spn_only`: `0.791488`
- `present_nibble_paligned_gated_mcnd`: `0.784897`
- `present_nibble_shuffled_paligned_gated_mcnd`: `0.784281`

The true-P gated model was weakly positive against N0, N1-v1, and shuffled control, but it failed all planned continuation gates:

- N1-v2 AUC vs N0: `+0.000356`, below required `+0.002`.
- N1-v2 AUC vs N1-v1: `+0.000598`, below required `+0.001`.
- N1-v2 AUC vs shuffled: `+0.000615`, below required `+0.001`.
- N1-v2 calibrated accuracy vs N0: `-0.000530`.

The strongest diagnostic signal came from `SPN-only`, not MCND fusion. This suggests the PRESENT nibble/P-layer structure view contains useful real-vs-random signal, but the current gate/fusion design dilutes or fails to inject it into the MCND backbone.

This is medium diagnostic evidence, not formal multi-seed `>=1000000/class` evidence. However, it is enough to reject scaling the exact current `present_nibble_paligned_gated_mcnd` design as the main next route.

### Suggested Action
For the next Innovation 1 experiment, design a minimal SPN transition-aware backbone around the SPN-only signal. Keep the benchmark fixed and compare at the same `262144/class` evidence scale:

1. Keep N0 baseline and current SPN-only as anchors.
2. Add N2 transition-aware models that treat 16 PRESENT nibbles and P-layer transitions as the primary representation.
3. Include true-P versus shuffled-P controls for attribution.
4. Do not change validation data, labels, negative mode, metric computation, or Zhang/Wang Case2 sample construction.
5. Only consider 1M/multi-seed scaling if N2 beats SPN-only and the true-P route beats shuffled-P by the predeclared gate.

### Metadata
- Source: conversation
- Related Files: docs/experiments/innovation1-n1v2-structure-ablation-plan.md, outputs/remote_results/i1_spn_n1v2_ablation_r7_262k_seed0_gpu0_20260627/
- Tags: innovation1, spn, present, n1v2, spn-only, transition-aware-backbone, evidence-gates
- See Also: LRN-20260621-001, LRN-20260623-001, LRN-20260627-001
- Pattern-Key: innovation1.spn_present.prioritize_transition_backbone_after_n1v2
- Recurrence-Count: 1
- First-Seen: 2026-06-27
- Last-Seen: 2026-06-27

---

## [LRN-20260628-001] correction

**Logged**: 2026-06-28T00:00:00+08:00
**Priority**: high
**Status**: promoted
**Area**: infra

### Summary
After a new training experiment smoke test passes, automatically continue to the corresponding non-smoke remote training launch instead of stopping at the smoke report.

### Details
The user corrected the experiment workflow after the SPN-only attribution smoke test. For this project, when the user asks to "推进" a new training experiment and a smoke matrix is part of the setup, the smoke is only a gate. If smoke passes and code/config/docs have been verified, committed, and pushed, the agent should proceed to launch the planned medium/formal remote run using the pushed GitHub commit and the established tmux monitor/retrieval workflow.

Correct behavior:

- Treat local smoke as a readiness gate, not the final deliverable, unless the user explicitly asks for smoke/local verification only.
- After smoke passes, commit and push scoped changes.
- Create or verify the matching remote config/launcher for the non-smoke run.
- Audit remote rules: `cmd.exe /c`, all artifacts under `G:\lxy`, disk-backed dataset cache/progress for `>=65536/class`, GitHub-pushed commit source.
- Launch remote training and start/verify a local tmux monitor for automatic retrieval.
- Report the run as `running`/`planned` with run id and monitor details, not as complete.

### Suggested Action
Promote this to `AGENTS.md` under research execution or remote workflow. Future "推进训练实验" tasks should run the full pipeline: implement -> smoke -> verify -> commit/push -> remote launch -> tmux monitor handoff.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, configs/experiment/innovation1/, configs/remote/, scripts/generated/remote/, docs/experiments/
- Tags: experiments, smoke, remote-training, tmux-monitor, workflow, innovation1
- See Also: LRN-20260622-001, LRN-20260627-001
- Pattern-Key: workflow.training.smoke_then_remote_launch
- Recurrence-Count: 1
- First-Seen: 2026-06-28
- Last-Seen: 2026-06-28
- Promoted: AGENTS.md

---

## [LRN-20260628-002] research

**Logged**: 2026-06-28T17:40:34+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
After SPN-only attribution, prioritize an InvP-centered PRESENT/SPN distinguisher route over generic DeltaC+InvP concatenation.

### Details
The completed remote run `i1_spn_only_attr_r7_262k_seed0_gpu1_20260628` showed that the strongest diagnostic row was `present_nibble_invp_only_spn_only`, not the previous `DeltaC + InvP(DeltaC)` anchor.

At `262144/class`, strict `encrypted_random_plaintexts` negatives, Zhang/Wang Case2 official MCND protocol:

- `present_zhang_wang_keras_mcnd`: AUC `0.783228`
- `present_nibble_delta_only_spn_only`: AUC `0.782918`
- `present_nibble_shuffled_paligned_spn_only`: AUC `0.784487`
- `present_nibble_paligned_spn_only`: AUC `0.790665`
- `present_nibble_invp_only_spn_only`: AUC `0.792536`

Key attribution deltas:

- InvP-only vs baseline: AUC `+0.009308`
- InvP-only vs DeltaC-only: AUC `+0.009617`
- InvP-only vs shuffled-P: AUC `+0.008048`
- InvP-only vs DeltaC+InvP anchor: AUC `+0.001871`

This supports the interpretation that inverse-P aligned `DeltaC` is the dominant useful signal in the current SPN-only family. Raw `DeltaC` concatenation may dilute or fail to improve the signal under the simple token-mixer architecture.

This is medium diagnostic single-seed evidence, not formal `>=1000000/class` multi-seed evidence.

### Suggested Action
Advance Innovation 1 as an InvP-centered route:

1. Add a compact InvP pair-set consistency model that reuses the existing InvP-only encoder and changes only pair aggregation.
2. Compare against baseline, current SPN-only anchor, InvP-only, DeltaC-only, and shuffled-P under the same `262144/class` protocol.
3. Treat local smoke as a launch gate; if smoke passes, automatically commit/push and launch the remote medium diagnostic with tmux monitor retrieval.
4. Only consider 1M/class multi-seed scaling after InvP-only or InvP-centered consistency remains stable across at least one additional 262k seed or clearly beats the current InvP-only anchor.

### Metadata
- Source: conversation
- Related Files: docs/experiments/innovation1-spn-only-attribution-plan.md, outputs/remote_results/i1_spn_only_attr_r7_262k_seed0_gpu1_20260628/
- Tags: innovation1, spn, present, invp, p-layer, attribution, pair-consistency
- See Also: LRN-20260627-002, LRN-20260628-001
- Pattern-Key: innovation1.spn_present.invp_centered_route
- Recurrence-Count: 1
- First-Seen: 2026-06-28
- Last-Seen: 2026-06-28

---

## [LRN-20260628-003] correction

**Logged**: 2026-06-28T18:20:00+08:00
**Priority**: high
**Status**: promoted
**Area**: docs

### Summary
Before starting or advancing meaningful training tasks, proactively write or update the experiment plan instead of waiting for the user to request it.

### Details
The user corrected the experiment workflow: future training launches should not depend on the user saying "写计划" first. For this project, a meaningful training task starts with a project-facing plan under `docs/experiments/`, then proceeds through implementation, smoke/readiness validation, scoped commit/push, remote launch from the pushed commit, local tmux monitoring, retrieval, result parsing, experiment-doc update, and final scoped commit/push.

Correct behavior:

- If the task is a meaningful new run, route, scale-up, or ablation, create or update the relevant `docs/experiments/` plan before launching.
- The plan should include the research question, fixed protocol, same-budget baseline, rows/models, scale, evidence gate, artifact paths, cache/progress expectation, remote device/run id expectations, and next action.
- Smoke/local checks remain readiness gates; if they pass and the user did not request smoke-only, continue to the planned non-smoke run.
- Do not mechanically update `docs/research/` for every run; update it only when the broader research route, theory, or method blueprint changes.
- Do not ask whether to write the plan for a meaningful training task; ask only when ownership of the document is genuinely ambiguous.

### Suggested Action
Promote this to `AGENTS.md` under Research Execution Style. Future experiment advancement should follow: plan first -> implement -> smoke -> commit/push -> remote launch -> tmux monitor -> retrieve -> update experiment docs -> commit/push.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, docs/experiments/, .learnings/LEARNINGS.md
- Tags: experiments, planning, docs, remote-training, workflow
- See Also: LRN-20260628-001, LRN-20260627-001
- Pattern-Key: workflow.training.proactive_experiment_plan_first
- Recurrence-Count: 1
- First-Seen: 2026-06-28
- Last-Seen: 2026-06-28
- Promoted: AGENTS.md

---

## [LRN-20260628-004] correction

**Logged**: 2026-06-28T19:25:00+08:00
**Priority**: high
**Status**: promoted
**Area**: research

### Summary
Do not default every remote training experiment to large multi-model comparison matrices; keep incremental runs lean.

### Details
The user pointed out that repeatedly comparing many models in every remote experiment is cumbersome. The correct workflow is to separate attribution/audit experiments from incremental model-selection experiments:

- Full comparison matrices are useful for stage gates, attribution studies, protocol audits, and checking whether a control invalidates a route.
- Incremental model changes should usually compare only the new candidate against the strongest current same-protocol anchor, plus the minimum necessary baseline/control rows.
- A normal incremental remote matrix should target 2-3 models and rarely exceed 4.
- Do not keep re-running historical weaker controls in every new experiment once the attribution route is stable, unless the research question specifically requires them.
- Existing already-launched remote jobs should normally be allowed to finish; apply the lean-matrix rule to the next planned runs.

For Innovation 1, after InvP attribution has already shown `InvP-only` as the current strongest route, follow-up architecture experiments should usually compare:

```text
new candidate
current strongest InvP anchor
optional Zhang/Wang baseline or one critical control
```

instead of automatically including DeltaC-only, shuffled-P, old DeltaC+InvP, and every previous baseline in each run.

### Suggested Action
Promote this to `AGENTS.md` under Research Execution Style. Future experiment plans should explicitly justify any matrix larger than 3-4 rows and prefer lean comparisons for iteration speed and clarity.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, docs/experiments/, configs/experiment/innovation1/
- Tags: experiments, model-matrix, remote-training, efficiency, innovation1
- See Also: LRN-20260628-002, LRN-20260628-003
- Pattern-Key: workflow.training.lean_experiment_matrix
- Recurrence-Count: 1
- First-Seen: 2026-06-28
- Last-Seen: 2026-06-28
- Promoted: AGENTS.md

---

## [LRN-20260629-001] correction

**Logged**: 2026-06-29T14:55:00+08:00
**Priority**: high
**Status**: promoted
**Area**: infra

### Summary
Remote tmux monitoring should be delegated to a sub-agent or watcher; the main thread should not repeatedly loop over tmux status.

### Details
The user corrected a workflow drift: after remote GPU launch, I repeatedly checked tmux sessions and monitor logs from the main thread. This is not the intended project workflow. The correct pattern is:

- Main thread launches or verifies exactly enough to hand off the remote run.
- A tmux monitor, watcher, or sub-agent owns the monitoring loop and retrieval.
- Main thread should not repeatedly inspect `tmux ls`, `monitor.log`, or remote progress just to see whether training finished.
- Main thread resumes result processing only when local artifacts have arrived, when the user explicitly asks for a status check, or when a monitor health failure is detected by the delegated watcher/sub-agent.
- If additional post-processing is needed after retrieval, it should also be delegated to a watcher/sub-agent where possible, not manually polled from the main thread.

This refines the existing "do not SSH-poll from the main thread" rule: do not replace SSH polling with main-thread tmux polling. Monitoring loops belong outside the main research/implementation thread.

### Suggested Action
Promote to `AGENTS.md` under Remote Monitoring And Retrieval. Future remote launches should report the monitor/sub-agent handoff and then continue with non-monitoring work or wait for retrieved artifacts. If the user asks "continue" while a remote run is active, avoid repetitive tmux checks; do a single local artifact check if needed, then proceed with planning/implementation that does not require the running result.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, .learnings/LEARNINGS.md, outputs/remote_results/
- Tags: remote-training, tmux, subagent, monitoring, workflow
- See Also: LRN-20260624-001
- Pattern-Key: remote_training.delegate_tmux_monitoring_to_subagent
- Recurrence-Count: 1
- First-Seen: 2026-06-29
- Last-Seen: 2026-06-29
- Promoted: AGENTS.md

---

## [LRN-20260705-001] best_practice

**Logged**: 2026-07-05T19:00:00+08:00
**Priority**: high
**Status**: promoted
**Area**: docs

### Summary
Update project guidance to match the current `self-improvement` workflow and its split from `self-healing`.

### Details
The user asked to update the current `self-improvement` usage. The installed skill now separates active runtime recovery from passive learning capture:

- Use `self-healing` for active failures that need diagnosis, patching, verification, and verified `HEAL-` entries during the current task.
- Use `self-improvement` for user corrections, outdated knowledge, recurring best practices, historical or external/tool failures, missing capabilities, and promotion of recurring self-healing handoffs.
- Route learnings by artifact type: `.learnings/LEARNINGS.md` for corrections, knowledge gaps, and best practices; `.learnings/ERRORS.md` for historical or external/tool failures; `.learnings/FEATURE_REQUESTS.md` for missing capabilities.
- Search existing `.learnings/` before adding a new entry, link related records with `See Also`, and use `Pattern-Key` plus recurrence fields for recurring patterns.
- Promote concise prevention rules only when they are broadly useful; for recurring patterns, prefer the current threshold of `Recurrence-Count >= 3`, at least two distinct tasks, and a 30-day window.
- Consider skill extraction when a learning becomes a reusable, verified, non-obvious workflow.

### Suggested Action
Promote this concise workflow into `AGENTS.md` so future agents apply the current self-improvement/self-healing split and promotion threshold without relying on chat context.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, .learnings/LEARNINGS.md, /home/fate/.agents/skills/self-improvement/SKILL.md
- Tags: self-improvement, self-healing, learnings, workflow, project-rules
- See Also: LRN-20260621-002
- Pattern-Key: workflow.self_improvement.current_usage
- Recurrence-Count: 1
- First-Seen: 2026-07-05
- Last-Seen: 2026-07-05
- Promoted: AGENTS.md

---

## [LRN-20260706-004] best_practice

**Logged**: 2026-07-06T02:20:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Do not advance the aligned active-difference integral route into neural residual or ensemble work until a non-baseline residual signal appears.

### Details
The composite residual probe tested whether deterministic feature-bank
combinations improve over the fixed `pair_xor_column_sum_variance` baseline on
three aligned single-nibble PRESENT r8 routes. At `2048/class`, seed `23`:

```text
Zhang/Wang active0 baseline AUC = 0.8878759145736694
Zhang/Wang best baseline+one delta = 0.0
AutoND active6 baseline AUC = 0.8747416734695435
AutoND best baseline+one delta = 0.0
Entropy active5 baseline AUC = 0.8852955102920532
Entropy best baseline+one delta = 0.0
```

The equal-weight composite diluted the signal to about `0.603-0.619` AUC, and
the best `baseline + one additional statistic` scan gave no improvement over
the baseline. This makes the route a deterministic SPN/multiset feature
baseline, not a current neural residual or diverse-expert candidate.

Correct framing:

- Keep `pair_xor_column_sum_variance` as the explicit comparator for this
  aligned active-difference route.
- Do not spend the next remote or neural slot on residual learning for this
  route unless a later local probe finds non-baseline signal.
- Continue searching for genuinely different SPN feature/input routes before
  diverse expert aggregation.

### Suggested Action
Use the composite residual audit as a gate before neural residual follow-up.
If both equal composite and best baseline+one fail to beat the fixed baseline,
record the route as deterministic-only and move to a different representation
family.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-integral-parity-control-plan.md, src/blockcipher_nd/cli/audit_integral_parity_signal.py
- Tags: innovation1, spn, present, integral, residual-probe, deterministic-baseline
- See Also: LRN-20260706-001, LRN-20260706-003, LRN-20260705-003
- Pattern-Key: innovation1.spn_present.aligned_active_difference_no_composite_residual
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-005] best_practice

**Logged**: 2026-07-06T02:45:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Do not reopen candidate-trail as a main route from weak local low-dimensional feature probes alone.

### Details
After the aligned active-difference route was reduced to a deterministic
baseline, candidate-evidence was rechecked as a possible non-neighbor feature
family. The local low-dimensional probe on PRESENT r7 official Case2 found
repeated weak top-axis composite signal:

```text
seed17 top-axis composite AUC = 0.5382152795791626
seed17 best axis = 285, AUC advantage = 0.03317534923553467
seed18 top-axis composite AUC = 0.5390714406967163
seed18 best axis = 64, AUC advantage = 0.025446653366088867
```

This is useful route-selection information, but it does not overturn the
retrieved 262144/class candidate-trail gate:

```text
best candidate-trail AUC = 0.703854276799
InvP anchor AUC = 0.793651987187
shuffled-cell control AUC = 0.702488259296
decision = stop_candidate_trail_route
```

Correct framing:

- The local probe is weak-positive but semantically/positionally unstable.
- It is not enough to reopen candidate-trail seed1 or remote scale.
- Candidate-trail remains stopped as a main route unless a genuinely different
  representation or control beats the InvP anchor and shuffled-cell control.

### Suggested Action
Use candidate-evidence low-dimensional probes only as cheap route-selection
screens. Before spending remote time, require stability across seeds and a
control that separates true candidate evidence from shuffled-cell evidence.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-candidate-trail-consistency-plan.md, src/blockcipher_nd/tasks/innovation1/spn_feature_audit.py
- Tags: innovation1, spn, present, candidate-trail, feature-probe, diverse-experts
- See Also: LRN-20260706-004, LRN-20260705-003, LRN-20260705-002
- Pattern-Key: innovation1.spn_present.candidate_trail_lowdim_probe_not_reopen_gate
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-014] correction

**Logged**: 2026-07-06T09:45:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Do not let Innovation 1 route selection passively follow the user's latest suggested direction.

### Details
The user explicitly corrected the research process: the agent should not simply
agree with a proposed direction such as "multiple neural networks" or "ensemble"
without independently checking local evidence and external research signals.

Current corrected interpretation:

- Multiple neural networks are still a valid later mechanism, but only as a
  diversity-gated score aggregation route.
- The next main experiment should not be a wider near-neighbor ensemble unless
  there is at least one compatible weak-positive non-neighbor expert with
  frozen score artifacts and low error overlap.
- External and local evidence currently rank SPN-aware representation/data
  search above generic model piling.
- Active-pattern auxiliary supervision has now stopped at `262144/class`
  seed0 retry, with candidate AUC `0.786112642265` below the InvP anchor AUC
  `0.793651987187`, so it should not be used as the next ensemble expert.

### Suggested Action
Before proposing or launching the next Innovation 1 SPN experiment, do an
independent route arbitration: inspect current artifacts, check stopped/held
routes, compare against the strongest same-protocol anchor, and only then decide
whether the next step is representation search, architecture, or ensemble.

### Metadata
- Source: user_feedback, experiment_audit, literature_recheck
- Related Files: docs/research/innovation1-spn-independent-route-recheck-20260706.md, docs/research/innovation1-spn-adaptation-literature-refresh-20260705.md, outputs/remote_results/i1_active_auxiliary_r7_262k_seed0_gpu1_retry1_20260704/
- Tags: innovation1, spn, route-selection, ensemble, independent-judgment, active-auxiliary
- See Also: LRN-20260706-011, LRN-20260706-013, LRN-20260705-003
- Pattern-Key: innovation1.spn_route_selection.independent_arbitration_before_following_user_direction
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-015] best_practice

**Logged**: 2026-07-06T10:15:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Do not use the current r8 GPD-style beamstats feature as the next diverse expert candidate.

### Details
After the user asked for independent Innovation 1 SPN route selection rather
than passively following the multi-network idea, the r8 GPD-style
beamstats4/deep3 feature was rechecked as a possible non-neighbor expert
source.

The higher-sample local attribution gate used:

```text
plan = configs/experiment/innovation1/innovation1_spn_present_r8_gpd_style_beamstats_512_seed0.csv
row_index = 3
feature = present_delta_paligned_sinv_sboxddt_beamstats4deep3_cell_matrix_bits
sample_structure = plaintext_integral_nibble_difference_matched_negative
samples_per_class = 4096
key_split = validation
audit_seeds = 0, 1, 2
```

Results:

| Seed | Best semantic scalar | AUC | AUC advantage | Best threshold accuracy |
|---:|---|---:|---:|---:|
| 0 | `score_max` | `0.509702891111` | `0.009702891111` | `0.528930664062` |
| 1 | `score_max` | `0.516063421965` | `0.016063421965` | `0.533813476562` |
| 2 | `confidence_std` | `0.510016858578` | `0.010016858578` | `0.522338867188` |

The predeclared gate required best semantic scalar AUC advantage `>= 0.02` on
all three seeds plus enough semantic stability to name the family. The route
failed that gate.

Correct interpretation:

- The current beamstats feature remains weak local route-selection evidence
  only.
- It is not a qualified non-neighbor expert for diverse score aggregation.
- Do not launch a `65536/class` remote beamstats run from this evidence.
- Do not build a multi-network ensemble around this feature.

### Suggested Action
Return to broader SPN representation/data search for a cleaner non-neighbor
expert. Reopen GPD-style features only if the representation changes
substantially or a new control shows stable non-baseline signal.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-gpd-style-beamstats-plan.md, outputs/local_audits/i1_present_r8_gpd_beamstats_attribution_seed0_4096.json, outputs/local_audits/i1_present_r8_gpd_beamstats_attribution_seed1_4096.json, outputs/local_audits/i1_present_r8_gpd_beamstats_attribution_seed2_4096.json
- Tags: innovation1, spn, present, gpd, beamstats, diverse-experts, route-selection
- See Also: LRN-20260706-014, LRN-20260706-011, LRN-20260706-004
- Pattern-Key: innovation1.spn_present.gpd_beamstats_not_current_diverse_expert
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-016] best_practice

**Logged**: 2026-07-06T10:55:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Treat r8 trail-position beamstats as a promising local SPN representation route, but require attribution before remote launch.

### Details
After the current GPD-style global beamstats feature failed its higher-sample
semantic attribution gate, a more position-aware existing model was tested:

```text
model = present_trail_position_stats_pairset
feature = present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits
sample_structure = plaintext_integral_nibble_difference_matched_negative
rounds = 8
pairs_per_sample = 16
negative_mode = encrypted_random_plaintexts
```

The 128/class smoke showed a large positive margin over a same-input
global-statistics control:

| Seed | Global-stat control AUC | Trail-position candidate AUC |
|---:|---:|---:|
| 0 | `0.608642578125` | `0.9423828125` |
| 1 | `0.62890625` | `0.9541015625` |

The 512/class local diagnostic preserved and strengthened the result:

| Seed | Global-stat control AUC | Trail-position candidate AUC |
|---:|---:|---:|
| 0 | `0.813568115234375` | `0.98883056640625` |
| 1 | `0.7928619384765625` | `0.9859771728515625` |

Correct interpretation:

- This is the strongest local non-neighbor SPN representation signal currently
  found in the GPD/beamstats branch.
- It is not a remote-launch basis yet.
- It is not a PRESENT r8 breakthrough claim.
- Because the sample structure is a matched-negative r8 integral setting, the
  next requirement is attribution/control: determine whether deterministic
  position statistics alone explain the signal.

### Suggested Action
Before any remote training, implement or run a local position-statistics
attribution/control audit, including deterministic baselines and
pair-order/active-nibble/difference controls. Keep the route as a promising
SPN representation/data candidate, not yet a qualified diverse ensemble expert.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-trail-position-beamstats-smoke-plan.md, configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_smoke.csv, configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_512_local.csv
- Tags: innovation1, spn, present, trail-position, beamstats, gpd, representation-search
- See Also: LRN-20260706-015, LRN-20260706-014, LRN-20260706-004
- Pattern-Key: innovation1.spn_present.trail_position_beamstats_promising_requires_attribution
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-017] best_practice

**Logged**: 2026-07-06T11:35:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Use a deterministic position-statistics baseline before scaling r8 trail-position beamstats.

### Details
The `present_trail_position_stats_pairset` route showed very strong local
neural AUC at 512/class on the r8 matched-negative integral setting:

```text
seed0 neural AUC = 0.98883056640625
seed1 neural AUC = 0.9859771728515625
```

A new local attribution audit extracted the deterministic position-statistics
vector used by the model and evaluated scalar statistics plus a fixed top-16
oriented z-score composite:

| Scale | Seed | Best scalar | Directional best-scalar AUC | Top-16 composite AUC |
|---:|---:|---|---:|---:|
| 512/class | 0 | `cell_span_cell6` | `0.6741142272949219` | `0.9032249450683594` |
| 512/class | 1 | `depth_word_span_depth2_trailword3` | `0.6828956604003906` | `0.8659286499023438` |
| 2048/class | 0 | `depth_word_span_depth2_trailword1` | `0.6629594564437866` | `0.8734362125396729` |
| 2048/class | 1 | `depth_word_span_depth1_trailword1` | `0.6627322435379028` | `0.8486461639404297` |

Correct interpretation:

- The route is promising but not cleanly a neural architecture win.
- Deterministic position-statistics features explain much of the signal.
- The neural candidate remains above the deterministic top-16 composite at
  512/class, so there may be nonlinear residual value.
- No remote launch should happen before deterministic baseline and
  pair-order/active-nibble/difference controls are in place.

### Suggested Action
For any future trail-position beamstats experiment, compare against the fixed
deterministic position-statistics composite or equivalent baseline. Treat
larger neural training as invalid for route claims unless it beats this
baseline under matched controls.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-trail-position-beamstats-smoke-plan.md, src/blockcipher_nd/tasks/innovation1/spn_feature_audit.py, outputs/local_audits/i1_present_r8_trail_position_attribution_seed0_2048.json, outputs/local_audits/i1_present_r8_trail_position_attribution_seed1_2048.json
- Tags: innovation1, spn, present, trail-position, beamstats, attribution, deterministic-baseline
- See Also: LRN-20260706-016, LRN-20260706-015, LRN-20260706-004
- Pattern-Key: innovation1.spn_present.trail_position_beamstats_requires_deterministic_baseline
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-018] best_practice

**Logged**: 2026-07-06T12:45:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Use train-selected position-statistics controls when judging r8 trail-position beamstats.

### Details
The first trail-position attribution audit selected top position-statistics
axes directly on validation labels. A stricter split baseline was implemented
and run:

```text
audit = present_trail_position_split_baseline
selection_split = train
evaluation_split = validation
combiner = train_selected_position_stat_oriented_zscore_mean
feature = present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits
sample_structure = plaintext_integral_nibble_difference_matched_negative
rounds = 8
pairs_per_sample = 16
negative_mode = encrypted_random_plaintexts
```

The split baseline fits top-k axis selection, axis orientation, and z-score
normalization on the train key only, then applies the fixed composite to the
validation key. Results:

| Scale | Seed | Train composite AUC | Validation composite AUC | Validation best accuracy |
|---:|---:|---:|---:|---:|
| 512/class | 0 | `0.8651924133300781` | `0.7695465087890625` | `0.703125` |
| 512/class | 1 | `0.9015998840332031` | `0.8455047607421875` | `0.7734375` |
| 2048/class | 0 | `0.8498256206512451` | `0.8056130409240723` | `0.735595703125` |
| 2048/class | 1 | `0.8753311634063721` | `0.8421728610992432` | `0.766845703125` |

Correct interpretation:

- The trail-position signal is not merely a validation-label top-k selection
  artifact.
- Deterministic depth/word/cell span statistics explain a large fraction of
  the route's signal.
- The 512/class neural candidate remains higher than the split deterministic
  baseline, so a nonlinear residual may exist, but future neural claims must
  beat this baseline under matched controls.
- This is local diagnostic evidence only, not remote-launch evidence and not a
  PRESENT r8 breakthrough claim.

### Suggested Action
Before any larger trail-position neural training, include a same-protocol
train-selected deterministic position-statistics baseline or postprocess gate.
Add active-nibble, pair-order, and difference controls before treating the route
as a qualified diverse ensemble expert or remote-launch candidate.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-trail-position-beamstats-smoke-plan.md, src/blockcipher_nd/tasks/innovation1/spn_feature_audit.py, outputs/local_audits/i1_present_r8_trail_position_split_baseline_seed0_2048.json, outputs/local_audits/i1_present_r8_trail_position_split_baseline_seed1_2048.json
- Tags: innovation1, spn, present, trail-position, beamstats, split-baseline, deterministic-control
- See Also: LRN-20260706-017, LRN-20260706-016, LRN-20260706-011
- Pattern-Key: innovation1.spn_present.trail_position_requires_train_selected_split_baseline
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-019] best_practice

**Logged**: 2026-07-06T13:35:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Trail-position controls show active/difference alignment matters more than pair order.

### Details
The r8 trail-position deterministic control suite was run at `512/class` for
seeds `0` and `1`:

```text
audit = present_trail_position_control_baseline
baseline = train-selected deterministic position-statistics split baseline
controls = active_nibble_1, input_difference_0x90, pair_order_reverse
feature = present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits
sample_structure = plaintext_integral_nibble_difference_matched_negative
negative_mode = encrypted_random_plaintexts
```

Results:

| Seed | Baseline AUC | Active-nibble control AUC | Difference control AUC | Pair-order reverse AUC |
|---:|---:|---:|---:|---:|
| 0 | `0.7695465087890625` | `0.49724578857421875` | `0.5139007568359375` | `0.7695465087890625` |
| 1 | `0.8455047607421875` | `0.49993133544921875` | `0.5224685668945312` | `0.8455047607421875` |

Correct interpretation:

- The trail-position deterministic signal collapses when the active plaintext
  nibble is moved or when the input difference is moved to `0x90`.
- Therefore it is not merely generic matched-negative integral leakage.
- The pair-order reverse control exactly matches the baseline because the
  selected statistics are dominated by order-invariant span/range features.
- Pair-order-sensitive trail models are not the next best model slot unless a
  new audit identifies pair-order-sensitive residual signal.

### Suggested Action
Keep trail-position as a controlled local SPN/integral representation candidate,
but require active-nibble and input-difference mismatch controls before any
larger neural claim. Do not prioritize pair-order models for this route unless
new evidence shows pair-order sensitivity.

### Metadata
- Source: experiment_audit
- Related Files: docs/experiments/innovation1-present-r8-trail-position-beamstats-smoke-plan.md, docs/research/innovation1-spn-independent-route-recheck-20260706.md, src/blockcipher_nd/tasks/innovation1/spn_feature_audit.py, outputs/local_audits/i1_present_r8_trail_position_control_baseline_seed0_512.json, outputs/local_audits/i1_present_r8_trail_position_control_baseline_seed1_512.json
- Tags: innovation1, spn, present, trail-position, controls, active-nibble, input-difference, pair-order
- See Also: LRN-20260706-018, LRN-20260706-017, LRN-20260706-016
- Pattern-Key: innovation1.spn_present.trail_position_controls_active_difference_not_pair_order
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-020] best_practice

**Logged**: 2026-07-06T05:22:32+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Use the trail-position residual gate before treating high neural AUC as SPN architecture gain.

### Details
The r8 trail-position 512/class neural diagnostic was gated against the
train-selected deterministic position-statistics baseline, same-input
global-stat neural control, and active-nibble/input-difference mismatch
controls:

```text
gate = scripts/gate-trail-position-residual
output = outputs/local_audits/i1_present_r8_trail_position_residual_gate_512.json
decision = support_trail_position_neural_residual_local
pair_order_assessment = pair_order_not_bottleneck
min_candidate_margin_vs_deterministic_auc = 0.140472412109375
min_candidate_margin_vs_global_auc = 0.175262451171875
min_deterministic_margin_vs_mismatch_auc = 0.255645751953125
```

Per-seed margins:

| Seed | Candidate AUC | Deterministic baseline AUC | Global control AUC | Max mismatch control AUC |
|---:|---:|---:|---:|---:|
| 0 | `0.98883056640625` | `0.7695465087890625` | `0.813568115234375` | `0.5139007568359375` |
| 1 | `0.9859771728515625` | `0.8455047607421875` | `0.7928619384765625` | `0.5224685668945312` |

Correct interpretation:

- The 512/class local diagnostic supports possible neural residual over the
  deterministic position-statistics baseline.
- The route remains active/difference-aligned and order-invariant under the
  current selected statistics.
- This justifies another controlled local or medium diagnostic, not a remote
  launch, PRESENT r8 breakthrough claim, or Zhang/Wang r7 Case2 claim.
- Diverse neural aggregation should remain a later validator until this or
  another non-neighbor route emits compatible frozen scores and low-overlap
  evidence.

### Suggested Action
Before scaling or reporting a trail-position neural route, run the residual
gate or an equivalent same-protocol comparison against deterministic
position-statistics, same-input global-stat, and active/difference mismatch
controls. Treat pair-order reverse parity as "pair_order_not_bottleneck", not
as route failure.

### Metadata
- Source: experiment_audit
- Related Files: src/blockcipher_nd/planning/trail_position_residual_gate.py, scripts/gate-trail-position-residual, docs/experiments/innovation1-present-r8-trail-position-beamstats-smoke-plan.md, docs/research/innovation1-spn-independent-route-recheck-20260706.md
- Tags: innovation1, spn, present, trail-position, residual-gate, deterministic-baseline, neural-residual
- See Also: LRN-20260706-019, LRN-20260706-018, LRN-20260706-011
- Pattern-Key: innovation1.spn_present.trail_position_neural_residual_gate_before_scale
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-021] best_practice

**Logged**: 2026-07-06T06:03:50+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
Trail-position neural residual remains positive at 2048/class, but global-stat control is also strong.

### Details
The r8 trail-position residual-gated diagnostic was extended locally from
`512/class` to `2048/class`:

```text
plan = configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_2048_local.csv
results = outputs/local_smoke/i1_present_r8_trail_position_beamstats_2048/results.jsonl
gate = outputs/local_audits/i1_present_r8_trail_position_residual_gate_2048.json
decision = support_trail_position_neural_residual_local
pair_order_assessment = pair_order_not_bottleneck
```

Gate margins:

```text
min_candidate_margin_vs_deterministic_auc = 0.1573951244354248
min_candidate_margin_vs_global_auc = 0.10353946685791016
min_deterministic_margin_vs_mismatch_auc = 0.2893033027648926
```

Key metrics:

| Seed | Candidate AUC | Global control AUC | Deterministic baseline AUC | Max mismatch control AUC |
|---:|---:|---:|---:|---:|
| 0 | `0.9991159439086914` | `0.8932428359985352` | `0.8056130409240723` | `0.5163097381591797` |
| 1 | `0.999567985534668` | `0.8960285186767578` | `0.8421728610992432` | `0.5250661373138428` |

Correct interpretation:

- The trail-position candidate still clears deterministic position-statistics,
  same-input global-stat neural control, and active/difference mismatch
  controls at `2048/class`.
- The same-input global-stat control rises to about `0.895` AUC, so this
  setting exposes strong integral/statistical structure even without position
  detail.
- The residual result supports controlled local/medium continuation, not a
  remote launch, PRESENT r8 breakthrough, or Zhang/Wang r7 Case2 claim.
- Before ensemble promotion, this route needs frozen-score artifacts and
  error-overlap/diversity checks against the r7 InvP/P-layer anchor and
  near-neighbor controls.

### Suggested Action
Treat trail-position residual as the current best local SPN/integral candidate.
The next meaningful step is not a wider neural ensemble; it is a cache-ready
medium diagnostic design plus later frozen-score diversity evaluation if the
medium result holds.

### Metadata
- Source: experiment_audit
- Related Files: configs/experiment/innovation1/innovation1_spn_present_r8_trail_position_beamstats_2048_local.csv, docs/experiments/innovation1-present-r8-trail-position-beamstats-smoke-plan.md, docs/research/innovation1-spn-independent-route-recheck-20260706.md
- Tags: innovation1, spn, present, trail-position, residual-gate, local-diagnostic, medium-readiness
- See Also: LRN-20260706-020, LRN-20260706-019, LRN-20260706-018
- Pattern-Key: innovation1.spn_present.trail_position_residual_2048_positive_but_controlled
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-022] best_practice

**Logged**: 2026-07-06T19:10:00+08:00
**Priority**: high
**Status**: pending
**Area**: research

### Summary
PRESENT r8 trail-position beamstats completed remote seed0 `65536/class` training and still beats the same-input global-stat control.

### Details
The remote medium diagnostic completed the two-row seed0 training matrix:

```text
run_id = i1_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706
source_commit = cc8197a83ae5ce7f7edfb484ea1d281110f3b7fa
results = outputs/remote_results/i1_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706/results/train_matrix.jsonl
train_samples_per_class = 65536
validation_samples_per_class = 32768
pairs_per_sample = 16
input_bits = 39936
negative_mode = encrypted_random_plaintexts
sample_structure = plaintext_integral_nibble_difference_matched_negative
```

Metrics:

| Model | AUC | Accuracy | Calibrated accuracy | Best epoch |
|---|---:|---:|---:|---:|
| `present_pairset_global_stats` | `0.9916146486066282` | `0.954833984375` | `0.9550933837890625` | `20` |
| `present_trail_position_stats_pairset` | `0.9999999953433871` | `0.9999542236328125` | `0.999969482421875` | `19` |

Margins:

```text
candidate_auc_margin_vs_global = +0.008385346736758947
candidate_accuracy_margin_vs_global = +0.0451202392578125
candidate_calibrated_accuracy_margin_vs_global = +0.0448760986328125
```

Correct interpretation:

- This is positive remote medium diagnostic evidence for the trail-position
  route on seed0.
- The global-stat control is very strong at `0.9916` AUC, so the dataset has
  strong global statistical signal even without trail-position detail.
- The candidate still wins, but the AUC residual over global control is under
  `0.01`; do not overstate the margin.
- This is not formal SPN/PRESENT evidence, not multi-seed evidence, not a
  `262144/class` or `1000000/class` result, not a Zhang/Wang r7 Case2
  reproduction, and not a multi-network aggregation result.

### Suggested Action
Keep the trail-position route active, but require the corrected watcher to
retrieve score artifacts and run/record same-protocol residual controls before
stronger claims. The next research choice should be either controlled scale-up
or a genuinely different expert family that first clears its own same-input
global-stat control, not near-neighbor averaging.

### Metadata
- Source: remote_result_retrieval
- Related Files: docs/experiments/innovation1-present-r8-trail-position-beamstats-smoke-plan.md, configs/remote/generated/monitor_i1_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706.sh
- Tags: innovation1, spn, present, trail-position, remote-medium, seed0
- See Also: LRN-20260706-021, LRN-20260706-020, LRN-20260621-001
- Pattern-Key: innovation1.spn_present.trail_position_65k_seed0_positive_medium
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260706-023] correction

**Logged**: 2026-07-06T19:12:00+08:00
**Priority**: high
**Status**: pending
**Area**: infra

### Summary
Remote monitors that wait for score artifacts must not treat `train_done.marker` as final completion.

### Details
The local monitor for `i1_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706`
exited with:

```text
completed_missing_or_incomplete_results rows=2
```

Root cause: the monitor checked:

```bash
compgen -G "${LOCAL_ROOT}/logs/*done.marker"
```

The remote launcher writes `train_done.marker` immediately after training and
before score export. The glob therefore matched `train_done.marker` and allowed
the monitor to exit before waiting for:

```text
score_artifacts/global_stats_control/models.json
score_artifacts/trail_position/models.json
<RUN_ID>_done.marker
```

The trail-position monitor was patched to wait for the exact final marker:

```bash
${LOCAL_ROOT}/logs/${RUN_ID}_done.marker
```

### Suggested Action
For future generated monitors with multi-stage post-training work, use exact
terminal markers such as `${RUN_ID}_done.marker` instead of broad `*done.marker`
globs. Treat `train_done.marker`, `score_export_done.marker`, and similar stage
markers as progress signals only. Add tests that assert monitors do not contain
`${LOCAL_ROOT}/logs/*done.marker` when stage markers exist.

### Metadata
- Source: remote_monitor_bug
- Related Files: configs/remote/generated/monitor_i1_present_r8_trail_position_beamstats_65k_seed0_gpu0_20260706.sh, tests/test_project_structure.py
- Tags: remote-monitor, score-artifacts, watcher, trail-position
- See Also: LRN-20260706-022, ERR-20260705-001
- Pattern-Key: remote.monitor.exact_final_done_marker_required
- Recurrence-Count: 1
- First-Seen: 2026-07-06
- Last-Seen: 2026-07-06

---

## [LRN-20260707-003] correction

**Logged**: 2026-07-07T13:30:00+08:00
**Priority**: high
**Status**: promoted
**Area**: infra

### Summary
The user does not want conversational approval loops before normal project `git push` attempts.

### Details
During the Innovation 1 residual-focus remote-launch preparation, repeated
`git push origin main` attempts were blocked by the sandbox reviewer because
the push would transfer many local commits to an external GitHub remote and the
reviewer wanted exact-current-payload approval. The user then clarified:

```text
遇到git push你提交得了不用等我审批
```

Correct future behavior:

```text
After scoped commits and verification, proactively attempt the normal configured
push instead of stopping in chat to ask for permission. If the platform
reviewer rejects the push, do not use workarounds such as dirty overlay,
alternate external transfer routes, or remote launches from unpublished code.
Report the exact rejected payload/head and continue with safe local preparation
or wait for a platform-acceptable approval path.
```

This preference changes the agent's conversational behavior, not the platform
safety boundary.

### Suggested Action
When repository rules say to push after a completed commit, run the normal push
command with the required escalation request immediately. Do not ask the user
in chat first. If the escalation reviewer rejects the external transfer, treat
that as a tool/policy blocker for publication only and keep progressing on
non-publication local work when possible.

### Metadata
- Source: user_feedback
- Related Files: AGENTS.md, .learnings/ERRORS.md
- Tags: git, push, approval, source-publication, remote-launch
- See Also: ERR-20260706-002
- Pattern-Key: workflow.git_push.proactive_attempt_after_commit
- Recurrence-Count: 1
- First-Seen: 2026-07-07
- Last-Seen: 2026-07-07
- Promoted: AGENTS.md

---
