# Innovation 1 SPN Bit-Sensitivity Non-Neighbor Route

**Date:** 2026-07-07

**Status:** conditional research route; no remote launch while the active
trail-position 262144/class watchers are running

## Current State

The active PRESENT-80 r8 trail-position route is still waiting for retrieved
medium-scale evidence:

```text
active_branch = wait_for_trail_position_262k_results
active_runs = seed0, seed1
samples_per_class = 262144
status = running
train_matrix_rows = 0
postprocess_allowed = false
```

That means the next useful work is not another remote SPN launch. The safe
parallel work is route design that can consume the trail-position score
artifacts after they exist.

## External Evidence Check

The direction is based on current local evidence plus the external literature
thread already used by this project:

- Zhang and Wang 2022, "Improving Differential-Neural Distinguisher Model For
  DES, Chaskey, and PRESENT", arXiv:2204.06341,
  <https://arxiv.org/abs/2204.06341>.
- Zhang et al. 2025, "Neural-Inspired Advances in Integral Cryptanalysis",
  arXiv:2505.10790, <https://arxiv.org/abs/2505.10790>.
- Tashkilkhanova et al. 2025, "Enhancing RX Neural Cryptanalysis: Advanced
  Data Formatting and Weighted Differential Techniques for Block Ciphers",
  arXiv:2511.06336, <https://arxiv.org/abs/2511.06336>.
- Zhang et al. 2025, "More Efficient Deep Learning-Based Distinguishing Attacks
  with Multiple Ciphertext Pairs", arXiv:2505.10792,
  <https://arxiv.org/abs/2505.10792>.
- The local paper index also flags bit selection, GPD, PRESENT entropy, and
  multi-ciphertext-pair work as representation/feature-search evidence rather
  than broad model-ensemble evidence.

The 2026-07-07 re-check adds one concrete implementation lesson. Recent
RX-neural work uses bit sensitivity tests to reduce the input data format, then
uses the freed input budget to include more ciphertext-pair evidence. That
pattern fits this route better than a wider generic ensemble: first identify
stable, train-only informative axes; then test whether the compact evidence can
support a non-neighbor expert or higher pair count without changing labels,
negative construction, or validation keys.

The takeaway is not "add more weak networks." The stronger route is:

```text
SPN-aware data/feature representation
train-only bit/axis sensitivity compression
multi-pair evidence only under the same protocol
then structure-aware architecture over that representation
then diverse aggregation only after a real non-neighbor expert exists
```

## Why This Route

The current trail-position route is strong locally, but it is not enough to
start averaging nearby variants:

```text
near-neighbor frozen-score aggregation = no gain over best trail-position expert
cell-value histogram = weak-positive but loses to same-input global control
raw matrix Inception = fails the both-seed global-control gate
handwritten InvP aggregate statistics = held
```

The missing ingredient for the user's multi-network idea is not count. It is a
third expert family that is both:

```text
weak-positive under the same protocol
low-overlap / non-neighbor relative to trail-position
```

Bit-sensitivity-guided projection is a better candidate than another generic
network because it can use the trail-position artifacts to choose a compact,
frozen representation and then test whether that representation carries a
different error pattern.

## Proposed Non-Neighbor Expert

Name:

```text
present_r8_bit_sensitivity_projection_expert
```

Hypothesis:

```text
The trail-position route may contain stable depth/cell/word sensitivity axes.
A compact projection expert built from those axes can become a non-neighbor
weak-positive expert, or can explain that the trail-position signal is already
too concentrated for useful aggregation.
```

The expert must be selected on training-split evidence only, then frozen before
validation:

```text
selection_input = trail-position candidate/control frozen scores plus features
selection_split = train only
validation_split = held-out validation key
allowed_projection_sources = depth, word, cell, prefix/trail blocks, activity
disallowed_shortcuts = label leakage, validation-selected masks, changed negatives
```

The feature-export tooling added for this route now makes that discipline
executable:

```text
train export -> features.npy / labels.npy / sample_ids.npy / metadata.json
validation export -> same files, plus reference-artifact alignment check
selector -> train-only frozen mask
scorer -> validation-only frozen projection scores
gate -> same-protocol margin and error-overlap checks
```

This differs from the held SGP route because SGP searched unstable raw axes
directly. This route starts from the current strongest controlled
trail-position residual and asks which axes explain its residual errors.

## 2026-07-07 V0 Raw-Axis Screen

A seed0 local 2048/class screen tested the simplest executable version of this
idea: select 64 individual raw feature axes on the train split, freeze the
mask, score the held-out validation split, and compare against the existing
global-control and trail-position frozen score artifacts.

```text
gate = outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_seed0_gate.json
decision = hold_projection_duplicate_or_weak
global_control_auc = 0.8542919158935547
trail_position_anchor_auc = 0.9985876083374023
projection_auc = 0.49335479736328125
projection_margin_vs_global_auc = -0.36093711853027344
projection_margin_vs_anchor_auc = -0.5052328109741211
```

The low error overlap with the trail-position anchor is not a diversity win:
the projection is near random. This rejects the raw individual-axis projection
as a candidate expert. It does not reject the broader bit-sensitivity idea,
because the external evidence points to feature-format compression and
multi-pair evidence, not necessarily single raw-axis masks.

Updated implication:

```text
do_not_expand_v0_raw_axis_projection
do_not_run_remote_or_seed1_for_this_exact_v0_screen
keep train/validation-aligned export and gate tooling
if revisiting projection, use grouped structural axes, residual summaries, or
multi-pair compressed formats rather than isolated raw feature columns
```

## 2026-07-07 V1 Grouped-Axis Screen

The next concrete step was implemented and screened locally, not launched as a
new remote experiment:

```text
scripts/select-bit-sensitivity-projection --group-size <n> --top-groups <k>
scripts/apply-bit-sensitivity-projection
projection_unit = contiguous_axis_group
mask fields = selected_groups, selected_axes, selected_group_count
scorer behavior = average each selected group as one frozen projection unit
```

This is a deliberately small correction to the v0 failure. It keeps the same
train-only mask discipline, but changes the projection unit from an isolated
raw scalar to a contiguous structure block. That is closer to the external
evidence thread:

```text
Zhang/Wang-style PRESENT gains are representation-driven.
Multi-pair neural distinguisher work treats input formatting as part of the
attack, not as a neutral preprocessing detail.
Bit-selection/sensitivity work is useful when it compresses structured input
so the saved budget can be spent on better evidence, not when it creates a
standalone single-bit oracle.
```

2048/class seed0 validation results:

| Variant | Projection units | AUC |
|---|---:|---:|
| group4/top16 | 16 | `0.5030102729797363` |
| group8/top8 | 8 | `0.5734086036682129` |
| group16/top4 | 4 | `0.5104804039001465` |
| group32/top2 | 2 | `0.5292434692382812` |

Anchors:

```text
same_input_global_control_auc = 0.8542919158935547
trail_position_anchor_auc = 0.9985876083374023
best_grouped_projection_auc = 0.5734086036682129
decision = hold_projection_duplicate_or_weak
hold_reason = does_not_clear_global_control
```

Important limitation:

```text
grouped-axis projection is a weak diagnostic signal, not an expert
no grouped-axis variant clears the same-input global control
no remote launch is justified from this exact contiguous-group mean variant
```

Updated implication:

```text
do_not_expand_v1_contiguous_group_mean_projection
do_not_run_remote_or_seed1_for_this_exact_v1_screen
preserve grouped tooling for future structure-aware residual summaries
next local-only route should change the representation, not merely the number
of contiguous axes averaged together
```

The first valid use is after watcher retrieval:

```text
1. postprocess 262144/class trail-position score artifacts
2. export train/validation bit-sensitivity features aligned to those artifacts
3. select grouped masks on train only
4. apply grouped frozen scorer on validation only
5. run postprocess-bit-sensitivity-projection against global and trail anchors
```

For the next variant, the hypothesis should move from simple contiguous group
means to structural residual summaries: for example, per-depth/per-cell
residual response maps, compressed multi-pair block statistics, or explicit
trail-family residual features. Those are more aligned with the literature
than adding more weak projection scorers to an ensemble.

## 2026-07-07 V2 Structural-Stats Projection Screen

The next variant implements that representation shift directly:

```text
feature_view = trail_position_stats
raw_input_bits = 39936
structural_stats_bits = 3708
selection = train split only
scoring = held-out validation split only
```

Instead of selecting isolated raw bits or contiguous raw columns, the selector
works on deterministic per-depth/per-word/per-cell statistics already used by
the strongest trail-position model family. This is still a frozen projection,
not a trained network.

2048/class local result:

```text
seed0 global_control_auc = 0.8542919158935547
seed0 structural_stats_projection_auc = 0.9096775054931641
seed0 trail_position_anchor_auc = 0.9985876083374023
seed0 margin_vs_global_auc = +0.055385589599609375
seed0 error_jaccard_with_anchor = 0.057441253263707574

seed1 global_control_auc = 0.8728437423706055
seed1 structural_stats_projection_auc = 0.9378452301025391
seed1 trail_position_anchor_auc = 0.9982948303222656
seed1 margin_vs_global_auc = +0.0650014877319336
seed1 error_jaccard_with_anchor = 0.08157099697885196

both decisions = projection_expert_ready_for_local_screen
```

Updated interpretation:

```text
V0 raw-axis projection failed.
V1 contiguous raw-axis group means remained weak.
V2 structural-stat projection clears the same-input global control on seed0
and seed1.
```

This is the strongest evidence so far for the non-neighbor projection idea,
but the claim is still narrow: two-seed local diagnostic only. The low
error-overlap pattern survives seed1, but naive frozen-score aggregation still
does not beat the best single trail-position expert:

```text
seed0 best_ensemble_delta_vs_best_single_auc = -0.001667022705078125
seed1 best_ensemble_delta_vs_best_single_auc = -0.001583099365234375
```

So the route should not be sold as "many networks already improve the best
model." The real result is better: a structurally different projection expert
now exists locally and clears strong controls on two seeds. The next work is to
test whether it survives larger retrieved trail-position artifacts and whether
more careful stacking/calibration can exploit its low-overlap errors.

That calibration question has now been tested once with a train-fitted
logistic stacking diagnostic:

```text
cli = scripts/evaluate-stacked-ensemble
fit_split = train frozen-score artifacts
eval_split = held-out validation frozen-score artifacts
feature_space = logits

seed0 stacked_validation_auc = 0.998295783996582
seed0 best_single_validation_auc = 0.9985876083374023
seed0 delta_stacked_vs_best_single_auc = -0.0002918243408203125
seed0 delta_stacked_vs_fixed_ensemble_auc = +0.0013751983642578125

seed1 stacked_validation_auc = 0.9975728988647461
seed1 best_single_validation_auc = 0.9982948303222656
seed1 delta_stacked_vs_best_single_auc = -0.0007219314575195312
seed1 delta_stacked_vs_fixed_ensemble_auc = +0.0008611679077148438

both decisions = stacked_ensemble_diagnostic_no_best_single_gain
```

This is the right kind of ensemble test because the weights are fitted on the
train split and scored on held-out validation artifacts. It improves over the
naive fixed ensemble, so calibration matters. It still does not beat the best
single trail-position expert, so the current result remains a diagnostic for a
real non-neighbor representation, not evidence that multi-network aggregation
has already improved the strongest local candidate.

A follow-up train-holdout selection diagnostic then selected calibration
settings only on a deterministic holdout carved from the train artifacts:

```text
train_holdout_fraction = 0.25
candidate_feature_spaces = logits, probabilities
candidate_l2 = 0.0, 0.0001, 0.001, 0.01
candidate_standardize = both

seed0 selected = logits, l2=0.0, standardize=false
seed0 stacked_validation_auc = 0.9985494613647461
seed0 best_single_validation_auc = 0.9985876083374023
seed0 delta_stacked_vs_best_single_auc = -0.00003814697265625

seed1 selected = probabilities, l2=0.0, standardize=false
seed1 stacked_validation_auc = 0.9984188079833984
seed1 best_single_validation_auc = 0.9982948303222656
seed1 delta_stacked_vs_best_single_auc = +0.0001239776611328125
```

This is an improvement over the default fixed stacking diagnostic, but it is
still mixed evidence: one seed nearly ties and one seed improves. It justifies
keeping train-side calibration selection in the toolbox for the retrieved
262144/class artifacts, but it still does not justify a remote launch or an
ensemble-success claim by itself.

A five-selection-seed stability sweep strengthened the interpretation:

```text
stability_cli = scripts/summarize-stacked-selection
selection_seeds = 0, 1, 2, 3, 4

seed0 delta_vs_best_single range =
  [-0.0000400543212890625, -0.00003814697265625]
seed0 positive_selection_seeds = 0 / 5

seed1 delta_vs_best_single range =
  [+0.0001239776611328125, +0.0001239776611328125]
seed1 positive_selection_seeds = 5 / 5

decision = stable_but_mixed_train_holdout_stacking_diagnostic
```

The route-level gate now combines those per-seed stability summaries instead
of letting one seed dominate the story:

```text
route_stability_cli = scripts/summarize-stacked-route
strict_route_report =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_trail_stats_stacking_route_stability_strict.json
relaxed_route_report =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_trail_stats_stacking_route_stability_relaxed.json

strict passed_seed_count = 0 / 2
relaxed passed_seed_count = 1 / 2
relaxed positive_seed_fraction = 0.5
relaxed delta_mean_vs_best_single_auc range =
  [-0.0000385284423828125, +0.0001239776611328125]

decision = stable_but_mixed_cross_seed_stacking_diagnostic
```

This makes the route more interesting, not more publishable: train-side
calibration selection is stable within each local seed, but the direction still
does not clear the two-seed same-protocol improvement gate. The useful progress
is methodological: the project now has a reusable cross-seed gate for future
frozen-score aggregation, so a future 262144/class result must pass the same
train-only selection discipline and the same route-level gate before it can be
called a multi-network improvement.

The next local iteration tested a different question: instead of freezing a
deterministic projection mask or stacking existing score artifacts, can the
compressed SPN structural-stat feature view itself train a tiny expert?

```text
cli = scripts/fit-compressed-feature-expert
feature_view = trail_position_stats
feature_count = 3708
fit_split = train feature artifacts
score_split = held-out validation feature artifacts
model = logistic
decision = compressed_feature_expert_local_screen_positive_needs_controls

seed0 validation_auc = 1.0
seed0 validation_accuracy = 0.99951171875
seed0 calibrated_validation_accuracy = 1.0

seed1 validation_auc = 1.0
seed1 validation_accuracy = 0.99951171875
seed1 calibrated_validation_accuracy = 1.0
```

A train-label shuffle control was added and run through the same CLI:

```text
control = --shuffle-train-labels --shuffle-seed 0
control_decision = compressed_feature_expert_shuffle_train_labels_control

seed0 control_validation_auc = 0.4979085922241211
seed0 control_calibrated_validation_accuracy = 0.51904296875

seed1 control_validation_auc = 0.5246953964233398
seed1 control_calibrated_validation_accuracy = 0.5322265625
```

This is a strong positive local signal for the representation itself: when the
tiny logistic expert sees the real train labels, the held-out validation split
is separated almost perfectly; when the fit labels are shuffled, the signal
collapses to near random. That is better evidence than the raw deterministic
projection, but it should be interpreted narrowly. It is still a 2048/class
local diagnostic with 1024/class validation score artifacts, not remote
evidence and not formal SPN/PRESENT evidence.

It also is not a multi-network improvement. Re-running the aligned frozen-score
comparison with global control, trail-position anchor, deterministic
structural-stat projection, and the compressed logistic expert gives:

```text
seed0 best_single = compressed_feature_logistic_expert
seed0 best_single_auc = 1.0
seed0 best_ensemble_auc = 1.0
seed0 delta_best_ensemble_vs_single_auc = 0.0

seed1 best_single = compressed_feature_logistic_expert
seed1 best_single_auc = 1.0
seed1 best_ensemble_auc = 0.999995231628418
seed1 delta_best_ensemble_vs_single_auc = -0.00000476837158203125
```

The route-level gate now records that interpretation as a machine-readable
decision:

```text
route_gate_cli = scripts/summarize-compressed-feature-expert
route_gate =
  outputs/local_audits/i1_present_r8_bit_sensitivity_projection_2048_trail_stats_compressed_expert_route_gate.json

normal_passed_seed_count = 2 / 2
shuffle_control_passed_seed_count = 2 / 2
ensemble_gain_passed_seed_count = 0 / 2
decision = compressed_feature_local_positive_controls_pass_not_ensemble_gain
```

The follow-up sparsity audit ranks dimensions using only train-split labels and
then scores sparse logistic experts on held-out validation:

```text
sparsity_cli = scripts/audit-compressed-feature-sparsity
ranking = train-only abs(class_mean_difference) / train_std

seed0 validation_auc_by_top_k:
  k=1   0.6603231430053711
  k=16  0.8615026473999023
  k=64  0.966217041015625
  k=128 0.9928836822509766
  k=256 0.9996299743652344

seed1 validation_auc_by_top_k:
  k=1   0.639317512512207
  k=16  0.8639602661132812
  k=64  0.9727020263671875
  k=128 0.995387077331543
  k=256 0.9996557235717773

both decisions = sparse_compressed_feature_local_screen_positive
```

This changes the route intuition. The compressed structural-stat signal is not
explained by one or two dominating scalar features; those are only weak
positive. It becomes strong around 128 train-ranked structural dimensions and
nearly saturated by 256. That points toward a medium-sparse SPN-structural
expert: preserve grouped trail-position/statistic structure and learn compact
combinations, rather than searching for a single magic statistic or averaging
more near-neighbor networks.

The sparse-feature decoder makes the medium-sparse signal more concrete:

```text
decode_cli = scripts/decode-compressed-feature-sparsity

seed0 top-256:
  depth_word_cell_span = 108
  depth_cell_span = 64
  word_span = 34
  depth_word_span = 33
  cell_span = 15
  depths = 44 / 51 / 54 / 56 across depth0..depth3

seed1 top-256:
  depth_word_cell_span = 119
  depth_cell_span = 61
  word_span = 30
  depth_word_span = 29
  cell_span = 15
  depths = 54 / 48 / 54 / 53 across depth0..depth3
```

That is the most useful architecture hint from this local diagnostic. The
strong dimensions are mostly span-type structural statistics, not raw bit
columns, isolated S-box cells, or one trail depth. The next model should act
like a grouped span/statistic-aware SPN expert: keep family, depth, word, and
cell structure visible to the network, then learn compact combinations across
those groups. This is more aligned with the evidence than either a single
hand-written scalar or a larger average of near-neighbor neural models.

A direct span-family restricted expert confirms that this is not just a
post-hoc reading of the top-256 sparse indices:

```text
span_family_cli = scripts/fit-compressed-feature-expert
span_family_filter = depth_word_cell_span + depth_cell_span + word_span +
  depth_word_span + cell_span
selected_feature_count = 731 / 3708

seed0 validation_auc = 0.9999723434448242
seed0 shuffle_train_labels_auc = 0.4802093505859375

seed1 validation_auc = 0.9999513626098633
seed1 shuffle_train_labels_auc = 0.47839784622192383
```

This makes the next architectural move sharper. A span/statistic-aware expert
should not start from all 3708 flat dimensions equally; it should explicitly
group span families, preserve depth/word/cell coordinates, and then learn
compact cross-group interactions. The result is still a local diagnostic, but
it is a better design signal than either the full compressed logistic expert or
near-neighbor network averaging.

The family attribution diagnostic narrows the design further:

```text
span_family_attribution_cli = scripts/audit-compressed-feature-families

single-family validation AUC:
  seed0 depth_word_cell_span = 0.9999923706054688
  seed1 depth_word_cell_span = 0.999969482421875
  seed0 depth_cell_span = 0.9865589141845703
  seed1 depth_cell_span = 0.9866390228271484
  seed0 word_span / depth_word_span / cell_span =
    0.9422855377197266 / 0.9392290115356445 / 0.8775739669799805
  seed1 word_span / depth_word_span / cell_span =
    0.9360342025756836 / 0.9317569732666016 / 0.8646869659423828

leave-out depth_word_cell_span validation AUC:
  seed0 = 0.9918107986450195
  seed1 = 0.9919548034667969
```

So the next grouped expert should have a clear backbone: depth_word_cell_span
is the primary channel. The other span families are not useless, because the
leave-one-out run without depth_word_cell_span still clears about 0.992 AUC,
but they should be modeled as auxiliary context or regularized lower-rank
channels. This is a sharper architecture hypothesis than "use all span features"
or "combine several networks".

The generic diverse-expert gate can mark the pool ready because there are now
multiple aligned families, but route-specific interpretation must be stricter:
the compressed logistic expert is probably a stronger same structural-stat
route, not yet proof that genuinely different networks combine to improve the
best expert. The next useful control is to carry this compressed expert into
the retrieved 262144/class artifacts only after those artifacts are complete,
and to add mismatch/permutation controls before any remote promotion.

Promotion remains hard:

```text
both seeds must clear same-input global control
both seeds must be weak-positive
error overlap with trail-position must be low for the right reason
mismatch controls must stay near random
```

## Activation Gate

Do not implement or launch this as a meaningful experiment until the active
trail-position 262144/class postprocess reaches one of these states:

```text
support_trail_position_score_residual_all_runs
mixed_trail_position_score_residual
hold_trail_position_score_residual_due_high_overlap
```

If the 262144/class artifacts are still missing, stale, or unverified, the only
allowed actions are documentation, local tooling, or tests that do not touch the
active remote runs.

## Local Screen Gate

The first executable step must be a local diagnostic screen, not a remote
scale-up:

```text
rounds = 8
samples_per_class <= 2048
pairs_per_sample = 16
sample_structure = plaintext_integral_nibble_difference_matched_negative
negative_mode = encrypted_random_plaintexts
train_key = 0x00000000000000000000
validation_key = 0x11111111111111111111
same_budget_baseline = present_pairset_global_stats
anchor = present_trail_position_stats_pairset
```

Advance only if the projection expert:

```text
beats same-input global control on both seeds
is weak-positive on both seeds
has lower error overlap with trail-position than the global control does
does not pass active-nibble or input-difference mismatch controls
exports compatible frozen scores
```

Hold immediately if it is a single-seed spike, loses to the same-input global
control, or simply duplicates the trail-position error set.

## Claim Scope

Allowed:

```text
conditional non-neighbor expert route
local diagnostic screen plan
representation-search hypothesis
```

Not allowed:

```text
do_not_launch_remote_now
do_not_claim_formal_spn_present_evidence
do_not_claim_breakthrough
do_not_claim_diverse_ensemble_ready
do_not_treat_262144_class_as_formal_evidence
```

## Next Action

Wait for the active 262144/class trail-position artifacts. When they are ready,
run score postprocess first, then decide:

```text
pass with low overlap -> prepare 1M trail-position confirmation first
pass but high overlap -> prioritize this projection expert as an explanation/diversity screen
mixed/weak -> use sensitivity analysis diagnostically, not as remote launch basis
```

## 2026-07-07 Literature-Integrated Next Route

The latest local evidence and the external literature point to the same next
research shape:

```text
not = wider pool of near-neighbor neural models
yes = better SPN evidence presentation, then calibrated aggregation
```

Relevant external pressure:

```text
Zhang/Wang 2022 PRESENT/DES/Chaskey work:
  derived multi-ciphertext-pair features and PRESENT-specific kernel choices
  matter for PRESENT distinguishers.

Neural-inspired integral cryptanalysis:
  neural models are useful as feature explorers for integral/SPN properties,
  so the right abstraction is often the state/data view, not just model size.

LLM-neural-distinguisher negative signal:
  a very different model family does not automatically improve a strong
  neural distinguisher; prompt/data representation carries much of the value.
```

Current local pressure:

```text
V0 raw-axis bit projection -> rejected
V1 contiguous raw-axis group projection -> rejected
V2 trail-position structural-stat projection -> clears same-input global
control on seed0 and seed1
train-fitted stacking -> improves over fixed ensemble but still loses to
best single trail-position anchor on both seeds
```

Therefore the next method-level route is:

```text
SPN compressed evidence pooling
```

This means converting the trail-position/bit-sensitivity signal into a compact
state-cell evidence table that can support more samples or more pair evidence
without simply concatenating more raw bits. The candidate should reuse
structural summaries already shown to matter:

```text
per depth / active cell
per word / P-layer position
S-box DDT-consistent beam statistics
train-only sensitivity-selected structural axes
optional pair-count or cell-count pooling summaries
```

The first implementation must be a local tooling/diagnostic step only while the
active 262144/class remote run is incomplete. Do not launch a new remote branch
or spend a GPU slot until the existing watcher retrieves both seed score
artifact sets.

Concrete next gate after 262144/class artifacts are ready:

```text
1. postprocess trail-position seed0/seed1 score artifacts
2. classify the anchor as pass / mixed / high-overlap hold
3. if pass or high-overlap hold, reuse the retrieved train/validation score
   artifacts to export the V2 structural-stat projection at the same scale
4. fit stacking only on train artifacts
5. score only on held-out validation artifacts
6. require improvement over best single anchor before calling aggregation useful
```

Do not run a validation hyperparameter sweep and call the best variant a
research gain. Any calibration choice beyond the current default should be
chosen by a train-side rule or a nested split before reporting validation AUC.

Promotion rule:

```text
promote_to_medium_scale_followup only if:
  same-budget global control is cleared
  deterministic/mismatch controls remain non-explanatory
  frozen scores are compatible
  error overlap with trail-position is low or explains a real residual
  train-fitted aggregation beats the best single validation AUC

otherwise:
  keep as representation diagnostic or discard
```

## 2026-07-07 Structured Span-Block Follow-Up

The span-family attribution result makes the next step more architectural than
ensembling. The useful local structural-stat signal is organized around:

```text
primary_backbone = depth_word_cell_span
auxiliary_context = depth_cell_span + word_span + depth_word_span + cell_span
```

That is different from "add several weak networks." It says the feature
coordinates themselves matter: depth, P-layer trail word, state cell, and
span/range statistic type should be preserved as model input structure.

The concrete tooling step is:

```text
cli = scripts/export-compressed-span-blocks
kind = compressed_spn_span_blocks
input = trail_position_stats features.npy / labels.npy / sample_ids.npy / metadata.json
outputs =
  depth_word_cell_span.npy [rows, depth, trailword, cell]
  depth_cell_span.npy [rows, depth, cell]
  word_span.npy [rows, word]
  depth_word_span.npy [rows, depth, trailword]
  cell_span.npy [rows, cell]
```

The exporter also writes a compact summary feature artifact:

```text
route_summary_cli = scripts/summarize-compressed-span-route
route_summary_decision = compressed_span_summary_retains_flat_signal_controls_pass
summary_feature_view = compressed_span_summary
summary_feature_count = 273
flat_span_feature_count = 731
feature_reduction_ratio = 0.3734610123119015
model = compressed_span_summary_logistic_expert
seed0 validation_auc = 0.9999141693115234
seed1 validation_auc = 0.9998435974121094
seed0 shuffle_train_labels_validation_auc = 0.5048818588256836
seed1 shuffle_train_labels_validation_auc = 0.4824662208557129
max_auc_drop_vs_flat_span = 0.00010776519775390625
```

That is the important bit: a 273-dimensional SPN-coordinate summary retains
nearly all local span-family signal while shuffled-label controls remain near
random. This does not add remote or formal evidence, but it changes the local
architecture hypothesis from "use all span columns" to "learn over compact
grouped span channels."

The primary/auxiliary prefix attribution sharpens the architecture split:

```text
filter_cli = scripts/fit-compressed-feature-expert --include-feature-prefix
primary_prefix = primary_
auxiliary_prefix = aux_
primary_feature_count = 133
auxiliary_feature_count = 140
seed0 primary_validation_auc = 0.9997234344482422
seed1 primary_validation_auc = 0.9992923736572266
seed0 auxiliary_validation_auc = 0.9964427947998047
seed1 auxiliary_validation_auc = 0.9976606369018555
primary_shuffle_validation_auc = 0.45572566986083984 / 0.5348129272460938
auxiliary_shuffle_validation_auc = 0.5421538352966309 / 0.5033082962036133
```

This means the `depth_word_cell_span`-derived primary channel is the backbone,
but the auxiliary context is not dead weight. The next learned model should not
spend capacity equally across all 273 features; it should privilege the primary
channel and use the auxiliary channels as lower-rank context/residual features.

The purpose is to turn the flat 731 selected span dimensions into SPN-coordinate
tensors so the next candidate can be a grouped span/statistic-aware expert:

```text
1. learn a compact primary channel over depth_word_cell_span
2. add lower-rank context channels for depth_cell/word/depth_word/cell spans
3. compare against the same frozen-feature logistic anchor and shuffle-label
   control at the same local budget
4. only consider remote scale after active 262144/class trail-position
   artifacts are retrieved and the local control gate still supports it
```

The first grouped expert is intentionally small:

```text
cli = scripts/fit-compressed-span-grouped-expert
model = compressed_span_grouped_logistic_expert
decision = compressed_span_grouped_expert_local_screen_positive_needs_controls
feature_model = two_branch_logistic
branch_inputs = primary_logit + auxiliary_logit
fit_split = train
score_split = validation
seed0 grouped_validation_auc = 0.9997968673706055
seed1 grouped_validation_auc = 0.9996414184570312
seed0 primary_branch_validation_auc = 0.9997234344482422
seed1 primary_branch_validation_auc = 0.9992923736572266
seed0 auxiliary_branch_validation_auc = 0.9964427947998047
seed1 auxiliary_branch_validation_auc = 0.9976606369018555
seed0 full_summary_validation_auc = 0.9999141693115234
seed1 full_summary_validation_auc = 0.9998435974121094
```

This result says something useful but narrow. The grouped `two_branch_logistic`
combiner beats both single branches on both local seeds, so the auxiliary
context is not decorative. The full 273-dimensional summary logistic still
wins by about `0.00012-0.00020` AUC, so the grouped expert is not the current
best scorer. Its value is that it converts a flat feature success into an
interpretable architecture pattern: primary SPN span backbone plus
lower-rank auxiliary residual context.

The obvious next question was whether finer semantic branch logits improve the
two-branch grouped result. They do not:

```text
semantic_feature_model = semantic_group_logistic
hybrid_feature_model = hybrid_group_logistic
seed0 semantic_group_validation_auc = 0.998713493347168
seed1 semantic_group_validation_auc = 0.9978828430175781
seed0 semantic_l2zero_validation_auc = 0.9994039535522461
seed1 semantic_l2zero_validation_auc = 0.9988164901733398
seed0 hybrid_group_validation_auc = 0.9992799758911133
seed1 hybrid_group_validation_auc = 0.9986753463745117
decision = semantic_or_hybrid_branch_logit_decomposition_hold
```

This is a useful negative result. The semantic groups themselves are
meaningful, but compressing each group into one prefit branch logit throws away
too much within-group information. The better next model should consume raw
group tensors or group-level raw summaries with structured regularization,
not a larger stack of prefit branch logits.

The raw-interaction follow-up is the first small positive step in that
direction:

```text
cli = scripts/fit-compressed-span-interaction-expert
feature_model = raw_plus_primary_auxiliary_interactions_logistic
raw_feature_count = 273
top_primary = 8
top_auxiliary = 8
interaction_count = 64
total_feature_count = 337
seed0 interaction_validation_auc = 0.9999170303344727
seed1 interaction_validation_auc = 0.9998636245727539
seed0 full_summary_validation_auc = 0.9999141693115234
seed1 full_summary_validation_auc = 0.9998435974121094
seed0 interaction_shuffle_validation_auc = 0.5185546875
seed1 interaction_shuffle_validation_auc = 0.48958492279052734
decision = raw_interaction_summary_tiny_positive_controls_pass_local
```

The result is tiny but directionally important. It keeps the full raw
span-summary vector, adds only train-selected primary/auxiliary interaction
terms, and beats the full linear summary anchor on both local seeds while
shuffle-label controls stay near random. This supports the next architectural
move: preserve raw SPN-coordinate group information and add structured
cross-group interactions, instead of compressing each group into a single logit.

The stricter train-internal holdout gate makes that conclusion more cautious:

```text
cli = scripts/fit-compressed-span-interaction-expert
added_gate = --selection-holdout-fraction 0.5 --selection-seed
selection_fit_split_mode = train_internal_holdout
interaction_selection_rows = 2048
fit_rows = 2048
seed0 holdout_interaction_validation_auc = 0.9998722076416016
seed1 holdout_interaction_validation_auc = 0.9998645782470703
seed0 delta_vs_full_summary_auc = -0.000041961669921875
seed1 delta_vs_full_summary_auc = 0.0000209808349609375
seed0 delta_vs_original_interaction_auc = -0.00004482269287109375
seed1 delta_vs_original_interaction_auc = 0.00000095367431640625
seed0 holdout_shuffle_validation_auc = 0.49273252487182617
seed1 holdout_shuffle_validation_auc = 0.5002899169921875
decision = raw_interaction_holdout_mixed_local_diagnostic
```

This means the exact flat cross-product bank should not be promoted as the next
remote-scale candidate by itself. The signal is still useful because shuffle
controls are random and seed1 survives the stricter split, but the mixed seed
outcome says the better next move is a block-preserving/raw group tensor
interaction model with explicit regularization, not more prefit branch logits
and not premature multi-network aggregation.

The first block-preserving follow-up used semantic primary/auxiliary groups
instead of train-selected individual feature pairs:

```text
cli = scripts/fit-compressed-span-block-interaction-expert
feature_model = raw_plus_semantic_block_interactions_logistic
raw_feature_count = 273
primary_group_count = 6
auxiliary_group_count = 6
block_pair_count = 36
block_interaction_stat_count = 4
block_interaction_feature_count = 144
total_feature_count = 417
seed0 block_interaction_validation_auc = 0.9999065399169922
seed1 block_interaction_validation_auc = 0.999908447265625
seed0 delta_vs_full_summary_auc = -0.00000762939453125
seed1 delta_vs_full_summary_auc = 0.000064849853515625
seed0 delta_vs_strict_flat_holdout_auc = 0.000034332275390625
seed1 delta_vs_strict_flat_holdout_auc = 0.0000438690185546875
seed0 block_interaction_shuffle_validation_auc = 0.5331153869628906
seed1 block_interaction_shuffle_validation_auc = 0.5094890594482422
l2_0.003_seed0_auc = 0.9999065399169922
l2_0.003_seed1_auc = 0.9999074935913086
l2_0.01_seed0_auc = 0.9998979568481445
l2_0.01_seed1_auc = 0.9999008178710938
decision = semantic_block_interaction_mixed_local_diagnostic
```

This is a better direction than the hand-selected flat interaction bank, but
not yet a stronger candidate. The block features are label-independent and
preserve semantic SPN coordinates, and they beat the strict flat-interaction
holdout on both seeds. However, they still do not clear the full-summary anchor
on seed0, and the seed0 shuffle-label control is mildly high at `0.5331`. The
next model should therefore keep the block/tensor idea but learn interactions
with a real block-preserving network or lower-rank constrained interaction
module, rather than simply adding more aggregate product statistics.
