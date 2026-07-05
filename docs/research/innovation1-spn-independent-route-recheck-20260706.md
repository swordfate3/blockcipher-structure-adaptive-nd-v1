# Innovation 1 SPN Independent Route Recheck

**Date:** 2026-07-06

**Status:** literature-and-local-evidence route correction; updated after SGP hold

## 2026-07-06 Update After SGP Audits

The proposed SGP route below has now been locally tested and held:

```text
raw-axis SGP audit = sgp_stable_axis_hold
grouped/orbit SGP audit = sgp_grouped_axis_hold
```

The strongest grouped signal was broad `word_bit_role` aggregation, with
`min composite AUC = 0.685741662979126`, but the top-k Jaccard was only
`0.14285714285714285`. Pair-slot-aware and P-layer orbit groups were less
stable (`0.0` Jaccard). This is not a projection-mask candidate.

Updated route priority:

```text
1. targeted group-distribution statistics over InvP(delta)
2. SPN-aware architecture only if it consumes those stable statistics directly
3. diverse expert aggregation only after a non-neighbor weak-positive artifact exists
4. SGP projection route is held unless a new stable source family appears
```

This does not mean InvP(delta) is useless. It means the weak evidence is broad
and distributional rather than localized to stable axes/groups. The next useful
experiment should treat it as a representation/statistics problem, not as a
mask-selection problem.

The existing `present_global_pairset_statistics` branch was also tested after
this update and held:

```text
invp_global_stats_audit = invp_global_stats_hold
best stat AUC min = 0.5180071592330933
composite AUC min = 0.5185081958770752
top-k Jaccard min = 0.06666666666666667
```

So the next statistics attempt should not be generic global activity. It should
preserve the grouped-SGP evidence more directly: group activities by
`cell`, `word_cell`, `word_bit_role`, and `p_layer_orbit`, then audit
distribution variance/span/top-k means and pair-slot consistency.

That targeted group-distribution attempt was also tested and held:

```text
invp_group_distribution_audit = invp_group_distribution_hold
best stat AUC min = 0.514545202255249
composite AUC min = 0.5135400295257568
top-k Jaccard min = 0.18518518518518517
```

Updated implication: deterministic aggregation around InvP(delta) should stop
for now. The grouped-SGP signal appears too weak and label-dependent for simple
handwritten statistics. The next route should either use a learned
pair/group-interaction representation with a same-budget local smoke, or move
to data/difference search instead of adding more handcrafted aggregate stats.

## Decision

Do not spend the next meaningful slot on:

```text
wider near-neighbor ensembles
SGP projection masks from the tested sources
more deterministic InvP(delta) aggregate statistics
renamed single-step Sinv/GPD features
the completed r9 difference screen
direct r8 pair-set scaling under the current protocol
```

The strongest still-supported route is the r7 InvP/P-layer aligned SPN view:

```text
route = present_nibble_invp_only_spn_only
evidence = two_seed_1000000_class_positive_with_attribution_control
seed0 AUC = 0.797470988906
seed1 AUC = 0.797347588554
max attribution-control AUC = 0.793621524954
```

The route-selection question has changed. It is no longer "which SGP mask do we
train?" or "how many networks do we aggregate?" The better question is:

```text
Which SPN representation or data construction can create a non-neighbor,
weak-positive, controllable expert beyond the already-supported r7 InvP-only
anchor?
```

## Why This Beats The Wider-Ensemble Default

The user's diverse-network idea remains valid, but the current evidence does
not justify making it the main next experiment. A real ensemble needs at least
one compatible, weak-positive, low-correlation non-neighbor expert. Current
candidate families are not there yet:

| Family | Current local/remote status | Route decision |
|---|---|---|
| `raw_mcnd` / `invp_cell` / `ddt_graph` | r7 near-neighbor ensemble improved AUC by only `+0.000949076843`, below the `+0.001` gate | Useful control, not the next main route |
| `ddt_graph` | two 262144/class seeds weak-positive but below no-DDT margin gate | Keep as weak diagnostic, not scale alone |
| `topology_aware_network` | seed1 gate stopped the route | Do not add another topology-only neighbor |
| `candidate_trail` | medium 262144/class stopped; local low-dimensional axes are weak-positive but unstable | Use for score-axis mining only |
| `gpd_beamstats` | local weak/unstable; beamstats AUC about `0.52-0.57` in recent diagnostics | Keep as candidate score source only |
| `projection_feature` | 65k weak hold; v2 local priors unstable across seeds | Do not continue hand-picked masks |
| `integral_active_difference` | neural follow-up stayed below fixed deterministic `pair_xor_column_sum_variance` baseline | Deterministic baseline, not a neural expert yet |

So the bottleneck is not "number of networks." The bottleneck is that the
available non-neighbor signals are too weak or too correlated to combine.

Additional completed gates now narrow the search:

| Route | Latest retrieved/local status | Decision |
|---|---|---|
| `r8_pairset_scale` | `1000000/class` seed0 pair-set row AUC `0.514032233534`, below r8 baseline AUC `0.554962712376` | stop or rethink current r8 pair-set scale |
| `r9_difference_screen` | `65536/class` seed0 screen ended `all_candidates_near_random_stop_difference_screen` | stop this screen; do not promote a difference |
| `r9_curriculum` | `262144/class` seed0 best candidate AUC `0.5018549287342466` | stop current curriculum route |
| `invp_group_distribution` | deterministic group-distribution audit composite AUC min `0.5135400295257568` | stop deterministic InvP aggregation |

## External Evidence Signal

The local paper cache and prior web refresh point in the same direction:

1. Zhang/Wang's PRESENT work and later multi-ciphertext-pair work emphasize
   derived data formats, not only model replacement.
2. Benamira et al.'s analysis shows neural distinguishers can rely on
   penultimate and antepenultimate-round distributions, motivating partial
   decryption and masked distribution views.
3. GPD-style work supports partial-decryption feature engineering, but in this
   repository the single-step `S^{-1}(P^{-1}(C))` semantics already exist as
   `present_pair_xor_paligned_sinv_cell_matrix_bits`; repeating it under a new
   name would not be a new method.
4. Recent SPN-oriented SKINNY/MIDORI work emphasizes state formatting,
   selective inverse substitution, and architecture matched to SPN state
   layout.
5. Score-distribution, bit-selection, and multi-pair lines all suggest that
   choosing what evidence reaches the model is at least as important as adding
   model capacity.

This now supports a route that searches for a cleaner representation source
before model scaling or ensemble aggregation.

## Next Route Arbitration

Ranked local actions:

| Priority | Route | Why | Gate |
|---:|---|---|---|
| 1 | learned pair/group-interaction diagnostic on a controlled SPN representation | Handwritten InvP statistics failed, but the literature and r7 evidence still support learned SPN-state representations. This must be a small diagnostic, not a remote launch. | Beat the same-input InvP-only or explicit deterministic baseline; include shuffled/topology or single-pair control. |
| 2 | cross-SPN transfer sanity for GIFT/SKINNY-style cell encoders | External SPN work emphasizes PRESENT/SKINNY/GIFT-style state formatting. The repository already has `Gift64`; a tiny smoke can test whether the route is PRESENT-specific before more PRESENT-only engineering. | Only proceed if the cipher implementation, difference profile, and strict negative path are validated locally. |
| 3 | frozen diverse expert pool | Keep the user's multi-network idea, but only after a non-neighbor expert has compatible scores and low error overlap. | Require weak-positive quality and diversity gates before aggregation. |

Non-actions:

```text
do_not_launch_remote_now
do_not_expand_near_neighbor_ensemble_now
do_not_continue_sgp_projection_from_current_sources
do_not_add_more_deterministic_invp_aggregate_stats
do_not_repeat_current_r9_difference_screen
```

The immediate implementation slot should therefore be a plan-and-smoke for one
small, controlled representation diagnostic. If that smoke is negative, the
route should be discarded quickly; if it is positive, only then prepare a
medium diagnostic ladder.

## Claim Scope

This is a route-selection note only. It does not claim a new PRESENT r8/r9
result, does not claim a breakthrough, and does not reinterpret local
projection, beamstats, SGP, difference-screen, or pair-set probes as formal
evidence.

## 2026-07-06 Trail-Position Split-Baseline Update

The independent route check should now rank the r8 trail-position beamstats
route ahead of broader near-neighbor ensembling, but only with a deterministic
control attached.

New local split-baseline evidence:

```text
audit = present_trail_position_split_baseline
selection_split = train
evaluation_split = validation
feature = present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits
sample_structure = plaintext_integral_nibble_difference_matched_negative
negative_mode = encrypted_random_plaintexts
```

| Scale | Seed | Validation composite AUC | Validation best accuracy |
|---:|---:|---:|---:|
| 512/class | 0 | `0.7695465087890625` | `0.703125` |
| 512/class | 1 | `0.8455047607421875` | `0.7734375` |
| 2048/class | 0 | `0.8056130409240723` | `0.735595703125` |
| 2048/class | 1 | `0.8421728610992432` | `0.766845703125` |

Decision:

```text
promote_trail_position_to_controlled_local_candidate
do_not_remote_launch_yet
do_not_call_neural_architecture_gain_without beating split deterministic baseline
```

Independent route ranking after this update:

| Rank | Route | Decision |
|---:|---|---|
| 1 | SPN/integral position-aware representation with deterministic split controls | Continue locally. The signal survives train-key selection and validation-key evaluation, and it is structurally different from raw/InvP/DDT near-neighbor variants. |
| 2 | Neural residual over trail-position statistics | Test only if the deterministic baseline is included in the same protocol. The 512/class neural candidate still exceeds the deterministic split baseline, so residual nonlinear value is plausible but unproven. |
| 3 | Diverse expert aggregation | Keep as a validator after a non-neighbor expert exists. Do not mechanically aggregate similar raw/InvP/DDT/trail-stat variants before score compatibility and error-overlap checks. |
| 4 | Generic model replacement or wider near-neighbor ensemble | Hold. External evidence and local gates favor representation/data construction over piling nearby models. |

This is a stronger answer to the user's correction: the project should not
simply follow the user's current ensemble suggestion, nor should it blindly
trust the current high neural AUC. The best current path is a controlled
SPN-aware representation route, with multi-network aggregation reserved for
later diversity validation.

### 512/Class Control Baseline Update

The first deterministic control suite was run for the trail-position route at
`512/class` on seeds `0` and `1`:

```text
baseline = train-selected deterministic position-statistics split baseline
controls = active_nibble_1, input_difference_0x90, pair_order_reverse
feature = present_delta_paligned_sinv_sboxddt_beamstats8deep4_cell_matrix_bits
sample_structure = plaintext_integral_nibble_difference_matched_negative
```

| Seed | Baseline AUC | Active-nibble control AUC | Difference control AUC | Pair-order reverse AUC |
|---:|---:|---:|---:|---:|
| 0 | `0.7695465087890625` | `0.49724578857421875` | `0.5139007568359375` | `0.7695465087890625` |
| 1 | `0.8455047607421875` | `0.49993133544921875` | `0.5224685668945312` | `0.8455047607421875` |

Route implication:

```text
The signal is specific to active-nibble/input-difference alignment and is not
just generic matched-negative integral leakage. However, it is order-invariant
under pair reversal, so pair-order sequence modeling is not the current
bottleneck.
```

Updated priority:

| Rank | Route | Decision |
|---:|---|---|
| 1 | Active/difference-aligned SPN position statistics | Continue locally with deterministic baseline and mismatch controls. |
| 2 | Neural residual over order-invariant span/range statistics | Test only against the split baseline; require active/difference mismatch controls. |
| 3 | Pair-order-sensitive trail models | Hold unless a new statistic shows pair-order sensitivity. Current selected features are span/range dominated. |
| 4 | Diverse expert aggregation | Still later. Require compatible frozen scores plus low error overlap before aggregation. |

## References Used

- `docs/research/innovation1-spn-adaptation-literature-refresh-20260705.md`
- `docs/research/spn_structured_nn_research_plan.md`
- `docs/research/neural_differential_models_survey.md`
- `papers/innovation_one/text/2021_benamira_deeper_look_ml_cryptanalysis.txt`
- `papers/innovation_one/text/2026_liu_spn_iot_friendly_neural_distinguisher_framework.txt`
- `papers/innovation_one/text/2026_polytopic_pdnd_simon_simeck_speck.txt`
- Zhang and Wang, *Improving Differential-Neural Distinguisher Model For DES, Chaskey, and PRESENT*: <https://arxiv.org/abs/2204.06341>
- Zhang et al., *Neural-Inspired Advances in Integral Cryptanalysis*: <https://arxiv.org/abs/2505.10790>
- Bellini et al., *Generic Partial Decryption as Feature Engineering for Neural Distinguishers*: <https://eprint.iacr.org/2025/1443>

## 2026-07-06 Active-Auxiliary And Route Arbitration Update

The active-pattern auxiliary-head retry has now completed, been retrieved, and
passed local gate validation:

```text
run_id = i1_active_auxiliary_r7_262k_seed0_gpu1_retry1_20260704
scale = 262144/class
decision = stop_active_auxiliary_route
candidate_model = present_nibble_invp_active_aux_spn_only
candidate_auc = 0.786112642265
anchor_auc = 0.793651987187
shuffled_auc = 0.784347117180
margin_vs_anchor_auc = -0.007539344922
margin_vs_shuffled_auc = +0.001765525085
```

This closes another tempting but underperforming structure route. The auxiliary
target was learnable, but it made the main real-vs-random decision worse than
the InvP anchor and did not create a qualified non-neighbor expert. Do not run
seed1 or 1M for this route unless a genuinely new auxiliary target is designed
and justified by a separate local gate.

The user also explicitly corrected the process: route selection must not merely
follow the latest suggested direction. In particular, the "multiple neural
networks" idea should be interpreted as a later diversity-gated validator, not
as the next default experiment. A useful ensemble needs structurally different
experts with compatible frozen score artifacts and low error overlap. It should
not be built by adding near-neighbor variants around the same raw/InvP/DDT
view.

The latest external check still points away from model piling and toward
SPN-aware representation/search:

```text
Zhang/Wang 2022: PRESENT gains came from multi-ciphertext-pair derived formats
and convolution/input changes, not only model capacity.

Zhang et al. 2025: neural methods were most useful as feature explorers for
integral properties and then fed back into cryptanalytic search.

Sakagami et al. 2026: switching to an LLM-style model did not improve over a
ResNet baseline on SPECK, while prompt/input representation affected outcomes.
```

Current arbitration:

| Rank | Route | Decision |
|---:|---|---|
| 1 | SPN-aware representation and difference/data search | Keep as the main next research level. Search for a controllable non-neighbor expert before scaling. |
| 2 | Structure-specific neural architecture | Only after the representation is clean and has an explicit same-input or deterministic baseline. |
| 3 | Diverse neural ensemble / score aggregation | Preserve the idea, but run only after at least one non-neighbor weak-positive expert has frozen scores and low-overlap evidence. |
| 4 | Near-neighbor ensemble, active auxiliary, deterministic InvP aggregate stats, current GIFT aligned route, current GPD beam route | Do not spend the next main slot here without a new hypothesis. |

The next practical action should be a lean local route-search plan rather than
a remote launch:

```text
question = Can a controlled SPN representation/search route produce a
           non-neighbor weak-positive expert that is not explained by the
           existing deterministic baseline?

required controls =
  same-budget InvP anchor or deterministic feature baseline
  strict encrypted-random-plaintext negatives
  fixed validation key and metric
  shuffled/topology/control row when the route claims structure
  diversity/error-overlap check before ensemble promotion
```

This is a process correction, not a new result claim.
