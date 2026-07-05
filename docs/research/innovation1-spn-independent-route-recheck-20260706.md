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

Do not spend the next meaningful slot on a wider near-neighbor ensemble, on the
current projection v2 variants, or on a renamed single-step Sinv/GPD feature.

The better next SPN/PRESENT route is:

```text
score-guided / sensitivity-guided SPN projection
before diverse expert aggregation
```

The core hypothesis is not "combine more neural networks." The hypothesis is:

```text
PRESENT r8/r9 signal is sparse and structure-aligned; the next gain is more
likely to come from selecting stable SPN-sensitive axes or score-distribution
features than from adding another similar graph/CNN expert.
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

This supports a route that mines stable SPN-sensitive evidence first, then uses
models or ensembles only after the evidence is real.

## Proposed Route: SGP

Name:

```text
SGP = score-guided / sensitivity-guided projection
```

The route has three layers.

### Layer 1: Stable Axis Audit

Run local audits under the same strict PRESENT protocol:

```text
cipher = PRESENT-80
negative_mode = encrypted_random_plaintexts
sample_structure = zhang_wang_case2_official_mcnd
rounds = 8 first, then r9 only if r8 gives stable evidence
```

Candidate feature families:

```text
InvP(delta) bit/cell axes
candidate-evidence score axes
beamstats confidence/margin/cumulative axes
trail-family score-distribution axes
projection_feature historical masks as controls only
```

Report for each axis family:

```text
single-axis AUC advantage
oriented top-k z-score composite AUC
top-k overlap across seeds
train-key vs validation-key stability
false-family / shuffled-cell control
```

### Layer 2: Mask Construction

Only build a projection mask from axes that pass stability checks. Avoid
hand-picked P-column masks unless they are rediscovered by the audit.

Predeclared candidate masks:

| Mask | Definition | Purpose |
|---|---|---|
| `sgp_top16_stable` | top 16 axes by two-seed stable score | compact signal check |
| `sgp_top32_stable` | top 32 axes by two-seed stable score | direct replacement for old projection v2 |
| `sgp_cell_balanced32` | best stable axes with no cell contributing more than a cap | prevents one-cell artifact |
| `sgp_false_family32` | same count from shuffled or false-family axes | control |
| `sgp_random32` | deterministic random mask with same feature width | chance/control floor |

### Layer 3: Neural Or Ensemble Use

Only after Layer 1 and Layer 2 pass:

```text
train a small same-budget MLP/CNN projection probe
compare against full raw anchor and old projection masks
export score artifacts only if projection is weak-positive and stable
then evaluate low-correlation expert-pool inclusion
```

This keeps diverse aggregation alive, but makes it evidence-gated.

## Gates

Local audit gate:

```text
top-k composite AUC >= 0.55 at 2048/class
and top-k overlap/Jaccard >= 0.35 across two seeds or key splits
and false-family control is lower by >= 0.01 AUC
```

Local neural smoke gate:

```text
SGP mask AUC > full raw anchor by >= 0.005
or SGP mask AUC > old best projection by >= 0.01
```

Remote diagnostic gate:

```text
Do not remote-launch until both local audit and local neural smoke pass.
First remote scale, if earned, is 65536/class diagnostic only.
No formal claim before >=1000000/class and multiple seeds.
```

Diverse expert gate:

```text
Only add SGP to the expert pool if its score artifact is weak-positive and its
error overlap with the best InvP/DDT/raw anchor is low enough to meet the
diverse expert plan.
```

## Immediate Next Experiment

The next small action should be a local SGP audit, not a remote launch:

```text
question = do any SPN score/feature axes remain stable across seeds and keys?
scale = 2048/class local diagnostic
models = no neural training in the first step
outputs = JSON audit with axis rankings, stability, controls, and candidate masks
decision = keep SGP, discard SGP, or implement a tiny projection smoke
```

If this audit fails, the route should be discarded quickly. If it passes, the
next implementation should generate a lean smoke CSV for `sgp_top32_stable`
against old projection v2 controls.

## Claim Scope

This is a route-selection plan only. It does not claim a new PRESENT r8/r9
result, does not claim a breakthrough, and does not reinterpret local
projection or beamstats probes as formal evidence.

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
