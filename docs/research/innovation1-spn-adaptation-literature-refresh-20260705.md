# Innovation 1 SPN Adaptation Literature Refresh

**Date:** 2026-07-05

**Status:** literature-informed route correction

## Why This Note Exists

The first recovered PRESENT neural ensemble screen showed mild complementarity
but did not pass its route gate:

```text
run_id = i1_present_neural_ensemble_r7_65k_seed0_gpu0_20260705
best_single = present_nibble_ddt_graph, AUC 0.789112608414
best_ensemble = probability_mean, AUC 0.790061685257
delta = +0.000949076843
gate = +0.001
decision = weak_neural_ensemble_positive_below_gate
```

That result should not push the project into mechanically wider ensembles. A
fresh external scan suggests the more promising SPN-adaptation direction is:

```text
primary = SPN-aware data / feature representation
secondary = structure-aware architecture
tertiary = diverse expert aggregation after weak-positive low-correlation experts exist
```

## External Evidence Checked

The scan used current web checks plus already extracted local research notes.
The saved lookup trace is:

```text
sources/research_spn_adaptation_20260705.md
```

The most relevant external pages successfully re-confirmed in this turn were:

1. Liu Zhang et al., **Neural-Inspired Advances in Integral Cryptanalysis**,
   arXiv:2505.10790, 2025.
   URL: <https://arxiv.org/abs/2505.10790>

   Key point for this project: neural networks are useful as feature explorers
   for integral properties, especially on SKINNY. The abstract reports that
   neural-discovered integral distinguishers reduce active plaintext bits for
   11-round SKINNY64/64, identify a 12-round key-dependent integral
   distinguisher, and enable more-round key recovery. This supports using
   SPN/integral structure as the main search direction, not merely as an
   ensemble member.

2. Liu Zhang and Zilong Wang, **Improving Differential-Neural Distinguisher
   Model For DES, Chaskey, and PRESENT**, arXiv:2204.06341, 2022.
   URL: <https://arxiv.org/abs/2204.06341>

   Key point for this project: derived features from multiple-ciphertext pairs
   and adjusted convolution kernels improve PRESENT 6-7 round neural
   distinguishers. This supports treating input/data format as a first-class
   mechanism.

3. Tatsuya Sakagami et al., **Do LLMs Make Neural Distinguishers Wise?**,
   arXiv:2606.10692, 2026.
   URL: <https://arxiv.org/abs/2606.10692>

   Key point for this project: a very different model class did not provide
   observable improvement over ResNet on SPECK-32/64, while prompt/data
   representation mattered. This is useful negative pressure against
   "try a bigger/different model" as the default strategy.

Additional locally extracted research notes in
`docs/research/neural_differential_models_survey.md` and
`docs/research/spn_structured_nn_research_plan.md` also point in the same
direction: two-difference / related-key formats, RX-neural formats,
score-distribution tests, and Generic Partial Decryption are primarily
input/feature/task design shifts, not evidence that broad model ensembling alone
is the next best SPN route.

## New Local Evidence After The Refresh

The r8 integral/inverse feature screen completed after the initial route note:

```text
run_id = i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705
status = retrieved / validated / postprocessed
samples_per_class = 65536
negative_mode = encrypted_random_plaintexts
sample_structure = plaintext_integral_nibble
```

Result:

| Row | Feature | AUC | Calibrated accuracy |
|---|---|---:|---:|
| Raw integral anchor | `ciphertext_pair_bits` | `0.999995831400` | `0.999786376953` |
| InvP matrix | `present_pair_xor_paligned_cell_matrix_bits` | `0.513465017546` | `0.528442382812` |
| InvP+Sinv matrix | `present_pair_xor_paligned_sinv_cell_matrix_bits` | `0.505787684582` | `0.532577514648` |

Gate decision:

```text
stop_integral_inverse_feature_screen_for_now
```

Interpretation:

```text
The screen supports "integral/multiset data construction exposes a very strong
high-round statistic" much more than "inverse-round matrix architecture wins."
```

This result strengthens the broad literature correction toward SPN-aware
feature/input search, but it also narrows the next action: do not scale the
current InvP/Sinv matrix candidate. The raw integral anchor is too strong and
likely contains deterministic multiset parity structure, so the next work
should audit and control the data construction before claiming an architecture
or neural representation gain.

That audit was then run locally at `2048/class` for train and validation keys.
The deterministic pair-xor parity statistic alone separated the current
positive-vs-random-right integral construction with `1.0` accuracy in all
checked seeds/splits:

```text
positive_pair_xor_parity_hw.zero_rate = 1.0
negative_pair_xor_parity_hw.zero_rate = 0.0
best threshold = parity_hw <= 0
accuracy = 1.0
```

This means the completed r8 raw integral anchor is now best understood as a
protocol/data-construction diagnostic. It is not a candidate to scale directly
as a neural distinguisher result.

## Route Correction

The literature scan changes the priority order:

### Priority 1: SPN-Aware Feature / Data Representation

Most promising next direction:

```text
PRESENT/SKINNY-style state-cell features
inverse linear layer / inverse P-layer views
partial inverse S-box or key-independent structural inverse-S features
integral / plaintext-active-cell structures
multiple-ciphertext-pair derived features with aggregation controls
```

This matches current local evidence:

```text
InvP-only has 1M/class two-seed positive evidence.
DDT graph is weak but not useless.
r8 integral/multiset data structure exposes a strong high-round signal, but
the current InvP/Sinv matrix variants did not beat the raw integral anchor.
```

### Priority 2: Structure-Aware Architecture

Architecture should follow the representation, not lead it. Useful candidates:

```text
cell-token encoders over SPN state
fixed inverse-linear/P-layer message passing
DDT-aware gates or priors
matrix/state-layout CNN blocks for PRESENT/SKINNY/GIFT-like ciphers
```

Avoid treating generic larger networks, LLM-like models, or broad model
ensembles as the main contribution unless they isolate a structure-specific
signal.

### Priority 3: Diverse Expert Pool

The diverse expert pool remains useful, but only after weak-positive artifacts
exist from genuinely different families:

```text
raw_mcnd
invp_cell
ddt_graph
pair_evidence
inverse_round_matrix
integral_feature
projection_feature
```

Until then, a pool such as raw + InvP + DDT graph is a near-neighbor control,
not the main Innovation 1 route.

## 2026-07-06 Independent Route Re-Rank

The latest route decision should not simply mirror the user's current
suggestion. Re-checking the local evidence against the literature-informed
ranking gives this priority:

| Priority | Route | Why it ranks here now | Next gate |
|---:|---|---|---|
| 1 | Controlled SPN feature/input route | Recent SPN neural-cryptanalysis work and local r8 evidence both point to representation/data construction. The strongest current r8 signal is a deterministic aligned active-difference statistic, not a new architecture. | Neural follow-up must beat or explain the fixed `pair_xor_column_sum_variance` AUC baseline. |
| 2 | SPN-aware architecture over a controlled representation | Architecture is valuable only after the input route is clean. r7 InvP/P-layer attribution remains the best completed structure evidence. | Same-input baseline and shuffled/topology controls. |
| 3 | Diverse expert pool | The near-neighbor ensemble was weak-positive but below gate. A real ensemble needs non-neighbor experts, score artifacts, and diversity/error-overlap checks. | At least one compatible weak-positive low-overlap expert beyond raw/InvP/DDT neighbors. |

This means the next small action is not "add more models to the ensemble." It
is a baseline-gated neural smoke on the aligned active-difference r8 route:

```text
plan = configs/experiment/innovation1/innovation1_spn_present_r8_integral_aligned_neural_followup_smoke.csv
purpose = test whether the neural probe exceeds or explains the deterministic baseline
claim_scope = local smoke only, no remote launch, no architecture claim
```

Reference deterministic AUCs from the fixed baseline audit at `2048/class`,
seed `23`:

| Difference route | Active nibble | Fixed baseline AUC |
|---|---:|---:|
| Zhang/Wang `0x9` | 0 | `0.8878759145736694` |
| AutoND `0x0d000000` | 6 | `0.8747416734695435` |
| Entropy `0x00d00000` | 5 | `0.8852955102920532` |

Any local neural AUC materially below these values is not evidence of SPN
architecture gain, even if the neural row looks above chance.

The local neural smoke was then run and stayed below all three fixed baseline
AUCs:

| Difference route | Active nibble | Neural smoke AUC | Fixed baseline AUC |
|---|---:|---:|---:|
| Zhang/Wang `0x9` | 0 | `0.67193603515625` | `0.8878759145736694` |
| AutoND `0x0d000000` | 6 | `0.71832275390625` | `0.8747416734695435` |
| Entropy `0x00d00000` | 5 | `0.78289794921875` | `0.8852955102920532` |

Decision:

```text
diagnostic_no_neural_architecture_gain
do_not_launch_remote_from_this_neural_followup
```

## Immediate Project Implication

Do not spend the next remote slot on a wider r7 ensemble. Also do not spend the
next slot scaling the current InvP/Sinv matrix candidate from the completed r8
integral/inverse screen. The better next action is:

```text
1. Treat the completed r8 integral result as a data-construction diagnostic,
   not an inverse-round architecture win.
2. Audit/control the integral raw anchor signal, especially deterministic
   pair-xor/multiset parity and active-nibble leakage.
3. If controls preserve a nontrivial high-round signal, write a lean
   confirmation plan for the controlled data route.
4. If controls collapse it, return to r9 weak-probe/curriculum/difference or
   a smaller local SPN-derived feature probe before new remote training.
5. Use diverse expert aggregation only after a non-neighbor family produces
   score artifacts under a matching protocol group.
```

## Candidate Next Experiments

### Candidate A: r8 Integral / Multiset Control Audit

Status:

```text
completed as i1_present_r8_integral_inverse_feature_screen_65k_seed0_gpu1_retry1_20260705
```

Reason:

```text
The raw integral anchor is nearly perfect, while the inverse-round matrix rows
are near-random. That makes the data construction itself the first object to
audit.
```

Action:

```text
Build a small local audit around deterministic multiset parity / pair-xor
statistics and define controls before any new remote scale-up.
```

### Candidate B: SPN Derived-Feature Probe Before New Remote Runs

Build a small local diagnostic that computes deterministic features from
existing ciphertext pairs:

```text
DeltaC
InvP(DeltaC)
active-cell mask
cell Hamming weights
DDT row/column legality summaries
simple inverse-S candidate statistics that do not use the true key
```

Train tiny linear/MLP probes at smoke scale to decide which feature blocks are
worth a remote screen. This can prevent wasting GPU time on broad architecture
variants.

### Candidate C: Diverse Expert Pool as Secondary Validator

Only after Candidate A or B yields a weak-positive non-neighbor expert, export
aligned scores and run the diverse expert gate. Do not treat near-neighbor
aggregation as evidence that the broader diverse route has failed.

## Recommended Next Concrete Step

Candidate A has now confirmed that the current raw anchor is explained by a
trivial deterministic parity statistic. Use the follow-up control plan next:

```text
docs/experiments/innovation1-present-r8-integral-parity-control-plan.md
```

The control plan should remove the positive-vs-random-right mismatch before
any remote scale-up. Control A now does remove the explicit pair-xor parity
separator:

```text
sample_structure = plaintext_integral_nibble_matched_negative
matched-negative parity audit accuracy = 0.5
positive parity zero_rate = 1.0
matched-negative parity zero_rate = 1.0
```

The matched-negative smoke/probe has now been run locally:

```text
plan = configs/experiment/innovation1/innovation1_spn_present_r8_integral_matched_negative_probe_smoke.csv
samples_per_class = 256
validation_samples_per_class = 128
raw-pair AUC = 0.805480957031
InvP matrix AUC = 0.530761718750
InvP+Sinv matrix AUC = 0.547485351562
```

A second local matched-negative smoke/probe has now been run with `seed = 1`:

```text
plan = configs/experiment/innovation1/innovation1_spn_present_r8_integral_matched_negative_probe_smoke_seed1.csv
samples_per_class = 256
validation_samples_per_class = 128
raw-pair AUC = 0.877990722656
InvP matrix AUC = 0.530029296875
InvP+Sinv matrix AUC = 0.574340820312
```

The follow-up deterministic audits at `2048/class`, audit seed `11`, show:

```text
pair-xor parity best accuracy = 0.5
pair-alignment best statistic = same_index_xor_hw_mean
pair-alignment best accuracy = 0.546630859375
```

A deterministic feature-bank audit then found a much stronger explanation for
the raw-pair residual:

```text
seed0 audit artifact = outputs/local_audits/r8_integral_matched_negative_feature_bank_audit_seed7_2048.json
seed0 best statistic = pair_xor_column_sum_variance
seed0 best threshold accuracy = 0.979248046875

seed1 audit artifact = outputs/local_audits/r8_integral_matched_negative_feature_bank_audit_seed11_2048.json
seed1 best statistic = pair_xor_column_sum_variance
seed1 best threshold accuracy = 0.982421875
```

This changes the immediate reading again: the raw matched-negative pair neural
smoke is likely learning a simple deterministic pair-xor column-distribution
variance statistic. That is not an architecture win, but it is a useful SPN
feature/input lesson. The branch should be kept as a deterministic
SPN/multiset feature candidate only after further controls such as
active-nibble variation, input-difference variation, pair-order scramble, and
same-budget anchor comparison. It should not consume the next remote neural
training slot. Keep Candidate C as a secondary validator, not the next remote
launch.

The first pair-order scramble control has now been run:

```text
control plan = configs/experiment/innovation1/innovation1_spn_present_r8_integral_pair_order_control_smoke.csv
matched anchor pair_xor_column_sum_variance accuracy = 0.979248046875
scrambled-positive pair_xor_column_sum_variance accuracy = 0.8818359375
```

This weakens but does not remove the statistic. The current best explanation
is therefore not "same-index pairing alone" and not "neural architecture
gain"; it is a deterministic SPN/integral multiset feature whose strength is
amplified by same-index fixed-difference pairing. The next local controls
should vary `integral_active_nibble` and `input_difference` before considering
any confirmation plan.

Those clean variation controls have now been run with
`plaintext_integral_nibble_difference_matched_negative`, which matches the
right-side `input_difference` shift across classes before testing the
feature-bank statistics:

```text
summary = outputs/local_audits/r8_integral_feature_variation_control_clean/summary_seed17_2048.json

active0 + Zhang/Wang diff 0x9:
  best statistic = pair_xor_column_sum_variance
  best accuracy = 0.81494140625

active1/7/15 + Zhang/Wang diff 0x9:
  best accuracy = 0.521240234375 / 0.520263671875 / 0.51611328125

active0 + AutoND / entropy / Wang-Jain differences:
  best accuracy = 0.525146484375 / 0.518310546875 / 0.514892578125
```

This narrows the hypothesis substantially: the useful deterministic statistic
is not a generic integral-multiset signal. It appears when the active integral
nibble is aligned with the fixed input-difference support. The next research
step should therefore test aligned active-nibble choices for each candidate
difference, and keep `pair_xor_column_sum_variance` as an explicit
deterministic baseline before using neural models.

The aligned active-difference audit has now been run:

```text
plan = configs/experiment/innovation1/innovation1_spn_present_r8_integral_aligned_difference_control_smoke.csv
summary = outputs/local_audits/r8_integral_aligned_difference_control/summary_seed17_2048.json

Zhang/Wang diff 0x9, active0:
  pair_xor_column_sum_variance accuracy = 0.81494140625

AutoND diff 0x0d000000, active6:
  pair_xor_column_sum_variance accuracy = 0.804443359375

Entropy diff 0x00d00000, active5:
  pair_xor_column_sum_variance accuracy = 0.804443359375

Wang/Jain diff 0x0700000000000700, active2 or active14:
  best accuracy = 0.51806640625 / 0.517578125
```

This is a meaningful refinement. The route is not merely "Zhang/Wang diff
special"; it appears to generalize across several single-nibble input
differences when the integral active nibble is aligned to the difference
support. It does not currently support a two-nibble difference under the
one-active-nibble construction.

A second local audit seed preserved the same split:

```text
artifact_dir = outputs/local_audits/r8_integral_aligned_difference_control_seed23/

Zhang/Wang diff 0x9, active0:
  pair_xor_column_sum_variance accuracy = 0.805908203125

AutoND diff 0x0d000000, active6:
  pair_xor_column_sum_variance accuracy = 0.79296875

Entropy diff 0x00d00000, active5:
  pair_xor_column_sum_variance accuracy = 0.8056640625

Wang/Jain diff 0x0700000000000700, active2 or active14:
  best accuracy = 0.51806640625 / 0.518798828125
```

This makes the next route choice sharper: build an explicit deterministic
baseline and multi-active-cell control before spending a neural training slot.
Diverse neural aggregation remains useful later, but only after this route
produces controlled, non-neighbor score artifacts.

The explicit deterministic baseline is now implemented:

```text
script = scripts/evaluate-integral-deterministic-baseline
api = integral_deterministic_baseline_from_task
default statistic = pair_xor_column_sum_variance
smoke artifact = outputs/local_audits/r8_integral_deterministic_baseline_smoke/row0_pair_xor_variance_seed23.json
smoke accuracy = 0.765625 at 64/class
```

This keeps the immediate Innovation 1 route honest: later neural rows must
beat or explain the fixed deterministic statistic, not merely rediscover it.
At `2048/class`, audit seed `23`, the fixed baseline AUC is already high:

```text
Zhang/Wang aligned active0 AUC = 0.8878759145736694
AutoND aligned active6 AUC = 0.8747416734695435
Entropy aligned active5 AUC = 0.8852955102920532
```

This raises the bar for the next neural follow-up: it must beat or explain the
fixed deterministic baseline, not simply reproduce a sub-0.89 signal.

The multi-active-cell construction for multi-nibble input differences has now
been tested locally on the Wang/Jain profile:

```text
sample_structure = plaintext_integral_multi_nibble_difference_matched_negative
plan = configs/experiment/innovation1/innovation1_spn_present_r8_integral_multi_active_difference_control_smoke.csv
pairs_per_sample = 256
seed29 pair_xor_column_sum_variance accuracy = 0.58203125
seed31 pair_xor_column_sum_variance accuracy = 0.59765625
seed29 feature-bank best accuracy = 0.58203125
```

This is a useful negative control: matching the two active cells to the
two-nibble difference did not turn the Wang/Jain route into a strong
deterministic feature candidate. The next meaningful slot should stay with the
single-nibble aligned active-difference route or a neural follow-up that must
beat the fixed deterministic baseline.

The current priority is therefore:

```text
single-nibble aligned active-difference SPN feature/input search > new SPN architecture variant > diverse ensemble
```

This is a route correction, not a claim of completion.
