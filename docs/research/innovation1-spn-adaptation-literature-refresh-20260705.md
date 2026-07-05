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

This changes the immediate reading slightly: the matched-negative raw-pair
signal is not explained by the two simplest deterministic controls checked so
far, and it survived a second tiny seed. However, it is still only a local
smoke with `256/class` training and `128/class` validation, so it should not
consume a remote slot yet. The next step is a local deterministic feature audit
or shallow-probe control for the raw matched-negative pair route. If that
identifies a nontrivial controlled statistic, write a lean confirmation plan.
If it collapses, switch to Candidate B and build a smaller SPN-derived feature
probe. Keep Candidate C as a secondary validator, not the next remote launch.

The current priority is therefore:

```text
SPN feature/input search > new SPN architecture variant > diverse ensemble
```

This is a route correction, not a claim of completion.
