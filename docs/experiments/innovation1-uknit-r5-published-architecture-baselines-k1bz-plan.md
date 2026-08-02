# Innovation 1 K1-BZ: uKNIT r5 Published Architecture Baselines

**Date:** 2026-08-02

**Status:** completed local diagnostic / gate `hold` / remote scale prohibited

## Research question

K1-BS compared the uKNIT structure expert with AutoND/DBitNet and two
project-internal generic SPN networks. It did not include a second published
differential-neural architecture or a published SPN-shaped Conv2D backbone.
K1-BZ asks:

> Under the frozen K1-BS uKNIT r5 protocol, do Zhang/Wang MCND-style or Liu
> et al. SPN Conv2D-style architectures recover signal that AutoND/DBitNet did
> not recover?

This is a two-seed local architecture diagnostic. It is not formal training,
paper-scale evidence, an exact reproduction of either paper, an attack result,
or evidence for a universal SPN network.

## Evidence anchor and single variable

K1-BZ reuses the completed, protocol-valid K1-BS rows as immutable context:

| Architecture | seed3 AUC | seed4 AUC |
|---|---:|---:|
| uKNIT structure expert | `0.902801514` | `0.932538986` |
| AutoND/DBitNet | `0.511321068` | `0.526423454` |

Only the trainable architecture changes. K1-BZ trains two new rows per seed:

| Model key | Literature role | Frozen adapter input |
|---|---|---|
| `spn_zhang_wang_mcnd_adapter` | Zhang/Wang 2022 multi-pair Inception-ResNet | raw ciphertext-pair bits |
| `spn_liu_case3_conv2d_adapter` | Liu et al. 2026 three-channel Conv2D-ResNet backbone | `C`, `C'`, raw `C xor C'` state matrices |

The Liu row intentionally uses raw `C xor C'` rather than a uKNIT inverse-round
operator. This holds the dataset and raw observable fixed while testing the
published architecture family. A later representation-attribution experiment
would require a true uKNIT inverse-linear view, the same Conv2D backbone on raw
input, and a shuffled-operator control. K1-BZ does not answer that separate
question.

## Frozen protocol

Train four independent rows: two seeds times two published architecture
adapters.

| Field | Frozen value |
|---|---|
| Cipher / rounds | uKNIT-BC r5 |
| Difference | cell11 role1, `0x0000400000000000` |
| Train | `2048/class`, `4096` total rows |
| Cross-key validation | `1024/class`, `2048` total rows |
| Seeds | `3`, `4` |
| Pairs/sample | `16` independent ciphertext pairs |
| Input width | `2048` bits/sample, `128` bits/pair |
| Negative definition | encrypted random plaintexts |
| Train/validation keys | exact K1-BS seed-matched fixed cross-key pairs |
| Epochs / batch | `10` / `64` |
| Loss / optimizer | MSE / Adam |
| Learning rate / weight decay | `1e-4` / `1e-5` |
| Checkpoint | restore best validation AUC |
| Device | local CPU, justified only because local CUDA is unavailable |

The run must reuse the completed K1-BS disk-backed train and validation cache.
No difference, key, label, negative definition, pair count, data budget,
optimizer, checkpoint rule, or validation metric may change.

## Architecture adapters

The Zhang/Wang adapter preserves the paper's network choices that are portable
across 4-bit-cell SPNs: three initial Conv1D branches with kernel sizes
`1/2/4`, five residual blocks with increasing odd kernels, global average
pooling, and a one-logit head. For 64-bit uKNIT it uses the same `8 x 16` basic
pair layout as the existing PRESENT port. For a future 128-bit Dialga row the
same implementation accepts an `8 x 32` layout. This is an MCND-style
architecture transfer, not a reproduction of the paper's PRESENT data or
reported accuracy.

The Liu adapter reshapes every basic pair into three `4 x state_cells`
channels: `C`, `C'`, and `C xor C'`. A `1 x 1` stem, three residual Conv2D
blocks, mean/max state pooling, pair projection, and mean/max multi-pair pooling
form the classifier. It transfers the paper's state-matrix Conv2D backbone but
does not claim the paper's inverse-round representation.

## Readiness gate

Before any optimizer step require:

1. exactly four frozen plan rows;
2. both models accept `[B, 2048]` and emit finite `[B, 1]` logits;
3. both models have finite nonzero backward gradients;
4. the Zhang/Wang adapter matches the existing PRESENT port exactly for a
   128-bit pair after loading the same parameters;
5. the Liu tensor is exactly `(C, C', C xor C')` with shape
   `[batch, pair, 3, 4, 16]`;
6. all task fields match K1-BS except the model key, network label, family and
   literature/evidence description;
7. local CUDA remains unavailable; otherwise use the local GPU.

Any failure prohibits training. Repair only the failed invariant and rerun the
unchanged readiness gate.

## Result gate

For each new architecture and seed record AUC, accuracy, parameter count,
restored-best epoch, and delta against the seed-matched K1-BS AutoND and
structure-expert anchors.

The strongest published adapter is eligible for a fresh, uniquely named remote
medium comparison only when both seeds satisfy:

```text
adapter AUC                  >= 0.550
adapter AUC - AutoND AUC     >= +0.010
```

If no adapter passes both clauses on both seeds, stop architecture scale-up and
report the rows as local diagnostics. Do not infer a paper-model ceiling from
`2048/class`. If one adapter passes, prepare a new three-model remote matrix
containing the structure expert, AutoND and only that adapter at
`65536/class`; do not modify or reuse the protocol-invalid K1-BT run id.

## Artifacts and next action

```text
run_id = i1_uknit_r5_published_architecture_baselines_k1bz_16pair_2048_seed3_seed4_20260802
```

The completed result root must contain the frozen plan hash, readiness report,
cache provenance, progress JSONL, four checkpoints, results JSONL, gate,
validation, comparison CSV, history CSV, summary, Chinese SVG, plot report and
rendered-pixel visual QA evidence. Refresh both recent-result indexes after the
run. The next action must be selected by the result gate, not by architecture
novelty or parameter count.

## Completed result

The frozen four-row run completed locally on 2026-08-02. Local CUDA was not
available, so the preregistered tiny diagnostic used CPU; no medium-or-larger
training was run locally. The exact K1-BS train and cross-key validation caches
were reused eight times with zero cache creation events.

| Architecture | Parameters | seed3 AUC | seed4 AUC | seed3 accuracy | seed4 accuracy |
|---|---:|---:|---:|---:|---:|
| uKNIT structure expert (K1-BS anchor) | `214316` | `0.902801514` | `0.932538986` | `0.811035156` | `0.857421875` |
| AutoND/DBitNet (K1-BS anchor) | `985985` | `0.511321068` | `0.526423454` | `0.504882812` | `0.516601562` |
| Zhang/Wang MCND adapter | `650177` | `0.493507862` | `0.493232727` | `0.500000000` | `0.498535156` |
| Liu raw Case-3 Conv2D adapter | `130945` | `0.526742458` | `0.505930424` | `0.516601562` | `0.500000000` |

Relative to the seed-matched AutoND anchor, the MCND adapter changed AUC by
`-0.017813206/-0.033190727`; the Liu adapter changed AUC by
`+0.015421390/-0.020493030`. Neither adapter reached AUC `0.550` on both seeds,
and neither beat AutoND by at least `+0.010` on both seeds. All frozen protocol
checks passed, including four result rows, finite AUC values, ten completed
epochs per row, restored-best checkpoints and cache reuse.

```text
gate status               = hold
decision                  = innovation1_uknit_k1bz_no_published_adapter_local_promotion
remote_scale              = no
selected_remote_candidate = null
```

The supported conclusion is narrow: under the same `2048/class` uKNIT r5
diagnostic protocol, these two architecture adapters did not recover a stable
cross-seed signal and therefore do not justify a remote scale step. This does
not reproduce the original Zhang/Wang or Liu protocols, and it does not show
that either published method fails at its native cipher, representation,
training scale or evaluation protocol.

The final Chinese SVG was rasterized and inspected at delivery size. After the
legend was moved away from the lower axis, the exact delivered figure had no
text overlap, clipping, missing glyphs, ambiguous axes or unreadable values.

## Evidence-backed next action

Retain K1-BZ as a completed local architecture diagnostic and stop both
published-adapter scale-up routes. Do not enlarge the dataset, add epochs or
tune either adapter post hoc. A future published-architecture comparison is
justified only as a new preregistered representation-attribution question, for
example by giving the Liu backbone a verified uKNIT inverse-linear view together
with the same raw-backbone and shuffled-operator controls. Such a study must use
a fresh run id and pass a local signal gate before any remote medium run.

## Literature boundary

- Zhang and Wang, *Improving Differential-Neural Distinguisher Model For DES,
  Chaskey and PRESENT*, 2022, arXiv:2204.06341.
- Liu, Li, Ren and Chen, *A Highly Efficient Neural Distinguisher Framework for
  IoT-Friendly Lightweight SPN Block Ciphers*, IEICE Transactions on
  Information and Systems, 2026, DOI:10.1587/transinf.2025EDP7070.
- Bellini et al., *A Cipher-Agnostic Neural Training Pipeline with Automated
  Finding of Good Input Differences*, ToSC 2023(3),
  DOI:10.46586/tosc.v2023.i3.184-212.

Reported paper accuracies are not imported into the K1-BZ result table because
the ciphers, negative construction, key sampling, training scale and evaluation
protocols are not aligned.
