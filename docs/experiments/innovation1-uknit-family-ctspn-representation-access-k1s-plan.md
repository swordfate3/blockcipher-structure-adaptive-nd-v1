# Innovation 1 uKNIT-Family CT-SPN Representation-Access Audit K1-S

**Date:** 2026-07-28
**Status:** completed / learned representation access not supported
**Execution:** local CPU, zero neural training

## 1. Research Question

K1-Q confirmed that uKNIT-BC r5 with the cell11 role-1 difference
`0x0000400000000000` contains a strong position-preserving deterministic
five-stage signal. Across untouched seed3/4 same-key and cross-key splits its
AUC was `0.806228-0.825591`. K1-R then trained the exact-composition K1-N model
and three equal-budget controls independently, but the exact model reached only
`0.493238-0.522282` on the same fresh splits and did not identify the correct
S-box semantics.

K1-S asks one question only:

> At which first representation stage inside each completed K1-R
> exact-composition checkpoint does the confirmed K1-Q position-preserving
> signal become inaccessible?

K1-S does not train or alter a model. It replays two frozen checkpoints on the
six frozen cell11 caches and applies the identical diagonal Fisher procedure
independently at each internal observation tap. This is a local mechanism
audit, not formal training, an attack, SOTA evidence, a uKNIT-family transfer
result, or a ceiling claim.

## 2. Frozen Sources

### 2.1 K1-Q data and deterministic upper anchor

```text
root = outputs/local_audit/
  i1_uknit_family_ctspn_difference_position_discovery_
  k1q_seed2_confirm_seed3_seed4_20260728
```

| Artifact | SHA-256 |
|---|---|
| `gate.json` | `1af79fa865736635d40f729fe6621e677a4378e64c6779fc449756ae48609f8b` |
| `dataset_manifest.jsonl` | `16d9549df5d1a6b2d88fd95e10ceec484e6f5443bd774f11d0f7d68dc85494f2` |
| `results.jsonl` | `faf78bc287b35f0237101869d53a89347451369d7e03d6a1253e32ab6f14bc91` |
| `feature_manifest.jsonl` | `e242aa8bd1f723954a6dcca14352b762163d1771f3e00ebdf1eb8afd7cf10868` |
| `scorer_manifest.jsonl` | `46f8e7823f11aabc03101fce3ba8ffdc20ddb35b8a1e88140305d94f5fac3261` |
| `validation.json` | `25b59f9b0eeab8eb894c4b3a40513437306a2c660f0c68f4ab478260689d8059` |

Require the exact pass decision
`innovation1_uknit_family_ctspn_k1q_confirmed_r5_difference_position_supported`,
cell11 in `confirmed_cells`, and the six confirmation cache payloads for
seed3/4 and all three splits. K1-S must replay the existing K1-Q cell11 T0
feature, scorer and AUC digests rather than merely reproduce approximately
similar metrics.

### 2.2 K1-R exact-composition neural checkpoints

```text
root = outputs/local_diagnostic/
  i1_uknit_family_ctspn_cell11_neural_attribution_
  k1r_2048_seed3_seed4_20260728
```

| Artifact | SHA-256 |
|---|---|
| frozen K1-R CSV | `8e612988656163602db20a80241b7b4cfdf01a7c16c37e3ae1e30447f2a4ab00` |
| `gate.json` | `73371777ddef3369b58132939a0d85bf5021e8a5233e6c6c549f1d506f37e299` |
| `checkpoint_manifest.json` | `a4ac9044df2dd1e0276ff88449e21c3c1d2b0a16c3113e7a059a9273adb04b2f` |
| `results.jsonl` | `b3ad0aaf3de1a974c149f2fd546e48696f4ae9ad7dabb8e364000c536dd57cf3` |
| `controls.jsonl` | `167fc68b40762d7a0781c78c5ed8d2ca4be427a5741cd96f4748bbb23d965acd` |
| `validation.json` | `161e4c64cc692955711b61e5ec9291a2d95c45e15a310f5e0540d5af7917d7b2` |

Require the clean K1-R hold decision
`innovation1_uknit_family_ctspn_k1r_cell11_neural_signal_not_supported` and
all protocol checks true. Load only these exact-composition checkpoints:

```text
seed3 SHA-256 = 030d280458654dcbda6a38aafe77f39c3d9f43cdee6ec350742e3d36252071e4
seed4 SHA-256 = a64b3f326795adf955aba6ee87ebc9b9a5b44861322aaa6a7087ea75c9c45e21
```

Strict state loading must preserve each state-dictionary digest. K1-S may add
an introspection path that returns intermediate tensors, but it must not add,
remove or change parameters, and the ordinary forward logits before and after
the introspection refactor must be bit-exact on deterministic fixtures.

## 3. Frozen Data And Evaluation Protocol

| Field | Frozen value |
|---|---|
| Cipher / rounds | uKNIT-BC r5 |
| Input difference | cell11 role1, `0x0000400000000000` |
| Seeds | `3`, `4` |
| Splits | `train_seen`, `same_key_fresh`, `cross_key_validation` |
| Train rows | `2048/class`, `4096` total |
| Fresh rows | `1024/class`, `2048` total per split |
| Pairs per sample | `4` |
| Feature input | four raw ciphertext pairs, `512` bits/sample |
| Negative definition | encrypted random plaintexts |
| Checkpoint | exact K1-R best-validation checkpoint for the same seed |
| Neural training | none: `optimizer_steps=0`, `epochs=0` |
| Tap scorer | K1-Q diagonal Fisher, variance floor `1e-6` |
| Scorer fit | independently per seed and tap on `train_seen` only |
| Control | deterministic label shuffle, independently per seed and tap |
| Execution | local CPU, batched frozen inference |

All taps for one seed/split consume the same K1-Q dataset digest. All neural
taps for one seed consume the same exact K1-R checkpoint digest. No model
selection, threshold tuning or tap-dependent scorer tuning is allowed after
seeing a fresh result.

## 4. Single Experimental Variable: Observation Tap

The only variable is where the frozen exact-composition forward path is
observed. Pair order, bit order, native cell order and hidden-channel order must
remain fixed when a tensor is flattened for the diagonal Fisher scorer.

| Tap | Exact tensor before flattening | Feature dimension | Meaning |
|---|---|---:|---|
| `T0_exact_position_histogram` | `[5 stages, 16 cells, 16 nibble values]` | `1280` | K1-Q deterministic upper anchor; independent of learned weights |
| `T1_bit_encoder_position` | `[4 pairs, 64 native bits, 32 hidden]` | `8192` | learned composition-bit encoder output before cell aggregation |
| `T2_topology_delta_position` | `[4 pairs, 16 native cells, 32 hidden]` | `2048` | learned topology update minus initial cell state before invariant pooling |
| `T3_invariant_cell_pool` | `[4 pairs, 96 pooled hidden]` | `384` | `mean/max/RMS` over the 16 cells, before residual pair projection |

T3 deliberately preserves the four pair slots while erasing native cell
position. This changes only the questioned cell-position aggregation between
T2 and T3. The already recorded K1-R exact logits remain the downstream neural
reference; K1-S does not introduce a fifth trained representation.

For every tap, fit a second scorer to the same features with a deterministic
permutation of the train labels. Evaluate the interpreted and shuffled-label
scorers on all three unmodified splits.

## 5. Protocol And Replay Gate

Interpret no metric unless every item passes:

1. every frozen K1-Q and K1-R artifact hash matches Section 2;
2. K1-Q and K1-R decisions, validation and protocol checks match exactly;
3. exactly six K1-Q cell11 caches are bound by payload row count and digest;
4. exactly two K1-R exact checkpoints are bound by path and SHA-256;
5. strict checkpoint loading retains the state-dictionary digest;
6. introspection leaves model parameter geometry and ordinary logits bit-exact;
7. all `2 seeds x 3 splits x 4 taps x 2 scorer modes = 48` result rows exist;
8. all `2 seeds x 3 splits x 4 taps = 24` feature rows and
   `2 seeds x 4 taps x 2 scorer modes = 16` scorer rows exist;
9. every row has the frozen row count, pair count, negative definition,
   dataset digest, checkpoint binding when applicable, zero epochs and zero
   optimizer steps;
10. T0 feature/scorer/AUC digests exactly replay the corresponding K1-Q cell11
    confirmation rows for both seeds and all splits;
11. every feature and score is finite, every label shuffle changes assignment
    while preserving class counts, and no fresh labels are used for fitting;
12. feature dimensions equal the frozen values in Section 4.

Any failure makes K1-S `invalid` and authorizes only repair of the failed
source, extraction or artifact binding followed by an unchanged rerun.

## 6. Research Gates

Apply every gate separately to seed3 and seed4 on both fresh splits; do not use
averages to hide a failure.

```text
T0 exact replay                                      required
interpreted tap AUC - label-shuffle AUC             >= +0.030
candidate position-preserving tap AUC               >= 0.550
candidate position-preserving tap AUC - T3 AUC      >= +0.030
```

Each learned tap passes only when its AUC and label-shuffle gates hold on every
fresh seed/split. Invariant cell pooling is destructive only when T2 passes,
T3 fails, and T2-minus-T3 passes everywhere. T0 is an upper anchor, not a
learned candidate.

## 7. Frozen Decisions And Required Next Action

- **T2 passes, T3 fails and T2 beats T3 everywhere:** identify invariant cell
  pooling as the first supported bottleneck. Next implement one active-relative,
  runtime-topology-derived position-preserving readout while freezing K1-R
  data, checkpoints, base branch, parameter budget and controls.
- **T1 passes but T2 fails:** identify cell aggregation or topology message
  updates before pooling as the first supported bottleneck. Next audit only the
  bit-to-cell encoder versus the two ordered update slots before redesign.
- **T3 passes while K1-R exact logits remain weak:** retain invariant pooling
  and isolate only residual pair projection, bounded-gate fusion and classifier
  scaling using the same checkpoint/data evidence.
- **Neither T1 nor T2 passes although T0 replays:** hold the K1-N learned
  representation route and retain a bounded deterministic position-histogram
  residual as the next single-variable architecture hypothesis. A T1 miss
  alone does not identify the bit encoder if T2 reconstructs accessible signal.
- **T0 does not replay:** mark the protocol invalid and repair source/tap
  binding only.

Blocked regardless of result: remote training, more samples, pairs, positions,
seeds or epochs; MoE; DDT/trail features; another cipher; another architecture
family; and any mechanical scale-up before the first destructive stage is
identified.

## 8. Run And Required Artifacts

```text
run_id = i1_uknit_family_ctspn_representation_access_
         k1s_seed3_seed4_20260728

output_root = outputs/local_audit/
  i1_uknit_family_ctspn_representation_access_
  k1s_seed3_seed4_20260728
```

Required artifacts:

```text
preflight.json
dataset_manifest.jsonl
checkpoint_manifest.json
feature_manifest.jsonl
scorer_manifest.jsonl
results.jsonl
tap_attribution.csv
gate.json
validation.json
summary.json
progress.jsonl
curves.svg
plot_report.json
visual_qa_passed.marker
```

After completion, render the Chinese tap-attribution chart, pass
`visual-qa-redraw` on a pixel rendering, refresh `outputs/00_RECENT_RESULTS.md`
and `outputs/00_RECENT_RESULTS.json`, record the observed decision and its exact
next action here, then commit and push only K1-S files.

## 9. Completed Result

K1-S completed all frozen zero-training rows and passed every protocol gate:

```text
feature rows = 24 / 24
scorer rows  = 16 / 16
result rows  = 48 / 48
optimizer steps = 0
neural epochs   = 0
failed protocol checks = []
T0 feature/scorer/AUC replay = exact
ordinary logits before/after every tap extraction = bit-exact
checkpoint state before/after every split = unchanged
```

Fresh interpreted AUC was:

| Seed | Split | T0 exact histogram | T1 bit encoder | T2 topology delta | T3 invariant pool | K1-R final logits |
|---:|---|---:|---:|---:|---:|---:|
| 3 | same-key fresh | `0.825591` | `0.558223` | `0.525655` | `0.510297` | `0.522282` |
| 3 | cross-key | `0.809460` | `0.536091` | `0.519584` | `0.521881` | `0.493238` |
| 4 | same-key fresh | `0.806228` | `0.529227` | `0.519121` | `0.539947` | `0.513252` |
| 4 | cross-key | `0.820532` | `0.557749` | `0.557799` | `0.533344` | `0.516200` |

T0 beat its deterministic label-shuffle control by
`+0.278514` to `+0.315833` on all four fresh rows. T1 retained a descriptive
`+0.042659` to `+0.079441` label-shuffle margin, but missed the `0.550` AUC
floor on seed3 cross-key and seed4 same-key. T2 passed both access thresholds
only on seed4 cross-key. T3 missed the AUC floor everywhere. T2-minus-T3 ranged
from `-0.020825` to `+0.024455`, so it never reached the frozen `+0.030` margin.

The frozen decision is:

```text
status = hold
decision = innovation1_uknit_family_ctspn_k1s_learned_representation_access_not_supported
remote_scale = no
```

K1-S therefore rejects the narrower claim that invariant cell pooling alone is
the first supported bottleneck. The learned 15-channel bit encoder sometimes
retains weak linearly accessible signal, but not consistently across seeds and
key scopes; the topology update does not repair that instability. The exact
five-stage position histogram remains strong and reproducible on every fresh
row, so the next experiment should retain that statistic explicitly rather
than ask the current learned compression to rediscover it from `2048/class`.

The first rendered chart was rejected because left-panel colorbar ticks
overlapped the repeated long tap labels on the right. The corrected chart uses
full Chinese tap names once on the left and aligned `T0-T3` labels on the
right. It passed rendered-pixel `visual-qa-redraw` at `1920x1056` with no text
overlap, clipping, missing glyphs, scale ambiguity or insufficient separation.

## 10. Recommended Next Action: K1-T Deterministic Position Residual

K1-T should ask one question:

> Does retaining the exact `stage x native-cell x nibble-value` histogram as a
> bounded residual make the confirmed uKNIT r5 cell11 signal learnable and
> attributable to the correct runtime S-box and diffusion semantics?

Freeze the six K1-Q caches, seed3/4 keys, `2048/class` train,
`1024/class` fresh splits, four pairs/sample, strict encrypted-random-plaintext
negatives, ten epochs, MSE/Adam, learning rate `1e-4`, weight decay `1e-5` and
best-validation-AUC restoration. Change one representation variable only:
project the deterministic five-stage position histogram into the existing
K1-N embedding through a bounded residual while retaining the existing base
branch and parameter budget as closely as possible.

Use a lean independently trained matrix:

```text
seed3/4 x {
  exact_position_histogram_residual,
  wrong_sbox_position_histogram_residual,
  invariant_histogram_residual,
  current_k1r_exact_anchor
}
```

The first three rows must have identical trainable geometry. The old K1-R exact
row is the same-data, same-optimizer anchor; it may be replayed from the frozen
result only if all dataset, checkpoint and metric digests remain exact. The
only structural controls are wrong S-box semantics and erasing native cell
position. Do not add more models unless readiness proves one of these controls
algebraically identical to the candidate.

Apply separately to both seeds and both fresh splits:

```text
exact residual AUC                         >= 0.600
exact residual - current K1-R exact        >= +0.050
exact residual - wrong S-box residual      >= +0.010
exact residual - invariant residual        >= +0.030
```

Run a small deterministic readiness fixture and parameter/logit control first,
then the frozen local `2048/class` diagnostic. A full local pass authorizes a
separate remote `65536/class` medium diagnostic with disk-backed cache,
progress and reuse; it remains non-formal and non-paper-scale. Any seed/split
miss holds remote scale and selects only the failed attribution axis for one
redesign. Do not add samples, positions, pairs, epochs, MoE, DDT/trails, cipher
identity, a new cipher, or another network family inside K1-T.
