# Innovation 1 uKNIT-Family Exact Partial-State Signal Audit K1-O

**Date:** 2026-07-28
**Status:** frozen / implementation pending
**Execution:** local CPU, deterministic zero-neural-training audit

## 1. Question

K1-N opened the topology residual gate and exposed the exact ordered inverse
linear/S-box composition, but every fresh uKNIT-BC r5 AUC remained below
`0.520`. Correct S-box composition also differed from wrong-S-box and no-S-box
controls by approximately `0.001` AUC or less. Dialga retained its strong
calibration signal, so the runner and optimization path were not generally
broken.

K1-O must resolve the remaining ambiguity before another neural redesign:

> Does the frozen uKNIT-BC r5 differential have a fresh-split association in
> exact, position-preserving cell-difference statistics, and does that
> association specifically require the correct S-box semantics?

This is a signal audit, not a neural experiment. It adds no network, gradient,
optimizer, epoch, sample, key, pair, or hyperparameter search.

## 2. Frozen Source Authority

```text
source_root = outputs/local_diagnostic/
  i1_uknit_family_ctspn_exact_operator_composition_k1n_2048_seed0_seed1_20260728
```

| Source artifact | SHA-256 |
|---|---|
| `gate.json` | `e2aed925c5d285f2856be791e1f6450b5e338f10e470572844539d86c1134a4f` |
| `dataset_manifest.jsonl` | `ecc990e4d724ec35fdce8bd52d947c78280db2140853feddee07189ade4341f0` |
| `validation.json` | `8743497b33f78c6e1bda7f49ca0900f78c7b669396d1b604c25d0a97a087634d` |
| `preflight.json` | `b87335f6d36b2eb377f0deb6999af1af4546553d01c9dc1d3bc049b42e824e8d` |

The source gate must retain status `hold`, all protocol checks, and decision:

```text
innovation1_uknit_family_ctspn_k1n_dialga_retained_uknit_signal_not_supported
```

K1-O uses only the six digest-bound uKNIT rows from the twelve-row source
dataset manifest. It must load the existing disk-backed arrays without
regeneration.

## 3. Frozen Data Protocol

| Field | Value |
|---|---|
| Cipher / rounds | uKNIT-BC r5 only |
| Seeds | `0`, `1` |
| Training | `2048/class`, `4096` total rows |
| Fresh same-key | `1024/class`, `2048` total rows |
| Cross-key | `1024/class`, `2048` total rows |
| Pairs per sample | `4` |
| Negative definition | encrypted random plaintexts |
| Keys / input differences | exact K1-N/K1-M values |
| Runtime window | two transitions, start `3` |
| Device | local CPU |

Train-seen rows fit the deterministic scorer. Both holdouts are evaluated once
with the fitted scorer. Train metrics are descriptive only. No result may
select a feature, threshold, regularizer, split, seed, or orientation.

## 4. One Audited Variable And Controls

For every ciphertext pair, reconstruct the five K1-N boundaries:

```text
ciphertext
-> inverse linear slot 1
-> inverse S-box slot 1
-> inverse linear slot 0
-> inverse S-box slot 0
```

At each retained boundary, convert the endpoint XOR into the native sixteen
4-bit cell values. For every sample, record the frequency of each value
`0..15`, averaging over the four pairs only. Position-preserving views retain
the separate `[stage, cell, value]` axes.

The frozen panel is:

| View | Features | Purpose |
|---|---:|---|
| `raw_position_histogram` | `1 x 16 x 16 = 256` | ciphertext-only anchor |
| `exact_five_stage_position_histogram` | `5 x 16 x 16 = 1280` | candidate |
| `no_sbox_five_stage_position_histogram` | `1280` | tests nonlinear contribution |
| `wrong_sbox_five_stage_position_histogram` | `1280` | tests correct heterogeneous S-box semantics |
| `exact_five_stage_invariant_histogram` | `5 x 16 = 80` | tests loss from pooling cell identity |
| `label_shuffled_exact_position_histogram` | `1280` | deterministic training-label control |

The wrong-S-box view uses the same deterministic uKNIT cell-assignment shuffle
as K1-N and leaves both linear matrices unchanged. The no-S-box view replaces
only the two inverse S-box applications with identity. The label-shuffle view
uses the exact candidate features, a seed-bound nonidentity permutation of
training labels, and the untouched true labels on evaluation rows.

## 5. Closed-Form Scorer

Each view independently fits one diagonal Fisher/LDA scorer on the frozen
training split:

```text
pooled_variance_j =
  ((n0 - 1) * variance0_j + (n1 - 1) * variance1_j) / (n0 + n1 - 2)

weight_j = (mean1_j - mean0_j) / (pooled_variance_j + 1e-6)

score(x) = (x - 0.5 * (mean0 + mean1)) dot weight
```

The variance floor is fixed at `1e-6`. Balanced priors fix the intercept; no
calibration or threshold search is allowed. AUC is computed directly from the
signed score, whose orientation is fixed by `mean1 - mean0`. Record score AUC,
zero-threshold accuracy, feature/scorer digests, class counts, weight norm,
nonzero-weight count, and dataset digest.

## 6. Protocol And Advance Gates

The audit is valid only if:

1. exactly `36` result rows exist (`2 seeds x 3 splits x 6 views`);
2. all source digests, dataset digests, row counts, labels, and runtime
   structures match K1-N;
3. all histogram rows are finite, nonnegative, and normalized on their declared
   stage/cell axes;
4. the exact and label-shuffled views share identical feature digests;
5. wrong S-boxes change the exact feature digest while preserving linear
   matrices;
6. every scorer is fitted only on its seed's train-seen rows;
7. training rows, optimizer steps, epochs, and neural parameters are all zero.

Apply every research threshold separately to both seeds and both fresh splits:

```text
exact position AUC                         >= 0.550
exact position - raw position AUC          >= +0.010
exact position - no-S-box AUC              >= +0.005
exact position - wrong-S-box AUC           >= +0.005
exact position - label-shuffled AUC         >= +0.030
```

Cell-position necessity is a separate attribution check:

```text
exact position - exact invariant AUC        >= +0.010
```

No mean may hide a failed seed or fresh split.

## 7. Decisions And Executable Next Action

- **All signal/semantic gates pass and position beats invariant:** implement a
  K1-P position-preserving cell-stage neural head. Keep K1-N data and budget,
  change only the final cell/stage aggregation, and retain raw, no-S-box,
  wrong-S-box, invariant, and label-shuffle controls.
- **All signal/semantic gates pass but invariant ties position:** implement a
  smaller K1-P invariant stage-histogram branch; do not add absolute cell
  identity because the audit did not justify it.
- **Exact signal clears `0.550` but correct and wrong/no-S-box views tie:** stop
  treating S-box semantics as the next variable. Localize the signal to raw or
  linear stages before another neural model.
- **All exact fresh AUC rows remain below `0.550`:** stop changing the network
  on the current uKNIT r5 differential. The next experiment must audit or
  replace the input differential/data protocol while preserving strict
  encrypted-random-plaintext negatives.
- **Seeds/splits disagree:** classify the signal as unstable and audit the
  frozen differential per key/split before any scale or architecture change.
- **Protocol invalid:** repair only the failed source, feature, scorer, or
  artifact invariant and rerun K1-O unchanged.

## 8. Run And Required Artifacts

```text
run_id = i1_uknit_family_ctspn_exact_partial_state_signal_audit_k1o_20260728
output = outputs/local_audit/<run_id>/
```

Required artifacts:

```text
preflight.json
dataset_manifest.jsonl
feature_manifest.jsonl
scorer_manifest.jsonl
results.jsonl
attribution.csv
validation.json
gate.json
summary.json
progress.jsonl
curves.svg
plot_report.json
visual_qa_passed.marker
```

The chart must explain the result in Chinese and pass rendered-pixel
`visual-qa-redraw`. After completion, refresh both recent-result indexes.

## 9. Claim Boundary And Blocked Routes

K1-O is a two-seed local `2048/class` deterministic mechanism audit. It cannot
establish formal scale, an attack, SOTA, a uKNIT ceiling, or arbitrary-SPN
transfer. Before K1-O is adjudicated, do not launch remote scale, add epochs,
samples, pairs, seeds, width, MoE, experts, DDT/trail inputs, cipher identity,
or another neural architecture.
