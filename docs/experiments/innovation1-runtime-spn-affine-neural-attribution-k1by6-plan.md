# Innovation 1 Runtime SPN Affine Neural Attribution K1-BY6

**Date:** 2026-08-01
**Status:** preregistered / not yet observed
**Execution:** local CPU fallback, two-row sub-medium diagnostic

## Research question

K1-BY5 proved that the affine wrong-endpoint control changes every frozen
PRESENT feature tap even after cell-order-invariant pooling. K1-BY6 changes one
thing only:

> With all data and optimization held fixed, does the learned primitive
> conditioner assign a consistently higher validation AUC to the correct
> PRESENT P layer than to this identifiable affine wrong P layer?

This is a neural attribution diagnostic. It is not formal-scale, remote-scale,
cross-cipher transfer, attack or SOTA evidence.

## Frozen source authority

K1-BY6 binds K1-BY3's plan, preflight, results, validation and gate by SHA-256.
K1-BY3 must remain a valid completed local diagnostic with:

```text
status   = hold
decision = innovation1_runtime_spn_k1by3_permutation_attribution_not_supported
```

Only these historical rows are used as read-only anchors:

| Seed | Correct routing AUC | No-conditioner AUC |
|---:|---:|---:|
| 2 | `0.6837368011474609` | `0.5437998771667480` |
| 3 | `0.6655435562133789` | `0.5272355079650879` |

K1-BY5's config, preflight, results, validation and gate are also bound by
SHA-256 and must retain:

```text
status   = pass
decision = innovation1_runtime_spn_k1by5_affine_endpoint_control_ready
```

Historical models are not retrained. The exact K1-BY3 disk cache is reused
read-only for both new rows.

## Single variable

The correct ordered primitive program is replaced by the frozen K1-BY5 source
endpoint map:

```text
u  = 4 * source_cell + source_role
u' = (5 * u + 1) mod 64
```

S-box tables, stage order, model shape, parameter count, initialization seed,
input data, labels and optimizer remain unchanged. Both correct and affine
programs must route exactly 32 S-box cells and 32 `linear_permutation` cells,
with zero `linear_gf2` cells and no cipher or absolute-cell identity.

## Frozen protocol

| Field | Value |
|---|---|
| Cipher / rounds | PRESENT-80 / 7 |
| Difference | `present_zhang_wang2022_mcnd:0`, input XOR `0x9` |
| Sample structure | `zhang_wang_case2_official_mcnd` |
| Negative definition | encrypted random plaintexts |
| Key schedule | fresh random key per pair |
| Pairs/sample | 16 |
| Train | `2048/class`, 4096 total |
| Validation | `1024/class`, 2048 total |
| Seeds | 2 and 3 |
| Epochs / batch | 10 / 64 |
| Loss / optimizer | MSE / Adam |
| LR / weight decay | `1e-4` / `1e-5` |
| Checkpoint | best `val_auc` |
| Parameters | 235780 |
| New training rows | exactly 2, affine wrong endpoint only |

Local CUDA was checked immediately before preregistration and was unavailable
(`cuda_available=false`, `device_count=0`). A CPU run is allowed only because
this is a two-row `2048/class` sub-medium diagnostic. No device change is
allowed after readiness or between rows.

## Readiness gate

Before any optimizer step, require all of the following:

- exactly two plan rows and exact K1-BY3 protocol equality;
- exact SHA bindings and required K1-BY3/K1-BY5 decisions;
- correct and affine program digests differ;
- affine program mode is `source_endpoint_affine_m5_b1_mod64`;
- equal 235780-parameter geometry and finite forward/backward values;
- identical expert usage: 32 S-box, 32 permutation and zero GF(2) cells;
- no cipher identity or absolute cell/bit identity;
- the exact K1-BY3 cache metadata and arrays exist for both seeds;
- execution device is frozen to CPU before optimization.

Any failed invariant makes the run invalid and authorizes repair only.

## Preregistered result gate

For each seed separately:

```text
correct K1-BY3 AUC - new affine-control AUC >= +0.005
```

Both seed2 and seed3 must pass. No average or stronger seed may rescue a failed
seed. Accuracy and calibrated accuracy are supporting metrics only.

## Decisions

- **Pass:** local permutation neural attribution is supported. Keep the affine
  control and test the validated permutation expert/control on GIFT at the same
  local diagnostic budget before any remote expansion.
- **Hold:** run a learned-access audit over the linear histogram, primitive
  expert output, cell fusion, invariant pooling and final taps using the frozen
  K1-BY3/K1-BY6 checkpoints. Do not increase data, epochs, pairs, width, seeds,
  rounds or ciphers.
- **Invalid:** repair only the failed source binding, plan, cache, model,
  parameter, device, progress or artifact invariant and rerun unchanged.

## Planned artifacts

```text
outputs/local_diagnostic/
i1_runtime_spn_affine_neural_attribution_k1by6_present_r7_16pair_2048_seed2_seed3_20260801/
  preflight.json
  results.jsonl
  condition_comparison.csv
  gate.json
  validation.json
  summary.json
  progress.jsonl
  history.csv
  curves.svg
  visual_qa_render_report.json
```

After observation, append exact AUC/accuracy values, per-seed margins, gate
status, claim boundary and the executable next action here, then refresh both
recent-result indexes.
