# Innovation 1 uKNIT-Family CT-SPN Residual Attribution K1-L

**Date:** 2026-07-28
**Status:** frozen before implementation
**Execution:** local CPU, zero-training mechanism audit

## 1. Decision Context And Question

K1-K retained K1-I's `0.954312-0.965747` Dialga AUC but did not make the
correct matrices beat every wrong-operator control by `0.005`. uKNIT r5 stayed
near chance on both fresh same-key rows. A preliminary read-only checkpoint
inspection after the K1-K decision exposed the learned raw residual gates:

```text
uKNIT seed0  = -0.000230862
uKNIT seed1  = -0.000305697
Dialga seed0 = +0.029130256
Dialga seed1 = -0.022540038
```

Because those values were inspected before this plan was written, K1-L is a
descriptive mechanism audit, not a blind confirmatory gate. It asks:

> Did uKNIT fail because exact-zero gate initialization starved the new edge
> branch of gradients, while Dialga opened the branch but used an
> operator-insensitive residual?

K1-L changes no weights, data, labels, keys, metrics or checkpoints and takes
no optimizer step.

## 2. Frozen Source Authority

```text
source_root = outputs/local_diagnostic/
  i1_uknit_family_ctspn_topology_edge_residual_k1k_2048_seed0_seed1_20260728
```

| Source artifact | SHA-256 |
|---|---|
| `gate.json` | `8922bd1d03de41547f33329b869204d2d05d664514674699f6661a7eaf758055` |
| `checkpoint_manifest.json` | `1c826e182c3762d389a6d575ddbc755331a6a0123fcba87dde7f856006b8473f` |
| `dataset_manifest.jsonl` | `ecc990e4d724ec35fdce8bd52d947c78280db2140853feddee07189ade4341f0` |
| `controls.jsonl` | `b08832d8f01fe0091a1a1f07e507dc830833662204c2fbc618f9702eca06d3a0` |
| `validation.json` | `02583f9045e8d4fd54900e3f54a40a56939d1516356beaa93a6e584e1d06fe8c` |

Frozen checkpoint digests:

| Cipher | Seed | SHA-256 |
|---|---:|---|
| uKNIT-BC | 0 | `f50a17dd09e5c7084081c1233f9b80bccc89aff342baa3d00a47c9f8ac320626` |
| uKNIT-BC | 1 | `fa58547e023c7a620b666d5e19a89790677f98d17f119abd7ad29f28542359c0` |
| Dialga-128 | 0 | `d5db221f288b18940744af13f38d7ba543715e45f46c326fb0511c2fbb35b7a4` |
| Dialga-128 | 1 | `b2515e623507ff1374cb9debb743b4ec1c030bed7759998d10f5c8b5d6403846` |

The source gate must remain a valid `hold` with decision:

```text
innovation1_uknit_family_ctspn_k1k_dialga_retained_operator_attribution_not_supported
```

## 3. Frozen Audit Panel

For each cipher, seed and cached split, strict-load the selected K1-K state and
evaluate these conditions without an optimizer:

| Condition | Intervention | Mechanism tested |
|---|---|---|
| `native_full` | trained gate and both native transition slots | exact K1-K replay |
| `gate_zero` | force only the scalar gate to zero | trained base path without residual fusion |
| `slot0_only` | retain only transition slot 0 in the residual | first operator contribution |
| `slot1_only` | retain only transition slot 1 in the residual | second operator contribution |
| `residual_row_shuffle` | deterministic label-blind row permutation of the native residual before fusion | sample-specific residual association |
| `reversed_full` | exchange transition slots | schedule attribution replay |
| `corrupted_full` | frozen source-column corruption | edge attribution replay |
| `no_topology_full` | identity operators | topology-removal replay |

Record full AUC, zero-gate AUC, residual logit contribution AUC, full-minus-zero
AUC, mean/RMS/max absolute residual logit contribution, residual embedding norm,
raw gate and `abs(tanh(gate))`. The exact, reversed, corrupted and no-topology
full AUC values must replay K1-K `controls.jsonl` within `1e-7`.

The row permutation is seeded only by cipher, seed and split identifiers. It
must be nonidentity, deterministic, label blind and applied to all residual
coordinates of a sample together. It may not inspect or rebalance labels.

## 4. Gradient-Path Proof

On one fixed digest-bound batch for every cipher/seed checkpoint, run backward
without an optimizer under two scalar-gate conditions:

1. gate forced to exact `0.0`;
2. gate forced to `atanh(0.05)` so the effective multiplier is exactly `0.05`.

Use the frozen MSE objective and labels. Record separate gradient norms for the
scalar gate, cell encoder, edge encoder, cell update and residual projection.
Restore the checkpoint state exactly afterward.

The algebraic starvation proof passes only if every residual-path parameter
except the scalar gate has gradient norm at most `1e-12` at exact zero, while at
least one edge/cell/residual parameter group has norm above `1e-8` at effective
gate `0.05`. This backward-only audit performs zero optimizer steps and must not
write a checkpoint.

## 5. Descriptive Mechanism Thresholds

The thresholds classify the already observed mechanism; they do not revise the
failed K1-K research gate.

```text
effectively closed gate: abs(tanh(gate)) <= 0.001
active gate:             abs(tanh(gate)) >= 0.010
operator-specific residual contribution:
  exact contribution AUC - each wrong-operator contribution AUC >= 0.005
  on both fresh splits and both seeds
sample-specific residual contribution:
  row shuffle explains at least 80% of abs(full AUC - zero-gate AUC)
```

No average may hide a failed seed or fresh split. Training-split effects are
descriptive only.

## 6. Decisions And Executable Next Action

- **uKNIT gates closed and zero-gate gradient starvation passes:** K1-M changes
  only the gate-opening schedule. Use a fixed small effective gate or a short
  residual-only warm-up, retain K1-K architecture/data/controls, and rerun the
  same local `2048/class`, two-seed budget. Do not add S-box semantics yet,
  because K1-K did not actually test a trainable uKNIT edge branch.
- **uKNIT gates active but residual carries no fresh-split association:** stop
  gate scheduling and pure linear-edge rescue. The next single variable is
  exact heterogeneous S-box/operator composition.
- **Dialga gate active but residual is not operator specific:** record that
  explicit linear edges alone are insufficient for correct-operator semantics;
  any later Dialga/uKNIT model must retain wrong-operator controls.
- **Residual is operator specific but fusion hides it:** change only bounded
  fusion/normalization, not the edge encoder or data.
- **Protocol or replay invalid:** repair only the failed binding or intervention
  and rerun K1-L unchanged.

## 7. Scale, Artifacts And Blocked Routes

```text
run_id = i1_uknit_family_ctspn_residual_attribution_k1l_20260728
output = outputs/local_audit/<run_id>/
```

Required artifacts:

```text
preflight.json
results.jsonl
gradient_attribution.jsonl
validation.json
gate.json
progress.jsonl
```

K1-L uses zero new samples, zero epochs and zero optimizer steps on local CPU.
It cannot establish formal scale, an attack, a SOTA result, arbitrary-SPN
transfer or an uKNIT ceiling. Do not launch remote scale, add data, epochs,
pairs, seeds, width, MoE, S-box/DDT/trail inputs, keys, cipher IDs, partial
decryption or a raw bypass before this mechanism audit is complete.
