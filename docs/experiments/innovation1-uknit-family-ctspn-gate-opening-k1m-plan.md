# Innovation 1 uKNIT-Family CT-SPN Gate Opening K1-M

**Date:** 2026-07-28
**Status:** completed / gate opened, uKNIT signal not supported
**Execution:** local CPU readiness followed by a fixed-budget local diagnostic

## 1. Research Question

K1-L showed that K1-K's exact-zero gate made all edge-residual parameter
gradients exactly zero at initialization. Both uKNIT checkpoints retained
effective gates below `0.00031`, so K1-K did not meaningfully train the new
branch. K1-M tests one change:

> Does initializing the bounded effective residual gate to `0.05` allow the
> unchanged position-preserving edge branch to learn stable uKNIT fresh-split
> signal while retaining Dialga and preferring the correct operators?

K1-M changes no architecture width, parameter shape, data, label, key, split,
sample count, pair count, epoch, optimizer, loss or metric.

## 2. Frozen Sources

```text
K1-K root = outputs/local_diagnostic/
  i1_uknit_family_ctspn_topology_edge_residual_k1k_2048_seed0_seed1_20260728
K1-L root = outputs/local_audit/
  i1_uknit_family_ctspn_residual_attribution_k1l_20260728
```

| Artifact | SHA-256 |
|---|---|
| K1-K gate | `8922bd1d03de41547f33329b869204d2d05d664514674699f6661a7eaf758055` |
| K1-K checkpoint manifest | `1c826e182c3762d389a6d575ddbc755331a6a0123fcba87dde7f856006b8473f` |
| K1-K dataset manifest | `ecc990e4d724ec35fdce8bd52d947c78280db2140853feddee07189ade4341f0` |
| K1-K controls | `b08832d8f01fe0091a1a1f07e507dc830833662204c2fbc618f9702eca06d3a0` |
| K1-L gate | `8be0ba47a207e4cf9af0c51b73c787e9d5c53c02c7ab9be47e4d57271fef6d70` |

All source protocol checks must remain true. K1-K must retain its exact hold
decision and K1-L must retain its exact starvation-supported decision.

## 3. Single Variable

Add one model option:

```text
residual_gate_initial_effective = 0.05
```

The raw learned scalar is initialized to `atanh(0.05)`, so its effective
multiplier is exactly `tanh(raw) = 0.05`. It remains a trainable scalar and the
existing bounded fusion remains:

```text
k1m_embedding = k1i_style_embedding
              + tanh(residual_gate) * tanh(edge_residual_embedding)
```

The default remains exact zero so K1-K behavior and checkpoints do not change.
No fixed gate, warm-up phase, auxiliary loss, extra optimizer group or altered
learning rate is allowed in K1-M.

## 4. Readiness Gate

Before training require:

1. exactly the K1-K architecture and parameter geometry (`128707` parameters)
   on both widths;
2. exact initial effective gate `0.05` within `1e-7` on all four candidate
   rows, while the K1-K default remains exact zero;
3. zeroing the gate reproduces the corresponding base-path logits within
   `1e-7`;
4. on a fixed digest-bound batch, every cell/edge/update/residual projection
   group has nonzero gradient above `1e-8` before any optimizer step;
5. correct, reversed, corrupted and no-topology models strict-load the same
   state and retain distinct Boolean-view and edge fingerprints;
6. both transition slots remain observable and joint cell relabeling remains
   within `1e-6`;
7. all twelve K1-K caches are exact-digest reusable; readiness consumes zero
   training rows and zero optimizer steps.

Any failure authorizes only repair of the failed binding or implementation and
an unchanged readiness rerun.

## 5. Frozen Local Diagnostic

| Field | Frozen value |
|---|---|
| Ciphers / rounds | uKNIT-BC r5; Dialga-128 r4 |
| Candidate | unchanged K1-K edge residual, effective gate initialized `0.05` |
| Same-budget anchor | completed K1-K exact candidate rows |
| Seeds | `0`, `1` |
| Samples | `2048/class` train; `1024/class` fresh same-key; `1024/class` cross-key |
| Pairs per sample | `4` |
| Negative definition | encrypted random plaintexts |
| Keys / differences | exact K1-K values |
| Epochs / batch | `10` / `64` |
| Loss / optimizer | MSE / Adam |
| Learning rate / weight decay | `1e-4` / `1e-5` |
| Checkpoint | best validation AUC, restored before controls |
| Device | local CPU |

All caches must be reused without regeneration. This is a local small mechanism
diagnostic, not formal training or a family conclusion.

## 6. Advance Gate

Apply every threshold separately to both fresh splits and both seeds.

For uKNIT-BC:

```text
candidate AUC                          >= 0.520
candidate AUC - same-row K1-K AUC     >= +0.005
candidate AUC - every topology control >= +0.005
abs(final effective gate)             >= 0.010
```

For Dialga-128:

```text
candidate AUC - same-row K1-K AUC     >= -0.005
candidate AUC - every topology control >= +0.005
```

Training-split rows are descriptive only. No mean may hide a failed row.

## 7. Decisions And Next Action

- **All gates pass:** retain K1-M and create a separate remote `65536/class`
  disk-cached diagnostic plan; do not call it formal evidence.
- **uKNIT gate opens but fresh rows fail:** gate starvation is repaired but the
  linear-edge representation remains insufficient. Stop gate schedules and
  make exact heterogeneous S-box/operator composition the next single variable.
- **uKNIT gate closes again:** discard trainable scalar opening and test one
  fixed bounded `0.05` fusion locally only if residual gradients stayed
  nonzero; otherwise stop this residual route.
- **Dialga anchor retained but controls fail:** preserve the K1-L conclusion
  that linear edges are not operator specific; no scale.
- **Dialga anchor lost:** discard K1-M and return to K1-K/K1-I calibration.
- **Protocol invalid:** repair only the failed implementation or binding and
  rerun unchanged.

Do not add samples, epochs, pairs, seeds, width, experts, S-box/DDT/trail
features, keys, cipher IDs, partial decryption or a raw bypass inside K1-M.

## 8. Run IDs And Artifacts

```text
readiness_run_id = i1_uknit_family_ctspn_gate_opening_k1m_readiness_20260728
training_run_id  = i1_uknit_family_ctspn_gate_opening_k1m_2048_seed0_seed1_20260728
```

Readiness must produce preflight, results, validation, gate and progress files.
Training must produce the same complete artifact family as K1-K, including four
checkpoints, sixty evaluation rows, final learned gate values, gate, validation,
history and progress. A result chart is required and must pass rendered-pixel
`visual-qa-redraw`. Refresh both recent-result indexes after readiness and after
the completed diagnostic.

## 9. Completed Result

The zero-training readiness gate passed before optimization:

```text
decision = innovation1_uknit_family_ctspn_k1m_execution_authorized
initial effective gate = 0.049999997
parameter count = 128707
all residual parameter groups receive nonzero gradients = true
K1-K default effective gate remains exact zero = true
all twelve source caches are digest-bound and reusable = true
training rows = 0
optimizer steps = 0
```

The fixed local diagnostic then completed without regenerating any training or
validation cache:

```text
training rows = 4 / 4
evaluation rows = 60 / 60
validation status = pass
errors = []
```

Fresh-split candidate AUC and same-row K1-K deltas were:

| Cipher | Seed | Split | K1-M AUC | K1-K anchor | Delta |
|---|---:|---|---:|---:|---:|
| uKNIT-BC r5 | 0 | same-key fresh | `0.508395` | `0.506373` | `+0.002022` |
| uKNIT-BC r5 | 0 | cross-key | `0.518564` | `0.514879` | `+0.003685` |
| uKNIT-BC r5 | 1 | same-key fresh | `0.490501` | `0.484057` | `+0.006444` |
| uKNIT-BC r5 | 1 | cross-key | `0.509111` | `0.505805` | `+0.003307` |
| Dialga-128 r4 | 0 | same-key fresh | `0.966957` | `0.965747` | `+0.001210` |
| Dialga-128 r4 | 0 | cross-key | `0.959439` | `0.957891` | `+0.001548` |
| Dialga-128 r4 | 1 | same-key fresh | `0.960375` | `0.958886` | `+0.001489` |
| Dialga-128 r4 | 1 | cross-key | `0.956191` | `0.954312` | `+0.001879` |

The learned effective gates remained active:

```text
uKNIT seed0 = 0.049259
uKNIT seed1 = 0.050336
Dialga seed0 = 0.080679
Dialga seed1 = 0.095307
```

K1-M therefore repaired K1-K's gradient-starvation mechanism and preserved the
Dialga calibration signal. It did not pass the frozen research gate: every
uKNIT fresh AUC stayed below `0.520`, improvement over K1-K was not at least
`+0.005` on every fresh row, and correct Dialga operators did not beat every
wrong-operator control by `0.005` on either fresh split.

```text
status = hold
decision = innovation1_uknit_family_ctspn_k1m_gate_opened_uknit_signal_not_supported
remote_scale = no
```

This is evidence that nonzero gate initialization is not the remaining primary
bottleneck. The claim is limited to the two-seed local `2048/class` mechanism
diagnostic; it is not a formal-scale failure or a uKNIT-family ceiling.

## 10. Recommended Next Action

Proceed to K1-N with one architectural variable: retain the K1-M bounded
position-preserving edge residual and effective-gate initialization, then add
the exact runtime order of heterogeneous inverse S-box and inverse linear
operators. The deterministic stage sequence must expose both ciphertext-pair
endpoints and their XOR difference through:

```text
ciphertext
-> inverse linear slot 1
-> cell-specific inverse S-box slot 1
-> inverse linear slot 0
-> cell-specific inverse S-box slot 0
```

Use K1-M as the same-budget anchor and include correct S-box/correct-linear,
shuffled-S-box/correct-linear, correct-S-box/reversed-linear,
correct-S-box/corrupted-linear, no-S-box-composition and no-topology controls.
Keep uKNIT-BC r5, Dialga-128 r4, `2048/class`, both fresh splits, four pairs,
ten epochs, batch 64 and seeds 0/1 unchanged. K1-N must pass its local mechanism
gate before any remote scale; do not add samples, epochs, pairs, seeds, width,
MoE, DDT/trail features, key/cipher IDs, partial decryption or a raw bypass.
