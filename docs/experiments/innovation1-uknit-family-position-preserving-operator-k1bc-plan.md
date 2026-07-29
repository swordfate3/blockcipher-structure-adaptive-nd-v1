# Innovation 1 K1-BC Position-Preserving Operator Training

**Status:** completed / hold
**Date:** 2026-07-29
**Run ID:** `i1_uknit_family_position_preserving_operator_k1bc_2048_replica0_replica1_20260729`

## 1. Research Question

K1-BB proved that a shared edge-token encoder can distinguish K1-BA's
same-summary operators, transmit the difference through sample-conditioned edge
messages to the frozen K1-AZ classifier, exactly replay K1-AZ when disabled,
and share one `41088`-parameter geometry across 64- and 128-bit states.

K1-BC tests the necessary next hypothesis:

> When only this new shared operator encoder is trained, does it retain or
> improve K1-AZ on the same data budget while assigning higher fresh-split AUC
> to the correct actual GF(2) topology than to same-summary corrupted and
> compatible cross-cipher operators?

This separates representational observability from learned attribution. K1-BB
passing does not imply K1-BC will pass.

## 2. One Trainable Variable

For each replica:

1. restore the exact K1-AZ epoch-9 checkpoint;
2. freeze every K1-AZ parameter, including base encoder, GF(2) edge residual,
   S-box transition residual, scalar gates and classifier;
3. initialize only K1-BB's shared `41088`-parameter operator encoder with the
   preregistered replica seed;
4. train that encoder jointly across all three ciphers with one optimizer and
   equal batches per cipher.

The enabled edge branch is:

```text
frozen K1-AZ global edge bias
  + 0.05 * tanh(shared position-preserving sample modulation)
  -> channelwise scale of the frozen exact GF(2) edge residual
```

The S-box transition branch remains the exact K1-AZ branch. No cipher ID,
per-cipher parameter, learned position table, adapter, router, expert or MoE is
allowed.

## 3. Same-Budget Protocol

| Field | Frozen value |
|---|---|
| Ciphers / rounds | uKNIT-BC r5; Midori64 r4; Dialga-128 r4 |
| Replicas | `0/1` |
| Dataset seeds | replica0: `3/6/0`; replica1: `4/7/1` |
| Train | `2048/class/cipher` = 4096 total rows per cipher |
| Fresh same-key | `1024/class/cipher` |
| Fresh cross-key | `1024/class/cipher` |
| Pairs per sample | `4` |
| Negative definition | encrypted random plaintexts |
| Epochs | `10` |
| Batch size | `64` |
| Cipher batches per epoch | `64` each, interleaved equally |
| Optimizer steps | `192/epoch`, `1920/replica` |
| Loss / optimizer | MSE / Adam |
| Learning rate / weight decay | `1e-4` / `1e-5` |
| Checkpoint metric | minimum cross-key AUC across the three ciphers |
| Device | local CPU |

All eighteen K1-AZ disk-backed datasets must be reused by exact digest. No row
may be regenerated or relabeled.

## 4. Frozen Same-Checkpoint Controls

Restore each selected K1-BC checkpoint and evaluate every cipher/split without
an optimizer step:

| Condition | Operator supplied to K1-BB | Runtime encryption / K1-AZ structure |
|---|---|---|
| `correct_operator` | correct native operator | correct |
| `same_summary_corrupted_operator` | K1-BA column-permuted operator | correct |
| `cross_cipher_operator` | compatible foreign operator below | correct |
| `disabled_k1az` | new path disabled | exact K1-AZ replay |

Cross-cipher operators preserve sample width:

```text
uKNIT-BC 64-bit  <- Midori64 native operator
Midori64 64-bit <- uKNIT-BC native operator
Dialga-128       <- block diagonal of two uKNIT-BC 64-bit operators
```

The lifted Dialga control is invertible and uses Dialga's correct cell, S-box
and encryption runtime. Only the operator supplied to the new modulation path
changes.

Expected evaluation is:

```text
2 replicas x 3 ciphers x 2 splits x 4 conditions = 48 rows
```

## 5. Protocol Gates

Require all K1-BB and K1-AZ artifact hashes, both source checkpoints and all
dataset digests to match. Each replica must complete exactly ten epochs and
`1920` optimizer steps on only the `41088` new parameters. The selected
checkpoint must be restored before all controls. Evaluation must use zero
optimizer steps, one immutable state and the correct encryption runtime.

The native and lifted cross operators must be binary, invertible, width
compatible and distinct from the correct operator.

## 6. Research Gates

For each replica, compute the macro AUC over the three cross-key
`correct_operator` rows and compare with K1-AZ's exact same-data macro:

```text
candidate cross-key macro - K1-AZ macro >= 0.0
```

For all twelve replica/cipher/split panels:

```text
correct candidate - K1-AZ anchor >= -0.005
```

For each topology control independently:

```text
correct candidate - control >= +0.001
passing panels                  >= 10/12
passing panels per cipher       >= 3/4
both replicas and both splits must appear among passing panels
```

`disabled_k1az` is the same-budget anchor, not a topology mismatch control.
Macro averages may not hide per-panel harm or a cipher with fewer than three
passing attribution panels.

## 7. Decisions

- **All gates pass:** retain K1-BB as the local uKNIT-family candidate and
  preregister a remote cache/resume readiness audit for `65536/class/cipher`.
  Do not launch until disk-backed cache and Git source gates pass.
- **AUC retains/improves but topology controls fail:** hold K1-BB and run a
  frozen-checkpoint optimization-attribution audit. Do not scale a shortcut.
- **Topology controls pass but macro/no-harm fails:** hold the family claim and
  inspect which cipher's shared gradient updates cause harm. Do not add
  capacity or loss balancing yet.
- **Both performance and attribution fail:** discard K1-BC training as
  unsupported at this local diagnostic and return to representation/optimizer
  analysis, not mechanical scale.
- **Protocol invalid:** repair only the failed binding and resume/replay the
  unchanged protocol.

No branch authorizes 16 pairs, more local data, epochs, seeds or width,
unfreezing K1-AZ, per-cipher modules, MoE, PCGrad or a benchmark change.

## 8. Required Artifacts

Write under `outputs/local_diagnostic/<run_id>/`:

```text
preflight.json
dataset_manifest.jsonl
operator_controls.json
results.jsonl
controls.jsonl
checkpoint_manifest.json
history.csv
gate.json
validation.json
summary.json
progress.jsonl
curves.svg
visual_qa_render_report.json
visual_qa_passed.marker
checkpoints/
```

After completion, append exact metrics and the evidence-backed next action to
this document, refresh both recent-result indexes, run focused regressions,
commit only K1-BC files and verify the exact pushed SHA.

## 9. Completed Result

The local diagnostic completed on 2026-07-29 with all protocol checks valid.
Both replicas completed exactly ten epochs and `1920` optimizer steps, and the
same selected checkpoint was restored for all zero-step controls.

Cross-key macro AUC did not retain the exact same-data K1-AZ anchor:

| Replica | K1-BC candidate | K1-AZ anchor | Delta | Gate |
|---|---:|---:|---:|---|
| 0 | `0.736684322` | `0.739590327` | `-0.002906005` | fail |
| 1 | `0.762597561` | `0.763059139` | `-0.000461578` | fail |

The per-panel no-harm gate also failed. The failing panel was uKNIT-BC
replica0 on fresh same-key data:

```text
candidate AUC = 0.624919891
K1-AZ AUC     = 0.631846428
delta         = -0.006926537
allowed floor = -0.005000000
```

Correct-topology attribution failed decisively for both required controls:

| Topology control | Passing panels | Required | Per-cipher passing panels |
|---|---:|---:|---|
| Same-summary corrupted operator | `0/12` | `>=10/12` | uKNIT `0/4`; Midori `0/4`; Dialga `0/4` |
| Cross-cipher operator | `0/12` | `>=10/12` | uKNIT `0/4`; Midori `0/4`; Dialga `0/4` |

The correct-versus-control margins were generally only `1e-6` to `1e-5` AUC
and sometimes negative, far below the preregistered `+0.001` requirement. The
new representation is observable before training, as established by K1-BB,
but the trained classifier output is effectively insensitive to which actual
operator the new path receives.

## 10. Verdict And Next Action

```text
status   = hold
decision = innovation1_uknit_family_k1bc_position_preserving_operator_training_not_supported
```

This is a local `2048/class/cipher`, four-pair, two-replica mechanism
diagnostic. It is not formal training, a failure at scale, an attack,
arbitrary-SPN generalization, unseen-cipher transfer or SOTA evidence.

The next experiment must reuse the frozen K1-BC checkpoints and change no
training budget. It will audit three quantities separately for each cipher and
replica:

1. gradient norm entering the shared operator encoder;
2. learned channelwise modulation magnitude at the frozen K1-AZ residual;
3. pairwise cosine similarity of per-cipher encoder gradients.

The decision unlocked by that audit is whether K1-BC failed because the fixed
`0.05 * tanh(...)` path receives too little optimization signal or because
joint uKNIT/Midori/Dialga gradients cancel in the shared encoder. Until that
question is resolved, do not run 16 pairs, larger data, more epochs, wider
models, extra seeds, remote GPU, loss balancing, PCGrad, per-cipher modules,
routers or MoE.
