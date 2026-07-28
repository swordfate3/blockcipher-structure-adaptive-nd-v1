# Innovation 1 K1-AO Multi-Cipher Shared-Weight Readiness

**Status:** completed / pass / shared runtime ready
**Date:** 2026-07-29
**Run ID:** `i1_uknit_family_multicipher_shared_weight_k1ao_readiness_20260729`

## 1. Decision Context

K1-AM showed that the correct Midori64 S-box affects a fixed checkpoint, but an
independently trained wrong-S-box model can learn a substitute. K1-AN removed
the learned transition coordinate system with a fixed Walsh basis; the branch
remained useful, but fresh AUC fell by `0.0275` to `0.0930` relative to K1-AK
and correct-S-box identification was still not stable.

The next single hypothesis is therefore weight ownership, not another feature
family or larger query budget:

```text
old protocol = one independently optimized backbone per runtime descriptor
K1-AO       = one backbone shared by uKNIT-BC, Midori64, and Dialga-128
```

If a wrong descriptor cannot receive its own compensating weights, correct
runtime semantics may become identifiable without discarding K1-AK's useful
learned transition representation.

## 2. Exact Question

Can one K1-AK parameter geometry accept a runtime descriptor on every batch
across 64-bit/16-cell and 128-bit/32-cell SPNs, while keeping the same parameter
objects and excluding cipher identity, absolute cell identity, keys, labels and
difference metadata from the neural input?

This run answers only implementation and evidence readiness. It performs no
optimizer step and produces no neural quality claim.

## 3. Frozen Authorities

| Cipher | Surface | Seeds | Difference | Runtime window | Input |
|---|---|---:|---|---|---|
| uKNIT-BC r5 | K1-T/K1-Q confirmed caches | 3, 4 | cell11 role1, `0x0000400000000000` | start 3, two transitions | four pairs, 512 bits |
| Midori64 r4 | K1-AK/K1-AH confirmed caches | 6, 7 | cell8 role1, `0x0000000400000000` | start 0, two transitions | four pairs, 512 bits |
| Dialga-128 r4 | K1-N/K1-W source caches | 0, 1 | `0x40` | start 2, two transitions | four pairs, 1024 bits |

Each seed binds `train_seen`, `same_key_fresh`, and
`cross_key_validation`, for exactly 18 read-only cache payloads. The config
freezes the gate, validation and dataset-manifest SHA-256 values for all three
source roots.

## 4. Single Implementation Variable

Retain K1-AK's learned compact S-box-transition path and all parameter shapes.
Expose branch enablement as an explicit argument of the existing
`logits_with_runtime` call so a shared checkpoint can be evaluated under:

```text
correct runtime
wrong S-box runtime, same checkpoint
transition branch off, same checkpoint
```

The intervention must not mutate model parameters or persistent runtime state.
No cipher-specific expert, adapter, output head, embedding or shape branch is
allowed.

## 5. Readiness Gate

All checks must pass:

1. The frozen JSON configuration and all nine source file digests match.
2. Exactly 18 selected cache rows exist and every `features.npy`/`labels.npy`
   payload recomputes to the manifest dataset hash without generation.
3. All three K1-AK instances expose identical parameter names and shapes,
   exactly `219320` trainable parameters and `52` state entries.
4. Seed 29 creates the same tensor-state hash for all three descriptors.
5. One uKNIT state dict strict-loads into Midori64 and Dialga models.
6. One shared model produces finite `[3,1]` logits for input widths 512, 512
   and 1024 using runtime-switched calls.
7. Correct/wrong-S-box/branch-off calls change only the intended runtime
   semantics or branch flag and produce observable finite logits.
8. The shared state hash is identical before and after every intervention.
9. The model declares no cipher identity, absolute cell/bit identity or native
   cell slots, and the call accepts only ciphertext features, runtime structure,
   S-box enablement and transition-branch enablement.
10. Training rows, validation rows, optimizer steps and generated datasets all
    remain zero.

Any failed check yields `hold`; no partial average may open training.

## 6. Artifacts

The readiness run must write under:

```text
outputs/local_readiness/
  i1_uknit_family_multicipher_shared_weight_k1ao_readiness_20260729/
```

Required artifacts are `preflight.json`, `runtime_manifest.jsonl`,
`dataset_manifest.jsonl`, `results.jsonl`, `comparison.csv`, `gate.json`,
`validation.json`, `summary.json`, `progress.jsonl`, and a Chinese explanatory
`curves.svg` that passes `visual-qa-redraw`.

## 7. Advance and Stop Rules

If readiness passes, preregister two local shared-training replicas:

```text
replica0 = uKNIT seed3 + Midori seed6 + Dialga seed0
replica1 = uKNIT seed4 + Midori seed7 + Dialga seed1
```

The frozen training budget is `2048/class/cipher`, `1024/class` fresh
evaluation per cipher, four pairs, ten epochs, batch size 64, MSE, Adam
`1e-4`, weight decay `1e-5`, equal batches per cipher and strict encrypted
random-plaintext negatives. Evaluate correct, wrong-S-box and branch-off
runtimes at the same shared checkpoint. Every cipher/replica panel must retain
its same-budget anchor and prefer the correct runtime; Dialga's high AUC may not
average over a failed uKNIT or Midori panel.

If readiness fails, repair only the exact geometry, runtime switching or cache
binding mismatch. Do not train, run remotely, add 16 pairs, increase samples or
epochs, introduce MoE/cipher IDs, scan Walsh coefficients, or create a new
feature family.

## 8. Completed Result

The zero-training run completed with every protocol and evidence check passing:

```text
status   = pass
decision = innovation1_uknit_family_k1ao_shared_weight_runtime_ready
training rows / validation rows / optimizer steps = 0 / 0 / 0
dataset generation = false
remote scale = no
```

The exact evidence counts are:

```text
source artifact digests = 9/9 exact
selected cache payloads = 18/18
recomputed dataset hashes = 18/18 exact
runtime geometry rows = 3/3
same-state intervention rows = 9/9
failed protocol checks = 0
failed evidence checks = 0
```

All three models have `219320` trainable parameters and `52` state-dict
entries. Seed 29 produced the same initial state hash for uKNIT-BC, Midori64
and Dialga-128:

```text
a14429446450c90e3e16fd44924ac92f1fb38a170a4e556134b77035cf7c631d
```

One uKNIT state dict strict-loaded into all three runtime geometries. The same
shared model produced finite `[3,1]` logits for `[3,512]`, `[3,512]`, and
`[3,1024]` ciphertext inputs. On the actual bound cache rows, the maximum
absolute logit changes were:

| Cipher | Wrong S-box, same state | Branch off, same state |
|---|---:|---:|
| uKNIT-BC | `0.000710130` | `0.012779534` |
| Midori64 | `0.005098104` | `0.011598557` |
| Dialga-128 | `0.001494378` | `0.013046116` |

These nonzero values establish only that runtime interventions reach the shared
computation. They are not AUC, training quality, semantic preference or
cross-cipher transfer evidence. The shared tensor-state hash and the model's
persistent runtime binding remained bit-exact before and after all nine calls.

Artifacts:

```text
outputs/local_readiness/
  i1_uknit_family_multicipher_shared_weight_k1ao_readiness_20260729/
```

The Chinese SVG was rendered to `2016x1080` pixels. The first visual inspection
found a legend colliding with the Dialga value label; the chart was redrawn as
six directly labeled bars. The final `visual-qa-redraw` inspection passed with
no overlap, clipping, missing glyphs, ambiguous control mapping or misleading
AUC language.

## 9. Evidence-Backed Next Action

Readiness opens one experiment only: **K1-AO shared training**, using the two
replicas and budgets frozen in section 7. Implement the smallest mixed-cipher
trainer that alternates equal numbers of batches and calls
`logits_with_runtime` explicitly for each cipher. Use one optimizer and one
checkpoint per replica; do not create per-cipher heads, adapters or optimizer
states.

At each shared best checkpoint, run 18 zero-training fresh evaluations:

```text
3 ciphers x 2 replicas x {
  correct runtime,
  wrong S-box same checkpoint,
  transition branch off same checkpoint
}
```

Advance only if every cipher/replica fresh panel retains its registered
same-budget independent anchor within `-0.010 AUC`, and the correct runtime
beats both same-checkpoint controls by at least `+0.005 AUC`. These numeric
training gates must be frozen in the separate K1-AO training plan before the
first optimizer step. A failed cipher cannot be rescued by macro averaging.

Do not pursue remote scale, 16 pairs, larger samples/epochs, MoE, cipher IDs,
separate experts or another feature family before this local shared-training
gate is adjudicated.
