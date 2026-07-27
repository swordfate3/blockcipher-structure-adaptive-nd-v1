# Innovation 1 uKNIT-Family CT-SPN Cell/Path Hypergraph K1-F

**Date:** 2026-07-28
**Readiness Run ID:** `i1_uknit_family_ctspn_cell_path_hypergraph_k1f_readiness_20260728`
**Training Run ID:** `i1_uknit_family_ctspn_cell_path_hypergraph_k1f_2048_seed0_seed1_20260728`
**Status:** completed / hold / shared-cell hypergraph not supported
**Prerequisite:** completed K1-E split-specific relative-path overfit confirmation

## 1. Question

K1-E proved that K1-D strongly attributes correct relative paths on both uKNIT
training caches but loses that preference on unseen validation rows. K1-F changes
one representation mechanism:

> Does permutation-equivariant message passing between paths that share source,
> middle or target cells generalize better than treating the paths as an anonymous
> set, while never exposing numeric cell identity to the network?

## 2. One Model Variable

K1-F retains K1-D's exact 76-value path token, directed two-transition path,
pair handling, pair attention, classifier family, data, optimizer and checkpoint
rule. It replaces the two anonymous path mixer blocks with two cell/path blocks:

```text
path token
  -> aggregate separately at shared source, middle and target cells
  -> broadcast three cell messages back to each incident path
  -> residual path/channel update
  -> repeat twice
  -> mean/max/RMS path pooling
```

Cell indices are used only by `scatter`/`gather` routing. They are never converted
to floating-point features, embedded, normalized or supplied to a learned layer.
A global cell relabeling may permute nodes and paths but must leave the final logit
unchanged.

K1-F uses token width `64`, pair width `128` and two message blocks. Its parameter
count must not exceed the frozen Runtime-E4 cap `442466`, and 64-/128-bit instances
must have identical state-dict geometry.

## 3. Relation-Isolation Control

In addition to the frozen K1-D controls, K1-F adds `incidence_shuffled`:

```text
path token values       = exactly unchanged as a multiset
path count              = unchanged
source/middle/target IDs = used only as routing keys
shared-cell incidence   = deterministically shuffled
```

This control isolates the new hypothesis. Before training, it must preserve the
sorted path-token SHA exactly while changing the routing SHA, post-message pooled
summary and same-weight logit.

## 4. Frozen Protocol

| Field | uKNIT-BC | Dialga-128 |
|---|---:|---:|
| Prefix rounds | 5 | 4 |
| Runtime transitions | 3-4 | 2-3 |
| Training | `2048/class` = 4096 total | `2048/class` = 4096 total |
| Validation | `1024/class` = 2048 total | `1024/class` = 2048 total |
| Seeds | 0, 1 | 0, 1 |
| Pairs/sample | 4 | 4 |
| Epochs | 10 | 10 |
| Batch size | 64 | 64 |
| Loss/optimizer | MSE / Adam | MSE / Adam |
| Learning rate/weight decay | `1e-4` / `1e-5` | `1e-4` / `1e-5` |
| Checkpoint | best validation AUC | best validation AUC |
| Negative mode | encrypted random plaintexts | encrypted random plaintexts |

Keys, input difference `0x40`, dataset construction, sample order, label balance,
metric and K1-D cache identities remain frozen.

## 5. Zero-Training Readiness Gate

Readiness passes only if:

1. the exact K1-E pass decision and all protocol checks are present;
2. the four frozen source tasks match the two-cipher/two-seed protocol;
3. path schema contains no cell/cipher identity value;
4. cell indices are declared and exercised as routing only;
5. uKNIT and Dialga contain shared-cell incidence with degree above one;
6. deterministic global cell relabeling preserves sorted token sets and logits;
7. incidence shuffle preserves the token set exactly, changes routing and changes
   post-message summary/logit under one strict-loaded state dict;
8. repeat-last, rotated, corrupted and no-topology change routing or path evidence
   and the same-weight post-message result for both ciphers;
9. 64-/128-bit state geometry is identical and parameter count is within cap;
10. readiness uses zero data rows, zero optimizer steps and writes no checkpoint.

## 6. Training Controls And Gate

Each selected K1-F checkpoint is evaluated without training on the same validation
rows under:

```text
correct_ordered
repeat_last
rotated
corrupted
no_topology
incidence_shuffled
```

For both uKNIT seeds:

```text
candidate AUC >= 0.520
candidate - max(Runtime-E4, K1-D) >= +0.005
candidate - each of five controls >= +0.005
```

For both Dialga seeds:

```text
candidate >= K1-D - 0.005
candidate - each of five controls >= +0.005
```

No macro average may hide a cipher, seed or control failure.

## 7. Decisions

- **All gates pass:** retain cell/path hypergraph routing and only then consider a
  separate K2 nonlinear cell-semantics experiment.
- **Readiness fails:** repair only the failed invariance, relation-isolation or
  geometry property; do not train.
- **Training attribution succeeds only on source caches:** close K1-F as another
  split shortcut before any scale-up.
- **uKNIT remains weak or a wrong relation wins:** hold the family claim and close
  this exact two-transition hypergraph parameterization.

## 8. Blocked Routes

- No remote launch or mechanical sample, pair, epoch, width or seed increase.
- No absolute cell values, cipher identity, learned router or MoE.
- No K2 S-box truth table, ANF, DDT, trail, partial decryption or guessed key.
- No attack, SOTA, arbitrary-SPN, transfer or uKNIT-ceiling claim.

## 9. Execution Order

1. Implement model, relation-isolation control and deterministic readiness tests.
2. Run readiness locally and index the completed result.
3. Only if readiness passes, create the frozen four-row plan and run local training.
4. Evaluate six same-checkpoint controls, write JSONL/CSV/SVG/gates, perform
   `visual-qa-redraw`, update this record and refresh the result index.

## 10. Completed Result

K1-F completed locally on 2026-07-28. All four frozen training rows and all
twenty-four same-checkpoint control rows completed. Every protocol check passed:

```text
status   = hold
decision = innovation1_uknit_family_ctspn_k1f_hypergraph_not_supported
training rows = 4
control rows  = 24
failed protocol checks = []
```

The selected best-validation checkpoints were:

| Cipher | Seed | Best epoch | Best-epoch train AUC | Validation AUC | Same-budget anchor | Candidate - anchor |
|---|---:|---:|---:|---:|---:|---:|
| uKNIT-BC r5 | 0 | 9 | `0.797085` | `0.498477` | `0.526651` | `-0.028174` |
| uKNIT-BC r5 | 1 | 5 | `0.701712` | `0.521642` | `0.528809` | `-0.007166` |
| Dialga-128 r4 | 0 | 10 | `0.979754` | `0.960718` | `0.958499` | `+0.002219` |
| Dialga-128 r4 | 1 | 10 | `0.974036` | `0.958663` | `0.957774` | `+0.000889` |

The relation-isolation margins were:

```text
uKNIT seed0 correct - incidence-shuffled = +0.000207
uKNIT seed1 correct - incidence-shuffled = +0.002453

Dialga seed0 correct - incidence-shuffled = +0.001827
Dialga seed1 correct - incidence-shuffled = +0.001048
```

All four margins remained below the frozen `+0.005` gate. uKNIT seed0 also
missed the `0.520` AUC floor, and both uKNIT seeds remained below the strongest
same-budget anchor. Dialga retained high absolute AUC, but repeated-last,
rotated and incidence-shuffled controls remained too close to establish the
new shared-cell relation as the cause of that signal. Dialga therefore remains
a mechanism calibration and cannot hide the uKNIT failure.

K1-F shows that adding learned message passing over shared source, middle and
target cells did not repair uKNIT generalization. The train/validation gaps are
large, but the current protocol changes both plaintext rows and fixed key
between those splits. K1-F alone therefore cannot distinguish sample-specific
overfit from key-specific signal.

## 11. Claim Scope And Next Action

This is a two-seed `2048/class` local mechanism diagnostic for uKNIT-BC prefix-r5
and Dialga-128 prefix-r4. It is not formal scale, an attack, SOTA evidence,
arbitrary-SPN transfer evidence or a uKNIT ceiling claim. No remote scale-up,
extra samples, extra width, MoE, K2 semantics, DDT, trail or local-decryption
route is authorized from this result.

The evidence-backed next action is K1-G: strict-load the four K1-F checkpoints
and compare the original training rows, fresh plaintext rows under the same
training key, and the original cross-key validation rows under all six relation
conditions. This zero-training audit resolves whether the next architecture
should remove absolute ciphertext/key shortcuts or close the learned hypergraph
and move to a constrained exact operator-tied propagation model.

Artifacts:

```text
outputs/local_diagnostic/
  i1_uknit_family_ctspn_cell_path_hypergraph_k1f_2048_seed0_seed1_20260728/
```

The Chinese `curves.svg` was rendered at `1600 x 1020` and passed
`visual-qa-redraw`: no unintended overlap, clipping, missing glyph, ambiguous
title, misleading scale or unreadable close-curve labels remained.
