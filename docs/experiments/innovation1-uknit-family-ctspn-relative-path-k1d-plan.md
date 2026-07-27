# Innovation 1 uKNIT-Family CT-SPN Relative Cross-Transition Path K1-D

**Date:** 2026-07-28
**Readiness Run ID:** `i1_uknit_family_ctspn_relative_path_k1d_readiness_20260728`
**Training Run ID:** `i1_uknit_family_ctspn_relative_path_k1d_2048_seed0_seed1_20260728`
**Status:** planned / zero-training readiness required

## 1. Question

K1-C proved that K1-B fits correct topology on both uKNIT training caches but loses
that preference on unseen validation rows. K1-D tests one narrower hypothesis:

> Can adjacent heterogeneous transitions be composed into a permutation-equivariant
> path set, so the model uses which observed endpoint values are connected across
> layers instead of memorizing absolute native cell numbers?

## 2. One Model Variable

K1-D retains K1-B's exact canonical state views, pair handling, hidden dimensions,
pair attention, classifier, optimizer, data and controls. It replaces only the
per-transition endpoint representation and its pre-pooling composition.

K1-B performs:

```text
transition edge
  + absolute source/target native cell position
  + source/target bit role
  -> encode each transition independently
  -> invariant edge pooling
  -> temporal mixer
```

K1-D performs:

```text
adjacent transition t and t+1
  -> connect paths through the intervening native S-box cell
  -> attach observed source/intermediate/target endpoint values
  -> attach directed source/intermediate/target bit roles
  -> shared path encoder
  -> invariant path pooling
  -> unchanged pair aggregation and classifier
```

The path token contains no cipher name and no absolute native cell index. A shared
renaming of native cells therefore only permutes the path set; it cannot change the
pooled representation. Repeated-last, rotated and corrupted schedules rewire which
observed values share a path and must remain observable.

The intervening nonlinear layer is represented only by cell reachability: all four
bits in one native S-box cell may interact. K1-D does not add an S-box truth table,
learned S-box descriptor, DDT, trail, partial decryption or guessed key. Explicit
canonical S-box composition remains a later K2 hypothesis and is blocked here.

## 3. Frozen Protocol

| Field | uKNIT-BC | Dialga-128 |
|---|---:|---:|
| Prefix rounds | 5 | 4 |
| Runtime transitions | 3-4 | 2-3 |
| Train samples/class | 2048 | 2048 |
| Validation samples/class | 1024 | 1024 |
| Seeds | 0, 1 | 0, 1 |
| Pairs/sample | 4 | 4 |
| Epochs | 10 | 10 |
| Batch size | 64 | 64 |
| Negatives | encrypted random plaintexts | encrypted random plaintexts |
| Device | local CPU diagnostic | local CPU diagnostic |

The same fixed keys, input difference `0x40`, cached datasets, MSE/Adam settings,
learning rate `1e-4`, weight decay `1e-5` and best-validation-AUC checkpoint rule
from K1-B remain frozen.

## 4. Required Controls

Every trained checkpoint is replayed without optimizer steps under:

```text
correct_ordered
repeat_last
rotated
corrupted
no_topology
```

The same-budget anchors are the completed Runtime-E4 and K1-B rows on the identical
validation datasets. No macro average may hide a cipher, seed or control failure.

## 5. Zero-Training Readiness Gate

Before training, deterministic probes must establish:

1. exactly one adjacent-transition composition is built for each two-transition
   runtime window;
2. every path is connected through one native S-box cell and contains no absolute
   cell identifier or cipher identity;
3. shared native-cell relabeling only permutes the path set and leaves its sorted
   token fingerprint unchanged;
4. repeated-last, rotated and corrupted controls change the relative path
   fingerprint for both ciphers;
5. correct versus each wrong control changes a fixed-state pooled path summary and
   final logit under strict shared weights;
6. 64-bit and 128-bit models have identical state-dict geometry;
7. trainable parameters do not exceed the Runtime-E4 anchor `442466`;
8. readiness performs zero optimizer steps and creates no training checkpoint.

Failure closes this exact path construction before any dataset inference or training.

## 6. Training Gate

For both uKNIT seeds:

```text
candidate AUC >= 0.520
candidate - max(Runtime-E4, K1-B) >= +0.005
candidate - repeat_last >= +0.005
candidate - rotated     >= +0.005
candidate - corrupted   >= +0.005
candidate - no_topology >= +0.005
```

For both Dialga seeds:

```text
candidate >= K1-B - 0.005
candidate - repeat_last >= +0.005
candidate - rotated     >= +0.005
candidate - corrupted   >= +0.005
candidate - no_topology >= +0.005
```

- **All gates pass:** retain the relative path representation and only then plan a
  separately controlled nonlinear K2 candidate.
- **Training attribution passes but validation fails:** close this relative path
  parameterization; do not scale it.
- **uKNIT fails while Dialga passes:** hold the family claim and inspect which uKNIT
  path equivalences remain collapsed at zero training.
- **Protocol failure:** repair only the binding and rerun unchanged.

## 7. Blocked Routes

- No remote launch or mechanical increase in data, pairs, epochs, width or seeds.
- No absolute cell positions, cipher identity, MoE or expert routing.
- No K2 S-box truth-table/ANF/DDT conditioning, trail, partial decryption or key guess.
- No arbitrary-SPN, attack, SOTA, transfer or uKNIT-ceiling claim from this diagnostic.

## 8. Execution Order

1. Implement the relative path constructor and deterministic invariance/control
   readiness audit.
2. Run readiness locally and record JSONL, gate and validation artifacts.
3. Only if every readiness check passes, add one K1-D model key and the frozen four-row
   experiment matrix.
4. Run the local `2048/class` diagnostic and twenty-row frozen control panel.
5. Produce JSONL/CSV/SVG/gate artifacts, run `visual-qa-redraw`, update the result
   index and document the next action.
