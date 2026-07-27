# Innovation 1 uKNIT-Family CT-SPN Relative Cross-Transition Path K1-D

**Date:** 2026-07-28
**Readiness Run ID:** `i1_uknit_family_ctspn_relative_path_k1d_readiness_20260728`
**Training Run ID:** `i1_uknit_family_ctspn_relative_path_k1d_2048_seed0_seed1_20260728`
**Status:** completed / hold

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

The frozen two-transition window produces exactly one composed path layer. K1-D
therefore removes K1-B's temporal convolution after path pooling: on a length-one
axis it cannot encode transition order and would only add a content transform. The
directed `source -> intermediate -> target` path is the order-sensitive object.
Pair aggregation, classifier dimensions, optimizer and all data settings remain
unchanged.

Each path token has exactly `76` values:

```text
source/intermediate/target left-right-xor role values = 3 x 12 = 36
source-target product and XOR role values             = 2 x 12 = 24
source-role -> target-role boolean reachability       = 4 x 4  = 16
total                                                   = 76
```

The temporal removal compensates for the wider semantic token rather than adding
capacity. The candidate must remain below the Runtime-E4 parameter cap. This
implementation refinement was frozen before readiness or training and does not alter
the single hypothesis: compose adjacent transitions before invariant pooling without
absolute cell identity.

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

## 9. Readiness Result

The zero-training readiness completed on 2026-07-28:

```text
run_id   = i1_uknit_family_ctspn_relative_path_k1d_readiness_20260728
status   = pass
decision = innovation1_uknit_family_ctspn_k1d_relative_path_execution_authorized
training rows = 0
optimizer steps = 0
parameters = 409954 <= Runtime-E4 442466
```

All protocol and structural checks passed. The correct uKNIT window produced `224`
relative paths and Dialga produced `882`. Reversing every native cell name preserved
the sorted path-token set exactly; maximum same-weight logit differences were
`1.15e-7` for uKNIT and `1.19e-7` for Dialga. Repeated-last, rotated, corrupted and
no-topology controls changed the token fingerprint, pooled summary and random-weight
logit for both ciphers. Every recorded path matched the exact boolean composition of
the two supplied GF(2) layers through one intervening four-bit cell.

This establishes implementation feasibility only. It authorizes the frozen local
four-row `2048/class` diagnostic but supplies no trained AUC or efficacy evidence.
The readiness artifacts are indexed as entry `001` at completion:

```text
outputs/local_readiness/i1_uknit_family_ctspn_relative_path_k1d_readiness_20260728/
  results.jsonl
  gate.json
  validation.json
  summary.json
  progress.jsonl
```

## 10. Completed Training Result

K1-D completed locally on 2026-07-28 with four training rows and twenty frozen
control rows. All protocol checks passed, but the research gate returned `hold`:

```text
decision = innovation1_uknit_family_ctspn_k1d_relative_path_not_supported

uKNIT seed0:
  candidate                         = 0.518386
  candidate - strongest anchor      = -0.008265
  candidate - repeat/rotated        = +0.007745 / +0.024916
  candidate - corrupted/no-topology = +0.020204 / -0.014898

uKNIT seed1:
  candidate                         = 0.515869
  candidate - strongest anchor      = -0.012940
  candidate - repeat/rotated        = +0.026663 / +0.022419
  candidate - corrupted/no-topology = -0.000401 / +0.020160

Dialga seed0/seed1 candidate         = 0.958499 / 0.957774
Dialga candidate - K1-B anchor       = -0.001607 / -0.003600
```

The relative path representation removed absolute cell-number input and retained
Dialga's high AUC, but neither uKNIT seed reached the frozen `0.520` floor or the
same-budget anchor margin. uKNIT seed0 was beaten by no-topology and seed1 was
slightly beaten by corrupted topology. Dialga also missed repeated-last attribution
on both seeds. Therefore the correct topology is not consistently responsible for
the unseen-data prediction.

This is a `2048/class` local mechanism diagnostic, not formal-scale failure or a
uKNIT ceiling. It does not authorize remote scale-up, more samples, extra width,
K2, MoE, DDT/trail or partial decryption.

The evidence-backed next action was K1-E: reuse all four selected checkpoints and
eight exact caches to compare topology attribution on the training and validation
splits without optimizer steps. K1-E has since confirmed split-specific relative-path
overfitting, so K1-D is closed and must not be scaled.

Artifacts:

```text
outputs/local_diagnostic/
  i1_uknit_family_ctspn_relative_path_k1d_2048_seed0_seed1_20260728/
```
