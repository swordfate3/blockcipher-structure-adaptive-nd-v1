# Innovation 1 uKNIT-Family CT-SPN Exact Operator Composition K1-N

**Date:** 2026-07-28
**Status:** completed / Dialga retained, uKNIT signal not supported
**Execution:** local CPU readiness followed by a fixed-budget local diagnostic

## 1. Research Question

K1-M repaired the exact-zero gate starvation and kept its topology residual
active, but uKNIT-BC r5 fresh AUC remained `0.490501-0.518564`. Dialga-128 r4
kept `0.956191-0.966957` AUC, while correct operators still failed to beat all
same-checkpoint wrong-operator controls by `0.005`. The remaining candidate
bottleneck is representation rather than gate scheduling:

> Does exposing the exact ordered composition of cell-specific inverse S-boxes
> and inverse GF(2) linear operators let the unchanged K1-M residual learn
> fresh uKNIT signal and identify the correct runtime operators?

K1-N is a local mechanism diagnostic. It cannot establish a uKNIT-family
ceiling, formal attack, paper-scale result, SOTA result or arbitrary-SPN
transfer claim.

## 2. Frozen Source

```text
K1-M root = outputs/local_diagnostic/
  i1_uknit_family_ctspn_gate_opening_k1m_2048_seed0_seed1_20260728
```

| Artifact | SHA-256 |
|---|---|
| K1-M gate | `0cf3714eaf2dbc0f052b04f4255ca007b16ab00b1c8abb93c270c3831a1876c6` |
| K1-M checkpoint manifest | `848a701087a6b0cca342461af97081f2a85cf4778a979660fa9a3b32b901e139` |
| K1-M dataset manifest | `ecc990e4d724ec35fdce8bd52d947c78280db2140853feddee07189ade4341f0` |
| K1-M controls | `289c611e8a41dc9d9a2e60868f6dab1313d7cf382f10e2a0052bb39cf433d2bd` |

The source must retain exact decision
`innovation1_uknit_family_ctspn_k1m_gate_opened_uknit_signal_not_supported`,
valid protocol checks and complete `4/60` training/evaluation rows.

## 3. Single Architectural Variable

Retain the complete K1-M base, position-preserving edge residual, bounded
fusion and initial effective gate `0.05`. Replace only the edge-residual bit
encoder input with a deterministic exact-composition view. For every pair,
construct left endpoint, right endpoint and their XOR at five ordered stages:

```text
stage 0: ciphertext
stage 1: inverse linear operator slot 1
stage 2: cell-specific inverse S-box slot 1
stage 3: inverse linear operator slot 0
stage 4: cell-specific inverse S-box slot 0
```

This gives `5 stages x 3 values = 15` binary channels per physical bit. Fixed
channel positions encode stage identity. Native cell membership and ordered bit
roles remain explicit when the four bit latents are assembled into each cell.
The existing K1-M source/target edge messages, two transition slots, cell
updates, invariant final pooling, classifier and gate are otherwise unchanged.

No learned S-box embedding, DDT, trail feature, cipher identity, key identity,
absolute cell ID, raw bypass, auxiliary loss, extra training stage or extra
fusion gate is allowed.

## 4. Same-Checkpoint Controls

Train only the exact candidate. Strict-load the same checkpoint into every
control with identical parameter names and shapes:

| Condition | S-box schedule | Linear schedule | Purpose |
|---|---|---|---|
| `exact_composition` | correct heterogeneous tables | correct ordered matrices | candidate |
| `wrong_sbox_semantics` | uKNIT cell assignments shuffled; homogeneous Dialga input domain permuted | correct | tests exact S-box use |
| `reversed_linear_schedule` | correct, unchanged slots | linear slots reversed only | tests linear order |
| `corrupted_linear_operators` | correct | deterministic source-column corruption | tests exact matrices |
| `no_sbox_composition` | S-box stages replaced by identity | correct | tests nonlinear contribution |
| `no_topology` | identity S-boxes | identity matrices | removes runtime operator semantics |
| `k1m_anchor` | completed K1-M exact row | completed K1-M exact row | same-budget anchor |

uKNIT has heterogeneous cell tables, so its wrong-S-box control shuffles cell
assignments within each slot. Dialga's normalized native cells share one table,
so assignment shuffling is an exact no-op; its wrong-S-box control instead
applies a fixed nonidentity permutation to the four-bit S-box input domain for
every cell and slot. Both variants change only S-box semantics and must change
the composition fingerprint. Reversing linear order must leave S-box slots
unchanged. Corruption must leave S-boxes unchanged. Every control must consume
the same dataset digest and strict-loaded candidate state.

## 5. Readiness Gate

Before optimization require all of the following:

1. exactly four candidate tasks for uKNIT-BC r5 and Dialga-128 r4, seeds 0/1;
2. exact K1-M source hashes, decision, protocol checks and cache bindings;
3. exact stage order and `15` binary channels on 64-bit and 128-bit states;
4. vectorized inverse composition equals an independently stepped reference;
5. forward replay of both transitions exactly reconstructs every fixed binary
   fixture after the inverse composition;
6. candidate, wrong-S-box, reversed-linear, corrupted-linear,
   no-S-box and no-topology models have identical state geometry and strict-load
   the same state;
7. each control changes only its declared descriptor component and has a
   distinct composition fingerprint;
8. both linear transition slots and both inverse-S-box slots change at least
   one fixed fixture;
9. the composition encoder and every retained residual parameter group receive
   nonzero gradient above `1e-8` at effective gate `0.05`;
10. all twelve K1-M caches are digest-bound and reused; readiness performs zero
    optimizer steps and consumes zero training rows.

Any failed readiness item authorizes only repair of that item and an unchanged
readiness rerun.

## 6. Frozen Local Diagnostic

| Field | Frozen value |
|---|---|
| Ciphers / rounds | uKNIT-BC r5; Dialga-128 r4 |
| Candidate | exact heterogeneous S-box/operator composition edge residual |
| Same-budget anchor | completed K1-M exact candidate rows |
| Seeds | `0`, `1` |
| Samples | `2048/class` train; `1024/class` fresh same-key; `1024/class` cross-key |
| Pairs per sample | `4` |
| Negative definition | encrypted random plaintexts |
| Keys / differences | exact K1-M values |
| Epochs / batch | `10` / `64` |
| Loss / optimizer | MSE / Adam |
| Learning rate / weight decay | `1e-4` / `1e-5` |
| Checkpoint | best validation AUC, restored before controls |
| Device | local CPU |

All datasets must reuse the K1-M cache with no regeneration. Training labels,
validation splits, metric calculation and checkpoint selection remain frozen.

## 7. Advance Gate

Apply each threshold separately to both fresh splits and both seeds. Training
rows are descriptive and no average may hide a failed row.

For uKNIT-BC:

```text
candidate AUC                              >= 0.520
candidate AUC - same-row K1-M AUC         >= +0.005
candidate AUC - every semantic control AUC >= +0.005
abs(final effective gate)                 >= 0.010
```

For Dialga-128:

```text
candidate AUC - same-row K1-M AUC         >= -0.005
candidate AUC - every semantic control AUC >= +0.005
```

## 8. Decisions And Next Action

- **All gates pass:** retain K1-N and write a separate remote `65536/class`
  disk-cached diagnostic plan. This remains medium diagnostic evidence.
- **uKNIT improves but misses a narrow control:** hold K1-N and run a zero-step
  contribution audit on the failed control before any scale.
- **uKNIT remains near chance while Dialga retains signal:** stop deterministic
  operator-view expansion. Audit whether the frozen differential itself has a
  measurable uKNIT r5 relation under exact partial-state statistics before
  changing the neural architecture again.
- **Dialga signal collapses:** discard K1-N because the new bottleneck erased a
  known strong mechanism; return to K1-M.
- **Candidate and wrong S-box are indistinguishable:** do not add capacity or
  MoE. Audit S-box assignment equivariance and cell grouping first.
- **Protocol invalid:** repair only the failed implementation/binding and rerun
  readiness or training unchanged.

Do not increase samples, epochs, pairs, seeds, width or experts; do not add
MoE, DDT/trail inputs, key/cipher IDs, partial decryption or a raw bypass before
the K1-N local gate is adjudicated.

## 9. Run IDs And Required Artifacts

```text
readiness_run_id = i1_uknit_family_ctspn_exact_operator_composition_k1n_readiness_20260728
training_run_id  = i1_uknit_family_ctspn_exact_operator_composition_k1n_2048_seed0_seed1_20260728
```

Readiness must produce preflight, results, validation, gate and progress files.
Training must produce four checkpoints, `84` evaluation rows, checkpoint and
dataset manifests, controls, split CSV, gate, validation, summary, history and
progress. Generate a Chinese explanatory SVG, inspect its rendered pixels with
`visual-qa-redraw`, then refresh `outputs/00_RECENT_RESULTS.md` and
`outputs/00_RECENT_RESULTS.json` before reporting the result.

## 10. Completed Result

The zero-training readiness gate passed before optimization:

```text
decision = innovation1_uknit_family_ctspn_k1n_execution_authorized
parameter count = 131875
composition channels per bit = 15
independent inverse-stage reference exact = true
two-transition forward round trip exact = true
all six semantic controls strict-load identical state = true
all composition and residual parameter groups receive gradient = true
training rows = 0
optimizer steps = 0
```

The frozen diagnostic completed all training and control rows while reusing all
training and validation caches:

```text
training rows = 4 / 4
evaluation rows = 84 / 84
validation status = pass
errors = []
cache regeneration = none
```

Fresh-split candidate AUC, K1-M delta, weakest semantic-control margin and
final effective gate were:

| Cipher | Seed | Split | K1-N AUC | K1-N - K1-M | Weakest control margin | Gate |
|---|---:|---|---:|---:|---:|---:|
| uKNIT-BC r5 | 0 | same-key fresh | `0.511873` | `+0.003478` | `-0.008980` | `0.048728` |
| uKNIT-BC r5 | 0 | cross-key | `0.516853` | `-0.001710` | `+0.000357` | `0.048728` |
| uKNIT-BC r5 | 1 | same-key fresh | `0.484239` | `-0.006262` | `-0.027189` | `0.051413` |
| uKNIT-BC r5 | 1 | cross-key | `0.506254` | `-0.002858` | `-0.000157` | `0.051413` |
| Dialga-128 r4 | 0 | same-key fresh | `0.967677` | `+0.000720` | `-0.000334` | `0.065301` |
| Dialga-128 r4 | 0 | cross-key | `0.959750` | `+0.000311` | `-0.000776` | `0.065301` |
| Dialga-128 r4 | 1 | same-key fresh | `0.959320` | `-0.001055` | `-0.000010` | `0.073340` |
| Dialga-128 r4 | 1 | cross-key | `0.954737` | `-0.001454` | `+0.000014` | `0.073340` |

Every uKNIT fresh AUC stayed below `0.520`. K1-N did not improve every K1-M
row by `+0.005`, and it did not beat every semantic control by `+0.005`.
Dialga retained the K1-M anchor within the allowed `-0.005`, but also failed
semantic attribution. In particular, exact composition versus no-S-box or
wrong-S-box composition differed by less than `0.001` AUC on every fresh
Dialga row and every fresh uKNIT row except for other linear controls. The
branch stayed open, so this is not another gradient-starvation result.

```text
status = hold
decision = innovation1_uknit_family_ctspn_k1n_dialga_retained_uknit_signal_not_supported
remote_scale = no
```

The claim is limited to this two-seed local `2048/class` mechanism diagnostic.
It does not prove that uKNIT r5 is impossible, that the family route has reached
a ceiling, or that more appropriate data and representations cannot work.

The first post-training manifest attempt failed only after all four checkpoints
were complete because a reused helper hardcoded the K1-K candidate model key.
The helper was parameterized and regression-tested, then the same four
checkpoints resumed at the evaluation stage. No model was retrained and no
cache was regenerated.

## 11. Recommended Next Action: K1-O Signal Audit

Before another neural redesign, freeze a local deterministic audit over only
the two uKNIT-BC r5 seeds and their existing three splits. The research question
is whether the frozen differential has any fresh-split association in exact
partial-state cell-difference statistics.

Use the same `2048/class` training rows and `1024/class` same-key/cross-key
holdouts, four pairs per sample, exact keys, strict negative definition and
digest-bound K1-M/K1-N caches. Fit the same closed-form diagonal Fisher/LDA
scorer separately to these feature views:

```text
raw_position_histogram
exact_five_stage_position_histogram
no_sbox_five_stage_position_histogram
wrong_sbox_five_stage_position_histogram
exact_five_stage_invariant_histogram
label_shuffled_exact_position_histogram
```

Each position-preserving view is the per-sample frequency of all 16 nibble XOR
values for every native cell and stage, averaged only over the four pairs. The
invariant control additionally pools cells. The scorer uses training-class
means and diagonal pooled variance only; no gradient optimizer, neural network,
extra sample, epoch or hyperparameter search is allowed.

Advance toward a new neural cell-stage architecture only if, for both seeds and
both fresh splits:

```text
exact position AUC                         >= 0.550
exact position - raw position AUC          >= +0.010
exact position - no-S-box AUC              >= +0.005
exact position - wrong-S-box AUC           >= +0.005
exact position - label-shuffled AUC        >= +0.030
```

If exact position signal exists but invariant pooling loses at least `0.010`,
the next neural variable is a position-preserving cell-stage head. If signal is
present but exact and wrong/no-S-box controls tie, S-box semantics are not the
priority. If every exact fresh row remains below `0.550`, stop modifying this
network on the current uKNIT r5 differential and audit the differential/data
protocol itself before more architecture or remote scale.
