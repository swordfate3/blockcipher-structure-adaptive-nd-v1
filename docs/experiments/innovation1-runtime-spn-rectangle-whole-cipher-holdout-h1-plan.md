# Innovation 1 Runtime-SPN RECTANGLE Whole-Cipher Holdout H1 Plan

Date: 2026-07-26

```text
status = completed / hold
execution = local sub-medium readiness then diagnostic
remote_scale = no
```

## Research Question

Can the already supported, conditioner-free Runtime-E4 backbone learn reusable
SPN structure primitives from GIFT, SKINNY, uKNIT and Dialga, then distinguish
RECTANGLE without any RECTANGLE training rows, target-head training,
fine-tuning or target-based checkpoint selection?

This is the first direct whole-cipher holdout for the current method objective.
X3/X4 showed that a frozen formal SKINNY representation exposes RECTANGLE
signal after a RECTANGLE-labelled target head is trained. H1 removes that
remaining target supervision and reuses the same shared classifier.

## Why RECTANGLE First

RECTANGLE is the highest-evidence first holdout:

- X3-A/X3-A2 passed two target seeds with frozen SKINNY representations;
- X4 showed that only a `384 -> 1` linear target readout was needed;
- its non-contiguous cell layout differs materially from the standard
  contiguous 4-bit cells used by GIFT, SKINNY and uKNIT;
- unlike Dialga, its local AUC is not so high that one easy target can hide a
  weak structural attribution result.

Those results motivate H1 but are not same-protocol baselines because their
target heads used RECTANGLE labels.

## Frozen Model

All roles use one unmodified `RuntimeE4EquivariantSpnDistinguisher` geometry:

```text
parameters = 442466
hidden = 64
pair embedding = 128
processor steps = 2
cell input = state_triplet
S-box context = edge_gate
round window = recurrent_window
primitive Adapter = none
True FiLM = none
typed relation residual = none
cipher ID/name/task head = none
```

The same state dict and classifier handle all source structures and the held-out
RECTANGLE structure.

## Zero-Leakage Contract

The runner must enforce all of the following:

1. `rectangle80` is absent from every training task and every source validation
   task used for checkpoint selection.
2. The best checkpoint is selected only by the four-source validation macro
   AUC.
3. The RECTANGLE training cache is never loaded by H1.
4. The RECTANGLE validation cache is loaded only after all source roles have
   finished training and restored their source-selected checkpoints.
5. No target-specific parameter, head, optimizer step, threshold calibration
   or fine-tuning is allowed.
6. Target AUC uses the fixed shared logits; calibrated accuracy is descriptive
   only and cannot drive the gate.

## Three Source Roles

Each role uses byte-identical four-source datasets, initialization policy,
parameter count, optimizer and budget:

| Role | Source structure | Source relation mode |
| --- | --- | --- |
| `correct` | exact external descriptors | exact inverse GF(2) |
| `corrupted` | deterministic corrupted descriptors | exact processing of the wrong descriptor |
| `no_topology` | exact cell/S-box metadata | inverse GF(2) relation disabled |

After source training, five target evaluations are performed:

```text
candidate_correct              = correct source checkpoint + correct RECTANGLE
candidate_corrupted_target     = same candidate checkpoint + corrupted RECTANGLE
candidate_no_topology_target   = same candidate checkpoint + no RECTANGLE topology
corrupted_source_control       = corrupted source checkpoint + corrupted RECTANGLE
no_topology_source_control     = no-topology source checkpoint + no RECTANGLE topology
```

The first three rows are same-checkpoint counterfactuals. They isolate target
descriptor use from training randomness and capacity.

## Fixed Budget

```text
source ciphers = GIFT-64 r6, SKINNY-64/64 r7,
                 uKNIT-BC prefix-r5, Dialga-128 prefix-r4
holdout = RECTANGLE-80 r6
train = 2048/class/source cipher
source validation = 1024/class/source cipher
target evaluation = 1024/class
pairs/sample = 4 independent ciphertext pairs
negative = encrypted random plaintexts
seeds = 0, 1
epochs = 10
batch = 256
optimizer = Adam, lr 1e-4, weight decay 1e-5
loss = MSE
checkpoint = best four-source validation macro AUC
cache = exact completed five-cipher disk cache
execution = local diagnostic only
```

## Readiness Gate

Before H1 training:

1. parse the frozen config and exact five runtime descriptors;
2. prove the source set is exactly four ciphers and excludes RECTANGLE;
3. prove the cache contains source train/validation and only target validation
   paths required by H1;
4. prove all three roles contain exactly `442466` parameters and strictly load
   the same initial state;
5. prove one state handles 64-bit, 128-bit, contiguous and non-contiguous cells;
6. prove the target correct/corrupted/no-topology paths produce distinct logits;
7. run a one-epoch synthetic four-source joint smoke, restore the source-only
   checkpoint and only then evaluate the target;
8. preserve existing Runtime-E4 and result-index regressions.

Any failure authorizes only a focused readiness repair.

## Advance And Stop Gate

For each seed independently, require all of:

```text
candidate_correct target AUC >= 0.55
candidate_correct - candidate_corrupted_target >= +0.005
candidate_correct - candidate_no_topology_target >= +0.005
candidate_correct - corrupted_source_control >= +0.005
candidate_correct - no_topology_source_control >= +0.005
correct source validation macro - corrupted source macro >= +0.005
correct source validation macro - no-topology source macro >= +0.005
all leakage, parameter, cache and checkpoint checks pass
```

A two-seed pass supports the first local claim that the shared Runtime-E4
checkpoint uses composable structure on an unseen cipher and opens a second
independent holdout design. A hold means the current shared objective does not
generalize zero-shot under this budget; it does not authorize target-head
training, Adapter/MoE revival, extra epochs or remote scale as a rescue. A
protocol failure permits only evidence repair.

## Evidence-Backed Next Action

Run readiness first. If it passes, execute exactly this local H1 matrix. If H1
passes both seeds, preregister one second whole-cipher holdout with the same
backbone and controls before any universality claim. If H1 holds, audit whether
the gap is source-task calibration or representation alignment; do not replace
zero-shot evaluation with X3/X4-style target supervision.

## Completed Readiness

The frozen readiness run completed at:

```text
outputs/local_readiness/i1_runtime_spn_rectangle_whole_cipher_holdout_h1_readiness_20260726/
status = pass
decision = innovation1_runtime_spn_rectangle_holdout_readiness_passed
checks = 11/11
```

The gate verified all required cache paths, excluded the historical RECTANGLE
training cache, matched all three roles at `442466` parameters, exercised one
state dict on 64-bit, 128-bit, contiguous and non-contiguous cells, and restored
a source-selected checkpoint before the first target evaluation. No
target-specific trainable state or target optimizer step exists.

## Completed H1 Diagnostic

The preregistered two-seed run completed at:

```text
outputs/local_diagnostic/i1_runtime_spn_rectangle_whole_cipher_holdout_h1_2048_seed0_seed1_20260726/
results.jsonl rows = 34
history rows = 60
checkpoints = 6
parameter counts = [442466]
target training rows = 0
protocol validation = pass
```

All ten leakage, checkpoint, parameter, strict-negative and same-checkpoint
counterfactual checks passed. The held-out RECTANGLE cache was loaded only after
all source-role training completed.

Target AUC and candidate margins were:

| Seed | Correct | Same-weight corrupted target | Same-weight no topology | Corrupted-source control | No-topology-source control |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.674040 | 0.634780 | 0.624584 | 0.633519 | 0.535845 |
| 1 | 0.588227 | 0.588502 | 0.606210 | 0.623055 | 0.537390 |

```text
seed0 candidate margins = +0.039260, +0.049456, +0.040521, +0.138195
seed1 candidate margins = -0.000276, -0.017983, -0.034829, +0.050837
```

Seed 0 passed the complete gate. Seed 1 retained an absolute unseen-cipher
signal above the `0.55` floor but failed all three attribution comparisons
against corrupted-target, no-topology-target and corrupted-source controls.

The source macro AUC also hid a material imbalance. For the seed-1 correct
checkpoint, Dialga reached `0.942334`, while SKINNY reached `0.534819`, GIFT
`0.477778` and uKNIT `0.483970`. Equal task loss therefore did not establish
equal shared-backbone contribution.

```text
status = hold
decision = innovation1_runtime_spn_rectangle_holdout_not_supported
claim scope = local 2048/class/source diagnostic only
```

This result does not support stable composable-primitive learning across seeds,
but it also does not show absence of unseen-cipher signal. It prohibits target
supervision, remote scaling, extra epochs, Adapter/FiLM/GNN/MoE revival and a
universality claim.

## Final Next Action

Run a no-training source-task gradient and representation-alignment audit on the
H1 seed-0 and seed-1 correct checkpoints. Use fixed, parameter-matched batches
from GIFT, SKINNY, uKNIT and Dialga to record per-task shared-backbone gradient
norms, pairwise cosine similarities, Dialga gradient share and per-cipher AUC.
The one variable under investigation is source-task optimization balance; the
model, data, labels, negatives and holdout protocol remain frozen.

If seed 1 shows Dialga gradient domination or reproducible negative cosine with
GIFT/uKNIT, preregister one same-budget task-normalized or gradient-balanced
training gate. If the audit does not support that mechanism, stop optimizer
balancing and audit representation alignment instead. Do not launch remote
training, enlarge H1, train a RECTANGLE head or add MoE before this decision.

## Visual QA

The final `curves.svg` was rendered and inspected at `1800x1012` and `1280x720`
under the `visual-qa-redraw` workflow. Both views passed title, label, legend,
reference-line, overlap, clipping and readability checks.
