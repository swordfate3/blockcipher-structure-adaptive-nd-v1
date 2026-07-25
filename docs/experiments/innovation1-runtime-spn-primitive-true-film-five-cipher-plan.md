# Innovation 1 Runtime-SPN Primitive True-FiLM Five-Cipher Plan

Date: 2026-07-26

## Status

```text
phase = completed local sub-medium diagnostic
implementation = complete
readiness = passed, 11/11 checks
training = complete
status = hold
decision = innovation1_runtime_spn_primitive_true_film_not_supported
execution = local sub-medium diagnostic only
remote_scale = prohibited
whole_cipher_holdout = prohibited
```

## Research Question

Can a parameter-matched, cell-local True FiLM conditioner make the shared
Runtime-E4 transition use runtime S-box and GF(2) diffusion descriptors more
effectively than the completed low-rank additive and multiplicative effects?

This is the final low-complexity differentiated-conditioning candidate before
stopping the Adapter/FiLM branch. It is not a sample-scale experiment and it
does not authorize learned MoE.

## Evidence That Motivates This Candidate

The completed five-cipher additive Adapter and multiplicative gate both held.
Post-hold audits established that:

1. the two routed modules received traffic and gradients;
2. their effective rank did not collapse;
3. same-route and shared-backbone gradients were not primarily conflicting;
4. the additive effect was functionally weak on seed1;
5. simply amplifying the additive scale reduced training macro AUC;
6. changing the same low-rank output from addition to multiplication did not
   pass the frozen two-seed control gate.

The remaining narrow hypothesis is that a descriptor-conditioned affine
transformation must act directly on the shared transition before token mixing,
rather than entering as a low-rank hidden-state residual after the mixer.

## One Variable

Only the conditional computation mechanism changes.

```text
old additive:
    mixed = Mixer(transition)
    next  = mixed + 0.1 * Adapter_route(mixed)

old multiplicative:
    mixed = Mixer(transition)
    next  = mixed * (1 + 0.1 * tanh(Adapter_route(mixed)))

new True FiLM:
    gamma, beta = Conditioner(local_runtime_descriptor)
    conditioned = transition * (1 + 0.1 * tanh(gamma))
                  + 0.1 * tanh(beta)
    next = Mixer(conditioned)
```

The shared Runtime-E4 backbone, exact GF(2) inverse view, S-box edge gate,
training loop, loss, optimizer, datasets, keys, negative definition,
checkpoint selection and evaluation metrics remain frozen.

## Runtime Descriptor

Every cell receives a 128-dimensional descriptor for the current runtime
round. It is computed only from the externally supplied structure object and
contains no cipher name, cipher ID, explicit block width or global structure
fingerprint.

```text
64 dimensions = the cell's 4-bit S-box truth-table bits
64 dimensions = four GF(2) local statistics for each
                target-bit-role x source-bit-role pair
```

The four diffusion statistics are:

1. normalized incoming edge count;
2. GF(2) parity of that count;
3. whether the same cell contributes that source role;
4. mean normalized source fan-out over connected source bits.

These quantities are invariant to a global relabeling of cells while still
distinguishing local S-box and diffusion primitives. The complete descriptor
must be verified against all five runtime structures before training.

## Parameter Matching

The True FiLM conditioner is:

```text
Linear(128, 10, bias=False)
GELU
Linear(10, 2 * 128, bias=True)
```

It contains exactly:

```text
128 * 10 + 10 * 256 + 256 = 4096 parameters
```

This equals the completed pair of rank-8 low-rank Adapters. The expected full
model count is therefore `446562` for every role and remains equal to the old
additive source. No hidden-width reduction, extra expert, task-specific head
or cipher-specific trainable state is allowed.

## Frozen Role Panel

All four roles use the same model geometry and active compute:

| Role | FiLM descriptor |
| --- | --- |
| `dense` | one fixed, structure-independent 128-vector at every cell |
| `correct` | the correct cell-local S-box and GF(2) descriptor |
| `uniform` | the per-round mean descriptor repeated at every cell |
| `shuffled` | the correct descriptor with fixed within-half feature permutations |

The `shuffled` control preserves descriptor value marginals but breaks the
semantic alignment of S-box bits and diffusion-role channels. It must not use
cell IDs or a permutation tied to cell numbering, because cell relabeling
equivariance is a required invariant.

## Frozen Five-Cipher Protocol

Use the exact completed additive/gated panel:

| Group | Cipher task | Rounds/window | Difference |
| --- | --- | ---: | ---: |
| core | GIFT-64 | r6, runtime window 2 | `0x40` |
| core | SKINNY-64/64 | r7, runtime window 2 | `0x2000` |
| core | RECTANGLE-80 | r6, runtime window 2 | `0x2100010020` |
| stress | uKNIT-BC | prefix-r5, runtime rounds 3--4 | `0x40` |
| stress | Dialga-128 | prefix-r4, runtime rounds 2--3 | `0x40` |

Frozen training protocol:

```text
samples_per_class = 2048 per cipher and seed
validation_samples_per_class = 1024 per cipher and seed
pairs_per_sample = 4
negative_mode = encrypted_random_plaintexts
train_key = all-zero key at each cipher's real key width
validation_key = fixed 0x11...11 key at each cipher's real key width
seeds = 0, 1
epochs = 10
batch_size = 256
optimizer = Adam
learning_rate = 1e-4
weight_decay = 1e-5
loss = MSE
checkpoint_metric = five-task validation macro AUC
restore_best_checkpoint = true
device = local CPU
```

Reuse the completed additive experiment's parameter-matched disk caches. Do
not regenerate a different benchmark and do not run this sub-medium diagnostic
remotely.

## Readiness Gate

Readiness must fail closed unless all checks pass:

1. all five external descriptors load exactly and yield finite `cells x 128`
   local descriptors for both runtime rounds;
2. the descriptor is cell-relabeling equivariant;
3. changing only an S-box or GF(2) relation changes the affected descriptor;
4. correct, uniform and shuffled controls produce distinct conditioned outputs
   on a mixed synthetic structure;
5. dense/correct/uniform/shuffled have identical state geometry, exactly
   `446562` parameters and the same active conditioner compute;
6. one shared state dictionary handles all five 64/128-bit structures;
7. every True FiLM parameter receives finite nonzero gradients;
8. no task-specific trainable state, cipher ID, explicit block width or global
   fingerprint is supplied;
9. cell relabeling preserves logits to absolute tolerance `1e-6`;
10. a one-epoch five-task smoke has equal `0.2` task weights, equal optimizer
    contributions, finite metrics, strict negatives and one shared checkpoint;
11. existing Runtime-E4, recurrent-window, Dialga and Adapter regressions pass.

## Joint Advance Gate

For each seed separately, `correct` must satisfy all conditions below.

Against each new-run control (`dense`, `uniform`, `shuffled`):

```text
correct - control >= +0.005 core macro validation AUC
correct - control >= +0.005 stress macro validation AUC
correct - control >= -0.005 on every individual cipher
```

Against the completed additive `correct` source under the same data and
budget:

```text
film_correct - additive_correct >= +0.005 core macro validation AUC
film_correct - additive_correct >= +0.005 stress macro validation AUC
film_correct - additive_correct >= -0.005 on every individual cipher
```

The result is a full joint pass only when both seeds pass every condition and
the validation archive proves 40/40 rows, 8/8 checkpoints, equal parameter
counts, strict negatives and the exact reused cache root.

## Decisions

- Full two-seed pass: keep True FiLM and preregister whole-cipher holdouts.
  Retrain from scratch with one entire cipher absent; do not fine-tune the
  held-out cipher and do not expose cipher identity.
- Core pass but stress hold: report `core_supported_new_cipher_hold`; do not
  enter holdouts or remote scale.
- Any control/source failure: discard True FiLM and stop deterministic
  Adapter/FiLM/MoE scaling. Do not increase rank, scale, experts, samples or
  epochs. Re-rank the typed R-GCN/GNN-FiLM alternative against method
  consolidation.
- Protocol/readiness failure: repair and rerun readiness only; metrics are not
  research evidence.

## Required Artifacts

```text
manifest.jsonl
smoke-results.json
validation.json
results.jsonl
history.csv
gate.json
summary.json
curves.svg
progress.jsonl
checkpoints/seed{0,1}-{dense,correct,uniform,shuffled}.pt
visual_qa_passed.marker
```

After each completed result-producing run, refresh
`outputs/00_RECENT_RESULTS.md` and `outputs/00_RECENT_RESULTS.json`.

## Readiness Result

The local readiness run completed before the real diagnostic:

```text
run_id = i1_runtime_spn_primitive_true_film_five_cipher_readiness_20260726
artifact_root = outputs/local_readiness/i1_runtime_spn_primitive_true_film_five_cipher_readiness_20260726/
status = pass
decision = innovation1_runtime_spn_primitive_true_film_readiness_passed
checks = 11/11 passed
```

Observed implementation evidence:

```text
all role parameter counts = 446562
old additive source parameter count = 446562
descriptor shape = cells x 128 for every cipher and round
GIFT/RECTANGLE descriptor collision = distinguished
uKNIT/Dialga descriptor collision = distinguished
all three FiLM parameter tensors = finite nonzero gradients
cell relabeling = descriptor equivariance and logit invariance passed
shared widths = GIFT/SKINNY/RECTANGLE/uKNIT 64-bit and Dialga 128-bit passed
```

This result authorizes only the frozen local `2048/class/cipher` matrix. It is
not evidence that True FiLM improves AUC and does not authorize remote scale or
whole-cipher holdouts.

## Completed Joint Result

The frozen two-seed matrix completed with valid protocol evidence:

```text
artifact_root = outputs/local_diagnostic/i1_runtime_spn_primitive_true_film_five_cipher_joint_2048_seed0_seed1_20260726/
result_rows = 40/40
checkpoints = 8/8
parameter_counts = [446562]
parameter_matched = true
strict_negative_mode = encrypted_random_plaintexts
cache_source_root = completed additive experiment cache
source_anchor_valid = true
visual_qa_redraw = pass
```

Correct True FiLM validation AUCs were:

| Seed | GIFT-64 | SKINNY-64/64 | RECTANGLE-80 | uKNIT-BC | Dialga-128 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.538209 | 0.491071 | 0.725286 | 0.515963 | 0.946774 |
| 1 | 0.528462 | 0.542784 | 0.695083 | 0.515000 | 0.947990 |

High Dialga AUC is not a True FiLM gain: the controls are similarly high. The
frozen decision depends on matched deltas, not raw task difficulty.

Macro validation-AUC deltas for `correct - comparison` were:

| Seed | Comparison | Core three-cipher macro | Stress two-cipher macro |
| ---: | --- | ---: | ---: |
| 0 | dense | +0.001635 | +0.005936 |
| 0 | uniform | +0.000376 | +0.002453 |
| 0 | shuffled | -0.003714 | +0.006420 |
| 0 | old additive correct | -0.002109 | +0.007195 |
| 1 | dense | +0.002331 | +0.003535 |
| 1 | uniform | +0.002667 | -0.005511 |
| 1 | shuffled | -0.000405 | +0.000329 |
| 1 | old additive correct | -0.001451 | +0.000864 |

Neither seed passed the core gate. Seed0 also regressed on SKINNY by
`-0.017939` versus shuffled and `-0.012104` versus the old additive source.
Seed1 regressed on uKNIT by `-0.010385` versus uniform. The failure is therefore
not a missing-gradient or inactive-conditioner artifact: True FiLM was active,
but its local descriptor effect did not produce stable, attributable gains.

```text
status = hold
decision = innovation1_runtime_spn_primitive_true_film_not_supported
core_pass = false
full_pass = false
whole_cipher_holdout = no
remote_scale = no
```

### Evidence-Backed Next Action

Discard this True FiLM candidate and stop deterministic Adapter/FiLM/MoE
scaling. Do not change rank, scale, expert count, samples or epochs as a rescue.
The next action is a design audit, not immediate training:

```text
research question = can typed relation-specific message passing around the
                    exact GF(2) view express local transition heterogeneity
                    without losing XOR semantics?
same-budget anchor = supported shared Runtime-E4 plus this 446562-parameter panel
required controls = relation-type correct, relation-type shuffled,
                    relation-agnostic matched graph path, shared Runtime-E4 anchor
one variable = relation-specific message/update operator only
initial scale = local synthetic/readiness probes, then 2048/class/cipher only
advance gate = both seeds exceed every matched control by +0.005 on core and
               stress with every cipher >= -0.005
stop gate = any cell-relabel, exact-GF(2), parameter, gradient or two-seed gate failure
blocked = learned MoE, remote scale, more samples/epochs, cipher-ID routing
```

Before implementing that candidate, compare its XOR-semantics risk and added
complexity against consolidating the already supported Runtime-E4 method. Only
a bounded typed GNN-FiLM design that preserves exact GF(2) information deserves
another training slot.
