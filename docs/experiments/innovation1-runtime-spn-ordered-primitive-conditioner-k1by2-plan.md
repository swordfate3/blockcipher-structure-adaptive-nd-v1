# Innovation 1 Runtime SPN Ordered Primitive Conditioner K1-BY2

**Date:** 2026-08-01
**Status:** completed / pass
**Execution:** local sub-medium fresh-seed confirmation; local CUDA unavailable, so use documented CPU fallback

## Research question

K1-BY1 found a large two-seed separation between correct ordered primitive
routing and both wrong-order and no-conditioner controls on uKNIT r5. K1-BY2
asks one confirmation question before the route is exposed to another cipher:

> Does the same correct-route advantage reproduce on untouched seeds and
> untouched fixed train/validation keys without changing the difference,
> network, sample budget or optimization protocol?

This is a local `2048/class` confirmation, not formal training, paper-scale
evidence, cross-cipher transfer or a SOTA claim.

## Frozen source authority

K1-BY2 must bind the completed K1-BY1 artifacts exactly:

```text
source run     = i1_runtime_spn_ordered_primitive_conditioner_k1by1_16pair_2048_seed3_seed4_20260801
source status  = pass
source decision= innovation1_runtime_spn_k1by1_compiler_conditioner_supported
```

The K1-BY1 plan, gate, result and validation SHA-256 digests are checked before
optimization. Any source drift invalidates readiness.

## Single variable

Only the seed and fixed key pair change. K1-BY1's strongest wrong-routing
control by two-seed mean AUC was wrong stage order, so the lean matrix contains:

| Condition | Role |
|---|---|
| `correct_compiler_routing` | K1-BY1 candidate and same-budget anchor |
| `wrong_order_routing` | strongest wrong-routing control |
| `no_compiler_conditioner` | identical parameter geometry with structure residual disabled |

All three conditions instantiate `235780` trainable parameters and use the same
raw-pair backbone, primitive experts, GRU and classifier geometry.

## Frozen protocol

Train six independent rows: two fresh seeds times three conditions.

| Field | Frozen value |
|---|---|
| Cipher / rounds | uKNIT-BC r5 |
| Difference | cell11 role1, `0x0000400000000000` |
| Train | `2048/class`, `4096` total rows |
| Cross-key validation | `1024/class`, `2048` total rows |
| Seeds | `5`, `6` |
| seed5 train / validation key | `0x8888...8888` / `0x9999...9999` |
| seed6 train / validation key | `0xaaaa...aaaa` / `0xbbbb...bbbb` |
| Pairs/sample | `16` independent ciphertext pairs |
| Input width | `2048` bits/sample, `128` bits/pair |
| Negative definition | encrypted random plaintexts |
| Runtime window | round start `3`, two transitions |
| Epochs / batch | `10` / `64` |
| Loss / optimizer | MSE / Adam |
| Learning rate / weight decay | `1e-4` / `1e-5` |
| Checkpoint | restored best validation AUC |
| Device | local CPU fallback, recorded before optimization |

The same seed must reuse identical disk-backed datasets across its three
conditions. No difference scan, key tuning, model tuning or seed-dependent
configuration is allowed.

## Readiness gate

Before optimizer steps require:

1. exact K1-BY1 source digests and pass decision;
2. exactly six frozen plan rows and the two fresh key pairs;
3. equal `235780` trainable parameters across all conditions;
4. finite `[B, 1]` outputs and finite nonzero total gradients;
5. correct and wrong-order semantic digests differ;
6. only no-conditioner disables the primitive residual;
7. no model consumes cipher identity or absolute cell/bit identity.

## Result gate

For seed5 and seed6 separately require:

```text
correct AUC                    >= 0.550
correct - no conditioner       >= +0.010
correct - wrong order          >= +0.005
```

Every clause must pass on both seeds. Averaging cannot rescue a failed seed.

## Decisions

- **Both seeds pass all clauses:** confirm the uKNIT route and preregister a
  same-contract PRESENT/GIFT permutation-expert diagnostic.
- **Correct signal passes but a margin fails:** hold structure attribution and
  inspect the failed control only; do not scale or add ciphers.
- **Either correct AUC fails:** hold the route as seed/key dependent and audit
  deterministic feature distributions without retraining a larger model.
- **Protocol invalid:** repair only the failed invariant and rerun unchanged.

Blocked: remote execution, more samples, more epochs, more pairs, new
differences, wrong-binding reintroduction, model width changes and publication
claims.

## Planned artifacts

```text
run_id = i1_runtime_spn_ordered_primitive_conditioner_k1by2_fresh_seed5_seed6_20260801
```

The completed result must include readiness, caches, progress, six checkpoints,
results, gate, validation, summary, comparison CSV, history CSV, Chinese SVG,
plot report and rendered-pixel visual QA evidence. Refresh both recent-result
indexes and append the measured result and next executable action here.

## Completed result

The frozen six-row run completed without protocol failures. Validation found
exactly six result rows and no missing, mismatched or duplicate rows.

| Condition | seed5 AUC | seed6 AUC |
|---|---:|---:|
| correct compiler routing | `0.999890327` | `1.000000000` |
| wrong stage order | `0.502291203` | `0.508229256` |
| no compiler conditioner | `0.500099659` | `0.526427746` |

The per-seed attribution margins were:

| Margin | seed5 | seed6 |
|---|---:|---:|
| correct - wrong order | `+0.497599125` | `+0.491770744` |
| correct - no conditioner | `+0.499790668` | `+0.473572254` |

Correct-route validation accuracy was `0.999023438` for seed5 and
`0.999511719` for seed6. The selected best epochs were 9 and 7 respectively;
seed6 reached best calibrated accuracy `1.000000000` at its selected
checkpoint.

Every preregistered signal and margin clause passed independently on both
fresh seeds and cross-key validation sets. The decision is therefore:

```text
status       = pass
decision     = innovation1_runtime_spn_k1by2_fresh_seed_confirmed
remote_scale = no
```

This confirms that the K1-BY1 uKNIT r5 advantage is not confined to its first
two seed/key pairs. It remains local `2048/class` diagnostic evidence. It does
not establish formal-scale performance, cross-cipher transfer, attack-round or
SOTA performance.

## Attribution limits

The compiled uKNIT window invoked `sbox4_table` and `linear_gf2`; it invoked no
`linear_permutation` expert. The result therefore validates correct stage order
and GF(2) target binding under the integrated deterministic-execution plus
learned-expert route, but it does not validate the permutation expert.

The present matrix also does not isolate deterministic compiled inverse
execution from the learned primitive descriptors. A later ablation must compare
deterministic execution alone against deterministic execution plus learned
primitive descriptors before attributing the full gain to learned structure.

## Recommended next action

Do not scale uKNIT or add epochs. First run a bounded K1-BY3 evidence-surface
selection audit over existing PRESENT and GIFT artifacts. The audit must identify
one already-supported permutation-SPN cipher/round/difference protocol with
strict encrypted-random-plaintext negatives and fresh same-protocol
difference-position evidence. This resolves whether PRESENT or GIFT is the
cleaner test surface without spending a training slot or tuning a difference
per model.

After that choice is frozen, preregister one local sub-medium diagnostic with
the same parameter geometry and exactly three conditions:

```text
correct compiled permutation routing
wrong permutation/target binding
no compiler conditioner
```

Use two untouched seeds, `16` pairs, `2048/class` training,
`1024/class` cross-key validation and `10` epochs unless the selected existing
anchor requires a stricter already-frozen same-budget protocol. Change only the
runtime cipher descriptor and its already-supported difference surface. Require
per seed `correct AUC >= 0.550`, `correct - no conditioner >= +0.010` and
`correct - wrong routing >= +0.005`. Passing unlocks a cross-cipher shared-weight
or transfer test; failure holds the permutation expert for local redesign. No
remote scale, difference scan, larger matrix or mechanical uKNIT scale-up is
authorized by K1-BY2.
