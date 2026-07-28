# Innovation 1 uKNIT-Family Midori64 Qualification K1-AG

**Date:** 2026-07-29
**Status:** completed / passed / Midori64 qualified for position calibration
**Execution:** local CPU; no dataset generation, optimizer step, or remote launch

## Research question

K1-AB established a positive uKNIT r5 diagnostic at sixteen ciphertext pairs,
while K1-AC through K1-AF showed that Dialga's near-perfect score is dominated
by GF(2) signal and multi-query aggregation rather than correct nonlinear
semantics. The next useful family member must therefore change schedule
regularity without introducing a new component vocabulary.

Midori64 is the narrowest such qualification surface. It uses the shared
involutory `Sb0` substitution and the native MIDORI column-mixing primitive
already used by the uKNIT canonical factorization, but its whole round repeats
one homogeneous `ShuffleCell` plus `MixColumn` schedule. K1-AG asks only:

> Can the repository represent a specification-faithful Midori64 cipher and
> its complete runtime SPN transition without changing the fixed K1-AA model
> geometry?

This is an implementation and compatibility qualification, not a neural
performance experiment.

## Frozen evidence

The cipher identity is Banik et al., *Midori: A Block Cipher for Low Energy*,
ASIACRYPT 2015, DOI `10.1007/978-3-662-48800-3_17`, ePrint `2015/1142`.
The two frozen full-encryption vectors are:

```text
PT=0000000000000000
K =00000000000000000000000000000000
CT=3c9cceda2bbd449a

PT=42c20fd3b586879e
K =687ded3b3c85b3f35b1009863e2a8cbf
CT=66bcdc6270d901cd
```

For the second vector, the frozen round-0 and round-1 stages are recorded in
[the K1-AG configuration](../../configs/experiment/innovation1/innovation1_uknit_family_midori64_qualification_k1ag_20260729.json).
They were cross-checked against two independent public implementations before
this plan was opened. Third-party source is not copied into the repository.

## One variable

The only new research variable is a native homogeneous Midori64 schedule. The
existing runtime descriptor schema, fixed virtual-slot K1-AA architecture,
metric code and benchmark semantics are unchanged.

The implementation must distinguish two linear objects:

```text
MIDORI canonical primitive = MixColumn
MIDORI whole round linear   = MixColumn o ShuffleCell
```

The first must equal the existing canonical `midori_linear_layer` on all 64
basis vectors. The runtime descriptor must encode the second, because it is the
actual post-S-box transition used by Midori64 encryption.

## Qualification matrix

No train/validation matrix is allowed. The local audit contains three rows:

| Row | Question | Required evidence |
| --- | --- | --- |
| cipher adapter | Is Midori64 implemented correctly? | two full vectors, frozen round-0/1 stages, all prefixes 1..16 |
| runtime descriptor | Does JSON reproduce the true transition? | all S-box values and all 64 linear basis vectors |
| fixed model geometry | Does K1-AA remain size-independent? | identical state geometry and `214316` trainable parameters at 4 and 16 pairs |

The adapter must also prove that `Sb0`, `ShuffleCell` and `MixColumn` have the
declared inverse properties. These are exact Boolean gates, not approximate
numeric comparisons.

## Frozen gate

K1-AG passes only if every protocol and research check is true:

```text
two public ciphertext vectors exact
all frozen intermediate states exact
round-prefix trace has exactly 16 valid states
Sb0 is self-inverse over all 16 inputs
ShuffleCell inverse is exact over all 16 positions
MixColumn is self-inverse over all 64 basis vectors
native MixColumn equals canonical MIDORI on all 64 basis vectors
runtime JSON S-box equals native Sb0 for every cell and input
runtime JSON linear matrix equals MixColumn o ShuffleCell on all 64 bases
factory and profile registration are exact
K1-AA 4-pair and 16-pair state geometry is identical
both K1-AA models have exactly 214316 trainable parameters
training_rows = validation_rows = optimizer_steps = 0
```

Any failure produces `status=fail`, `remote_scale=no`, and authorizes only a
repair of the failed adapter/descriptor invariant. It does not authorize a
different vector convention, relaxed equality, neural training, or a remote
run.

## Claim boundary

A passing result means only that Midori64 is a correctly implemented,
component-compatible candidate for the uKNIT-family runtime architecture. It
does not establish a useful differential, a Midori neural distinguisher,
cross-cipher transfer, shared-weight generalization, an attack, or SOTA.

## Next action if passed

Run a separate preregistered Midori64 difference-position calibration. Freeze
one reduced-round prefix, difference value/bit role, strict encrypted-random-
plaintext negatives, pair count and data budget; sweep only the sixteen native
cell positions on one discovery seed, then confirm the frozen winner on fresh
seeds and same-key/cross-key scopes. Only a confirmed signal surface may open a
zero-Midori-training shared-checkpoint attribution experiment.

Do not train Midori in K1-AG, do not remote-scale it, do not introduce MANTIS
tweak/reflection semantics, and do not revisit Dialga r5 as a substitute.

## Completed result

The zero-training qualification completed locally under:

```text
run_id = i1_uknit_family_midori64_qualification_k1ag_20260729
status = pass
decision = innovation1_uknit_family_midori64_k1ag_qualified
protocol_valid = true
remote_scale = no
training_rows = validation_rows = optimizer_steps = 0
```

Observed exact evidence:

| Surface | Result |
| --- | ---: |
| Public full-encryption vectors | `2/2` exact |
| Frozen intermediate states | `10/10` exact |
| Prefix states | `16/16` available and adapter-consistent |
| Native/canonical MixColumn basis vectors | `64/64` exact |
| Runtime/full-round linear basis vectors | `64/64` exact |
| Runtime S-box entries | `256/256` exact |
| Runtime transition types | `1` homogeneous repeated transition |
| K1-AA parameters at 4 pairs | `214316` |
| K1-AA parameters at 16 pairs | `214316` |
| K1-AA state geometry | identical, `52` tensors |
| K1-AA forward output at both pair counts | `[2, 1]` |

Every frozen protocol and research check passed. The complete evidence is under:

```text
outputs/local_readiness/
i1_uknit_family_midori64_qualification_k1ag_20260729/
```

No chart was generated because K1-AG contains exact Boolean qualification
checks rather than curves or comparable continuous metrics.

## Final adjudication and executable next step

Keep the Midori64 adapter and descriptor. The result establishes a correct
homogeneous component-equivalent surface for the fixed K1-AA network, but it
does not yet establish any learnable cipher signal.

Next create K1-AH as a separate local position-calibration audit:

```text
question      = which native Midori64 cell preserves a confirmed strict signal?
anchor        = the same difference value/bit role at the default cell
one variable  = native cell position only
discovery     = one frozen seed, all 16 compatible positions
confirmation  = untouched seeds and fresh same-key/cross-key scopes
pairs         = fixed before discovery; no per-position pair tuning
negative      = encrypted random plaintexts
model         = none; deterministic/exact signal statistic plus shuffled labels
execution     = local audit only
advance gate  = frozen winner beats raw and shuffled controls on every
                confirmation seed/scope
stop gate     = no candidate confirms; do not train or remote-scale Midori64
```

K1-AH must preregister its exact reduced-round prefix, difference value,
sample budget, seeds and statistic after checking the existing uKNIT/Dialga
calibration utilities for protocol reuse. Only a passing K1-AH may open a
same-budget neural attribution matrix with correct, wrong-S-box and
wrong-linear controls.
