# Innovation 1 uKNIT-Family Canonical Component Factorization K0

Date: 2026-07-27

```text
run_id = i1_uknit_family_canonical_component_factorization_k0_20260727
status = completed / pass
execution = local CPU
training_rows = 0
optimizer_steps = 0
remote = no
```

## 1. Question

Before building the Canonical-Transition SPN Network (`CT-SPN`), can the native
uKNIT-BC and Dialga-128 components be compiled exactly into a shared
MANTIS/MIDORI component basis?

The audit tests representation correctness only. It does not train a neural
network and cannot establish differential signal, transfer, attack performance or
a universal-SPN claim.

## 2. Frozen Anchors

The same-budget anchor is exact evaluation of the existing native implementations,
which already pass published vectors:

```text
uKNIT anchor = 12 x 16 native four-bit S-box tables + 11 native 64-bit linear maps
Dialga anchor = 4 native byte S-box types + 4 native 128-bit linear round types
```

Canonical primitives:

```text
nonlinear = MANTIS/MIDORI four-bit S-box
uKNIT linear = MIDORI 64-bit involutory diffusion
Dialga linear = fixed Midori-style MixColumns
```

The uKNIT alternative-representation source is pinned only as a specification
cross-reference and external oracle:

```text
repository = https://github.com/syllab-ntu/UKNIT.git
commit = f6493014fb7326cf3fffa2bb642b26cd59650e4f
source = uknit-implementations/uKNIT-BC_alternative_representation.cpp
```

The project implementation remains independent. The audit recovers uKNIT factors
from the published native component tables and verifies the resulting operators;
it does not copy GPL encryption logic.

## 3. One Hypothesis

```text
H_K0:
uKNIT and Dialga apparent heterogeneous components are exact bit/byte-permutation
compositions of shared MANTIS/MIDORI primitives.
```

No data protocol, neural architecture, optimizer, sample count or training setting
is changed because K0 has no training.

## 4. Exact Checks

### 4.1 uKNIT-BC

1. Recover a deterministic input and output four-bit permutation for each of the
   `12 x 16 = 192` native S-box tables by exhaustive search over `24 x 24`
   permutation pairs.
2. Reconstruct every table for all 16 inputs: `192 x 16 = 3072` exact probes.
3. Recover input/output bit permutations between each native 64-bit linear map and
   the MIDORI diffusion by colour-preserving bipartite graph isomorphism.
4. Reconstruct all eleven maps on all 64 unit vectors: `11 x 64 = 704` exact probes.
5. Rebuild prefix and full encryption from the recovered canonical components and
   require all four official full-round vectors plus all eleven official zero-key
   prefix states to match the native implementation.
6. Repeat factor recovery and require identical manifests.
7. Require a wrong nontrivial bit permutation and shuffled transition order to
   change operator fingerprints.

### 4.2 Dialga-128

1. Reconstruct all four byte S-box types as bit-permutation conjugates of two
   parallel MANTIS S-boxes for every byte value: `4 x 256 = 1024` exact probes.
2. Reconstruct all four 128-bit linear round types as the published byte
   permutation followed by fixed MixColumns on all unit vectors:
   `4 x 128 = 512` exact probes.
3. Run the published scheduler with canonical replacement operators and require
   all four published full vectors and the complete published 16-round trace to
   match the native implementation.
4. Require wrong bit/byte permutations and shuffled linear round order to change
   fingerprints.

## 5. Evidence And Gate

Required local artifacts:

```text
outputs/local_audit/i1_uknit_family_canonical_component_factorization_k0_20260727/
  results.jsonl
  validation.json
  gate.json
  summary.json
  progress.jsonl
```

No chart is generated: exhaustive equality counts and SHA-256 operator fingerprints
are stronger evidence than a visualization for this exact gate.

Advance only if every protocol and exact-reconstruction check passes:

```text
uKNIT S-box probes                 = 3072/3072
uKNIT linear unit probes           = 704/704
uKNIT official full vectors        = 4/4
uKNIT official prefix states       = 11/11
Dialga byte S-box probes           = 1024/1024
Dialga linear unit probes          = 512/512
Dialga official full vectors       = 4/4
Dialga official trace states       = 16/16
recovery manifest deterministic    = true
all wrong controls distinct        = true
training rows / optimizer steps    = 0 / 0
```

Any failed equality makes K0 invalid and permits only factorization, indexing or
boundary-semantics repair. It does not authorize tuning the gate or neural training.

## 6. Claim Boundary

A K0 pass supports only:

> The project can exactly compile the native uKNIT/Dialga components into shared
> MANTIS/MIDORI-family primitives with explicit runtime permutations.

It does not show that a neural network can exploit these views, that uKNIT and
Dialga are one formally named cipher family, or that the method generalizes to MSX
or arbitrary SPNs.

## 7. Evidence-Dependent Next Action

If K0 passes, retain the canonical compiler and prepare K1 as the frozen local
`2048/class`, two-seed, same-protocol neural diagnostic already defined in the
research blueprint. K1 changes only raw transition fusion to canonical exact-state
view fusion and compares it with the strongest Runtime-E4 anchor plus frozen
repeated, shuffled, corrupted and no-topology evaluations.

Do not launch remote uKNIT training, enlarge K1, add a learned MoE, or start K2
nonlinear conditioning from a K0 pass. If K0 fails, repair only the exact component
factorization and rerun K0 unchanged.

## 8. Completed Result

K0 completed locally on 2026-07-27 with:

```text
status = pass
decision = innovation1_uknit_family_canonical_component_factorization_supported
recent-results index = 001

uKNIT S-box factors/probes       = 192; 3072/3072
uKNIT linear factors/probes      = 11; 704/704
uKNIT full vectors/prefix states = 4/4; 11/11
uKNIT repeated manifest          = identical
uKNIT wrong controls             = both distinct

Dialga byte S-box probes         = 1024/1024
Dialga linear probes             = 512/512
Dialga full vectors/trace states = 4/4; 16/16
Dialga wrong controls            = all distinct

training rows / optimizer steps  = 0 / 0
validation errors                = []
```

Artifacts:

```text
outputs/local_audit/i1_uknit_family_canonical_component_factorization_k0_20260727/
```

The deterministic uKNIT factor manifest is:

```text
d66983062701799cabf66f17ac168e0c0ecfdd9b6efe58d1d4f79d4aa2bb1592
```

### Adjudication

Retain the exact canonical compiler. The earlier unconstrained truth-table/matrix
conditioning problem is now separated from component semantics: every native
uKNIT/Dialga operator used by the proposed family route has an exact shared-primitive
representation, and the frozen wrong controls do not collapse to that representation.

This is readiness evidence, not neural performance evidence. The next experiment is
K1, which must test whether canonical exact-state-view fusion improves the same
uKNIT-r5 and Dialga-r4 differential protocol over the strongest Runtime-E4 anchor.
Do not proceed to K2, remote scaling, a learned MoE or an MSX claim from K0 alone.
