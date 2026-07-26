# Innovation 1 Runtime-SPN Multiscale Orbit-Basis C4 Plan

```text
status = completed / pass / zero-training audit only
run_id = i1_runtime_spn_multiscale_orbit_basis_c4_20260726
remote = no
training_rows = 0
optimizer_steps = 0
decision = innovation1_runtime_spn_multiscale_orbit_basis_feasible
```

## Research Question

Does a fixed multiscale bank of exact inverse GF(2) runtime views expose
non-collapsed structure information beyond Runtime-E4's current one-step view,
while remaining sensitive to correct topology and heterogeneous transition
order and equivariant to cell relabeling?

C4 does not modify the running C3 experiment. It does not generate ciphertext
data or train a neural network. It is a representation feasibility audit that
must pass before a multiscale orbit model may be proposed.

## Same-Method Anchor

Current Runtime-E4 computes a current state and one exact inverse-linear state:

```text
anchor depths = 0, 1
```

C4 changes one representation variable by retaining:

```text
candidate depths = 0, 1, 2, 4, 8
```

No learned parameter, S-box route, data protocol or metric is changed because
C4 performs zero training and zero cipher-data generation.

## Frozen Structure Panel

| Cipher | Width | Runtime window |
| --- | ---: | ---: |
| PRESENT-80 | 64 | one homogeneous P layer |
| GIFT-64 | 64 | one homogeneous P layer |
| RECTANGLE-80 | 64 | one homogeneous ShiftRows layer |
| SKINNY-64/64 | 64 | one homogeneous ShiftRows/MixColumns layer |
| uKNIT-BC | 64 | all ten heterogeneous transitions |
| Dialga | 128 | one complete four-transition cycle |

Cipher names are audit metadata only. The orbit computation receives only the
runtime linear matrices and fixed depths.

## Exact Orbit

For every structure, form the complete all-unit-bit probe basis. Because the
operator is linear, its output over this basis is the exact matrix of the
composed inverse view rather than a sample estimate.

```text
O_0 = I
O_d = L_(r-d)^-1 ... L_(r-1)^-1
depths = 0, 1, 2, 4, 8
```

Traversal starts at the last supplied transition and moves backward. It cycles
through the frozen runtime window only when depth exceeds the window length.
The five-view axis is independent of block width and number of supplied rounds.

## Controls

For all six ciphers:

- `corrupted`: use `RuntimeSpnStructure.corrupted(seed=20260724)` and the same
  depths;
- `no_topology`: replace every nonzero-depth view with identity.

For uKNIT and Dialga only:

- `repeat_last`: repeat the final transition across the complete window;
- `rotated_window`: rotate the supplied ordered transition window by one while
  preserving every matrix and the same view depths.

Support distance is Jaccard distance over all nonzero-depth matrix entries:

```text
distance(A, B) = count(A XOR B) / count(A OR B)
```

## Protocol Checks

C4 is valid only when all checks pass:

1. exactly six structures load with the frozen widths and round counts;
2. depths are exactly `0,1,2,4,8` and every depth-zero view is identity;
3. every exact, corrupted, repeat-last and rotated-window view is binary and
   full GF(2) rank;
4. depth one is bit-exact with each structure's existing `exact_inverse`;
5. correct and corrupted structures have distinct deterministic fingerprints;
6. repeat-last and rotated controls are used only for uKNIT and Dialga and are
   distinct from their ordered windows;
7. rotating every cell label by one conjugates every orbit view exactly;
8. every artifact metric is finite and computed from the complete unit basis;
9. a repeated same-config audit has the same manifest SHA-256;
10. training rows, optimizer steps and remote execution remain zero.

Protocol failure is invalid evidence and permits repair only of the failed
invariant.

## Frozen Research Gate

Every cipher must satisfy:

```text
unique exact views among depths 0/1/2/4/8 >= 3
new distinct views at depths 2/4/8 beyond depths 0/1 >= 1
correct-corrupted support distance >= 0.25
correct-no-topology support distance >= 0.25
```

Both uKNIT and Dialga must additionally satisfy:

```text
correct-repeat-last support distance >= 0.10
correct-rotated-window support distance >= 0.10
```

The gate is cipher-balanced: all six ciphers are required, and the ten uKNIT
transitions do not receive ten votes.

## Decisions

Pass:

```text
decision = innovation1_runtime_spn_multiscale_orbit_basis_feasible
next = after C3 completes, preregister C5 same-budget local neural diagnostic
```

C5 must compare a single orbit-basis candidate against unchanged Runtime-E4
and necessary correct/corrupted/no-topology orbit controls under an unchanged
sub-medium data protocol. C4 does not authorize C5 implementation or training.

Hold:

```text
decision = innovation1_runtime_spn_multiscale_orbit_basis_not_ready
next = close this operator basis; keep C3 as the only active training route
```

Do not tune depths, corruption seed, controls, metrics or thresholds after
reading C4.

## Claim Boundary

C4 can establish operator-level non-collapse, topology/control separation,
ordered-window sensitivity and cell-relabel equivariance on six known
structures. It cannot establish differential signal, learnability, neural
performance, zero-shot transfer, unseen-cipher generalization, nonlinear S-box
composability, an attack, SOTA or a breakthrough.

## Artifacts

```text
outputs/local_audit/i1_runtime_spn_multiscale_orbit_basis_c4_20260726/
  results.jsonl
  validation.json
  gate.json
  summary.json
  progress.jsonl
```

No visualization is planned. Exact per-cipher tables and protocol checks are
the appropriate evidence; a two-dimensional projection would be ornamental.

## Completed Result

The frozen C4 audit completed locally on 2026-07-26. It emitted six result
rows, all ten protocol checks passed, and the progress log ended with
`run_done`.

| Cipher | Unique views | New multihop views | Correct-corrupted | Correct-no-topology | Correct-repeat-last | Correct-rotated-window |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PRESENT-80 | 3 | 1 | 0.986139 | 0.967742 | n/a | n/a |
| GIFT-64 | 3 | 1 | 0.994106 | 0.623656 | n/a | n/a |
| RECTANGLE-80 | 5 | 3 | 0.988142 | 0.769231 | n/a | n/a |
| SKINNY-64/64 | 5 | 3 | 0.889303 | 0.966667 | n/a | n/a |
| uKNIT-BC | 5 | 3 | 0.728573 | 0.986077 | 0.683315 | 0.713592 |
| Dialga | 5 | 3 | 0.931696 | 0.942609 | 0.529862 | 0.682353 |

The weakest all-cipher topology separations were GIFT's
correct-no-topology distance `0.623656` and uKNIT's correct-corrupted distance
`0.728573`, both above the frozen `0.25` threshold. The weakest heterogeneous
window separation was Dialga's correct-repeat-last distance `0.529862`, above
the frozen `0.10` threshold. Every cipher retained at least three unique exact
views and at least one view beyond depths zero and one.

```text
validation = pass
protocol checks = 10/10
research checks = 6/6
result rows = 6
decision = innovation1_runtime_spn_multiscale_orbit_basis_feasible
```

This result keeps the multiscale orbit basis as a candidate representation. It
does not show that a neural network can learn from the views or that they carry
differential signal. It also does not reopen the failed S-box descriptor,
synthetic-topology, Adapter, FiLM, MoE or larger-backbone routes frozen by C2.

## Evidence-Backed Next Action

Do not start C5 while C3 is still running. After C3 completes and is retrieved,
preregister one sub-medium, same-budget local neural diagnostic with exactly
one representation change:

```text
anchor = unchanged Runtime-E4 depths 0/1
candidate = Runtime-E4 with frozen exact orbit depths 0/1/2/4/8
required controls = corrupted orbit / no-topology orbit
data, labels, negative definition, pairs, seed, epochs and optimizer = unchanged
advance gate = candidate beats anchor and both controls under the frozen C5 plan
stop gate = candidate ties or loses to anchor, or correct orbit fails a control
remote = no until the local C5 gate passes
```

The reason to wait for C3 is attribution: C3 first determines whether the
existing one-step exact-topology model survives at formal PRESENT r7 scale. C5
then tests only whether retaining exact multihop views improves that same model,
without mixing representation and benchmark changes or consuming another
remote slot prematurely.
