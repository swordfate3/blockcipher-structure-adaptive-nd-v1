# Innovation 1 uKNIT-Style Canonical-Component SPN Network

Date: 2026-07-27

```text
status = K0 exact factorization passed / K1 planning
priority = next Innovation 1 architecture study after active PRESENT evidence closes
primary cipher = uKNIT-BC
adjacent validation cipher = Dialga-128
remote training = not authorized
```

## 1. Research Decision

Innovation 1 will not continue by forcing one unconstrained descriptor-conditioned
network across every SPN. The next architecture study targets a narrower structural
class:

```text
heterogeneous, non-round-aligned SPNs whose apparent per-round components are
runtime permutations or conjugates of a small set of canonical primitives
```

The method-level question is:

> Can a cipher-name-free neural distinguisher canonicalize each supplied round into
> shared primitive coordinates before learning, so that uKNIT-style changes of S-box
> assignment and linear wiring become aligned computations rather than unrelated
> conditioning vectors?

This is not a universal-SPN claim. It is a structure-adaptive framework for an
explicitly defined component-equivalence class.

## 2. Why uKNIT And Dialga Form A Useful Mechanism Panel

### 2.1 uKNIT-BC

The primary specification is:

```text
Kai Hu, Mustafa Khairallah, Thomas Peyrin and Quan Quan Tan
uKNIT: Breaking Round-Alignment for Cipher Design
IACR Transactions on Symmetric Cryptology, 2026(2)
DOI: 10.46586/tosc.a0zo-4njsuvm
local PDF: papers/算法/uKNIT（轻量级算法设计）.pdf
```

uKNIT-BC has twelve substitution layers and eleven distinct sparse invertible
64-bit GF(2) linear layers. Its important extra structure is not visible when every
truth table and matrix is treated as an unrelated descriptor:

```text
S_(r,c) = D_(r,c) o S_MANTIS o B_(r,c)
```

Every four-bit uKNIT S-box is the same MANTIS/MIDORI base S-box with input and
output bit transpositions. Appendix C also gives an alternative whole-cipher
representation in which the uKNIT linear layers are bit-permutation equivalent to
the MIDORI linear layer. The full data path can therefore be expressed using shared
MANTIS substitutions, shared MIDORI diffusion and round-specific bit permutations.

### 2.2 Dialga

The adjacent specification is:

```text
Subhadeep Banik et al.
Dialga: A Family of Low-Latency Tweakable Block Ciphers Using Multiple Linear Layers
IACR Transactions on Symmetric Cryptology, 2025(4), pages 70-124
DOI: 10.46586/tosc.v2025.i4.70-124
local PDF: papers/算法/（超低时延密码算法）Family of Low-Latency Tweakable.pdf
```

Dialga is also built from Midori-like components. Its round function uses a fixed
matrix multiplication together with round-dependent byte permutations, while its
substitution layer combines a base four-bit S-box with selected bit permutations.
Dialga is not identical to uKNIT, but it supplies an independent 128-bit test of the
same broader mechanism: shared canonical components composed through changing
runtime permutations.

### 2.3 Boundary

```text
uKNIT = primary non-round-aligned mechanism anchor
Dialga = independent Midori-derived heterogeneous stress cipher
GIFT/SKINNY/RECTANGLE = later homogeneous retention controls
MSX = excluded; generalized Feistel with word arithmetic, not this SPN family
```

The phrase `uKNIT-style family` in this project means the structural class above. It
does not claim that the uKNIT paper formally defines several named uKNIT block
ciphers beyond uKNIT-BC.

## 3. Why The Existing Network Is Mismatched

The current Runtime-E4 line has supported exact GF(2)-topology attribution on
SKINNY at the completed two-seed `1000000/class` project-formal scale. That exact
operator path must be retained. The uKNIT evidence nevertheless exposes three
different limitations.

1. The default `last_transition` path sees only the final loaded transition even when
   more descriptors are available.
2. The `recurrent_window` path executes all loaded inverse linear transforms but uses
   one shared hidden-state update over raw round coordinates. On uKNIT prefix-r5 its
   correct-window AUC was `0.501017/0.527083`; only seed1 passed attribution, while
   seed0 was near chance and below corrupted/no-topology controls.
3. Truth-table MLP, inverse-triplet, delta-U, ANF, Adapter, multiplicative gate, True
   FiLM and typed GNN-FiLM routes did not establish correct nonlinear semantics or
   stable joint gains. They must not be reopened under new names.

The current truth-table path asks the network to learn that many superficially
different S-boxes are related. The proposed route performs that known coordinate
alignment exactly before the learned encoder.

## 4. Proposed Architecture

Working name:

```text
Canonical-Transition SPN Network (CT-SPN)
```

The first implementation should remain small:

```text
ciphertext pairs
  -> exact runtime inverse-linear state views for every loaded real transition
  -> compile round descriptors into canonical primitive coordinates
       S_(r,c) -> input permutation B_(r,c)
                  shared MANTIS operator
                  output permutation D_(r,c)
       L_r     -> pre-permutation
                  shared MIDORI-family linear operator
                  post-permutation
  -> shared four-bit cell encoder for every canonical view
  -> small order-sensitive temporal convolution over the transition axis
  -> pair-invariant and cell-invariant pooling
  -> shared binary distinguisher head
```

The transition-axis processor must not read cipher name, cipher id, block width,
round count or a global fingerprint. Variable state width and transition length are
handled through masks and invariant pooling; they do not change trainable parameter
shapes.

The exact GF(2) operations stay outside the learned network. A real-valued message
passing layer must not be used as a substitute for XOR. The neural component learns
how to combine exact canonical state views, not how to approximate the cipher's
linear layer.

### 4.1 Difference From The Closed Recurrent Route

The closed uKNIT recurrent route accumulated raw-coordinate transition embeddings
inside one hidden state. CT-SPN instead exposes an explicit tensor:

```text
[sample, pair, real_transition, cell, canonical_cell_feature]
```

The network can compare adjacent real transitions without requiring one recurrent
state to remember which round-specific coordinate system produced an earlier view.
The first candidate uses a shallow shared temporal convolution, not a larger GRU,
LSTM or Transformer.

### 4.2 Extending To A New Cipher

For a new cipher using the same canonical primitive family, adaptation requires only
a verified factorization descriptor. No new cipher-specific backbone or head is added.

If a new cipher uses a genuinely different canonical S-box or linear primitive, it may
later add one primitive expert only after an independent semantic and attribution
gate. Experts are indexed by verified primitive mechanism, not cipher name. A large
learned MoE is not part of the first study.

## 5. Staged Evidence Plan

### K0: Exact Factorization Audit

Before training, implement a zero-training audit that proves the canonical descriptors
are cryptographically exact.

For uKNIT require:

- every one of the `12 x 16` S-box factorizations reconstructs the published truth
  table for all sixteen inputs;
- every one of the eleven linear factorizations matches the existing 64-bit matrix on
  all 64 unit vectors;
- reconstructed round and full-cipher traces match the existing published-vector-
  verified implementation;
- shuffling a nontrivial input/output permutation or round order changes the operator
  fingerprint;
- no trainable parameters, training rows or optimizer steps are created.

For Dialga require the corresponding bit/byte-permutation and matrix identities for
all loaded round types and the existing public-vector checks.

K0 failure permits only factorization repair. It does not authorize neural training.

K0 completed on 2026-07-27. All `3072` uKNIT S-box probes, `704`
uKNIT linear unit probes, `1024` Dialga byte-S-box probes and `512` Dialga
linear unit probes matched exactly. The canonical operators also matched all four
uKNIT vectors, eleven uKNIT prefix states, all four Dialga vectors and the complete
sixteen-state Dialga trace. Repeated recovery produced the same manifest and all
wrong bit/byte/order controls changed fingerprints. The gate passed with zero
training rows and zero optimizer steps. This establishes the compiler boundary only;
K1 is still required to test neural usefulness.

### K1: Linear-Schedule Specialist

K1 changes one learned hypothesis only: raw Runtime-E4 transition fusion versus
canonical exact-state-view fusion. Learned S-box conditioning is disabled in both
roles so that the already rejected nonlinear path cannot contaminate attribution.

Frozen family panel:

```text
primary = uKNIT-BC prefix-r5
secondary = Dialga-128 prefix-r4
train = 2048/class/cipher
validation = 1024/class/cipher
pairs/sample = 4
seeds = 0,1
epochs = 10
negative = encrypted random plaintext pairs
execution = local sub-medium diagnostic
```

Train only two roles per seed:

```text
1. strongest same-protocol Runtime-E4 anchor
2. parameter-matched CT-SPN candidate
```

Evaluate each frozen candidate checkpoint with:

```text
correct ordered transitions
same-length repeated-last transitions
shuffled transition order
deterministically corrupted topology
no topology
```

The controls change only runtime structure. Data, labels, checkpoint and head remain
fixed. The active trainable-parameter difference between candidate and anchor must be
at most one percent.

Advance only when both seeds satisfy:

```text
uKNIT candidate AUC >= 0.520
Dialga candidate AUC >= 0.550
candidate - same-budget anchor >= +0.005 per cipher
candidate - repeated-last >= +0.005 per cipher
candidate - shuffled-order >= +0.005 per cipher
candidate - corrupted/no-topology >= +0.005 per cipher
```

The uKNIT absolute floor preserves the completed U3 same-budget gate. Dialga's higher
floor reflects its completed strong local prefix-r4 signal. A two-cipher macro average
cannot hide failure on uKNIT.

### K2: Canonical S-box Composition

K2 is prohibited unless K0 and K1 pass. It changes only the nonlinear representation:

```text
raw/disabled S-box path
  -> exact B^-1 / shared involutory MANTIS S-box / D^-1 canonical operator path
```

The candidate must beat input-permutation, output-permutation, identity and
wrong-round-assignment controls from one frozen checkpoint. Responsiveness without
correct-semantic dominance is a hold, as established by S1 and S2.

### K3: Homogeneous Retention And New-Cipher Holdout

Only after the uKNIT/Dialga family gate passes should one homogeneous SPN be added as
a retention control and one genuinely unseen component-equivalent cipher be selected.
Whole-cipher holdout claims require zero target training rows, zero target checkpoint
selection and correct-versus-wrong factorization controls.

## 6. Explicitly Blocked Routes

- Do not launch another Adapter, FiLM, typed GNN, truth-table MLP, ANF-gate or learned
  soft/Top-k MoE rescue.
- Do not use cipher names or a cipher-identification auxiliary loss.
- Do not mix K1 with a new difference search, new negative definition, extra pairs or
  changed validation protocol.
- Do not call `2048/class` a formal result, attack, SOTA comparison or uKNIT ceiling.
- Do not remotely scale K1 unless both seeds pass the local family and control gates.
- Do not launch the prepared RECTANGLE RCT3 merely because it exists; the user's
  current priority is the uKNIT-style architecture study.
- Do not interrupt the active PRESENT formal evidence run. K0 is zero-training and K1
  remains pending until that run is locally retrieved and adjudicated.

## 7. Thesis-Safe Claim If Successful

A successful K0-K3 route would support:

> A runtime structure-adaptive neural distinguisher for component-equivalent,
> non-round-aligned SPNs. The method compiles round-specific substitutions and
> diffusion layers into shared canonical primitive coordinates, retains fixed
> backbone geometry, and distinguishes correct ordered factorizations from repeated,
> shuffled and corrupted controls.

It would not support an arbitrary-SPN or all-block-cipher claim. That narrower scope
is intentional and testable.

## 8. Immediate Next Action

Keep the active PRESENT seed1 monitor as the only remote training owner. K0 is now
complete and retained. Preregister and implement K1 readiness without changing the
frozen data protocol; execute its local training only after the PRESENT result is
locally retrieved and adjudicated. Do not launch RCT3 or a remote uKNIT run before
those gates are available.
