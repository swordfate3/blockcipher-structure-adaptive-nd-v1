# Innovation 1 Runtime-E4 Recurrent Runtime Window Readiness

Date: 2026-07-25

## Scope

This record closes an implementation gap identified by the runtime-SPN audit:
the existing `RuntimeE4EquivariantSpnDistinguisher` loaded a multi-round
runtime descriptor but used only the final inverse-linear map and final S-box
descriptor. The change in this record adds an opt-in processor that consumes
every loaded transition. It does not change the frozen RTG2/RTG3 E4 protocol,
and it is not a trained result.

## Interface

The existing behavior remains the default:

```text
round_window_mode = last_transition
```

The new candidate is selected explicitly:

```text
round_window_mode = recurrent_window
runtime_rounds    = number of runtime descriptor transitions to load
processor_steps   = number of shared E4 token-mixer blocks
runtime_structure_window_control = full | repeat_last
```

`runtime_rounds` is now separate from `processor_steps`. Changing the loaded
runtime window therefore does not change any trainable tensor shape. Existing
plans that omit both new options retain their previous structure window,
parameter count, checkpoint keys and forward path.

`repeat_last` preserves the loaded window length but replaces every S-box and
linear descriptor with the final transition. It is an equal-recursion-depth
control for a genuinely heterogeneous runtime window. On homogeneous
PRESENT/GIFT/SKINNY windows it is intentionally an exact no-op.

Every constructed model records the transformed structure actually received
by the backbone, not only the source descriptor identity:

```text
runtime_structure_transition_sha256s
runtime_structure_window_sha256
runtime_structure_unique_transition_count
runtime_structure_homogeneous
```

Each transition hash uses canonical JSON over cell membership, bit roles, that
round's S-box truth descriptors and its GF(2) linear matrix. The window hash is
order-sensitive. Corruption, S-box shuffling and repeated-final controls are
therefore distinguishable from the source descriptor and reproducible across
Linux and Windows.

## Processor Semantics

For each loaded transition, in reverse order, the processor:

1. aligns both ciphertext endpoints with that round's exact inverse GF(2)
   linear map when topology is enabled;
2. builds current/previous cell tokens with the existing shared cell encoder;
3. conditions those tokens on that round's per-cell 4-bit S-box truth table;
4. applies the existing E4 cell-equivariant mixer stack;
5. recurrently merges the transition state into one persistent cell-token
   state before advancing to the earlier transition.

The same trainable modules are reused at every transition. There are no
cipher-name embeddings, per-cipher heads, per-round parameter arrays or
runtime-width-dependent trainable tensors. The output state is pooled with the
same pair attention and classifier as the existing E4 model.

This is a learned structure processor, not exact multi-round partial
decryption. Successive inverse-linear views do not remove unknown round keys
or claim to invert nonlinear rounds. The S-box tables and GF(2) maps are
runtime conditioning information whose usefulness still requires controlled
training evidence.

## Deterministic Readiness Evidence

Focused tests establish the following invariants:

```text
legacy default preserved
    omitted round_window_mode selects last_transition

one-round reduction
    recurrent_window and last_transition produce bit-exact equal logits
    under identical weights and a one-round runtime structure

earlier-linear counterfactual
    changing only an earlier GF(2) layer leaves last_transition logits equal
    but changes recurrent_window logits

earlier-S-box counterfactual
    changing only an earlier S-box descriptor leaves last_transition logits
    equal but changes recurrent_window logits

heterogeneous-window control
    a uKNIT full window and same-length repeated-final window have equal model
    geometry and final descriptors but different earlier descriptors/logits

homogeneous-window boundary
    PRESENT, GIFT and SKINNY full windows are byte-identical to repeated-final
    windows; repeated-final therefore produces bit-exact equal logits

result evidence identity
    transformed per-transition hashes, ordered window hash, unique-transition
    count and homogeneous flag are JSON-serializable model metadata

cell relabeling
    jointly relabeling input cells and runtime structure preserves logits
    within 1e-6

fixed geometry
    one-round and three-round runtime windows have identical state-dict shapes
    and the same recurrent backbone also forwards a synthetic 128-bit SPN

general GF(2)
    recurrent forward and backward-compatible tests pass on SKINNY-64/64
```

The recurrent mode deliberately rejects the final-transition-only `delta_v`
and `delta_u` query heads. Their query semantics have not yet been defined for
multiple runtime transitions, so silently applying them would create an
ambiguous protocol.

## Evidence Boundary

This readiness result proves that the implementation can consume earlier
runtime linear and S-box descriptors without changing parameter geometry. It
does not prove that the added transitions improve validation AUC, that learned
weights use each transition correctly, or that the model transfers across
ciphers. It is not an attack, paper-scale evidence, SOTA evidence or a
breakthrough claim.

The active RTG3-A formal SKINNY run continues to use:

```text
round_window_mode = last_transition
```

Therefore its result remains directly comparable with RTG2-B. This readiness
change must not be retroactively attributed to RTG2/RTG3 checkpoints.

## Recommended Next Action

First complete and jointly adjudicate RTG3-A seed0 and conditional seed1. Do
not launch recurrent-window training while that formal replication is active.

The built-in PRESENT, GIFT and SKINNY descriptors repeat the same S-box and
linear topology in every loaded round. A recurrent SKINNY experiment could
measure repeated structure-processing depth, but it cannot establish that the
network used distinct earlier-round topology. Do not use SKINNY full-window
versus repeated-final as that attribution claim.

If RTG3-A confirms the existing E4 topology margin, preregister one local
sub-medium uKNIT attribution gate. This is a new mechanism that directly
addresses the stopped U2-H route's documented last-transition limitation; it
does not revive the old delta-U query unchanged. Freeze uKNIT prefix r5,
difference `0x40`, four pairs, strict encrypted-random-plaintext negatives,
runtime window S3/L3 through S4/L4, dataset hashes, seeds, epochs, optimizer
and checkpoint rule. Use the two-input state-triplet geometry because
final-transition-only delta-query semantics remain prohibited in recurrent
mode. The matrix may use five rows because it is an attribution phase gate:

```text
same-geometry anchor       = last_transition + correct heterogeneous window
candidate                  = recurrent_window + correct heterogeneous window
equal-depth control        = recurrent_window + repeated-final window
topology control           = recurrent_window + corrupted heterogeneous window
no-topology control        = recurrent_window + linear topology disabled
```

The repeated-final control is required because recurrent processing applies
the shared mixer stack once per loaded transition. Without it, an AUC gain over
the old anchor could be caused by extra recurrent computation rather than
earlier-round structure. Advance beyond the local gate only if both seeds show
that the correct heterogeneous window beats the repeated-final, corrupted and
no-topology controls, while retaining a useful absolute AUC relative to the
last-transition anchor. A weak or control-invalidated local result must trigger
redesign, not remote scale-up.

## Fail-Closed Pre-Training Gate

The deterministic command
`scripts/check-runtime-spn-recurrent-window-readiness` now constructs every
model in a proposed two-seed panel before any dataset generation. It records
the transformed transition hashes, ordered window hash, unique-transition
count, parameter count and a full named-parameter shape fingerprint. The gate
requires exactly these five equal-budget roles per seed:

```text
last_transition correct anchor
recurrent_window correct full-window candidate
recurrent_window repeated-final equal-depth control
recurrent_window corrupted-topology control
recurrent_window no-topology control
```

It fails closed unless uKNIT r5 uses the frozen two-transition S3/L3--S4/L4
window, four ciphertext pairs, `2048/class`, two seeds, ten epochs and strict
encrypted-random-plaintext negatives. It also rejects homogeneous candidate
windows, non-homogeneous repeated-final controls, unequal final transitions,
equal full/repeated window hashes, parameter-geometry drift, wrong role modes,
or any data/training protocol drift. The gate contains no AUC threshold and
does not authorize training while RTG3-A remains unresolved.

The gate now uses fixed, seed-independent model initialization and a fixed
`512-bit` binary probe to execute forward and backward passes for all ten
planned models. It requires finite `[2, 1]` logits, finite full parameter-gradient
coverage, seed-invariant role output hashes, and distinct candidate logits from
the last-transition, repeated-final, corrupted and no-topology interventions.
This closes the gap between model construction and actual computation-graph
readiness. Different untrained logits prove only that the interventions are
active; they are not accuracy, AUC or evidence that one role is superior.
