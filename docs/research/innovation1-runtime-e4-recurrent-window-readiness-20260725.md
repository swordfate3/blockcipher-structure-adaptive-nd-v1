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
```

`runtime_rounds` is now separate from `processor_steps`. Changing the loaded
runtime window therefore does not change any trainable tensor shape. Existing
plans that omit both new options retain their previous structure window,
parameter count, checkpoint keys and forward path.

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

If RTG3-A confirms the existing E4 topology margin, preregister one local
sub-medium attribution gate that changes only `round_window_mode`. Freeze
SKINNY r7, difference `0x2000`, four pairs, strict encrypted-random-plaintext
negatives, dataset hashes, seeds, epochs, optimizer and checkpoint rule. The
matrix must include:

```text
same-protocol anchor       = last_transition + correct topology
candidate                  = recurrent_window + correct full window
topology control           = recurrent_window + deterministic corrupted window
no-topology control        = recurrent_window + linear topology disabled
equal-compute depth control= recurrent recurrence with only the final
                             transition repeated across the loaded window
```

The equal-compute depth control is required because recurrent processing
applies the shared mixer stack once per loaded transition. Without it, an AUC
gain over the old anchor could be caused by extra recurrent computation rather
than earlier-round structure. Advance beyond the local gate only if both seeds
show that the correct full window beats the repeated-final, corrupted and
no-topology controls, while retaining a useful absolute AUC relative to the
frozen last-transition anchor. A weak or control-invalidated local result must
trigger redesign, not remote scale-up.
