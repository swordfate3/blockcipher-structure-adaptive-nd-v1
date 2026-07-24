# Innovation 1 uKNIT Runtime-SPN Mechanism Reproduction

Date: 2026-07-24

## Research Question

Can the runtime-parameterized SPN implementation represent a real cipher whose
S-box assignment and general GF(2) diffusion both change by round, without
changing the neural backbone parameter geometry or inventing a cipher-specific
network?

uKNIT-BC is the selected mechanism anchor because it has sixteen 4-bit cells,
twelve round- and cell-specific S-box layers, and eleven distinct sparse
invertible 64-bit GF(2) linear layers. PRESENT, GIFT and SKINNY use one shared
4-bit S-box within a round and therefore cannot test cell-specific S-box
ownership.

## Verified Sources

Primary specification:

```text
Kai Hu, Mustafa Khairallah, Thomas Peyrin and Quan Quan Tan
uKNIT: Breaking Round-Alignment for Cipher Design
IACR Transactions on Symmetric Cryptology
paper license = CC BY 4.0
local PDF = papers/算法/uKNIT（轻量级算法设计）.pdf
```

External verification oracle:

```text
repository = https://github.com/syllab-ntu/UKNIT.git
commit     = f6493014fb7326cf3fffa2bb642b26cd59650e4f
license    = GPLv3
```

The project implementation is independently written from the published
specification. The GPLv3 code is not copied as implementation code; its four
published encryption traces and structural model are used as an external test
oracle. The S-box and linear tables are cryptographic parameters transcribed
from Appendix B of the CC BY 4.0 paper.

## Implemented Representation

Cipher implementation:

```text
src/blockcipher_nd/ciphers/spn/uknit.py
```

It implements:

- a 64-bit block and 128-bit key;
- the 13-round-key generalized STK schedule;
- twelve cell- and round-specific 4-bit S-box layers;
- eleven distinct sparse GF(2) linear layers;
- the full final `AddKey -> SBox -> AddKey` round;
- explicit input, key and round validation.

Reduced-round semantics are prefix semantics. For `rounds=1..11`, encryption
returns the state after the requested number of the real
`AddKey -> SBox -> Linear` transitions. Only `rounds=12` executes the real final
S-only layer and final key addition. No artificial final linear layer or
whitening key is introduced into a reduced prefix.

Runtime descriptor:

```text
configs/runtime/spn/uknit64.json
```

The descriptor intentionally contains eleven transition rounds:

```text
S0,L0 ... S10,L10
```

The twelfth S-box has no following diffusion layer and is therefore not
misrepresented as a topology transition. The generic loader now supports a
continuous window selected by:

```text
runtime_round_start
processor_steps
```

For example, `runtime_round_start=4, processor_steps=2` loads exactly S4/L4 and
S5/L5. A window outside rounds 0 through 10 fails closed. Existing one-round
PRESENT and SKINNY descriptors retain their `repeat_single_round=true`
behavior; a nonzero start is rejected because a repeated single layer has no
distinct round identity.

## Equal-Geometry Controls

The cipher-name-free training entries now include:

```text
runtime_spn_e4_equivariant_true
runtime_spn_e4_equivariant_corrupted
runtime_spn_e4_equivariant_sbox_shuffled
runtime_spn_e4_equivariant_independent
```

The new S-box control applies one deterministic non-identity cell permutation
to every loaded round's S-box descriptors while preserving:

- the complete S-box multiset in every round;
- the exact cell partition and bit roles;
- every linear matrix and inverse;
- model parameters and initialization geometry;
- input features, labels and training protocol.

This isolates S-box-to-cell ownership from linear-topology corruption and the
no-linear-topology control.

## Deterministic Evidence

The implementation passes all four official full-round vectors:

| Plaintext | 128-bit key | Ciphertext |
| --- | --- | --- |
| `0000000000000000` | `00000000000000000000000000000000` | `034af0b3c687e424` |
| `0123456789abcdef` | `0123456789abcdef0123456789abcdef` | `7d4ef882c1f42dba` |
| `ffffffffffffffff` | `ffffffffffffffffffffffffffffffff` | `db058583df8f186f` |
| `1111111111111111` | `fedcba98765432100123456789abcdef` | `7c8ddaf0fead3409` |

The zero-key vector also matches every official post-linear state from rounds
1 through 11, and the nonzero vector matches all thirteen published round
keys. Every one of the 192 S-box tables is a permutation. All eleven linear
matrices are exactly invertible over GF(2), and every tested descriptor window
matches the cipher-side structure tensor for tensor.

On the real uKNIT descriptors, moving the same per-round S-box multiset to
different cells is invisible to the global-mean `late_pair` path within
`1e-6`, while `late_cell` changes the fixed-weight logits by more than `1e-6`.
The correct, linear-corrupted, S-box-shuffled and independent adapters retain
identical state-dict geometry and complete forward passes.

## Neural Consumption Boundary

The descriptor and generic runtime structure preserve every round in the
selected window, but the current `RuntimeE4EquivariantSpnDistinguisher` does
not recurrently consume that full window. Its active feature path uses the
last loaded inverse linear matrix and the last loaded S-box descriptors. This
is distinct from the earlier `RuntimeParameterizedSpnDistinguisher`, which
walks backward through up to `processor_steps` loaded rounds.

A deterministic counterfactual held the final uKNIT S-box and linear layer
byte-identical while changing only the earlier linear layer in a two-round
window. With the same input tensor and fixed weights:

```text
last linear layer equal       = true
last S-box descriptors equal  = true
earlier linear layer changed  = true
Runtime-E4 max logit delta    = 0.0
round-wise Runtime max delta  = 0.005227193236351013
```

Therefore this reproduction proves full-window representation and validation,
but current Runtime-E4 neural evidence must be described as last-transition
conditioning. A future multi-round equivariant processor is a separate model
hypothesis; it must not be retroactively attributed to existing E4 results or
introduced into a frozen E4 protocol.

## Claim Boundary

This is a mechanism reproduction and readiness result. It proves that the
current 4-bit runtime SPN representation can load and validate a real
non-round-aligned structure, and that Runtime-E4 can condition on its final
loaded transition without changing parameter geometry. It does not prove that
Runtime-E4 consumes the complete multi-round window, nor does it provide a
trained distinguisher, AUC, cryptanalytic attack, cross-cipher transfer result,
paper-scale result or breakthrough. No result index entry is created because
no result-producing training run was executed.

## Recommended Next Action

Create a three-row local attribution plan before training:

```text
question          = does cell-preserving S-box conditioning exploit true uKNIT ownership?
candidate         = late_cell + correct S-box assignment
same-budget anchor= late_pair + correct assignment
required control  = late_cell + deterministic shuffled assignment
cipher/rounds     = uKNIT-BC prefix r4
runtime window    = round_start 2, processor_steps 2 (S2/L2 and S3/L3)
difference        = 0x0000000000000040 engineering calibration, not literature claim
pairs/sample      = 4
train scale       = 2048/class
seeds             = 0,1
epochs            = 10
negative mode     = encrypted_random_plaintexts
execution         = local sub-medium diagnostic
```

Before training, require identical dataset hashes, labels, parameter geometry
and initialization for matched rows, exact descriptor-window evidence and a
nonzero fixed-weight assignment-sensitivity check. Advance only if the correct
`late_cell` row beats both the global-mean anchor and shuffled-assignment
control on both seeds with a nontrivial margin. If it is near chance or does
not beat the shuffled control, redesign locally. Do not launch remote scale,
add DDT/trail features, change negative construction, or mix this attribution
with the active watcher-managed SKINNY RTG2-A run.

## Post-Recurrent-Window Implementation Update

The neural-consumption boundary above remains authoritative for every recorded
U1/U2 Runtime-E4 result: those checkpoints used the default
`last_transition` path and cannot be relabeled as multi-round evidence.

The later implementation readiness work adds an opt-in
`round_window_mode=recurrent_window` path. It walks every loaded uKNIT
transition with shared parameters and includes a same-length
`runtime_structure_window_control=repeat_last` control. Fixed-weight tests now
show that a full heterogeneous uKNIT window differs from repeated-final, while
PRESENT/GIFT/SKINNY repeated-round windows collapse exactly to that control.

This new mechanism directly addresses the earlier-layer invisibility proven
in this record, but it has no trained AUC yet. Any resumed uKNIT study must be
a separately preregistered recurrent-window attribution gate and must not
reuse the final-transition delta-query claim unchanged.
