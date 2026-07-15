# Innovation 1 Feistel Paper-to-Code Map

**Date:** 2026-07-16

**Scope:** source-verified map for DES, SM4, SIMON, and SIMECK structure-aware
neural distinguishers. This is a mechanism/local-diagnostic implementation
record, not an exact or paper-scale reproduction claim.

## Selected Method Cell

The first balanced-Feistel implementation targets the ordinary-key Lu et al.
SIMON64/128 and SIMECK64/128 protocol:

```text
raw input = 8 ciphertext pairs x 128 bits = 1024 bits/sample
candidate = cipher-correct previous-round relation inside the network
control   = fixed left/right branch swap with identical parameters
anchor    = multiscale_dense_resnet on the identical raw 1024-bit input
negative  = independently sampled plaintexts encrypted under the row key
```

The paper is available locally as
`papers/innovation_one/pdf/2022_lu_related_key_neural_distinguishers_simon_simeck_eprint.pdf`.
Canonical journal metadata is DOI `10.1093/comjnl/bxac195`, online 2023 and
print 2024. Author code is source-verified at
`github.com/JIN-smile/Improved-Related-key-Differential-based-Neural-Distinguishers`,
audited commit `602c664e649a4e3e8e56dc1961efb67400f5c7fb`.

## Paper-to-Code Mapping

| Paper element | Source protocol | Project mapping | Status |
|---|---|---|---|
| SIMON64/128 primitive | 32-bit AND-RX Feistel halves | `ciphers/feistel/simon.py` | implemented; deterministic formula tests required |
| SIMECK64/128 primitive | 32-bit AND-RX Feistel halves | `ciphers/feistel/simeck.py` | implemented; deterministic formula tests required |
| Input difference | `(0x00000000,0x00000040)` | `registry/difference_profiles.py` | implemented and tested |
| Eight pairs/sample | `s=8` | `independent_pairs`, `pairs_per_sample=8` | supported |
| Key sampling | one independent key/sample | `key_rotation_interval=1` | supported |
| Positive label | all eight plaintext pairs use fixed difference | `independent_pairs` positive row | supported |
| Paper negative | second plaintext randomized for negative labels | strict encrypted random plaintext pair | supported with stricter explicit semantics |
| Raw ciphertext fields | `C0_L,C0_R,C1_L,C1_R` | raw `ciphertext_pair_bits` | supported |
| Derived fields | `delta L`, `delta R`, previous-right and pseudo-previous-right differences | deterministic tensor channels in balanced-Feistel model | implemented and formula-tested |
| Paper network | five-block SE-ResNet, 120 epochs | small PyTorch pair encoder and residual stack, 10 epochs | mechanism adaptation, not exact architecture |
| Train/test scale | `2e7/2e6` total samples | `4096/4096` total train/validation rows in local diagnostic; fresh repeats separate | local diagnostic only |
| Published SIMON anchors | r11/r12/r13/r14 accuracy `0.9181/0.7117/0.5722/0.5148` | use r12 only as strong-signal calibration target | external reference only |
| Published SIMECK anchors | r14-r18 accuracy `0.9142/0.7663/0.6356/0.5577/0.5202` | use r15 only as strong-signal calibration target | external reference only |

## Exact Relation Formulas

For each ciphertext pair `(C0, C1)`, split every 64-bit ciphertext as
`(L,R)` and append eight 32-bit word channels in the author-code order:

```text
delta_L
delta_R
C0_L
C0_R
C1_L
C1_R
delta_previous_R
delta_pseudo_previous2_R
```

The derived channels are:

```text
prev0 = C0_L xor f(C0_R)
prev1 = C1_L xor f(C1_R)
delta_previous_R = prev0 xor prev1

pseudo0 = C0_R xor f(prev0)
pseudo1 = C1_R xor f(prev1)
delta_pseudo_previous2_R = pseudo0 xor pseudo1
```

Cipher functions:

```text
SIMON: f(x) = rol8(x) & rol1(x) xor rol2(x)
SIMECK: f(x) = rol5(x) & x xor rol1(x)
```

## Related Literature Boundary

| Route | Verified source | Why not first |
|---|---|---|
| Wang/Wang two-difference RKND | DOI `10.7717/peerj-cs.2566`; Zenodo `11178441` | changes to related-key/two-difference data protocol |
| Hou et al. multiscale/multi-pair | DOI `10.1038/s41598-025-98251-1` | mainly 32-bit SIMON/SPECK and adds key/intermediate materials |
| Liu et al. RX-neural | arXiv `2511.06336` | RX threat/data definition and 32-bit variants |
| Mirzaali et al. polytopic PDND | DOI `10.1186/s42400-025-00472-4`; GitHub `NeuralDistinguisher/Multiple-PDND` | replaces pairs with plaintext/key polytopes |
| Yu/Wu/Zhang SM4 Conv-ResNet | local PDF `2023_yu_wu_zhang_sm4_conv_resnet_analysis.pdf` | closer baseline repair needed; current r5 diagnostics are near chance |

These routes remain valid follow-ups, but combining any of them with the first
round-relation model would change both data and architecture and destroy
attribution.

## Reproduction Boundary

The project implements and tests the round-relation mechanism under a controlled
local budget. It does not reproduce the paper's TensorFlow SE-ResNet, 120 epochs,
`2e7/2e6` sample scale, random-ciphertext convention, five repetitions, staged
training, or published accuracy. Every report must keep external paper accuracy
separate from project AUC.

## Implemented Route Verdict

After repairing balanced-class key indexing and regenerating all affected data,
the cipher-correct pair-pool relation model produced these fresh-test AUCs at
`8192/class`:

| Cipher/round | seed0 true/shuffled | seed1 true/shuffled | Verdict |
|---|---|---|---|
| SIMON r11 | `0.658869 / 0.500258` | `0.689276 / 0.506655` | retained easier-round relation signal |
| SIMECK r14 | `0.860501 / 0.502065` | `0.855689 / 0.510210` | retained easier-round relation signal |
| SIMON r12 | `0.505086 / 0.500343` | not run | target round not ready |
| SIMECK r15 | `0.511122 / 0.500361` | not run | target round not ready |

The first structure-adaptive selection rule is conditional: use the
cipher-correct previous-round relation with the shared pair-pool encoder for
the demonstrated easier-round cells, reject the pair-axis Lu layout at this
small budget, and do not promote either architecture to r12/r15 or remote scale
without an equal-compute low-to-high curriculum attribution result.
