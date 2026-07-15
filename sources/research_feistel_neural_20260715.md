# Feistel Innovation 1 Literature Check (2026-07-15)

## Decision

Use reduced-round DES as the first Feistel architecture-matching cell. DES is a
balanced Feistel network, is named explicitly in the opening proposal, already
has a tested implementation in this repository, and has a directly relevant
multiple-pair differential-neural baseline. SIMON/SIMECK and SM4 remain
follow-up generalization ciphers because their AND-RX and generalized/unbalanced
round functions introduce additional mechanism changes.

## Primary Anchor

Liu Zhang and Zilong Wang, *Improving Differential-Neural Distinguisher Model
For DES, Chaskey and PRESENT*, arXiv:2204.06341 (2022), published as *Improving
the Accuracy of Differential-Neural Distinguisher for DES, Chaskey, and
PRESENT*, IEICE Transactions on Information and Systems 106-D(7), 2023,
DOI `10.1587/transinf.2022EDL8094`.

Relevant DES protocol and results:

```text
internal plaintext difference = (0x40080000, 0x04000000)
rounds                        = 5, 6, 7
group size m                  = 2, 4, 8, 16 ciphertext pairs/sample
DES initial kernels           = (1, 4, 6)
train                         = 10^7 grouped rows in Case2 (10^7 * m raw pairs)
test                          = 10^6 grouped rows in Case2 (10^6 * m raw pairs)
negative                      = independently sampled plaintext then encrypted
key sampling                  = random
```

Table 2 Case2 accuracy:

| DES rounds | m=2 | m=4 | m=8 | m=16 |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 0.7224 | 0.8424 | 0.9490 | 0.9939 |
| 6 | 0.5728 | 0.6213 | 0.6842 | 0.7603 |
| 7 | - | - | 0.5050 | 0.5106 |

The 7-round row uses staged training and is not a suitable first attribution
gate. The new project experiment therefore starts at six rounds and does not
claim exact paper reproduction.

## Later Literature

- Bellini, Gerault, Hambitzer, and Rossi, *A Cipher-Agnostic Neural Training
  Pipeline with Automated Finding of Good Input Differences*, ToSC 2023(3),
  DOI `10.46586/tosc.v2023.i3.184-212`. DBitNet is the generic counterpoint:
  it supports Feistel ciphers such as SIMON/HIGHT/TEA, but its purpose is a
  cipher-agnostic pipeline rather than explicit Feistel branch semantics.
- Lu et al., *Improved (Related-Key) Differential-Based Neural Distinguishers
  for SIMON and SIMECK Block Ciphers*, The Computer Journal, 2023. It supports
  later AND-RX Feistel generalization, but changes feature engineering and may
  use related-key settings.
- *Enhanced related-key differential neural distinguishers for SIMON and
  SIMECK block ciphers*, PeerJ Computer Science, 2024,
  DOI `10.7717/peerj-cs.2566`. This is not the first DES cell because related-key
  sampling changes the benchmark.
- *Improving deep learning-based neural distinguisher with multiple ciphertext
  pairs for Speck and Simon*, Scientific Reports, 2025,
  DOI `10.1038/s41598-025-98251-1`. This is the strongest recent prompt to add
  SIMON after DES, not to conflate SIMON's AND-RX behavior with balanced DES.

## Conditional SM4 Generalization Audit

Yu, Wu, and Zhang, *Analysis of SM4 Algorithm Based on Convolutional Residual
Network* (2023), is the closest project-held neural anchor for the opening
proposal's named SM4 target. Its reported protocol is:

```text
input difference = (0x00000000, 0x00000000, 0x00000000, 0x00000001)
input row         = one 256-bit ciphertext pair
positive          = fixed-difference plaintext pair encrypted under one key
negative          = independent second random plaintext encrypted under that key
train             = 1,000,000 total rows
test              = 100,000 total rows
epochs            = 25
optimizer         = Adam, learning rate 1e-4
loss              = MSE
batch size        = 5000
```

The paper describes a 32-channel convolutional residual tower, dropout `0.5`,
and two 64-unit fully connected layers. Table 1 reports validation accuracy
`0.999` at SM4 rounds 3--5, then `0.504`, `0.502`, and `0.496` at rounds 6--8.
These are accuracy results under the paper protocol, not directly comparable
to project AUC values.

The repository already has a reduced-round SM4 implementation and the fixed
profile `sm4_yu2023_conv_resnet`. Its strict
`encrypted_random_plaintexts` negative class matches the semantic paper
negative more closely than random ciphertext. The existing
`multiscale_dense_resnet` is only a literature-family baseline; it is not an
exact port of the paper's unspecified figure-level convolution details.

SM4 must not reuse the DES two-branch mapping. Its round recurrence is:

```text
X[i+4] = X[i] xor T(X[i+1] xor X[i+2] xor X[i+3] xor rk[i])
```

so a structure-attribution candidate needs four ordered 32-bit word roles,
the three-word round-function input relation, the feedback word, and an
equal-capacity shuffled-word/bit control. If the DES-r5 strong-signal gate
passes, freeze a separate SM4-r5 readiness and attribution plan around this
recurrence plus the Yu/Wu/Zhang-family baseline. Do not launch SM4 merely from
a DES topology margin, and do not claim generalized-Feistel transfer until the
SM4 candidate beats its controls under a same-budget protocol.

OpenAlex and arXiv metadata were checked on 2026-07-15. The search found the
published 2023 Zhang/Wang version and later SIMON/SIMECK or differential-linear
work, but no newer same-task DES multiple-pair neural architecture benchmark
that displaces Zhang/Wang as the first anchor.

## Official DES Code Audit

The public Google Drive folder linked by the paper was audited directly on
2026-07-15. The DES folder is `10o3vCJ9zs6gqdJh10T4o1lz46uJ5-sPb` and exposes
the following primary files:

| File | Drive file id | SHA-256 |
| --- | --- | --- |
| `deep_net_des.py` | `1ddRjxLRPDmtSfeKB1zhhXIwzpxZWV4lZ` | `c23f808d0f2f814f6963afd64c945acf0cd29ef93ac832589796814d84b5526e` |
| `des.py` | `1O0I3DhuBqHpEAhYVKjlIGkPJ7K-va8Ol` | `4e610e8ab99cbdedf806c5b1678a5d12724018bbf47bd5f89ac859c4ba4f2b14` |
| `eval.py` | `1CEC2il5C7i8gZJrcqh_JB924G8cMkWGC` | `4698f21ca56f711efa65bd80d0cc8c38cd6556c19d128a21a103b88a9c53f586` |

The audit resolves the two protocol ambiguities relevant to this project:

1. `make_target_diff_samples` generates a fresh 64-bit random DES key for every
   basic pair. `make_dataset_with_group_size` then reshapes independent raw pairs
   into groups of `m`; the pairs in one grouped row do not share a key.
2. For a negative raw pair, the second plaintext is independently random and
   both plaintexts are encrypted under the same per-pair key. This agrees with
   `encrypted_random_plaintexts`; random ciphertext is not the paper protocol.

The exact official network data path is:

```text
flat 128*m bits
  -> reshape (m, 4, 32) as C0L, C0R, C1L, C1R
  -> permute to (m, 32, 4)
  -> Conv1D kernels (1, 4, 6), 32 filters each, along 32 bit positions
  -> five residual blocks with kernels (3, 5, 7, 9, 11)
  -> global average pooling across pair and bit-position axes
  -> one sigmoid output
```

TensorFlow treats the `m` axis as an extended batch dimension for `Conv1D`, so
each pair is encoded with shared weights before `GlobalAveragePooling2D`
aggregates the whole pair set. The official training function supports arbitrary
`group_size`, although the checked-in `__main__` example launches only `m=2`.

There is also a source/text discrepancy that must remain visible: the paper DES
section says the L2 penalty is `8e-4`, while the public training function calls
`make_resnet(..., reg_param=1e-5)`. A paper-text reproduction and an
official-code reproduction therefore require separate labels.

## Paper-To-Code Map And Remaining Gaps

| Paper mechanism | Current project implementation | Evidence level / gap |
| --- | --- | --- |
| DES internal difference `(0x40080000,0x04000000)` | external profile `0x0000801000004000` followed by repository DES IP | unit-tested equivalent mapping |
| independent random key per basic pair | `zhang_wang_case2_official_mcnd` | cache metadata and generator tests verify `per_pair_random` |
| encrypted random-plaintext negatives | common differential dataset generator | same semantic negative class |
| input `(m,32,4)` and kernels `(1,4,6)` | canonical DES pair tensor and one-dimensional Inception branches | mechanism-aligned |
| five residual blocks `(3,5,7,9,11)` | current R1 paper-family row uses only three blocks `(3,5,7)` | not exact; controlled adaptation |
| global average pooling plus sigmoid | current R1 uses per-pair mean/max, pair-set mean/max, and an MLP head | not exact; aggregation/head changed |
| `10^7` grouped train and `10^6` grouped test rows | R1 uses `2048/class` train and three `2048/class` fresh tests | local diagnostic only, not paper scale |
| Table 2 DES-r6 `m=16` accuracy `0.7603` | project reports AUC and accuracy separately | no direct metric subtraction until an exact-protocol accuracy reproduction exists |

If the current R1 matrix has no signal, the first redesign should repair this
baseline gap rather than increase sample count mechanically: port the official
five-block/global-average backbone, then compare a branch-interaction candidate
and an equal-capacity shuffled control on that same backbone and budget. This
tests whether the weak result came from the adapted aggregation/head before any
remote scale exception is considered.

## Project Adaptation Boundary

The repository's `Des` implementation includes fixed IP/FP permutations, while
the Zhang/Wang DES description omits them around the reduced round function.
The profile `des_zhang_wang2022_mcnd` therefore injects external difference
`0x0000801000004000`, which becomes the paper's internal
`0x4008000004000000` after IP. The candidate model removes the public final
permutation and restores canonical `L_r || R_r` branch order internally.

The experiment retains the project-wide strict negative definition,
`encrypted_random_plaintexts`, which agrees with the paper's semantic negative
class. Report AUC and accuracy separately; do not compare project AUC directly
with Table 2 accuracy.
