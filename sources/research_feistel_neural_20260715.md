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

OpenAlex and arXiv metadata were checked on 2026-07-15. The search found the
published 2023 Zhang/Wang version and later SIMON/SIMECK or differential-linear
work, but no newer same-task DES multiple-pair neural architecture benchmark
that displaces Zhang/Wang as the first anchor.

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
