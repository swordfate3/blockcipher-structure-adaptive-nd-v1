# Innovation 2 Output-Prediction Literature Index

**Date:** 2026-07-15

**Scope:** source-verified initial corpus for neural-network output-prediction
attacks on block ciphers. This is an acquisition and route-definition record,
not yet a full systematic review.

The expanded corpus for the current high-round integral-neural target, including
PRESENT-specific papers and deterministic search baselines, is maintained in
`docs/research/innovation2-high-round-literature-corpus-20260716.md`. This file
remains the narrower output-prediction index and should not be read as the full
Innovation 2 literature set.

## Innovation Boundary

The opening proposal defines Innovation 2 as a separate task from neural
distinguishers:

```text
input/target = plaintext or state at round r -> ciphertext or state at round r+1
metric       = bit accuracy, whole-state success, convergence, critical round
question     = when does diffusion make the requested output no more predictable
               than a frozen random/majority/deterministic baseline?
```

The proposed project extension, `state_r -> state_(r+1)`, is not already
established by the papers below. Most prior work predicts final ciphertext or
plaintext from black-box input/output pairs. Singh 2025 uses round-wise
emulation, but removes key addition; it therefore models a public deterministic
SPN transform rather than a keyed encryption attack. This difference must stay
explicit when Innovation 2 is designed.

## Retrieved Core Papers

The following four PDFs and their extracted text are available locally under
`papers/innovation_two/`. File type, page count, title text, and SHA-256 were
checked after retrieval.

| Priority | Paper | Why it matters | Local PDF |
| ---: | --- | --- | --- |
| 1 | Kimura et al., *Output Prediction Attacks on Block Ciphers Using Deep Learning*, ACNS Workshops 2022, DOI `10.1007/978-3-031-16815-4_15`; correct ePrint `2021/401` | Foundational black-box ciphertext-prediction/plaintext-recovery protocol; covers toy SPN and Feistel ciphers and larger-state extensions | `papers/innovation_two/pdf/2021_kimura_output_prediction_block_ciphers.pdf` |
| 2 | Kimura et al., *A Deeper Look into Deep Learning-based Output Prediction Attacks Using Weak SPN Block Ciphers*, JIP 2023, DOI `10.2197/ipsjjip.31.550` | Connects prediction success to weak S-boxes and classical differential/linear resistance; important for deciding whether a critical-round curve measures a meaningful cipher property | `papers/innovation_two/pdf/2023_kimura_deeper_output_prediction_weak_spn.pdf` |
| 3 | Watanabe, Ito, and Ohigashi, *On the Effects of Neural Network-based Output Prediction Attacks on the Design of Symmetric-key Ciphers*, ePrint `2024/1310`, CSCML 2024 DOI `10.1007/978-3-031-76934-4_13`; expanded journal DOI `10.1016/j.jisa.2025.104016` | Extends the method from S-box ciphers to SIMON-like AND-RX structures and relates prediction success to diffusion and biased linear events | `papers/innovation_two/pdf/2024_watanabe_output_prediction_symmetric_ciphers.pdf` |
| 4 | Singh, *PRESENT Full Round Emulation: Structural Flaws and Predictable Outputs*, ePrint `2025/1069` | Closest published artifact to round-wise chained modeling, but removes the key schedule/AddRoundKey path; useful as a baseline design and as a warning about threat-model overclaiming | `papers/innovation_two/pdf/2025_singh_present_full_round_emulation.pdf` |

Verified files:

| File | Pages | SHA-256 |
| --- | ---: | --- |
| `2021_kimura_output_prediction_block_ciphers.pdf` | 33 | `b337605d20f7d43ebc628277050183377a4066bd028a26aece1ba9edd3ece299` |
| `2023_kimura_deeper_output_prediction_weak_spn.pdf` | 12 | `c7296c61c7b0c78f49a06737c66086c6b7a1c7c7734e5fcf16a9faabc57637c4` |
| `2024_watanabe_output_prediction_symmetric_ciphers.pdf` | 30 | `92f7aa9174215a84befbe58b35e30806249c1393088f073b0da6dc5d1ffaf055` |
| `2025_singh_present_full_round_emulation.pdf` | 14 | `5c763a7812946150859cbfc13f20a93ebb73ef8d987de8e09e649fa28b9249be` |

The current Kimura PDF was reused from the repository's previously verified
copy because IACR's PDF endpoint returned a Cloudflare challenge during this
acquisition. The Watanabe and Singh PDFs were retrieved from archived copies of
their canonical IACR URLs after the same failure. The canonical landing pages,
titles, authors, revision dates, and PDF hashes were checked separately.

## Related Background

These sources are relevant to the boundary of Innovation 2, but are not the
primary round-output-prediction protocol:

| Paper | Relation | Acquisition status |
| --- | --- | --- |
| Hu and Zhao, *Research on Plaintext Restoration of AES Based on Neural Network*, 2018, DOI `10.1155/2018/6868506` | Precursor plaintext-recovery task; useful for attack-goal and metric comparison | Metadata and official open-access URL verified; publisher PDF endpoint returned HTML in this environment |
| So, *Deep Learning-Based Cryptanalysis of Lightweight Block Ciphers*, 2020, DOI `10.1155/2020/3701067` | Recovers keys from known plaintext/ciphertext pairs under a restricted 64-character keyspace; a necessary threat-model counterexample | Metadata and official open-access URL verified; publisher PDF endpoint returned HTML in this environment |
| Idris et al., *A Deep Learning Approach for Active S-Box Prediction of Lightweight Generalized Feistel Block Ciphers*, 2021, DOI `10.1109/ACCESS.2021.3099802` | Predicts a security-related structural quantity rather than ciphertext bits; useful adjacent evidence for dynamic security assessment | Metadata verified; IEEE PDF endpoint returned HTML in this environment |
| Mishra, Krishna Murthy, and Pal, *Neural Network Based Analysis of Lightweight Block Cipher PRESENT*, 2018, DOI `10.1007/978-981-13-0761-4_91` | Historical PRESENT neural-analysis precursor; should not be substituted for an output-prediction protocol without full-text audit | Metadata verified; no open full text retrieved |
| Ge and Hu, *Neural Networks and Cryptography: An Overview*, 2021, DOI `10.13868/j.cnki.jcr.000432` | Chinese background review cited by the opening proposal | Metadata and journal landing URL verified; journal endpoint returned `403` |

## Opening-Proposal Citation Corrections

Three copied identifiers in the opening proposal do not resolve to the cited
papers and must not be reused:

1. The Kimura output-prediction paper is not IACR ePrint `2022/1724`.
   That identifier currently resolves to *Formal Analysis of SPDM: Security
   Protocol and Data Model version 1.2*. The verified output-prediction ePrint
   is `2021/401`, and the ACNS Workshops DOI is
   `10.1007/978-3-031-16815-4_15`.
2. The Watanabe CSCML chapter DOI ends in `_13`, not `_18`. DOI
   `10.1007/978-3-031-76934-4_18` resolves to an unrelated homomorphic
   encryption chapter.
3. IACR ePrint `2019/783` is *Dissecting the CHES 2018 AES Challenge*, not
   *Neural Network Based Full Round PRESENT Cipher Attack*. The latter opening-
   proposal entry remains unverified and must not be cited as an IACR paper.

## Initial Research Implication

The literature supports an output-prediction benchmark, but not an automatic
security claim. Before implementation, the experiment plan must freeze:

```text
attacker knowledge     = algorithm known/unknown; key known/unknown
key protocol           = fixed key, rotating keys, or train/test unseen keys
input                   = plaintext, ciphertext, state_r, or partial state
target                  = full state, selected bits, next round, or final output
deterministic baseline  = exact public round function when inputs make it computable
statistical baseline    = per-bit majority and calibrated random prediction
evaluation              = bit accuracy plus exact-state success and confidence intervals
critical-round rule     = predeclared return-to-baseline criterion across seeds/keys
```

The next research action is to read and protocol-audit the four core papers,
then write a lean Innovation 2 baseline plan. Training should not start until
the plan proves that the target is not trivially reconstructible from the
provided state and public round function.
