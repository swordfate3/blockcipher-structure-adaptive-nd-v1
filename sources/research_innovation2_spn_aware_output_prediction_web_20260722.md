# Innovation 2 SPN-aware output-prediction web search log

Date: 2026-07-22

Backend: Tavily advanced search through `web-search-plus`.

Included domains: `eprint.iacr.org`, `arxiv.org`, `springer.com`,
`ieeexplore.ieee.org`, `dl.acm.org`, `semanticscholar.org`.

The first attempted `research-lookup` academic backend did not run because the
environment had neither `PARALLEL_API_KEY` nor `OPENROUTER_API_KEY`. The first
sandboxed Tavily attempt was blocked by network permissions; the approved
network retry succeeded. One two-query parallel approval timed out without a
usable result, so both queries were rerun once serially.

## Query 1

```text
structure-aware neural output prediction block cipher SPN PRESENT P-layer cipher topology
```

| Score | Title | URL | Task assessment |
|---:|---|---|---|
| 0.584 | PRESENT Full Round Emulation: Structural Flaws and Predictable Outputs | https://eprint.iacr.org/2025/1069.pdf | Relevant nearest neighbor; removes key addition and chains generic MLPs |
| 0.572 | Output Prediction Attacks on SPN Block Ciphers using Deep Learning | https://eprint.iacr.org/2021/401.pdf | Core output-prediction baseline; generic LSTM/CNN, black-box network |
| 0.481 | A Tutorial on the Implementation of Block Ciphers | https://eprint.iacr.org/2020/1545.pdf | SPN implementation background, not neural output prediction |
| 0.431 | A Deep Learning Approach for Active S-Box Prediction of Lightweight Generalized Feistel Block Ciphers | https://eprint.iacr.org/2021/066.pdf | Predicts active-S-box security quantity, not ciphertext output values |
| 0.389 | Recent Advances of Neural Attacks against Block Ciphers | https://caislab.kaist.ac.kr/publication/paper_files/2020/scis2020_SG.pdf | Survey/background; reports generic dense cipher emulation |
| 0.331 | Improved Related-Key Differential Neural Distinguishers for SPN Block Ciphers | https://eprint.iacr.org/2026/535 | Related-key real/random differential classification, not output prediction |
| 0.315 | Lightweight Block Cipher Security Evaluation based on Machine Learning and Deep Learning | https://eprint.iacr.org/2020/1235.pdf | Active-S-box/security evaluation and prior generic PRESENT analysis |
| 0.239 | Encryption with Autoregressive Language Models | https://arxiv.org/pdf/2305.10445 | Unrelated use of SPN terminology |

Relevant excerpts preserved from the returned results:

- Singh: the model excludes the key schedule, chains round models, and reports
  full-round prediction above a random exact-match baseline.
- Kimura: the attack predicts ciphertext/plaintext from fixed-key pairs in a
  black-box setting and evaluates toy and full-size SPN/Feistel ciphers.
- ePrint 2026/535: the task is a related-key differential neural distinguisher
  on PRESENT/SKINNY, explicitly not true ciphertext-output regression.

## Query 2

```text
PRESENT neural network P-layer output prediction ciphertext prediction structure aware
```

| Score | Title | URL | Task assessment |
|---:|---|---|---|
| 0.622 | PRESENT Full Round Emulation: Structural Flaws and Predictable Outputs | https://eprint.iacr.org/2025/1069.pdf | Same nearest neighbor; generic MLP, all-zero/no-key round function |
| 0.598 | Improved integral neural distinguisher model for lightweight cipher PRESENT | https://link.springer.com/article/10.1186/s42400-024-00258-0 | MBConv/DenseNet structured-vs-random multiset classification, not output values |
| 0.536 | Output Prediction Attacks on SPN Block Ciphers using Deep Learning | https://eprint.iacr.org/2021/401.pdf | Same core LSTM/CNN output-prediction baseline |
| 0.419 | Oblivious Neural Network Predictions via MiniONN | https://eprint.iacr.org/2017/452.pdf | Privacy-preserving inference; unrelated ciphertext meaning |
| 0.392 | Using Modular Arithmetic Optimized Neural Networks To Crack Affine Cryptographic Schemes Efficiently | https://arxiv.org/html/2507.14229v1 | Classical affine cipher/key recovery, not modern SPN output prediction |
| 0.346 | Towards Deep Encrypted Training | https://arxiv.org/html/2604.16834v1 | Homomorphic encrypted neural training, unrelated |

Relevant excerpts preserved from the returned results:

- Wu/Guo: the network contains convolution, MBConv-modified DenseNet, and a
  binary prediction header; its input is ciphertext multisets and its target is
  an integral distinguisher class, not `P -> E_K(P)` output bits.
- Singh: the result claims predictable full-round behavior, but the verified
  local full text states that key addition is removed and key-based modeling is
  future work.

## Query 3

```text
cipher topology neural network output prediction block cipher wrong topology shuffled topology ablation
```

| Score | Title | URL | Task assessment |
|---:|---|---|---|
| 0.432 | On the Effects of Neural Network-based Output Prediction Attacks on the Design of Symmetric-key Ciphers | https://eprint.iacr.org/2024/1310 | Verified core output-prediction paper; generic LSTM on SIMON variants |
| 0.384 | Output Prediction Attacks on SPN Block Ciphers using Deep Learning | https://eprint.iacr.org/2021/401.pdf | Verified core LSTM/CNN baseline |
| 0.356 | Output Prediction Attacks on SPN Block Ciphers using Deep Learning (Semantic Scholar) | https://www.semanticscholar.org/paper/Output-Prediction-Attacks-on-SPN-Block-Ciphers-Deep-Kimura-Emura/5a68329606fc946517e0c428f19bb8c4b4a3ce20 | Duplicate metadata record |
| 0.305 | Scalable Neural Cryptanalysis of Block Ciphers in Federated Attack Environments | https://www.mdpi.com/2227-7390/14/2/373 | Crossref-verified federated EE/PR with FC and BiLSTM; no PRESENT or explicit topology |
| 0.089 | RF-Informed Graph Neural Networks for Accurate and Data-Efficient Circuit Performance Prediction | https://arxiv.org/html/2508.16403v3 | Circuit topology, unrelated to cipher output prediction |
| 0.088 | Tensor-view Topological Graph Neural Network | https://arxiv.org/html/2401.12007v3 | Generic topology model, unrelated to cryptanalysis |

The returned answer sentence claimed that shuffled/wrong topologies can weaken
ciphers, but none of the returned cryptanalysis records actually supplied a
same-parameter wrong-topology neural-output ablation. That synthesized sentence
is therefore not evidence and is excluded from the novelty judgment.

## Search-level conclusion

Across these three phrase families, no returned record matched all of:

```text
fixed unknown secret key
true ciphertext selected-output prediction
PRESENT/SPN structure explicitly encoded inside the neural network
exact P-layer versus identity/wrong-P same-parameter attribution
architecture-matched label shuffle
```

This is a bounded search result, not proof of novelty. It supports retaining the
project claim as provisional and conditional on OPA3, while requiring title,
author, year, landing-page, and full-text verification for any newly surfaced
candidate before final thesis wording.

## Metadata verification of newly surfaced candidates

The MDPI landing page returned `Access Denied`, so it was not treated as a
verified full-text source. Crossref metadata for DOI `10.3390/math14020373`
verified:

```text
Title: Scalable Neural Cryptanalysis of Block Ciphers in Federated Attack Environments
Authors: Ongee Jeong, Seonghwan Park, Inkyu Moon
Journal/year: Mathematics 14(2), 373, 2026
Task: Encryption Emulation and Plaintext Recovery plus distributed scaling
Ciphers: DES, SDES, AES-128, SAES, SPECK32/64
Models: fully connected networks and BiLSTM
```

The Crossref abstract does not include PRESENT or an explicit S-box/P-layer
network. Its 2024 predecessor, DOI `10.3390/math12131936`, was also verified:

```text
Title: Comprehensive Neural Cryptanalysis on Block Ciphers Using Different Encryption Methods
Authors: Ongee Jeong, Ezat Ahmadzadeh, Inkyu Moon
Journal/year: Mathematics 12(13), 1936, 2024
Tasks: EE, PR, key recovery, and ciphertext classification
Ciphers: DES, SDES, AES, SAES, SPECK
Models: fully connected, RNN, and Transformer comparisons
```

These papers occupy generic multi-architecture output emulation/recovery, so
the project novelty cannot be phrased as merely using a non-LSTM model.

Crossref also surfaced `Neural Cryptanalysis of Lightweight Block Ciphers Using
Residual MLPs`, DOI `10.1109/CSR64739.2025.11130149`, by Charis
Eleftheriadis et al. (IEEE CSR 2025). A targeted Tavily check of the IEEE record
and ResearchGate abstract identifies its task as all-in-one differential
cryptanalysis of SIMON and SPECK, not true ciphertext-output prediction. It is
therefore an architecture neighbor but not a matching task.

## Full-text verification of the Jeong paper family

The open-access publisher PDFs were retrieved from the MDPI resource host and
verified by PDF metadata, title, authors, DOI, and full-text protocol:

```text
2024: https://mdpi-res.com/d_attachment/mathematics/mathematics-12-01936/
      article_deploy/mathematics-12-01936-v2.pdf
2026: https://mdpi-res.com/d_attachment/mathematics/mathematics-14-00373/
      article_deploy/mathematics-14-00373.pdf
```

The 2024 paper's block-array Encryption Emulation task is full-output value
prediction: plaintext bits are the input and the complete ciphertext bit array
is the target. Plaintext Recovery reverses that mapping. Its two relevant model
families are:

```text
FCNN    = 512 -> 1024 -> 512 -> block-size sigmoid output
BiLSTM  = three bidirectional LSTM layers, hidden size 256,
          followed by a block-size sigmoid output
```

For DES and SPECK, the BiLSTM receives the two block halves as a sequence of
length two; this is a generic data-layout prior, not an S-box/P-layer topology.
Block-array models use BCE, AdamW, learning rate `0.001`, batch `128`, and `300`
epochs. Full-size-cipher training grows from `2^16` to `2^22` pairs; the
architecture comparison uses `2^22` training pairs and `2^15` test pairs. The
metric is average per-bit threshold accuracy (`BAPavg`) over the full output.
For SPECK32/64 r3 EE, Table 3 reports `0.587` for FCNN and `0.883` for BiLSTM.

The 2024 data-generation section does not state the key-freezing rule precisely
enough to infer it in isolation. The 2026 follow-up explicitly defines both the
centralized and federated attacks as fixed-unknown-secret-key KPA sessions and
describes the 2024 work as its centralized predecessor. The 2026 protocol uses:

```text
train/test pairs = 2^20 / 2^15 total
rounds           = 1, 2, 3, 4
models           = FCNN and three-layer BiLSTM-256
loss/optimizer   = BCE / Adam
federated loop   = 10 global rounds x 10 local epochs
batch/lr         = 4096 / 0.0001
edge servers     = 2, 4, 8, 16, 32 with fixed total training rows
```

Neither paper evaluates PRESENT. Neither architecture explicitly implements a
cipher S-box, the PRESENT P-layer, exact-versus-wrong topology, a position-bound
selected-output head, or an architecture-matched label-shuffle attribution.
Federated learning changes where model updates are trained and aggregated; it
does not add a cipher-structure representation.

## Consequence for Innovation 2

The Jeong papers narrow the novelty statement. The project cannot claim novelty
for generic plaintext-to-ciphertext prediction, per-bit output accuracy, FCNN,
BiLSTM, or merely comparing several neural architectures. Their useful role is
as an external full-output and larger-data architecture family.

They do not displace the running OPD1 question. OPD1 tests a narrower mechanism
not found in these papers: frozen easy ciphertext positions under PRESENT, a
position-bound head, and same-parameter exact-P/no-P/wrong-P/label-shuffle
controls. Because Jeong 2024 uses up to `2^22` training pairs and 300 epochs, an
OPD1 failure at `2^17`/100 epochs may stop the current same-budget position-head
route but cannot establish a universal output-prediction or architecture-family
ceiling. No Jeong-derived run should be inserted before OPD1 completes.
