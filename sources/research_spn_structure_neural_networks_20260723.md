# SPN-structure neural-network literature lookup (2026-07-23)

## Question

Which published or preprint works build neural distinguishers that are structurally aware of SPN ciphers, and which (if any) pass the cipher topology to one shared network at runtime?

## Search queries

- `SPN-aware neural distinguisher topology permutation layer neural network cipher graph cryptanalysis`
- `"graph neural network" block cipher cryptanalysis SPN S-box permutation topology neural distinguisher`
- `"Improved Related-Key Differential Neural Distinguishers for SPN Block Ciphers" network architecture Conv2D ResNet feature enhancement`
- `"SPN" "neural network" "permutation layer" cipher-agnostic neural distinguisher architecture SKINNY MIDORI PRESENT GIFT`

The searches used Tavily through the repository's web-search fallback. Results were checked against local full text, IACR ePrint landing pages, Crossref, and publisher metadata. A targeted graph-neural-network query did not surface a block-cipher neural-distinguisher paper that represents an arbitrary SPN round topology as a runtime graph input.

## Verified closest works

1. Jiashuo Liu, Manman Li, Jiongjiong Ren, and Shaozhen Chen, "A Highly Efficient Neural Distinguisher Framework for IoT-Friendly Lightweight SPN Block Ciphers," IEICE Transactions on Information and Systems E109.D(2), 238-248, 2026. DOI: https://doi.org/10.1587/transinf.2025EDP7070
   - Uses cipher-aware inverse round operations to expose a previous-round-like representation.
   - Reshapes `(Cbar, Cbar', delta Cbar)` as a `3 x 4 x n/4` tensor and uses a Conv2D residual network.
   - Evaluated on SKINNY and MIDORI families.
   - The input dimensions and inverse operations are adapted for the target cipher; no external adjacency/linear-layer specification is supplied to a shared topology-conditioned processor.

2. Chuchu Ge and Qichun Wang, "Improved Related-Key Differential Neural Distinguishers for SPN Block Ciphers," IACR ePrint 2026/535, 2026. https://eprint.iacr.org/2026/535
   - Official abstract describes a unified related-key framework for difference selection, dataset construction, network architecture, training, and evaluation.
   - Uses invertible SPN components for final-round-state feature enhancement and reuses plaintext pairs across related keys.
   - Evaluated on SKINNY-64/64 and PRESENT-64/80.
   - This is a related-key protocol. Its reported PRESENT four-pair accuracies are not directly comparable with single-key or independent-key strict-negative experiments.
   - The official page does not claim that an arbitrary SPN topology is passed as a runtime graph/matrix input.

3. Rui-Tao Su, Jiong-Jiong Ren, and Shao-Zhen Chen, "Improved Framework of Related-key Differential Neural Distinguisher and Applications to the Standard Ciphers," IACR ePrint 2025/537, 2025. https://eprint.iacr.org/2025/537
   - Uses multi-ciphertext, multi-difference data, cipher-structure-guided differential filtering, and a Deep Residual Shrinkage Network.
   - Evaluated on DES and PRESENT, with SIMECK used for a differential-combination validation experiment.
   - It is a cross-structure construction framework, not one topology-parameterized SPN message-passing network.

4. Emanuele Bellini, David Gerault, Anna Hambitzer, and Matteo Rossi, "A Cipher-Agnostic Neural Training Pipeline with Automated Finding of Good Input Differences," IACR Transactions on Symmetric Cryptology 2023(3), 184-212. DOI: https://doi.org/10.46586/tosc.v2023.i3.184-212
   - Introduces DBitNet and automated input-difference search.
   - DBitNet deliberately avoids cipher-specific components and uses dilated convolutions to cover short- and long-range bit dependencies.
   - It is cipher-agnostic because it ignores the cipher graph, not because it consumes a cipher topology description.

5. Wanqing Wu and Mingyu Guo, "Improved integral neural distinguisher model for lightweight cipher PRESENT," Cybersecurity 7, article 65, 2024. DOI: https://doi.org/10.1186/s42400-024-00258-0
   - Builds PRESENT-specific `invP` and `invS` features and uses DenseNet with MBConv.
   - Strong example of encoding known SPN inverse operations in the data representation.
   - It is a PRESENT-specific integral-classification protocol, not a generic runtime-topology network.

6. Generic Partial Decryption as Feature Engineering for Neural Distinguishers, IACR ePrint 2025/1443. https://eprint.iacr.org/2025/1443
   - Automates partial-decryption-derived features and evaluates Simon, Simeck, and the SPN cipher Aradi.
   - Strong evidence that generic, structure-derived features can help, but the neural backbones remain Gohr-style/DBitNet rather than graph-conditioned SPN processors.

## Related but less structurally similar

- A. Jain, V. Kohli, and G. Mishra, "Deep Learning based Differential Distinguisher for Lightweight Cipher PRESENT," IACR ePrint 2020/846. https://eprint.iacr.org/2020/846
  - PRESENT-specific MLP experiments; the network does not model the P-layer topology explicitly.
- "Neural differential distinguishers for GIFT-128 and ASCON," Journal of Information Security and Applications, 2024, DOI: https://doi.org/10.1016/j.jisa.2024.103758
  - Tests GIFT-128/ASCON using score-distribution and MLP methods; using an SPN cipher is not itself an SPN-topology-aware architecture.
- B. Zahednejad and L. Lyu, "An improved integral distinguisher scheme based on neural networks," International Journal of Intelligent Systems 37, 7584-7613, 2022.
  - Integral-neural framework across several structures; structurally related at the task/data level, not a runtime SPN graph processor.

## Evidence boundary

As of this lookup, the literature clearly contains SPN-specific feature engineering, SPN-shaped Conv2D networks, cipher-agnostic neural pipelines, and cross-cipher construction frameworks. This search did not find an explicit implementation of one shared neural processor that accepts, at runtime, all of the following as data: per-round S-box/cell types, arbitrary bit permutations or general GF(2) linear maps, and round-varying topology. This is a bounded literature-search result, not a proof that no such work exists.
