# E5 Literature Search: Neural-Cryptanalysis Transfer

```text
date     = 2026-07-15
provider = Tavily via web-search-plus 2.8.6
depth    = advanced
query    = neural cryptanalysis transfer learning pretrained distinguisher
           cross cipher block cipher one epoch adaptation
```

## Results

1. **Breaking Indistinguishability with Transfer Learning: A First Look at
   SPECK32/64 Lightweight Block Ciphers**
   - URL: https://arxiv.org/html/2405.19683v1
   - Tavily score: `0.664`
   - Search excerpt: uses a pretrained deep model as a feature extractor and
     trains a shallow XGBoost model; reports more than 94% lower data needs.
     The protocol is not automatically comparable to strict differential
     neural-distinguishing transfer.
2. **A Deeper Look at Machine Learning-Based Cryptanalysis**
   - URL: https://eprint.iacr.org/2021/287
   - Tavily score: `0.495`
   - Search excerpt: analyzes what the SPECK neural distinguisher learns and
     connects it to ciphertext and earlier-round differential distributions.
3. **Comprehensive Neural Cryptanalysis on Block Ciphers**
   - URL: https://www.mdpi.com/2227-7390/12/13/1936
   - Tavily score: `0.482`
   - Search excerpt: broad neural-cryptanalysis comparison across model types;
     not a controlled cross-cipher differential-transfer study.
4. **Quantum Neural Network based Distinguisher for Differential
   Cryptanalysis on Simplified Block Ciphers**
   - URL: https://eprint.iacr.org/2022/1671
   - Tavily score: `0.477`
   - Search excerpt: quantum/classical distinguishers for simplified DES,
     AES, and PRESENT variants; not checkpoint transfer.
5. **Using AI for Block Cipher Cryptanalysis**
   - URL: https://www.cryptool.org/media/post-contents/20-years-cryptool/CT20years_DeepLearningSpeck.pdf
   - Tavily score: `0.474`
   - Search excerpt: overview of SPECK neural-distinguisher training and key
     recovery; emphasizes millions of generated samples.
6. **Comprehensive Neural Cryptanalysis on Block Ciphers (preprint)**
   - URL: https://www.preprints.org/manuscript/202405.2022
   - Tavily score: `0.465`
   - Search excerpt: preprint version of the broad architecture comparison.
7. **Scalable Neural Cryptanalysis of Block Ciphers in Federated Settings**
   - URL: https://www.mdpi.com/2227-7390/14/2/373
   - Tavily score: `0.457`
   - Search excerpt: federated block-cipher neural analysis; not a strict
     source-to-target checkpoint experiment.
8. **Recent Advances of Neural Attacks against Block Ciphers**
   - URL: https://caislab.kaist.ac.kr/publication/paper_files/2020/scis2020_SG.pdf
   - Tavily score: `0.453`
   - Search excerpt: survey-style overview of neural attacks against block
     ciphers.

## E5 Use

The search found prior neural-cryptanalysis transfer and frozen-feature work,
so generic pretraining or freezing is not the Innovation 1 novelty. It did not
find a controlled cipher-spec-generated true-versus-shuffled SPN topology
objective followed by cross-SPN one-epoch adaptation.
