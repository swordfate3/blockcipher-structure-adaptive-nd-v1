# GIFT-64 Neural Distinguisher Comparison Lookup

Date: 2026-07-14

Question: How does the project's local GIFT-64 r6 E4-R2 result compare with
recent mainstream neural distinguishers for GIFT?

## Search Queries

- `"GIFT-64" "neural distinguisher" cipher`
- `"Neural Distinguishers on TinyJAMBU-128 and GIFT-64" 0.5754 17M 4M 6 round`
- `2025 2026 GIFT-64 neural differential distinguisher reduced rounds cryptography`
- `TRMSCPDKD GIFT-64 4-round 97% training samples validation protocol`

Backend: Web Search Plus with Tavily, cross-checked against repository-held
paper text and the 2024 neural differential cryptanalysis survey.

## Comparable Findings

1. The strongest credible like-for-like round-count reference found for
   GIFT-64 remains Sun et al. (ICONIP proceedings, 2022): an LSTM-based
   6-round GIFT-64 result with validation accuracy `0.5754`, approximately
   `17M` training samples, and `4M` validation samples. The 2024 survey reports
   this result without its critical-warning marker. The survey classifies its
   task as `3-2-CT-R`: three ciphertexts, two input differences, raw
   ciphertext input, and differential detection. The project's four-pair
   binary task is approximately `8-1-CT-R`, so it is not protocol-identical.
2. The same survey warns against using several nominally stronger GIFT claims
   as baselines. It rejects the evidentiary force of a full-round GIFT-64 claim
   with accuracy above 90%, and notes severe overfitting in a small GIFT-COFB
   study. It also notes that a claimed TweGIFT-128 7-round result had
   validation accuracy only `0.5002` despite higher training accuracy.
3. A 2025 Scientific Reports paper reports `97%` on 4-round GIFT-64 as a
   cross-cipher generalization check. This is a substantially easier round
   count and uses a richer dataset construction involving multiple-round
   ciphertext material and key-related fields, so it is not a direct r6
   benchmark.
4. Shen et al. (JISA 2024) concerns GIFT-128 rather than GIFT-64. It reports
   raw validation accuracy `0.5542` for 7-round GIFT-128 and application-level
   score-distribution aggregation up to `99.36%`. The latter is not a raw
   single-sample accuracy and is not directly comparable to the project's
   GIFT-64 AUC.

## Project Result Being Compared

```text
cipher/rounds       = GIFT-64 r6
target train rows   = 8192/class = 16384 total
target validation  = 4096/class = 8192 total
pairs/sample        = 4
negative mode       = encrypted_random_plaintexts
target seed         = 0
true-to-true AUC    = 0.569627493620
true-to-true accuracy = 0.541015625000
best/calibrated accuracy = 0.552978515625
```

The project matches the credible public GIFT-64 frontier in round count but
does not beat the published r6 accuracy, even when using its validation-set
calibrated threshold. Its distinctive evidence is
instead the much smaller training budget and the matched scratch,
source-shuffled, and target-shuffled attribution controls. Metrics and data
protocols remain different, so no SOTA claim is justified.

## Sources

- Bellini et al., `Survey: Six Years of Neural Differential Cryptanalysis`,
  IACR ePrint 2024/1300: https://eprint.iacr.org/2024/1300
- T. Sun, D. Shen, S. Long, Q. Deng, S. Wang, `Neural Distinguishers on
  TinyJAMBU-128 and GIFT-64`, DOI:
  https://doi.org/10.1007/978-981-99-1642-9_36
- `Improving deep learning-based neural distinguisher with multiple ciphertext
  pairs for speck and Simon`, Scientific Reports 15, 13696 (2025), DOI:
  https://doi.org/10.1038/s41598-025-98251-1
- D. Shen et al., `Neural differential distinguishers for GIFT-128 and ASCON`,
  Journal of Information Security and Applications 82 (2024) 103758, DOI:
  https://doi.org/10.1016/j.jisa.2024.103758
- Sun et al. supplementary code:
  https://github.com/ASC8384/Neural-Distinguishers
