# uKNIT cryptanalysis literature search (2026-07-29)

## Question

Whether an independent paper has improved the public attack coverage against uKNIT-BC,
and what the best published round counts currently are.

## Queries and sources checked

- IACR ePrint full-text metadata search: `uKNIT` and exact phrase `"uKNIT-BC"`.
  Both searches returned only ePrint 2024/1962.
  - https://eprint.iacr.org/search?q=uKNIT
  - https://eprint.iacr.org/search?q=%22uKNIT-BC%22
  - https://eprint.iacr.org/2024/1962
- DBLP publication search: `uKNIT`. It returned the ePrint record and the ToSC record
  for the same cryptographic work, plus one unrelated wearable-computing paper.
  - https://dblp.org/search/publ/api?q=uKNIT&h=50&format=json
  - https://dblp.org/rec/journals/iacr/HuKPT24
  - https://dblp.org/rec/journals/tosc/HuKPT26
- Crossref title/bibliographic searches: `uKNIT cryptanalysis` and `"uKNIT-BC"`.
  The only relevant cryptographic result was the original design paper. Crossref reported
  `is-referenced-by-count = 0` on 2026-07-29.
  - https://api.crossref.org/works?query.title=uKNIT%20cryptanalysis&rows=20
  - https://api.crossref.org/works/10.46586%2Ftosc.a0zo-4njsuvm
- Semantic Scholar DOI lookup reported `citationCount = 0` and an empty citations list on
  2026-07-29.
  - https://api.semanticscholar.org/graph/v1/paper/DOI:10.46586%2Ftosc.a0zo-4njsuvm
- ToSC formal publication page and DOI metadata were checked directly.
  - https://tosc.iacr.org/index.php/ToSC/article/view/13006
  - https://doi.org/10.46586/tosc.a0zo-4njsuvm
- The local formal-version PDF was inspected directly:
  `papers/算法/uKNIT（轻量级算法设计）.pdf`.

## Verified paper identity

Kai Hu, Mustafa Khairallah, Thomas Peyrin, and Quan Quan Tan,
"uKNIT: Breaking Round-Alignment for Cipher Design," IACR Transactions on Symmetric
Cryptology, 2026(2), published 2026-06-11, DOI
`10.46586/tosc.a0zo-4njsuvm`. The earlier public version is IACR ePrint 2024/1962.

uKNIT-BC has a 64-bit block, a 128-bit key, and 12 full rounds.

## Best attacks reported by the designers

Table 1 of the formal paper reports the following key-recovery attacks on reduced-round
uKNIT-BC:

| Attack | Covered rounds | Window | Time | Data |
|---|---:|---|---:|---:|
| Differential | 10/12 | `W(0,10)` | `2^110.6` operations | `2^55.6` chosen plaintexts |
| Impossible differential | 10/12 | `W(1,10)` | `2^98` operations | `2^63` chosen plaintexts |
| Demirci-Selcuk meet-in-the-middle | 9/12 | `W(0,9)` | `2^115` operations | `2^61` chosen plaintexts |
| Differential-linear | 9/12 | `W(1,9)` | `2^92.7` operations | `2^53.7` chosen plaintexts |

The same paper also reports:

- a 7-round impossible-differential distinguisher, extended to the 10-round key-recovery
  attack above;
- 7-round zero-correlation linear hulls;
- a longest detected 7-round integral/zero-sum property;
- no detected 8-round zero-sum property in any `W(i,8)` window and no detected 8-round
  one-sum property;
- no full 12-round attack satisfying the authors' security-claim limits.

The authors' security claim is specifically bounded: no full-round attack below `2^112`
time and at most `2^47` chosen or adaptively chosen plaintext-ciphertext pairs. Some
reduced-round attacks in Table 1 deliberately exceed the `2^47` data limit and are security
margin analyses, not violations of that bounded full-round claim.

## Search conclusion

As of 2026-07-29, no independent follow-up cryptanalysis paper was found that improves
the attack round coverage against uKNIT-BC. The best public results found are still those
in the designers' own paper: 10/12 rounds for key recovery and 7 rounds for the longest
reported integral/impossible-differential distinguishers. No public full 12-round break was
found.

This is a scoped literature-search conclusion, not proof that no unpublished, unindexed,
or paywalled result exists. The paper's formal publication is recent (2026-06-11), which
also limits the time available for independent follow-up work.

## Backend limitations

- OpenAlex could not be checked on this date because its daily API budget was exhausted.
- The available academic-search API credentials were unavailable, so the conclusion relies
  on the directly checked IACR, ToSC, DBLP, Crossref, Semantic Scholar, and local-paper
  evidence listed above.
