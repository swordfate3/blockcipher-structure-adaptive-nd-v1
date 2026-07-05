# Research Lookup: Innovation 1 SPN Adaptation

**Date:** 2026-07-05
**Purpose:** Re-check whether the next Innovation 1 SPN/PRESENT improvement
route should prioritize diverse neural-network aggregation or another
SPN-adaptive mechanism.

## Web-Checked Sources

1. Liu Zhang et al., **Neural-Inspired Advances in Integral Cryptanalysis**,
   arXiv:2505.10790, 2025.
   URL: <https://arxiv.org/abs/2505.10790>

   Project relevance: supports treating neural models as feature/property
   explorers for integral structure. This points toward SPN-aware data and
   feature construction, especially integral/active-cell representations,
   before wider model aggregation.

2. Liu Zhang and Zilong Wang, **Improving Differential-Neural Distinguisher
   Model For DES, Chaskey, and PRESENT**, arXiv:2204.06341, 2022.
   URL: <https://arxiv.org/abs/2204.06341>

   Project relevance: PRESENT improvement is tied to multiple-ciphertext-pair
   derived features and network/input-format changes. This supports making
   input representation a first-class mechanism.

3. Tatsuya Sakagami et al., **Do LLMs Make Neural Distinguishers Wise?**,
   arXiv:2606.10692, 2026.
   URL: <https://arxiv.org/abs/2606.10692>

   Project relevance: a very different model class did not automatically
   improve neural distinguishers over established neural baselines, while
   representation/prompt design mattered. This is negative pressure against
   "just combine or enlarge models" as the default next step.

4. Local paper note: **Improved integral neural distinguisher model for
   lightweight cipher PRESENT**.
   Local path:
   `papers/innovation_one/grobid_md/improved-integral-neural-distinguisher-model-for-lightweight-cipher-present.md`

   Project relevance: supports testing integral/multiset and previous-round
   inverse-permutation/inverse-S representations for PRESENT. This source was
   not newly verified by web in this turn; it is already present in local
   project notes.

5. Local paper note: **Generic Partial Decryption as Feature Engineering for
   Neural Distinguishers**.
   Local path:
   `papers/innovation_one/grobid_md/generic-partial-decryption-as-feature-engineering-for-neural-distinguishers.md`

   Project relevance: supports treating partial inverse/previous-round
   features as feature-engineering routes that must be tested with controls.
   This source was not newly verified by web in this turn; it is already
   present in local project notes.

## Route Judgment

The current route priority should be:

```text
SPN feature/input search > structure-aware architecture > diverse ensemble
```

Diverse neural aggregation remains useful only as a secondary validator after
there are compatible weak-positive experts from genuinely different families.
The near-neighbor r7 neural ensemble showed mild complementarity but did not
justify using the next remote slot on a wider ensemble by itself.
