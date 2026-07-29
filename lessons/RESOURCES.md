# Block Cipher Innovation Resources

## Knowledge

- [Six-SPN implementation lesson](0004-runtime-spn-cipher-architectures.html)
  Visual map of the project-authoritative PRESENT-80, GIFT-64, SKINNY-64/64, RECTANGLE-80, uKNIT-BC, and Dialga-128 encryption flows. It links each diagram back to the exact implementation.
- [SPN architecture cheat sheet](reference/spn-cipher-architecture-cheatsheet.html)
  Compressed operation order, state layout, and runtime-structure fields for the six Innovation 1 ciphers.
- [Project SPN implementations](../src/blockcipher_nd/ciphers/spn/)
  Primary authority for the encryption semantics used by experiments. Prefer these files over remembered textbook diagrams when reporting the current project.
- [uKNIT specification](../papers/%E7%AE%97%E6%B3%95/uKNIT%EF%BC%88%E8%BD%BB%E9%87%8F%E7%BA%A7%E7%AE%97%E6%B3%95%E8%AE%BE%E8%AE%A1%EF%BC%89.pdf)
  Published source for the 12-round non-round-aligned S-box and linear-layer schedule implemented by the project.
- [uKNIT principle lesson](0005-uknit-principle.html)
  Data-first explanation of the uKNIT design framework, uKNIT-BC encryption,
  key schedule, and the structure required by the current specialist network.
- [Official uKNIT implementation](https://github.com/syllab-ntu/UKNIT)
  Authors' reference repository linked from the specification. Use as an
  external oracle; the project implementation independently transcribes the
  published tables.

- [Paper: _On Extending Integral Distinguishers_](../papers/innovation_two/pdf/On%20Extending%20Integral%20Distinguishers.pdf)
  Primary source for Split-and-Cancel, the left-kernel criterion, pullback theorems, and PRESENT/GIFT results. Use for all claims about the paper.
- [Public implementation: `hadipourh/splitandcancel`](https://github.com/hadipourh/splitandcancel)
  Authors' code referenced by the paper. Use when moving from the conceptual matrix example to reproducibility or implementation auditing.
- [Current Innovation 2 authority](../docs/research/innovation2-output-prediction-thesis-boundary-20260721.md)
  Current fixed-unknown-key true-output-prediction contract, OP9--OPA3 evidence, claim boundary, and stopped OPA4/OPA5 route.
- [Historical integral-property blueprint](../docs/research/innovation2-structure-conditioned-integral-output-prediction-20260715.md)
  Historical structure-conditioned empirical integral-property task. Use only to keep that lesson's ranking problem separate from exact certification and current output prediction.
- [Innovation 2 experiment record](../docs/experiments/innovation2-present-r5-structure-conditioned-integral-parity-feasibility-plan.md)
  Frozen data protocol and E0-E3 evidence. Use when comparing the paper's exact output-combination search with the project's cross-key ranking task.
- [Historical PG-NBPO thesis consolidation](../docs/research/innovation2-thesis-consolidation-20260719.md)
  Frozen strict-label protocol, PG-NBPO architecture, dual-cipher results, and negative evidence for the superseded integral-balance branch.
- [E98 PRESENT r9 PU-ranking readiness](../docs/experiments/innovation2-present-r9-generalized-relation-pu-ranking-readiness-plan.md)
  Latest high-round data-readiness audit. Use to distinguish the confirmed four-round method from the held nine-round ranking route.

## Wisdom (Communities)

- [IACR Cryptology ePrint Archive](https://eprint.iacr.org/)
  Primary venue for checking the newest integral-cryptanalysis preprints and their revisions before making novelty claims.
- [Cryptography Stack Exchange](https://crypto.stackexchange.com/)
  Moderated technical community useful for testing precise questions about division property, monomial prediction, and integral terminology.

## Gaps

- The closest 2026 neural-guided and empirical-kernel papers still need a side-by-side protocol audit before a final Innovation 2 novelty claim.
- No published source reviewed so far specifies the project's exact uKNIT
  five-stage position-histogram neural residual; its evidence is project-local
  and must be reported separately from the cipher designers' claims.
