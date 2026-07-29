# Mission: Understand And Defend Structure-Aware Neural Cryptanalysis

## Why

Build enough concrete understanding of block-cipher structure, neural
distinguishers and symbolic integral cryptanalysis to judge both thesis
innovations correctly, design defensible experiments, and explain why a model
uses genuine cipher semantics rather than shortcuts.

## Success looks like

- Hand-calculate a masked output bit and its integral XOR over a small input set.
- Explain how Split-and-Cancel turns unresolved ANF terms into a binary matrix.
- Distinguish empirical cross-key balance rate, exact zero-sum, and weak-key zero-sum.
- Decide which part of Innovation 2 is candidate ranking and which part needs exact certification.
- Trace a concrete state through an SPN round and explain the roles of the S-box,
  linear layer and round key.
- Explain why uKNIT breaks round alignment and why its neural representation must
  retain round, S-box, native-cell and GF(2) operator semantics.

## Constraints

- Start from small numerical tables before introducing ANF, kernels, or SAT terminology.
- Tie every lesson to the active Innovation 1 or Innovation 2 thesis route.
- Do not treat finite-key observations as proof for all keys.

## Out of scope

- A full tutorial on SAT/CP solver implementation.
- Reproducing every cipher result in the paper during the first lesson.
