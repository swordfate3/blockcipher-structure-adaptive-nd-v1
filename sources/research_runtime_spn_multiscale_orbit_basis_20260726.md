# Runtime-SPN Multiscale Orbit-Basis Source Record

Date: 2026-07-26

## Question

Is a fixed bank of exact multi-hop runtime linear-operator views a defensible
next representation hypothesis for Innovation 1 after the source-topology
diversity D1 hold?

## Lookup Record

The repository `research-lookup` primary executable was unavailable. The
configured Tavily fallback was then queried for higher-order graph filters, but
the provider returned HTTP 432 because its usage limit was exhausted; the Exa
fallback had no API key. No result from that failed aggregate query influenced
the route decision.

The following records were opened and verified on their official landing pages:

### MixHop

- Sami Abu-El-Haija, Bryan Perozzi, Amol Kapoor, Nazanin Alipourfard, Kristina
  Lerman, Hrayr Harutyunyan, Greg Ver Steeg, and Aram Galstyan.
- "MixHop: Higher-Order Graph Convolutional Architectures via Sparsified
  Neighborhood Mixing."
- Proceedings of the 36th International Conference on Machine Learning,
  PMLR 97:21-29, 2019.
- Official landing page:
  https://proceedings.mlr.press/v97/abu-el-haija19a.html
- arXiv record: https://arxiv.org/abs/1905.00067

The official abstract says that a single conventional graph convolution cannot
learn a general class of neighborhood-mixing relations. MixHop explicitly
mixes feature representations from several adjacency powers and can express
difference operators. This supports the general representation principle that
several exact operator distances can expose relations hidden by a single-hop
view. It does not provide cryptanalytic evidence.

### SIGN

- Fabrizio Frasca, Emanuele Rossi, Davide Eynard, Ben Chamberlain, Michael
  Bronstein, and Federico Monti.
- "SIGN: Scalable Inception Graph Neural Networks."
- arXiv:2004.11198, first submitted 2020, verified version v3.
- Official landing page: https://arxiv.org/abs/2004.11198

The official abstract describes a fixed bank of graph convolutional filters of
different sizes that can be precomputed, including different local graph
operators and diffusion matrices. This supports a fixed-shape operator-view
bank as a simple alternative to learned routing. It does not establish that
adjacency powers are useful for block-cipher distinguishers.

### Closest SPN Neural-Distinguisher Evidence

The existing verified repository lookup is:

```text
sources/research_spn_structure_neural_networks_20260723.md
```

Its closest work is Jiashuo Liu, Manman Li, Jiongjiong Ren, and Shaozhen Chen,
"A Highly Efficient Neural Distinguisher Framework for IoT-Friendly
Lightweight SPN Block Ciphers," IEICE Transactions on Information and Systems
E109.D(2), 238-248, 2026, DOI 10.1587/transinf.2025EDP7070. The paper uses
cipher-aware inverse round operations to expose a previous-round-like view,
then reshapes `(Cbar, Cbar', delta Cbar)` for a Conv2D residual network.

The local full-text extraction is:

```text
papers/innovation_one/grobid_md/
  a-highly-efficient-neural-distinguisher-framework-for-iot-friendly-lightweight-spn-block-ciphers.md
```

The paper supports inverse-layer feature engineering, but it adjusts inverse
operations and input layout per cipher. It does not pass arbitrary runtime
linear maps to one shared network and does not evaluate a fixed multiscale
operator orbit.

## Local Architecture Evidence

`RuntimeE4EquivariantSpnDistinguisher.encode` currently forms the output
difference and one exact inverse-linear difference before cell encoding. Its
recurrent-window mode processes transitions sequentially, but adds each
transition representation into one shared sequence instead of retaining an
explicit fixed bank of operator depths. This makes a multiscale orbit audit
non-duplicative at the representation interface.

## Evidence Boundary

MixHop and SIGN are architectural analogies, not cryptanalysis baselines. Liu
et al. is direct SPN neural-distinguisher evidence for an inverse-layer view,
not evidence for multi-hop runtime topology. The combination justifies only a
zero-training feasibility audit. It does not authorize a new neural model,
remote compute, a universal-SPN claim, or any performance expectation.
