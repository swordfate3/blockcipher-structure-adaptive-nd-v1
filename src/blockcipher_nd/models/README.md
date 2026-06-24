# Model Package Layout

This package keeps model implementations in canonical subpackages. Do not add
single-file top-level forwarding modules; use `baseline/`, `common/`, and
`structure/` directly.

## Canonical Implementation Paths

- `baseline/`: cipher-agnostic and paper baseline networks.
  - Examples: `mlp.py`, `cnn.py`, `dbitnet.py`, `gohr_speck.py`, `resnet_bitslice.py`.
- `common/`: reusable neural-network components shared by multiple model families.
  - Examples: activation builders, normalization layers, attention pooling.
- `structure/`: structure-aware innovation-one models.
  - `structure/adaptive_dbitnet.py`: generic adaptive and structure-conditioned DBitNet blocks.
  - `structure/arx/`: ARX-specific entry points and future ARX-specialized modules.
  - `structure/spn/`: SPN-specific cell, nibble, and token-mixer pair-set models.
  - `structure/feistel/`: reserved for Feistel-specific models.
  - `structure/moe.py`: structure-aware expert fusion models.
- `registry.py`: stable `model_key -> class` lookup used by experiments.

## Imports

Use these canonical imports:

```python
from blockcipher_nd.models.baseline.gohr_speck import GohrSpeckDistinguisher
from blockcipher_nd.models.common.components import build_activation
from blockcipher_nd.models.structure.arx import ArxStructureAdaptivePairSetDBitNetDistinguisher
from blockcipher_nd.models.structure.spn import SpnTokenMixerPairSetDistinguisher
from blockcipher_nd.models.structure.moe import StructureAwareMoEDistinguisher
```

Use `blockcipher_nd.registry.model_factory.build_model()` for experiment construction.
