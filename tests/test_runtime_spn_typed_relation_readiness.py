from __future__ import annotations

from pathlib import Path

from blockcipher_nd.tasks.innovation1.runtime_spn_typed_relation_readiness import (
    _control_probe,
    _forbidden_identity_probe,
    _relation_probe,
    _role_geometry,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_primitive_adapter_readiness import (
    _load_structures,
)


ROOT = Path(__file__).resolve().parents[1]


def test_typed_relation_design_probes_pass_for_frozen_five_cipher_panel() -> None:
    structures = _load_structures()

    assert _relation_probe(structures)["passed"] is True
    assert _role_geometry()["passed"] is True
    control = _control_probe(structures["uknit64"])
    assert control["support_preserved"] is True
    assert control["logits_distinct"] is True
    assert _forbidden_identity_probe()["passed"] is True


def test_typed_relation_readiness_script_exists() -> None:
    assert (ROOT / "scripts/check-runtime-spn-typed-relation-readiness").is_file()
