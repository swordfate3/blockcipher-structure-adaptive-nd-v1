from __future__ import annotations

import json
import subprocess
from pathlib import Path

from blockcipher_nd.cli.audit_innovation2_present_r5_strict_label_provider import (
    main as audit_main,
)
from blockcipher_nd.cli.plot_innovation2_present_r5_strict_label_provider import (
    main as plot_main,
)
from blockcipher_nd.tasks.innovation2.present_r5_strict_label_provider_coverage import (
    FROZEN_CLAASP_COMMIT,
    StrictLabelCoverageConfig,
    inspect_claasp_p1,
)


def _make_claasp_fixture(root: Path) -> None:
    present = root / "claasp/ciphers/block_ciphers/present_block_cipher.py"
    module = (
        root
        / "claasp/cipher_modules/models/milp/milp_models/Gurobi"
        / "monomial_prediction.py"
    )
    tests = (
        root
        / "tests/unit/cipher_modules/models/milp/milp_models/Gurobi"
        / "monomial_prediction_test.py"
    )
    present.parent.mkdir(parents=True)
    module.parent.mkdir(parents=True)
    tests.parent.mkdir(parents=True)
    present.write_text(
        "from claasp.name_mappings import INPUT_PLAINTEXT, INPUT_KEY\n"
        "class PresentBlockCipher:\n"
        "    def __init__(self, number_of_rounds=None):\n"
        "        cipher_inputs=[INPUT_PLAINTEXT, INPUT_KEY]\n",
        encoding="utf-8",
    )
    module.write_text(
        "from gurobipy import Model, GRB, Env\n"
        "class MilpMonomialPredictionModel:\n"
        "    '''This module can only be used if the user possesses a Gurobi license.'''\n"
        "    def find_superpoly_of_specific_output_bit(self):\n"
        "        poly = self.get_solutions()\n"
        "        return poly\n"
        "    def find_exact_degree_of_superpoly_of_specific_output_bit(self):\n"
        "        pass\n"
        "    def find_keycoeff_of_cube_monomial_of_specific_output_bit(self):\n"
        "        # Fix all other non-key input bits to 0\n"
        "        pass\n"
        "def _parse_cube_positions():\n"
        "    pass\n"
        "def check_correctness_of_keycoeff_of_cube_monomial_or_superpoly():\n"
        "    pass\n",
        encoding="utf-8",
    )
    tests.write_text(
        "import pytest\n"
        "@pytest.mark.skip(reason='Requires Gurobi license')\n"
        "def test_provider(): pass\n",
        encoding="utf-8",
    )


def _init_git_fixture(root: Path) -> str:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_audit_protocol_freezes_present80_r5_full_pool() -> None:
    config = StrictLabelCoverageConfig(run_id="e52")

    assert config.rounds == 5
    assert config.structure_count == 96
    assert config.witness_keys == 16
    assert config.offsets_per_structure == 8


def test_claasp_manifest_uses_full_superpoly_not_zero_offset_keycoeff(
    tmp_path: Path,
) -> None:
    root = tmp_path / "claasp"
    _make_claasp_fixture(root)
    result = inspect_claasp_p1(
        root,
        actual_commit=FROZEN_CLAASP_COMMIT,
        runtime={
            "sage_executable": "/usr/bin/sage",
            "sage_version": "test",
            "sage_modules": {"sage": True, "bitstring": True, "gurobipy": False},
            "claasp_present_importable": True,
            "claasp_present_import_error": None,
            "gurobi_license_status": "not_checked_package_missing",
            "relevant_docker_image_found": False,
        },
    )

    assert result["source_checks"][
        "full_superpoly_keeps_non_cube_public_variables_symbolic"
    ]
    assert result["source_checks"]["key_coefficient_is_zero_offset_only"]
    assert result["target_mapping"]["required_api"] == (
        "find_superpoly_of_specific_output_bit"
    )
    assert result["status"] == "execution_unavailable"


def test_smoke_cli_writes_separated_provider_artifacts(tmp_path: Path) -> None:
    claasp = tmp_path / "claasp"
    _make_claasp_fixture(claasp)
    _init_git_fixture(claasp)
    output = tmp_path / "output"

    exit_code = audit_main(
        [
            "--run-id",
            "i2_present_r5_strict_label_provider_smoke_test",
            "--output-root",
            str(output),
            "--claasp-root",
            str(claasp),
            "--mode",
            "smoke",
            "--structure-count",
            "4",
            "--witness-keys",
            "2",
            "--offsets-per-structure",
            "1",
        ]
    )

    assert exit_code == 0
    expected = {
        "provider_manifest.json",
        "structures.json",
        "masks.json",
        "labels.csv",
        "certificates.jsonl",
        "witnesses.jsonl",
        "results.jsonl",
        "gate.json",
        "summary.json",
        "metadata.json",
        "progress.jsonl",
    }
    assert expected.issubset(path.name for path in output.iterdir())
    assert not (output / "matched_contrast.csv").exists()
    gate = json.loads((output / "gate.json").read_text(encoding="utf-8"))
    assert gate["status"] == "hold"
    assert gate["decision"] == "innovation2_present_r5_strict_label_bank_not_ready"
    manifest = json.loads(
        (output / "provider_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["providers"][0]["status"] == "completed_insufficient"
    assert manifest["providers"][1]["status"] in {
        "protocol_mismatch",
        "execution_unavailable",
    }

    assert (
        plot_main(
            [
                "--summary",
                str(output / "summary.json"),
                "--output",
                str(output / "curves.svg"),
            ]
        )
        == 0
    )
    svg = (output / "curves.svg").read_text(encoding="utf-8")
    assert "创新2 E52" in svg
    assert "五轮" in svg
    assert "key-coefficient" in svg
