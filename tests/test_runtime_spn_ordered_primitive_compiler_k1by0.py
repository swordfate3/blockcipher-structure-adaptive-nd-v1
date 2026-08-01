from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from blockcipher_nd.cli.audit_runtime_spn_ordered_primitive_compiler_k1by0 import (
    render_k1by0_svg,
)
from blockcipher_nd.models.structure.spn.ordered_primitive_program import (
    EXPERT_CONTRACT,
    GF2_EXPERT,
    PERMUTATION_EXPERT,
    compile_ordered_primitive_program,
    permute_program_target_bindings,
    program_exactly_replays,
    replay_ordered_primitive_program,
    rotate_program_stages,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_ordered_primitive_compiler_k1by0 import (
    adjudicate,
    audit_programs,
    load_and_validate_config,
    load_source_authority,
    load_structures,
)
from blockcipher_nd.tasks.innovation1.runtime_spn_structure_program_pretrain_k1bw import (
    structure_variants,
)


def test_k1by0_binds_k1bx0_hold_and_zero_training_protocol() -> None:
    config = load_and_validate_config()
    source_gate, checks = load_source_authority(config)

    assert source_gate["status"] == "hold"
    assert all(checks.values())
    assert config["audit"]["training_steps"] == 0
    assert config["audit"]["ciphertext_rows"] == 0
    assert config["audit"]["control_seeds"] == [11, 23, 37, 53]


def test_k1by0_exactly_replays_seven_structures_with_shared_contract() -> None:
    config = load_and_validate_config()
    structures = load_structures(config)[0]

    programs = {
        name: compile_ordered_primitive_program(structure)
        for name, structure in structures.items()
    }

    assert len(programs) == 7
    assert all(
        program_exactly_replays(program, structures[name])
        for name, program in programs.items()
    )
    assert all(
        replay_ordered_primitive_program(program).window_sha256()
        == structures[name].window_sha256()
        for name, program in programs.items()
    )
    assert all(
        contract["uses_cipher_identity"] is False
        for contract in EXPERT_CONTRACT.values()
    )
    assert programs["gift64"].expert_usage[PERMUTATION_EXPERT] > 0
    assert programs["skinny64"].expert_usage[GF2_EXPERT] > 0
    assert programs["dialga128"].expert_usage[GF2_EXPERT] > 0


def test_k1by0_joint_cell_relabel_has_same_semantics_and_exact_replay() -> None:
    config = load_and_validate_config()
    structures = load_structures(config)[0]
    for structure in structures.values():
        original = compile_ordered_primitive_program(structure)
        relabeled, semantic_ids, _variants = structure_variants(structure, seed=11)
        transported = compile_ordered_primitive_program(
            relabeled,
            semantic_cell_ids=semantic_ids,
        )

        assert transported.semantic_sha256 == original.semantic_sha256
        assert program_exactly_replays(transported, relabeled)


def test_k1by0_rejects_applicable_wrong_order_and_all_wrong_bindings() -> None:
    config = load_and_validate_config()
    structures = load_structures(config)[0]
    applicable = []
    for name, structure in structures.items():
        program = compile_ordered_primitive_program(structure)
        rotated = rotate_program_stages(program)
        is_applicable = len(set(program.stage_content_sha256s)) > 1
        applicable.append(name) if is_applicable else None
        assert program_exactly_replays(rotated, structure.rotate_transitions())
        assert (rotated.semantic_sha256 != program.semantic_sha256) is is_applicable

        for seed in config["audit"]["control_seeds"]:
            wrong_binding = permute_program_target_bindings(program, seed=seed)
            assert wrong_binding.semantic_sha256 != program.semantic_sha256
            assert not program_exactly_replays(wrong_binding, structure)

    assert set(applicable) == {"uknit64", "dialga128"}


def test_k1by0_gate_distinguishes_pass_hold_and_invalid() -> None:
    config = load_and_validate_config()
    structures = load_structures(config)[0]
    rows, programs = audit_programs(config, structures)

    passed = adjudicate(
        config,
        rows,
        programs,
        protocol_checks={"source": True},
    )
    assert passed["status"] == "pass"

    held_rows = deepcopy(rows)
    target = next(
        row for row in held_rows if row["control"] == "wrong_target_binding"
    )
    target["passed"] = False
    held = adjudicate(
        config,
        held_rows,
        programs,
        protocol_checks={"source": True},
    )
    assert held["status"] == "hold"
    assert "all_wrong_bindings_rejected" in held["failed_research_checks"]

    invalid = adjudicate(
        config,
        rows,
        programs,
        protocol_checks={"source": False},
    )
    assert invalid["status"] == "invalid"


def test_k1by0_plot_explains_compiler_scope_in_chinese(tmp_path: Path) -> None:
    config = load_and_validate_config()
    structures = load_structures(config)[0]
    rows, programs = audit_programs(config, structures)
    gate = adjudicate(
        config,
        rows,
        programs,
        protocol_checks={"source": True},
    )
    output = tmp_path / "curves.svg"

    report = render_k1by0_svg(gate, programs, rows, output)

    text = output.read_text(encoding="utf-8")
    assert report["panels"] == 4
    assert "编译成可执行的神经网络积木顺序" in text
    assert "每种密码自动选择了哪些共享专家" in text
    assert "不是差分区分AUC" in text
    assert "uKNIT和Dialga" in text
