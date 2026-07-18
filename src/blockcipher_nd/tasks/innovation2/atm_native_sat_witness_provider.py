from __future__ import annotations

import importlib
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from blockcipher_nd.tasks.innovation2.deterministic_provider_contract import ATM_COMMIT


@dataclass(frozen=True)
class NativeSatProviderConfig:
    run_id: str
    mode: str = "audit"
    expected_commit: str = ATM_COMMIT
    projected_key_cap: int = 1 << 16
    trail_model_cap: int = 1 << 20

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if self.mode not in {"smoke", "audit"}:
            raise ValueError("mode must be smoke or audit")
        if len(self.expected_commit) != 40:
            raise ValueError("expected_commit must be a full hash")
        if min(self.projected_key_cap, self.trail_model_cap) <= 0:
            raise ValueError("SAT caps must be positive")


def algebraic_transition_coefficient(
    lookup_table: Sequence[int],
    *,
    input_bits: int,
    output_exponent: int,
    input_exponent: int,
) -> int:
    if len(lookup_table) != 1 << input_bits:
        raise ValueError("lookup table size must match input_bits")
    output_bits = max(1, max(lookup_table, default=0).bit_length())
    if not 0 <= input_exponent < 1 << input_bits:
        raise ValueError("input exponent is out of range")
    if not 0 <= output_exponent < 1 << output_bits:
        raise ValueError("output exponent is out of range")
    coefficients = [
        int((value & output_exponent) == output_exponent) for value in lookup_table
    ]
    for bit in range(input_bits):
        for mask in range(1 << input_bits):
            if mask & (1 << bit):
                coefficients[mask] ^= coefficients[mask ^ (1 << bit)]
    return coefficients[input_exponent]


def key_mask_from_projected_literals(
    projected_literals: Sequence[int], key_vars: Sequence[int]
) -> int:
    if len(projected_literals) != len(key_vars):
        raise ValueError("projected literals must match key vars")
    key_mask = 0
    for index, (literal, variable) in enumerate(zip(projected_literals, key_vars)):
        if abs(int(literal)) != int(variable):
            raise ValueError("projected literal order does not match key vars")
        if literal > 0:
            key_mask |= 1 << index
    return key_mask


def exponent_assumptions(
    input_vars: Sequence[int],
    output_vars: Sequence[int],
    input_exponent: int,
    output_exponent: int,
) -> tuple[int, ...]:
    if not 0 <= input_exponent < 1 << len(input_vars):
        raise ValueError("input exponent is out of range")
    if not 0 <= output_exponent < 1 << len(output_vars):
        raise ValueError("output exponent is out of range")
    return tuple(
        variable if input_exponent & (1 << index) else -variable
        for index, variable in enumerate(input_vars)
    ) + tuple(
        variable if output_exponent & (1 << index) else -variable
        for index, variable in enumerate(output_vars)
    )


def count_models_parity(
    model: Sequence[Sequence[int]],
    assumptions: Sequence[int],
    *,
    model_cap: int,
    solver_factory: Callable[..., Any],
) -> dict[str, Any]:
    if model_cap < 1:
        raise ValueError("model_cap must be positive")
    count = 0
    with solver_factory(bootstrap_with=model) as solver:
        for _ in solver.enum_models(assumptions=tuple(assumptions)):
            count += 1
            if count > model_cap:
                return {
                    "status": "unknown",
                    "reason": "trail_model_cap_exceeded",
                    "models_seen": count,
                    "parity": None,
                }
    return {
        "status": "exact",
        "reason": None,
        "models_seen": count,
        "parity": count & 1,
    }


def find_key_monomial_witness(
    model: Sequence[Sequence[int]],
    input_vars: Sequence[int],
    output_vars: Sequence[int],
    key_vars: Sequence[int],
    *,
    input_exponent: int,
    output_exponent: int,
    projected_key_cap: int,
    trail_model_cap: int,
    solver_factory: Callable[..., Any],
    projected_model_enumerator: Callable[..., Iterable[Sequence[int]]],
) -> dict[str, Any]:
    if not key_vars:
        return {
            "status": "no_witness",
            "reason": "model_has_no_key_variables",
            "projected_keys_seen": 0,
            "unknown_key_masks": 0,
            "witness": None,
        }
    assumptions = exponent_assumptions(
        input_vars, output_vars, input_exponent, output_exponent
    )
    nonzero_key_model = tuple(tuple(clause) for clause in model) + (tuple(key_vars),)
    unknown_key_masks = 0
    projected_seen = 0
    for projected in projected_model_enumerator(
        nonzero_key_model, tuple(key_vars), assumptions=assumptions
    ):
        projected_seen += 1
        if projected_seen > projected_key_cap:
            return {
                "status": "unknown",
                "reason": "projected_key_cap_exceeded",
                "projected_keys_seen": projected_seen,
                "unknown_key_masks": unknown_key_masks,
                "witness": None,
            }
        key_literals = tuple(int(value) for value in projected)
        parity = count_models_parity(
            nonzero_key_model,
            assumptions + key_literals,
            model_cap=trail_model_cap,
            solver_factory=solver_factory,
        )
        if parity["status"] != "exact":
            unknown_key_masks += 1
            continue
        if parity["parity"] == 1:
            key_mask = key_mask_from_projected_literals(key_literals, key_vars)
            return {
                "status": "witness",
                "reason": None,
                "projected_keys_seen": projected_seen,
                "unknown_key_masks": unknown_key_masks,
                "witness": {
                    "key_exponent_mask": key_mask,
                    "key_exponent_hex": f"0x{key_mask:X}",
                    "key_weight": key_mask.bit_count(),
                    "key_literals": list(key_literals),
                    "trail_models": parity["models_seen"],
                    "parity": 1,
                },
            }
    if unknown_key_masks:
        return {
            "status": "unknown",
            "reason": "one_or_more_key_masks_exceeded_trail_cap",
            "projected_keys_seen": projected_seen,
            "unknown_key_masks": unknown_key_masks,
            "witness": None,
        }
    return {
        "status": "no_witness",
        "reason": "all_projected_key_masks_exactly_even_or_absent",
        "projected_keys_seen": projected_seen,
        "unknown_key_masks": 0,
        "witness": None,
    }


def replay_key_monomial_parity(
    model: Sequence[Sequence[int]],
    input_vars: Sequence[int],
    output_vars: Sequence[int],
    key_vars: Sequence[int],
    *,
    input_exponent: int,
    output_exponent: int,
    key_exponent_mask: int,
    trail_model_cap: int,
    solver_factory: Callable[..., Any],
) -> dict[str, Any]:
    if not 0 <= key_exponent_mask < 1 << len(key_vars):
        raise ValueError("key exponent mask is out of range")
    assumptions = exponent_assumptions(
        input_vars, output_vars, input_exponent, output_exponent
    ) + tuple(
        variable if key_exponent_mask & (1 << index) else -variable
        for index, variable in enumerate(key_vars)
    )
    return count_models_parity(
        model,
        assumptions,
        model_cap=trail_model_cap,
        solver_factory=solver_factory,
    )


def inspect_author_source(atm_root: Path, paper_text: Path) -> dict[str, Any]:
    trails = (atm_root / "Modelling/Trails.py").read_text(encoding="utf-8")
    sat = (atm_root / "Tools/SATmodelling.py").read_text(encoding="utf-8")
    notebook = (atm_root / "Ciphers/PRESENT/PRESENT.ipynb").read_text(
        encoding="utf-8"
    )
    requirements = (atm_root / "requirements.txt").read_text(encoding="utf-8")
    paper = paper_text.read_text(encoding="utf-8")
    normalized_paper = " ".join(paper.lower().split())
    return {
        "checks": {
            "native_glucose4_import_present": (
                "from pysat.solvers import Glucose4 as Solver" in trails
            ),
            "projected_model_enumerator_present": "def enum_projected_models" in sat,
            "exact_key_dependence_api_present": "def is_key_dependent(" in trails,
            "key_polynomial_sum_api_present": "def get_sum(" in trails,
            "constant_coefficient_api_present": (
                "def get_key_independent_sum(" in trails
            ),
            "python_sat_is_frozen_requirement": "python-sat" in requirements.splitlines(),
            "get_sum_tuple_integer_membership_bug_detected": "if k in V:" in trails,
            "limited_key_dependence_cap_is_not_strict": (
                "if l > limit:\n                    return True" in trails
            ),
            "limited_constant_cap_is_not_strict": (
                "if c > limit:\n                return 1" in trails
            ),
            "paper_says_constant_value_is_unknown": (
                "the algorithm does not give any information on what this constant is"
                in normalized_paper
            ),
            "present_notebook_uses_fresh_round_key_masks": (
                "construct_iterated_cipher" in notebook
                and "[2**64-1]*(split[1]+1)" in notebook
            ),
            "present80_master_key_schedule_absent": (
                "master key" not in notebook.lower()
                and "key schedule" not in notebook.lower()
            ),
        },
        "claim_scope": (
            "official ATM native projected-SAT source contract; independent round "
            "keys, not actual PRESENT-80 master-key schedule"
        ),
    }


def install_single_process_qmc_compatibility_shim() -> None:
    qmc = importlib.import_module("LogicOptimisation.QMC")
    prop_models = importlib.import_module("Modelling.PropModels")
    cp_model = importlib.import_module("ortools.sat.python.cp_model")

    def qmc_optimise_cnf_single_process(
        truth_table: Any, dont_care: Any, max_search_time: int = 60, threads: int = 1
    ) -> list[list[int]]:
        prime_implicants: list[tuple[int, int]] = []
        for value in range(truth_table.size()):
            qmc.__f(truth_table, prime_implicants, value)
        model = cp_model.CpModel()
        variables = [
            model.NewBoolVar(f"x{index}")
            for index in range(len(prime_implicants))
        ]
        for value in range(truth_table.size()):
            if not truth_table.test(value) and not dont_care.test(value):
                model.AddAtLeastOne(qmc.__g(prime_implicants, variables, value))
        model.Minimize(sum(variables))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max_search_time
        solver.parameters.num_search_workers = 1
        status = solver.Solve(model)
        result: list[list[int]] = []
        if status in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
            for index, selected in enumerate(variables):
                if not solver.Value(selected):
                    continue
                assignment, omitted = prime_implicants[index]
                clause: list[int] = []
                for bit in range(int(math.log2(truth_table.size()))):
                    if omitted & 1 == 0:
                        clause.append((1 - 2 * (assignment & 1)) * (bit + 1))
                    assignment >>= 1
                    omitted >>= 1
                result.append(clause)
        else:
            result.append([])
        return result

    prop_models.QMC_optimise_CNF = qmc_optimise_cnf_single_process


def run_phase_a_calibration(
    atm_root: Path,
    *,
    trail_model_cap: int,
) -> dict[str, Any]:
    root = str(atm_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    components = importlib.import_module("Construction.Components")
    compound = importlib.import_module("Construction.CompoundFunction")
    iterated = importlib.import_module("Construction.IteratedCipher")
    sat_tools = importlib.import_module("Tools.SATmodelling")
    solvers = importlib.import_module("pysat.solvers")
    solver_factory = solvers.Glucose4
    install_single_process_qmc_compatibility_shim()

    present_sbox = (
        0xC,
        0x5,
        0x6,
        0xB,
        0x9,
        0x0,
        0xA,
        0xD,
        0x3,
        0xE,
        0xF,
        0x8,
        0x4,
        0x7,
        0x1,
        0x2,
    )
    sbox_component = components.SBox(4, 4, list(present_sbox))
    sbox_function = compound.CompoundFunction(4, 4)
    sbox_id = sbox_function.add_component(sbox_component)
    for bit in range(4):
        sbox_function.connect_components(compound.INPUT_ID, bit, sbox_id, bit)
        sbox_function.connect_components(sbox_id, bit, compound.OUTPUT_ID, bit)
    sbox_model, sbox_inputs, sbox_outputs, sbox_keys = sbox_function.to_model()

    sbox_rows: list[dict[str, Any]] = []
    for input_exponent in range(16):
        for output_exponent in range(16):
            expected = algebraic_transition_coefficient(
                present_sbox,
                input_bits=4,
                output_exponent=output_exponent,
                input_exponent=input_exponent,
            )
            observed = count_models_parity(
                sbox_model,
                exponent_assumptions(
                    sbox_inputs, sbox_outputs, input_exponent, output_exponent
                ),
                model_cap=trail_model_cap,
                solver_factory=solver_factory,
            )
            sbox_rows.append(
                {
                    "input_exponent": input_exponent,
                    "output_exponent": output_exponent,
                    "expected": expected,
                    "observed": observed["parity"],
                    "status": observed["status"],
                    "models_seen": observed["models_seen"],
                    "match": observed["status"] == "exact"
                    and observed["parity"] == expected,
                }
            )

    identity = components.SBox(1, 1, [0, 1])
    keyed_toy = iterated.construct_iterated_cipher([identity], [1, 0])
    toy_model, toy_inputs, toy_outputs, toy_keys = keyed_toy.to_model()
    witness = find_key_monomial_witness(
        toy_model,
        toy_inputs,
        toy_outputs,
        toy_keys,
        input_exponent=0,
        output_exponent=1,
        projected_key_cap=8,
        trail_model_cap=trail_model_cap,
        solver_factory=solver_factory,
        projected_model_enumerator=sat_tools.enum_projected_models,
    )
    replay = replay_key_monomial_parity(
        toy_model,
        toy_inputs,
        toy_outputs,
        toy_keys,
        input_exponent=0,
        output_exponent=1,
        key_exponent_mask=1,
        trail_model_cap=trail_model_cap,
        solver_factory=solver_factory,
    )
    constant = find_key_monomial_witness(
        toy_model,
        toy_inputs,
        toy_outputs,
        toy_keys,
        input_exponent=1,
        output_exponent=1,
        projected_key_cap=8,
        trail_model_cap=trail_model_cap,
        solver_factory=solver_factory,
        projected_model_enumerator=sat_tools.enum_projected_models,
    )
    forced_unknown = count_models_parity(
        ((1, -1),),
        (),
        model_cap=1,
        solver_factory=solver_factory,
    )
    matches = sum(bool(row["match"]) for row in sbox_rows)
    return {
        "sbox_rows": sbox_rows,
        "metrics": {
            "sbox_coefficients": len(sbox_rows),
            "sbox_matches": matches,
            "sbox_nonzero_coefficients": sum(
                int(row["expected"]) for row in sbox_rows
            ),
            "sbox_maximum_models_per_query": max(
                int(row["models_seen"]) for row in sbox_rows
            ),
            "sbox_median_models_per_query": statistics.median(
                int(row["models_seen"]) for row in sbox_rows
            ),
            "toy_key_variables": len(toy_keys),
        },
        "toy": {
            "key_dependent_xor_constant_term": witness,
            "witness_replay": replay,
            "constant_x_coefficient": constant,
            "forced_low_cap": forced_unknown,
        },
        "checks": {
            "all_256_sbox_coefficients_match": matches == 256,
            "toy_key_term_returns_witness": witness["status"] == "witness",
            "toy_witness_is_nonzero_key_monomial": bool(
                witness.get("witness")
                and int(witness["witness"]["key_exponent_mask"]) == 1
            ),
            "toy_witness_replay_is_exactly_odd": (
                replay["status"] == "exact" and replay["parity"] == 1
            ),
            "toy_constant_term_has_no_key_witness": constant["status"]
            == "no_witness",
            "cap_exhaustion_is_unknown": forced_unknown["status"] == "unknown",
        },
    }


def evaluate_phase_a(
    config: NativeSatProviderConfig,
    *,
    actual_commit: str,
    source: dict[str, Any],
    calibration: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    source_checks = {
        "atm_commit_matches_frozen_version": actual_commit == config.expected_commit,
        **source["checks"],
    }
    environment_checks = {
        "python_sat_available": bool(environment.get("python_sat_available")),
        "glucose4_available": bool(environment.get("glucose4_available")),
        "bitarrays_extension_available": bool(
            environment.get("bitarrays_extension_available")
        ),
    }
    calibration_checks = dict(calibration["checks"])
    if not all(source_checks.values()) or not all(environment_checks.values()):
        status = "fail"
        decision = "innovation2_atm_native_sat_source_or_environment_invalid"
        action = "repair the frozen source/dependency build before any provider probe"
    elif not all(calibration_checks.values()):
        status = "fail"
        decision = "innovation2_atm_native_sat_exact_calibration_failed"
        action = "repair adapter semantics; do not run the nine-round probe"
    else:
        status = "pass"
        decision = "innovation2_atm_native_sat_mechanism_ready_for_r9_probe"
        action = "run exactly one frozen nine-round relation mutation under hard caps"
    gate = {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "source_checks": source_checks,
        "environment_checks": environment_checks,
        "calibration_checks": calibration_checks,
        "metrics": calibration["metrics"],
        "claim_scope": (
            "ATM native projected-SAT mechanism reproduction on PRESENT S-box and "
            "one-bit keyed toy; not a nine-round witness, actual PRESENT-80 label, "
            "neural result, attack, or SOTA"
        ),
        "next_action": {
            "action": action,
            "r9_probe": status == "pass",
            "training": False,
            "remote_scale": False,
            "closed_routes": [
                "limited helpers interpreted as strict certificates",
                "not-in-basis candidates used as negatives",
                "actual PRESENT-80 claims from independent round-key witnesses",
                "neural training before relation-level witness width",
            ],
        },
    }
    result_rows = [
        {
            "run_id": config.run_id,
            "task": "innovation2_atm_native_sat_witness_provider_phase_a",
            "metric": key,
            "value": value,
            "status": status,
            "decision": decision,
            "training_performed": False,
        }
        for key, value in calibration["metrics"].items()
    ]
    return {"gate": gate, "result_rows": result_rows}


def serializable_config(config: NativeSatProviderConfig) -> dict[str, Any]:
    return asdict(config)
