from __future__ import annotations

import importlib
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from blockcipher_nd.tasks.innovation2.deterministic_provider_contract import ATM_COMMIT
from blockcipher_nd.tasks.innovation2.present_generalized_relation_precursor_boundary import (
    CanonicalRelation,
    precursor_plaintext_count,
)
from blockcipher_nd.ciphers.spn.present import Present80


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


def select_singleton_relation_mutation(
    relations: Sequence[CanonicalRelation],
    *,
    state_bits: int = 64,
) -> dict[str, Any]:
    coordinates = sorted({coordinate for relation in relations for coordinate in relation})
    coordinate_index = {coordinate: index for index, coordinate in enumerate(coordinates)}
    pivots = _gf2_pivots(
        sum(1 << coordinate_index[coordinate] for coordinate in relation)
        for relation in relations
    )
    singletons = sorted(
        (relation[0] for relation in relations if len(relation) == 1),
        key=lambda coordinate: (
            precursor_plaintext_count(coordinate[0]),
            coordinate,
        ),
    )
    mask = (1 << state_bits) - 1
    for input_exponent, output_exponent in singletons:
        for shift in range(1, state_bits):
            candidate_output = (
                (output_exponent << shift)
                | (output_exponent >> (state_bits - shift))
            ) & mask
            candidate = (input_exponent, candidate_output)
            if candidate in coordinate_index:
                in_positive_span = _is_in_span(
                    1 << coordinate_index[candidate], pivots
                )
            else:
                in_positive_span = False
            if not in_positive_span:
                return {
                    "source_relation": [
                        {
                            "input_exponent": input_exponent,
                            "input_exponent_hex": f"0x{input_exponent:016X}",
                            "output_exponent": output_exponent,
                            "output_exponent_hex": f"0x{output_exponent:016X}",
                        }
                    ],
                    "candidate_relation": [
                        {
                            "input_exponent": input_exponent,
                            "input_exponent_hex": f"0x{input_exponent:016X}",
                            "output_exponent": candidate_output,
                            "output_exponent_hex": f"0x{candidate_output:016X}",
                        }
                    ],
                    "output_rotation": shift,
                    "relation_size": 1,
                    "input_weight": input_exponent.bit_count(),
                    "source_output_weight": output_exponent.bit_count(),
                    "candidate_output_weight": candidate_output.bit_count(),
                    "source_is_public_positive": True,
                    "candidate_is_in_public_positive_span": False,
                }
    raise ValueError("no singleton marginal-matched mutation outside the positive span")


def _gf2_pivots(rows: Iterable[int]) -> dict[int, int]:
    pivots: dict[int, int] = {}
    for row in rows:
        value = int(row)
        while value:
            pivot = value.bit_length() - 1
            if pivot in pivots:
                value ^= pivots[pivot]
            else:
                pivots[pivot] = value
                break
    return pivots


def _is_in_span(value: int, pivots: dict[int, int]) -> bool:
    remainder = int(value)
    while remainder:
        pivot = remainder.bit_length() - 1
        if pivot not in pivots:
            return False
        remainder ^= pivots[pivot]
    return True


def build_present_independent_key_model(
    atm_root: Path, *, rounds: int
) -> dict[str, Any]:
    if rounds < 1:
        raise ValueError("rounds must be positive")
    root = str(atm_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    components = importlib.import_module("Construction.Components")
    compound = importlib.import_module("Construction.CompoundFunction")
    iterated = importlib.import_module("Construction.IteratedCipher")
    sat_tools = importlib.import_module("Tools.SATmodelling")
    solvers = importlib.import_module("pysat.solvers")
    install_single_process_qmc_compatibility_shim()

    start = time.monotonic()
    present_sbox = components.SBox(
        4,
        4,
        [0xC, 5, 6, 0xB, 9, 0, 0xA, 0xD, 3, 0xE, 0xF, 8, 4, 7, 1, 2],
    )
    bit_permutation = tuple(
        (16 * bit) % 63 if bit < 63 else 63 for bit in range(64)
    )
    round_function = compound.CompoundFunction(64, 64)
    sbox_ids = [round_function.add_component(present_sbox) for _ in range(16)]
    for bit in range(64):
        round_function.connect_components(
            compound.INPUT_ID, bit, sbox_ids[bit // 4], bit % 4
        )
        round_function.connect_components(
            sbox_ids[bit // 4], bit % 4, compound.OUTPUT_ID, bit_permutation[bit]
        )
    cipher = iterated.construct_iterated_cipher(
        [round_function] * rounds, [(1 << 64) - 1] * (rounds + 1)
    )
    model, input_vars, output_vars, key_vars = cipher.to_model()
    metadata = {
        "rounds": rounds,
        "key_additions": rounds + 1,
        "key_model": "independent_round_keys",
        "input_variables": len(input_vars),
        "output_variables": len(output_vars),
        "key_variables": len(key_vars),
        "cnf_clauses": len(model),
        "maximum_cnf_variable": max(
            (abs(literal) for clause in model for literal in clause), default=0
        ),
        "model_ready_seconds": time.monotonic() - start,
    }
    return {
        "model": model,
        "input_vars": input_vars,
        "output_vars": output_vars,
        "key_vars": key_vars,
        "solver_factory": solvers.Glucose4,
        "projected_model_enumerator": sat_tools.enum_projected_models,
        "metadata": metadata,
    }


def run_r9_singleton_probe(
    atm_root: Path,
    *,
    input_exponent: int,
    output_exponent: int,
    projected_key_cap: int,
    trail_model_cap: int,
) -> dict[str, Any]:
    start = time.monotonic()
    bundle = build_present_independent_key_model(atm_root, rounds=9)
    model = bundle["model"]
    input_vars = bundle["input_vars"]
    output_vars = bundle["output_vars"]
    key_vars = bundle["key_vars"]
    probe = find_key_monomial_witness(
        model,
        input_vars,
        output_vars,
        key_vars,
        input_exponent=input_exponent,
        output_exponent=output_exponent,
        projected_key_cap=projected_key_cap,
        trail_model_cap=trail_model_cap,
        solver_factory=bundle["solver_factory"],
        projected_model_enumerator=bundle["projected_model_enumerator"],
    )
    replay: dict[str, Any] | None = None
    if probe["status"] == "witness":
        replay = replay_key_monomial_parity(
            model,
            input_vars,
            output_vars,
            key_vars,
            input_exponent=input_exponent,
            output_exponent=output_exponent,
            key_exponent_mask=int(probe["witness"]["key_exponent_mask"]),
            trail_model_cap=trail_model_cap,
            solver_factory=bundle["solver_factory"],
        )
    return {
        "model": bundle["metadata"],
        "probe": probe,
        "replay": replay,
        "total_seconds": time.monotonic() - start,
    }


def run_present_relation_panel(
    atm_root: Path,
    *,
    rounds: int,
    input_exponent: int,
    output_exponents: Sequence[int],
    projected_key_cap: int,
    trail_model_cap: int,
    model_output: Path,
    panel_output: Path,
) -> dict[str, Any]:
    query_specs = [
        {
            "input_exponent": input_exponent,
            "output_exponent": output_exponent,
        }
        for output_exponent in output_exponents
    ]
    return run_present_relation_queries(
        atm_root,
        rounds=rounds,
        query_specs=query_specs,
        projected_key_cap=projected_key_cap,
        trail_model_cap=trail_model_cap,
        model_output=model_output,
        panel_output=panel_output,
    )


def run_present_relation_queries(
    atm_root: Path,
    *,
    rounds: int,
    query_specs: Sequence[dict[str, Any]],
    projected_key_cap: int,
    trail_model_cap: int,
    model_output: Path,
    panel_output: Path,
) -> dict[str, Any]:
    bundle = build_present_independent_key_model(atm_root, rounds=rounds)
    model_output.write_text(
        json.dumps(bundle["metadata"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    panel_output.write_text("", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for query_index, query_spec in enumerate(query_specs):
        input_exponent = int(query_spec["input_exponent"])
        output_exponent = int(query_spec["output_exponent"])
        query_start = time.monotonic()
        probe = find_key_monomial_witness(
            bundle["model"],
            bundle["input_vars"],
            bundle["output_vars"],
            bundle["key_vars"],
            input_exponent=input_exponent,
            output_exponent=output_exponent,
            projected_key_cap=projected_key_cap,
            trail_model_cap=trail_model_cap,
            solver_factory=bundle["solver_factory"],
            projected_model_enumerator=bundle["projected_model_enumerator"],
        )
        replay: dict[str, Any] | None = None
        label: int | None = None
        certificate = "unknown"
        if probe["status"] == "witness":
            replay = replay_key_monomial_parity(
                bundle["model"],
                bundle["input_vars"],
                bundle["output_vars"],
                bundle["key_vars"],
                input_exponent=input_exponent,
                output_exponent=output_exponent,
                key_exponent_mask=int(probe["witness"]["key_exponent_mask"]),
                trail_model_cap=trail_model_cap,
                solver_factory=bundle["solver_factory"],
            )
            if replay["status"] == "exact" and replay["parity"] == 1:
                label = 0
                certificate = "key_dependent_odd_witness"
        elif probe["status"] == "no_witness":
            replay = replay_key_monomial_parity(
                bundle["model"],
                bundle["input_vars"],
                bundle["output_vars"],
                bundle["key_vars"],
                input_exponent=input_exponent,
                output_exponent=output_exponent,
                key_exponent_mask=0,
                trail_model_cap=trail_model_cap,
                solver_factory=bundle["solver_factory"],
            )
            if replay["status"] == "exact":
                label = 1
                certificate = "constant_exhaustive_no_nonzero_key_monomial"
        row = {
            "query_index": query_index,
            "rounds": rounds,
            "input_exponent": input_exponent,
            "input_exponent_hex": f"0x{input_exponent:016X}",
            "output_exponent": output_exponent,
            "output_exponent_hex": f"0x{output_exponent:016X}",
            "label": label,
            "certificate": certificate,
            "probe": probe,
            "replay": replay,
            "elapsed_seconds": time.monotonic() - query_start,
            "query_metadata": {
                key: value
                for key, value in query_spec.items()
                if key not in {"input_exponent", "output_exponent"}
            },
        }
        rows.append(row)
        with panel_output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
    return {"model": bundle["metadata"], "rows": rows}


def present_two_round_input_cone(output_bit: int) -> tuple[int, ...]:
    if output_bit not in range(64):
        raise ValueError("output_bit must be in [0, 63]")
    permutation = tuple(
        (16 * bit) % 63 if bit < 63 else 63 for bit in range(64)
    )
    inverse = [0] * 64
    for source, destination in enumerate(permutation):
        inverse[destination] = source
    round_two_pre_permutation = inverse[output_bit]
    round_two_cell = round_two_pre_permutation // 4
    round_two_inputs = tuple(4 * round_two_cell + bit for bit in range(4))
    round_one_source_cells = {
        inverse[position] // 4 for position in round_two_inputs
    }
    return tuple(
        sorted(
            4 * cell + bit
            for cell in round_one_source_cells
            for bit in range(4)
        )
    )


def two_round_cone_matched_queries(output_bit: int = 0) -> tuple[dict[str, Any], ...]:
    cone = present_two_round_input_cone(output_bit)
    cone_set = set(cone)
    outside_bit = min(bit for bit in range(64) if bit not in cone_set)
    queries: list[dict[str, Any]] = []
    for weight in range(1, 9):
        inside_bits = cone[:weight]
        outside_bits = cone[: weight - 1] + (outside_bit,)
        for group, bits in (("inside", inside_bits), ("outside", outside_bits)):
            input_exponent = sum(1 << bit for bit in bits)
            queries.append(
                {
                    "input_exponent": input_exponent,
                    "output_exponent": 1 << output_bit,
                    "weight": weight,
                    "cone_group": group,
                    "all_input_bits_inside_cone": group == "inside",
                    "input_bits": list(bits),
                    "output_bit": output_bit,
                }
            )
    return tuple(queries)


def _binary_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return wins / (len(positives) * len(negatives))


def scalar_present_independent_key_coefficient(
    *,
    rounds: int,
    input_exponent: int,
    output_exponent: int,
    round_keys: Sequence[int],
) -> int:
    if len(round_keys) != rounds + 1:
        raise ValueError("round_keys must include one key per round plus post-whitening")
    accumulator = 0
    plaintext = input_exponent
    while True:
        state = plaintext
        for round_index in range(rounds):
            state ^= int(round_keys[round_index]) & ((1 << 64) - 1)
            state = Present80._permutation_layer(Present80._sbox_layer(state))
        state ^= int(round_keys[-1]) & ((1 << 64) - 1)
        accumulator ^= int((state & output_exponent) == output_exponent)
        if plaintext == 0:
            return accumulator
        plaintext = (plaintext - 1) & input_exponent


def evaluate_cone_matched_panel(
    config: NativeSatProviderConfig,
    *,
    phase_a_gate: dict[str, Any],
    model: dict[str, Any] | None,
    rows: Sequence[dict[str, Any]],
    planned_queries: int,
    worker_status: str,
    wall_clock_cap_seconds: int,
) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows if row.get("label") in {0, 1}]
    resolved_rows = [row for row in rows if row.get("label") in {0, 1}]
    constant_rows = sum(label == 1 for label in labels)
    key_dependent_rows = sum(label == 0 for label in labels)
    unknown_rows = planned_queries - len(resolved_rows)
    degree_auc_raw = _binary_auc(
        labels,
        [float(row["query_metadata"]["weight"]) for row in resolved_rows],
    )
    degree_auc = (
        None
        if degree_auc_raw is None
        else max(degree_auc_raw, 1.0 - degree_auc_raw)
    )
    cone_auc_raw = _binary_auc(
        labels,
        [
            0.0 if row["query_metadata"]["all_input_bits_inside_cone"] else 1.0
            for row in resolved_rows
        ],
    )
    cone_auc = (
        None if cone_auc_raw is None else max(cone_auc_raw, 1.0 - cone_auc_raw)
    )
    weights = {
        int(row["query_metadata"]["weight"]): set()
        for row in rows
    }
    for row in rows:
        weights[int(row["query_metadata"]["weight"])].add(
            str(row["query_metadata"]["cone_group"])
        )
    all_negative_witnesses_replay = all(
        row.get("label") != 0
        or (
            int(row["probe"]["witness"]["key_exponent_mask"]) != 0
            and row.get("replay", {}).get("status") == "exact"
            and row.get("replay", {}).get("parity") == 1
        )
        for row in rows
    )
    scalar_key_sets = (
        (0, 0, 0),
        (0x0123456789ABCDEF, 0xFEDCBA9876543210, 0x1111222233334444),
        (0xAAAAAAAAAAAAAAAA, 0x5555555555555555, 0xDEADBEEFCAFEBABE),
    )
    positive_scalar_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("label") != 1:
            continue
        scalar_values = [
            scalar_present_independent_key_coefficient(
                rounds=2,
                input_exponent=int(row["input_exponent"]),
                output_exponent=int(row["output_exponent"]),
                round_keys=keys,
            )
            for keys in scalar_key_sets
        ]
        expected = row.get("replay", {}).get("parity")
        positive_scalar_rows.append(
            {
                "query_index": row.get("query_index"),
                "expected_constant": expected,
                "scalar_values": scalar_values,
                "match": expected in {0, 1}
                and all(value == expected for value in scalar_values),
            }
        )
    all_positive_scalar_checks_match = bool(positive_scalar_rows) and all(
        row["match"] for row in positive_scalar_rows
    )
    source_checks = {
        "e58_phase_a_gate_passed": phase_a_gate.get("status") == "pass"
        and phase_a_gate.get("decision")
        == "innovation2_atm_native_sat_mechanism_ready_for_r9_probe",
        "model_rounds_are_two": bool(model and model.get("rounds") == 2),
        "model_has_three_key_additions": bool(
            model and model.get("key_additions") == 3
        ),
        "model_has_192_independent_key_variables": bool(
            model
            and model.get("key_variables") == 192
            and model.get("key_model") == "independent_round_keys"
        ),
        "each_weight_has_inside_outside_pair": len(weights) == 8
        and all(groups == {"inside", "outside"} for groups in weights.values()),
        "two_round_cone_is_bits_0_to_15": present_two_round_input_cone(0)
        == tuple(range(16)),
        "all_constant_rows_match_three_scalar_key_sets": (
            all_positive_scalar_checks_match
        ),
    }
    width_checks = {
        "completed_queries_at_least_12": len(rows) >= 12,
        "unknown_fraction_at_most_0p25": unknown_rows / planned_queries <= 0.25,
        "strict_constant_rows_at_least_4": constant_rows >= 4,
        "strict_key_dependent_rows_at_least_4": key_dependent_rows >= 4,
        "all_negative_witnesses_replay_odd": all_negative_witnesses_replay,
    }
    shortcut_checks = {
        "degree_only_auc_at_most_0p65": degree_auc is not None and degree_auc <= 0.65,
        "cone_membership_auc_at_most_0p80": cone_auc is not None and cone_auc <= 0.80,
    }
    metrics = {
        "planned_queries": planned_queries,
        "completed_queries": len(rows),
        "strict_constant_rows": constant_rows,
        "strict_key_dependent_rows": key_dependent_rows,
        "unknown_rows": unknown_rows,
        "degree_only_auc": degree_auc,
        "cone_membership_strongest_direction_auc": cone_auc,
        "worker_status": worker_status,
        "scalar_validated_constant_rows": sum(
            bool(row["match"]) for row in positive_scalar_rows
        ),
    }
    if not all(source_checks.values()):
        status = "fail"
        decision = "innovation2_atm_r2_cone_matched_panel_protocol_invalid"
        action = "repair cone/query/model ownership before interpreting labels"
    elif not all(width_checks.values()):
        status = "hold"
        decision = "innovation2_atm_r2_cone_matched_panel_width_not_ready"
        action = "do not train RCCA; singleton strict label width remains insufficient"
    elif not all(shortcut_checks.values()):
        status = "hold"
        decision = "innovation2_atm_r2_singleton_relation_shortcut_dominated"
        action = "close singleton relations and design multi-coordinate GF(2) cancellation"
    else:
        status = "pass"
        decision = "innovation2_atm_r2_cone_matched_panel_ready"
        action = "run the full strict label-width and shortcut audit before RCCA"
    gate = {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "source_checks": source_checks,
        "width_checks": width_checks,
        "shortcut_checks": shortcut_checks,
        "metrics": metrics,
        "scalar_validation": {
            "key_sets": len(scalar_key_sets),
            "rows": positive_scalar_rows,
        },
        "wall_clock_cap_seconds": wall_clock_cap_seconds,
        "projected_key_cap": config.projected_key_cap,
        "trail_model_cap": config.trail_model_cap,
        "claim_scope": (
            "PRESENT two-round independent-round-key cone-matched singleton relation "
            "label audit; not actual PRESENT-80 labels, neural training, attack, or SOTA"
        ),
        "next_action": {
            "action": action,
            "full_width_audit": status == "pass",
            "multi_coordinate_design": decision
            in {
                "innovation2_atm_r2_cone_matched_panel_width_not_ready",
                "innovation2_atm_r2_singleton_relation_shortcut_dominated",
            },
            "training": False,
            "remote_scale": False,
        },
    }
    result_rows = [
        {
            "run_id": config.run_id,
            "task": "innovation2_atm_r2_cone_matched_panel",
            "metric": key,
            "value": value,
            "status": status,
            "decision": decision,
            "training_performed": False,
        }
        for key, value in metrics.items()
        if key != "worker_status"
    ]
    return {"gate": gate, "result_rows": result_rows}


def evaluate_low_round_panel(
    config: NativeSatProviderConfig,
    *,
    phase_a_gate: dict[str, Any],
    model: dict[str, Any] | None,
    rows: Sequence[dict[str, Any]],
    planned_queries: int,
    rounds: int,
    worker_status: str,
    wall_clock_cap_seconds: int,
) -> dict[str, Any]:
    constant_rows = sum(row.get("label") == 1 for row in rows)
    key_dependent_rows = sum(row.get("label") == 0 for row in rows)
    explicit_unknown = sum(row.get("label") is None for row in rows)
    missing_rows = planned_queries - len(rows)
    total_unknown = explicit_unknown + missing_rows
    all_negative_witnesses_replay = all(
        row.get("label") != 0
        or (
            int(row["probe"]["witness"]["key_exponent_mask"]) != 0
            and row.get("replay", {}).get("status") == "exact"
            and row.get("replay", {}).get("parity") == 1
        )
        for row in rows
    )
    source_checks = {
        "e58_phase_a_gate_passed": phase_a_gate.get("status") == "pass"
        and phase_a_gate.get("decision")
        == "innovation2_atm_native_sat_mechanism_ready_for_r9_probe",
        "model_artifact_present": model is not None,
        "model_rounds_match": bool(model and int(model.get("rounds", -1)) == rounds),
        "model_key_additions_match": bool(
            model and int(model.get("key_additions", -1)) == rounds + 1
        ),
        "model_key_variables_match": bool(
            model and int(model.get("key_variables", -1)) == (rounds + 1) * 64
        ),
        "independent_round_key_model": bool(
            model and model.get("key_model") == "independent_round_keys"
        ),
    }
    width_checks = {
        "completed_queries_at_least_12": len(rows) >= 12,
        "unknown_fraction_at_most_0p25": total_unknown / planned_queries <= 0.25,
        "strict_constant_rows_at_least_4": constant_rows >= 4,
        "strict_key_dependent_rows_at_least_4": key_dependent_rows >= 4,
        "all_negative_witnesses_replay_odd": all_negative_witnesses_replay,
        "all_nonresolved_rows_are_unknown": all(
            row.get("label") in {0, 1, None} for row in rows
        ),
    }
    metrics = {
        "planned_queries": planned_queries,
        "completed_queries": len(rows),
        "strict_constant_rows": constant_rows,
        "strict_key_dependent_rows": key_dependent_rows,
        "explicit_unknown_rows": explicit_unknown,
        "missing_timeout_rows": missing_rows,
        "total_unknown_fraction": total_unknown / planned_queries,
        "worker_status": worker_status,
    }
    if not all(source_checks.values()):
        status = "fail"
        decision = "innovation2_atm_r2_strict_relation_panel_protocol_invalid"
        action = "repair source/model ownership before interpreting panel labels"
    elif all(width_checks.values()):
        status = "pass"
        decision = "innovation2_atm_r2_strict_relation_panel_ready"
        action = "expand to the frozen 1024-query label-width and shortcut audit"
    else:
        status = "hold"
        decision = "innovation2_atm_r2_strict_relation_panel_not_ready"
        action = (
            "do not train RCCA; inspect class diversity and provider completion, then "
            "decide whether multi-coordinate cancellation is required"
        )
    gate = {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "source_checks": source_checks,
        "width_checks": width_checks,
        "metrics": metrics,
        "wall_clock_cap_seconds": wall_clock_cap_seconds,
        "projected_key_cap": config.projected_key_cap,
        "trail_model_cap": config.trail_model_cap,
        "claim_scope": (
            "PRESENT two-round independent-round-key ATM strict relation-label panel "
            "readiness; not actual PRESENT-80 labels, neural training, attack, or SOTA"
        ),
        "next_action": {
            "action": action,
            "full_width_audit": status == "pass",
            "training": False,
            "remote_scale": False,
        },
    }
    result_rows = [
        {
            "run_id": config.run_id,
            "task": "innovation2_atm_r2_strict_relation_panel",
            "metric": key,
            "value": value,
            "status": status,
            "decision": decision,
            "training_performed": False,
        }
        for key, value in metrics.items()
        if key != "worker_status"
    ]
    return {"gate": gate, "result_rows": result_rows}


def evaluate_r9_probe(
    config: NativeSatProviderConfig,
    *,
    candidate: dict[str, Any],
    worker_status: str,
    worker_result: dict[str, Any] | None,
    wall_clock_cap_seconds: int,
) -> dict[str, Any]:
    candidate_checks = {
        "source_is_public_positive": bool(candidate["source_is_public_positive"]),
        "candidate_is_outside_public_positive_span": not bool(
            candidate["candidate_is_in_public_positive_span"]
        ),
        "relation_size_preserved": int(candidate["relation_size"]) == 1,
        "input_weight_preserved": True,
        "output_weight_preserved": int(candidate["source_output_weight"])
        == int(candidate["candidate_output_weight"]),
    }
    strict_witness = False
    if worker_status == "completed" and worker_result is not None:
        probe = worker_result["probe"]
        replay = worker_result.get("replay")
        strict_witness = bool(
            probe["status"] == "witness"
            and probe.get("witness")
            and int(probe["witness"]["key_exponent_mask"]) != 0
            and replay
            and replay["status"] == "exact"
            and replay["parity"] == 1
        )
    witness_checks = {
        "worker_completed_within_wall_clock_cap": worker_status == "completed",
        "nonzero_key_exponent_witness_found": strict_witness,
        "relation_level_replay_is_exactly_odd": strict_witness,
        "key_model_is_independent_round_keys": bool(
            worker_result
            and worker_result.get("model", {}).get("key_model")
            == "independent_round_keys"
        ),
    }
    if not all(candidate_checks.values()):
        status = "fail"
        decision = "innovation2_atm_native_sat_r9_candidate_protocol_invalid"
        action = "repair the frozen marginal-matched mutation; do not search alternatives"
    elif strict_witness:
        status = "pass"
        decision = "innovation2_atm_native_sat_r9_strict_negative_found"
        action = "expand to at most 32 frozen matched candidates for label-width audit"
    elif worker_status == "timeout":
        status = "hold"
        decision = "innovation2_atm_native_sat_r9_wall_clock_cap_exceeded"
        action = "close the native exact r9 witness route at the frozen cap"
    elif worker_status == "completed":
        status = "hold"
        decision = "innovation2_atm_native_sat_r9_negative_not_proven"
        action = "retain the candidate as unknown; do not relabel or mutate post hoc"
    else:
        status = "fail"
        decision = "innovation2_atm_native_sat_r9_worker_failed"
        action = "diagnose the worker failure before interpreting the provider"
    gate = {
        "run_id": config.run_id,
        "status": status,
        "decision": decision,
        "candidate_checks": candidate_checks,
        "witness_checks": witness_checks,
        "worker_status": worker_status,
        "wall_clock_cap_seconds": wall_clock_cap_seconds,
        "projected_key_cap": config.projected_key_cap,
        "trail_model_cap": config.trail_model_cap,
        "claim_scope": (
            "one frozen PRESENT nine-round independent-round-key generalized-relation "
            "mutation probe; not an actual PRESENT-80 master-key negative, neural "
            "training result, attack, or SOTA"
        ),
        "next_action": {
            "action": action,
            "label_width_audit": strict_witness,
            "training": False,
            "remote_scale": False,
            "closed_routes": [
                "timeout or cap interpreted as a negative",
                "post-hoc candidate mutation",
                "actual PRESENT-80 claims",
                "neural training before 256/256 strict label width",
            ],
        },
    }
    result_rows = [
        {
            "run_id": config.run_id,
            "task": "innovation2_atm_native_sat_r9_singleton_probe",
            "metric": "strict_negative_found",
            "value": strict_witness,
            "status": status,
            "decision": decision,
            "training_performed": False,
        },
        {
            "run_id": config.run_id,
            "task": "innovation2_atm_native_sat_r9_singleton_probe",
            "metric": "worker_completed",
            "value": worker_status == "completed",
            "status": status,
            "decision": decision,
            "training_performed": False,
        },
    ]
    return {"gate": gate, "result_rows": result_rows}


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
