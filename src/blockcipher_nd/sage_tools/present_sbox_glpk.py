from __future__ import annotations

import json
from time import perf_counter
from typing import Any

from blockcipher_nd.ciphers.spn.present import PRESENT_SBOX_ANF


def enumerate_sbox_output_exponent_glpk(output_exponent: int) -> dict[str, Any]:
    from sage.all import MixedIntegerLinearProgram
    from sage.numerical.mip import MIPSolverException

    output_exponent = int(output_exponent)
    if output_exponent not in range(1, 16):
        raise ValueError("output_exponent must be in [1, 15]")
    selected = [bit for bit in range(4) if output_exponent & (1 << bit)]
    program = MixedIntegerLinearProgram(maximization=False, solver="GLPK")
    choices = program.new_variable(binary=True)
    input_bits = program.new_variable(binary=True)
    keys = [
        (output_bit, term_index)
        for output_bit in selected
        for term_index in range(len(PRESENT_SBOX_ANF[output_bit]))
    ]
    for output_bit in selected:
        program.add_constraint(
            sum(
                choices[output_bit, term_index]
                for term_index in range(len(PRESENT_SBOX_ANF[output_bit]))
            )
            == 1
        )
    for input_bit in range(4):
        occurrences = [
            choices[output_bit, term_index]
            for output_bit, term_index in keys
            if PRESENT_SBOX_ANF[output_bit][term_index] & (1 << input_bit)
        ]
        if occurrences:
            for occurrence in occurrences:
                program.add_constraint(input_bits[input_bit] >= occurrence)
            program.add_constraint(input_bits[input_bit] <= sum(occurrences))
        else:
            program.add_constraint(input_bits[input_bit] == 0)
    program.set_objective(0)
    counts: dict[int, int] = {}
    solutions = 0
    started = perf_counter()
    while True:
        try:
            program.solve(log=0)
        except MIPSolverException:
            break
        selected_keys = [
            key for key in keys if float(program.get_values(choices[key])) > 0.5
        ]
        input_exponent = int(
            sum(
                1 << bit
                for bit in range(4)
                if float(program.get_values(input_bits[bit])) > 0.5
            )
        )
        direct_exponent = 0
        for output_bit, term_index in selected_keys:
            direct_exponent |= int(PRESENT_SBOX_ANF[output_bit][term_index])
        if input_exponent != direct_exponent:
            raise AssertionError("GLPK input exponent disagrees with selected terms")
        counts[input_exponent] = int(counts.get(input_exponent, 0) + 1)
        solutions += 1
        selected_key_set = set(selected_keys)
        program.add_constraint(
            sum(
                1 - choices[key] if key in selected_key_set else choices[key]
                for key in keys
            )
            >= 1
        )
    expected_solutions = 1
    for output_bit in selected:
        expected_solutions *= len(PRESENT_SBOX_ANF[output_bit])
    return {
        "output_exponent": output_exponent,
        "selected_output_bits": selected,
        "backend": type(program.get_backend()).__name__,
        "seconds": float(perf_counter() - started),
        "solutions": int(solutions),
        "expected_solutions": int(expected_solutions),
        "complete": solutions == expected_solutions,
        "counts": {str(key): int(value) for key, value in sorted(counts.items())},
    }


def enumerate_sbox_output_exponent_glpk_json(output_exponent: int) -> str:
    return json.dumps(
        enumerate_sbox_output_exponent_glpk(int(output_exponent)), sort_keys=True
    )
