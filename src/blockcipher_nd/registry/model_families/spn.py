from __future__ import annotations

import torch
from torch import nn

from blockcipher_nd.models.structure import (
    GiftAlignedTokenMixerRawInputDistinguisher,
    GiftCrossSpnTypedCellE5FromPresentOffDistinguisher,
    GiftCrossSpnTypedCellE5FromPresentShuffledPlaceboDistinguisher,
    GiftCrossSpnTypedCellE5FromPresentTrueShuffledDistinguisher,
    GiftCrossSpnTypedCellE5ScratchDistinguisher,
    GiftCrossSpnTypedCellE6FromPresentFunctionalMarginDistinguisher,
    GiftCrossSpnTypedCellE6FromPresentOffDistinguisher,
    GiftCrossSpnTypedCellE6FromPresentShuffledPlaceboDistinguisher,
    GiftCrossSpnTypedCellE6ScratchDistinguisher,
    GiftCrossSpnTypedCellEquivariantMixerDistinguisher,
    GiftCrossSpnTypedCellNoPositionDistinguisher,
    GiftCrossSpnTypedCellSharedViewEncoderDistinguisher,
    GiftCrossSpnTypedCellRawDistinguisher,
    GiftCrossSpnTypedCellShuffledFromPresentTrueDistinguisher,
    GiftCrossSpnTypedCellShuffledDistinguisher,
    GiftCrossSpnTypedCellTrueFromPresentShuffledDistinguisher,
    GiftCrossSpnTypedCellTrueFromPresentTrueDistinguisher,
    GiftCrossSpnTypedCellTrueDistinguisher,
    Gift64GohrStyleResNetPairSetDistinguisher,
    Gift64SunStyleLstmPairSetDistinguisher,
    PresentInceptionMCNDDistinguisher,
    PresentInceptionMCNDGlobalMatrixDistinguisher,
    PresentInceptionMCNDMatrixDistinguisher,
    PresentInceptionMCNDPairStackMatrixDistinguisher,
    PresentCrossSpnTypedCellRawDistinguisher,
    PresentCrossSpnTypedCellE5OffDistinguisher,
    PresentCrossSpnTypedCellE5ShuffledPlaceboDistinguisher,
    PresentCrossSpnTypedCellE5TrueShuffledDistinguisher,
    PresentCrossSpnTypedCellE6FunctionalMarginDistinguisher,
    PresentCrossSpnTypedCellE6OffDistinguisher,
    PresentCrossSpnTypedCellE6ShuffledPlaceboDistinguisher,
    PresentCrossSpnTypedCellShuffledDistinguisher,
    PresentCrossSpnTypedCellTrueDistinguisher,
    PresentMatrixTrailHybridPairSetDistinguisher,
    PresentNibbleDDTGraphDistinguisher,
    PresentNibbleDeltaStateMatrixConv2DSpnOnlyDistinguisher,
    PresentNibbleDeltaTopologyResidualSpnOnlyDistinguisher,
    PresentNibbleCase3InvPTopologyResidualSpnOnlyDistinguisher,
    PresentNibbleCase3RawTopologyResidualSpnOnlyDistinguisher,
    PresentNibbleCase3ShuffledPTopologyResidualSpnOnlyDistinguisher,
    PresentInvPDBitNet2023Distinguisher,
    PresentRawDeltaDBitNet2023Distinguisher,
    PresentShuffledPDBitNet2023Distinguisher,
    PresentNibbleNoDDTGraphDistinguisher,
    PresentPairSetGlobalStatsDistinguisher,
    PresentPairSetGlobalStatsHybridDistinguisher,
    PresentPairSetHistogramHybridDistinguisher,
    PresentPairSetStatsHybridDistinguisher,
    PresentActiveCellGraphPairSetDistinguisher,
    PresentPLayerMixerPairSetDistinguisher,
    PresentNibbleDeltaOnlySpnOnlyDistinguisher,
    PresentNibbleInvPActiveAuxSpnOnlyDistinguisher,
    PresentNibbleInvPNoDDTGateDistinguisher,
    PresentNibblePAlignedGatedMCNDDistinguisher,
    PresentNibbleInvPPLayerGraphSpnOnlyDistinguisher,
    PresentNibbleInvPPairConsistencySpnOnlyDistinguisher,
    PresentNibbleInvPPairMixerConsistencySpnOnlyDistinguisher,
    PresentNibbleInvPOnlySpnOnlyDistinguisher,
    PresentNibbleInvPStateMatrixConv2DSpnOnlyDistinguisher,
    PresentNibbleInvPTopologyResidualSpnOnlyDistinguisher,
    PresentNibbleInvPShuffledPLayerGraphSpnOnlyDistinguisher,
    PresentNibbleInvPShuffledSboxPriorGateDistinguisher,
    PresentNibbleInvPSboxPriorGateDistinguisher,
    PresentNibblePAlignedMCNDDistinguisher,
    PresentNibblePAlignedSpnOnlyDistinguisher,
    PresentNibblePAlignedTransitionDistinguisher,
    PresentNibblePAlignedTransitionResidualDistinguisher,
    PresentNibbleShuffledDDTGraphDistinguisher,
    PresentNibbleShuffledPAlignedGatedMCNDDistinguisher,
    PresentNibbleShuffledPAlignedSpnOnlyDistinguisher,
    PresentNibbleShuffledPStateMatrixConv2DSpnOnlyDistinguisher,
    PresentNibbleShuffledPTopologyResidualSpnOnlyDistinguisher,
    PresentNibbleShuffledTransitionResidualDistinguisher,
    PresentTrailPositionStatsPairSetDistinguisher,
    PresentTrailMixerPairSetDistinguisher,
    PresentZhangWangKerasMCNDDistinguisher,
    SpnCellPairSetDBitNetDistinguisher,
    SpnLiuCase3Conv2DAdapterDistinguisher,
    SpnNibbleConvPairSetDistinguisher,
    SpnTokenMixerPairSetDistinguisher,
    SpnZhangWangMCNDAdapterDistinguisher,
)
from blockcipher_nd.models.structure.spn.runtime_parameterized import (
    FixedRuntimeSpnProtocolAdapter,
    RuntimeParameterizedSpnSpec,
)
from blockcipher_nd.models.structure.spn.canonical_transition import (
    CanonicalTransitionSpnSpec,
    FixedCanonicalTransitionSpnProtocolAdapter,
)
from blockcipher_nd.models.structure.spn.canonical_relative_path import (
    FixedRelativePathSpnProtocolAdapter,
    RelativePathSpnSpec,
)
from blockcipher_nd.models.structure.spn.canonical_cell_path_hypergraph import (
    CellPathHypergraphSpnSpec,
    FixedCellPathHypergraphSpnProtocolAdapter,
)
from blockcipher_nd.models.structure.spn.gf2_boolean_view import (
    FixedGf2BooleanViewSpnProtocolAdapter,
    Gf2BooleanViewSpnSpec,
)
from blockcipher_nd.models.structure.spn.operator_tied_latent import (
    FixedOperatorTiedLatentSpnProtocolAdapter,
    OperatorTiedLatentSpnSpec,
)
from blockcipher_nd.models.structure.spn.topology_edge_residual import (
    FixedTopologyEdgeResidualSpnProtocolAdapter,
    TopologyEdgeResidualSpnSpec,
)
from blockcipher_nd.models.structure.spn.exact_operator_composition import (
    FixedExactOperatorCompositionSpnProtocolAdapter,
)
from blockcipher_nd.models.structure.spn.ordered_primitive_conditioner import (
    FixedOrderedPrimitiveConditionedSpnProtocolAdapter,
    OrderedPrimitiveConditionerSpec,
)
from blockcipher_nd.models.structure.spn.ordered_primitive_program import (
    compile_ordered_primitive_program,
    permute_program_source_endpoints_affine,
    permute_program_target_bindings,
    rotate_program_stages,
)
from blockcipher_nd.models.structure.spn.position_histogram_residual import (
    FixedCompactInvariantHistogramResidualSpnProtocolAdapter,
    FixedCompactSboxTransitionResidualSpnProtocolAdapter,
    FixedComponentSeparatedDualPathSboxTransitionResidualSpnProtocolAdapter,
    FixedDualStructureConditionedSboxTransitionResidualSpnProtocolAdapter,
    FixedPositionHistogramResidualSpnProtocolAdapter,
    FixedStructureConditionedSboxTransitionResidualSpnProtocolAdapter,
    PositionHistogramResidualSpnSpec,
    SboxTransitionResidualSpnSpec,
)
from blockcipher_nd.models.structure.spn.runtime_structure_factories import (
    gift64_runtime_structure,
    present_runtime_structure,
    skinny64_runtime_structure,
)
from blockcipher_nd.models.structure.spn.runtime_structure import (
    RuntimeSpnStructure,
    load_runtime_spn_descriptor,
    runtime_spn_structure_from_truth_bits,
)
from blockcipher_nd.registry.model_options import (
    int_option,
    int_tuple_option,
    matrix_kernel_size_option,
)


def _apply_runtime_structure_window_control(
    structure: RuntimeSpnStructure,
    options: dict[str, object],
) -> tuple[RuntimeSpnStructure, str]:
    control = str(options.get("runtime_structure_window_control", "full"))
    if control == "full":
        return structure, control
    if control == "repeat_last":
        return structure.repeat_last_transition(), control
    if control == "rotated":
        return structure.rotate_transitions(), control
    raise ValueError(
        "runtime_structure_window_control must be full, repeat_last, or rotated"
    )


def build_spn_model(
    name: str,
    input_bits: int,
    hidden_bits: int,
    pair_bits: int | None,
    options: dict[str, object],
) -> nn.Module | None:
    ordered_primitive_models = {
        "runtime_spn_k1by1_compiler_correct": "correct",
        "runtime_spn_k1by1_compiler_wrong_order": "wrong_order",
        "runtime_spn_k1by1_compiler_wrong_binding": "wrong_binding",
        "runtime_spn_k1by1_compiler_affine_wrong_endpoint": (
            "affine_wrong_endpoint"
        ),
        "runtime_spn_k1by1_no_compiler_conditioner": "no_conditioner",
        "runtime_spn_k1by13_anchor_correct": "correct",
        "runtime_spn_k1by13_adapter_correct": "correct",
        "runtime_spn_k1by13_adapter_affine": "affine_wrong_endpoint",
        "runtime_spn_k1by13_adapter_shuffled": "correct",
        "runtime_spn_k1by14_paired_correct": "correct",
        "runtime_spn_k1by14_paired_affine": "affine_wrong_endpoint",
    }
    if name in ordered_primitive_models:
        descriptor_path = options.get("runtime_structure_path")
        if not isinstance(descriptor_path, str) or not descriptor_path.strip():
            raise ValueError(f"model {name} requires runtime_structure_path")
        runtime_rounds = int_option(options, "runtime_rounds", 2)
        runtime_round_start = int_option(options, "runtime_round_start", 0)
        assert runtime_rounds is not None
        assert runtime_round_start is not None
        descriptor = load_runtime_spn_descriptor(
            descriptor_path,
            rounds=runtime_rounds,
            round_start=runtime_round_start,
        )
        program = compile_ordered_primitive_program(descriptor.structure)
        control = ordered_primitive_models[name]
        if control == "wrong_order":
            program = rotate_program_stages(program)
        elif control == "wrong_binding":
            wrong_binding_seed = int_option(options, "wrong_binding_seed", 11)
            assert wrong_binding_seed is not None
            program = permute_program_target_bindings(
                program,
                seed=wrong_binding_seed,
            )
        elif control == "affine_wrong_endpoint":
            affine_multiplier = int_option(
                options,
                "affine_endpoint_multiplier",
                5,
            )
            affine_offset = int_option(options, "affine_endpoint_offset", 1)
            assert affine_multiplier is not None
            assert affine_offset is not None
            program = permute_program_source_endpoints_affine(
                program,
                multiplier=affine_multiplier,
                offset=affine_offset,
            )
        pair_embedding_dim = int_option(options, "pair_embedding_dim", 128)
        primitive_hidden_dim = int_option(
            options,
            "primitive_hidden_dim",
            hidden_bits,
        )
        assert pair_embedding_dim is not None
        assert primitive_hidden_dim is not None
        source_permutation_value = options.get(
            "post_expert_source_cell_permutation"
        )
        if source_permutation_value is None:
            source_permutation = None
        elif isinstance(source_permutation_value, list):
            source_permutation = tuple(int(value) for value in source_permutation_value)
        else:
            raise ValueError(
                "post_expert_source_cell_permutation must be a JSON list"
            )
        spec = OrderedPrimitiveConditionerSpec(
            hidden_dim=primitive_hidden_dim,
            pair_embedding_dim=pair_embedding_dim,
            dropout=float(options.get("dropout", 0.0)),
            initial_effective_gate=float(
                options.get("primitive_gate_initial_effective", 0.05)
            ),
            linear_histogram_mode=str(
                options.get("linear_histogram_mode", "local")
            ),
            post_expert_residual_mode=str(
                options.get("post_expert_residual_mode", "none")
            ),
            post_expert_adapter_mode=str(
                options.get("post_expert_adapter_mode", "none")
            ),
            post_expert_adapter_bottleneck_dim=int(
                options.get("post_expert_adapter_bottleneck_dim", 16)
            ),
            post_expert_source_cell_permutation=source_permutation,
        )

        def build_ordered_adapter(
            runtime_program,
        ) -> FixedOrderedPrimitiveConditionedSpnProtocolAdapter:
            return FixedOrderedPrimitiveConditionedSpnProtocolAdapter(
                input_bits=input_bits,
                pair_bits=(
                    2 * descriptor.structure.block_bits
                    if pair_bits is None
                    else pair_bits
                ),
                program=runtime_program,
                spec=spec,
                descriptor_name=descriptor.name,
                descriptor_path=str(descriptor.path),
                descriptor_sha256=descriptor.sha256,
                descriptor_round_start=descriptor.round_start,
                descriptor_available_rounds=descriptor.available_rounds,
                conditioner_enabled=control != "no_conditioner",
            )

        model = build_ordered_adapter(program)
        if name in {
            "runtime_spn_k1by14_paired_correct",
            "runtime_spn_k1by14_paired_affine",
        }:
            counterfactual_program = compile_ordered_primitive_program(
                descriptor.structure
            )
            if control == "correct":
                counterfactual_program = permute_program_source_endpoints_affine(
                    counterfactual_program,
                    multiplier=int(options.get("affine_endpoint_multiplier", 5)),
                    offset=int(options.get("affine_endpoint_offset", 1)),
                )
            model.configure_runtime_contrast(
                orientation=str(options.get("runtime_contrast_orientation", "")),
                counterfactual_model=build_ordered_adapter(counterfactual_program),
                scale=float(options.get("runtime_contrast_scale", 0.25)),
                margin=float(options.get("runtime_contrast_margin", 0.02)),
            )
        return model
    position_histogram_models = {
        "runtime_spn_ct_k1t_position_histogram_true": "true",
        "runtime_spn_ct_k1t_position_histogram_wrong_sbox": "wrong_sbox",
        "runtime_spn_ct_k1t_position_histogram_invariant": "invariant",
        "runtime_spn_ct_k1w_compact_histogram_true": "true",
        "runtime_spn_ct_k1w_compact_histogram_wrong_sbox": "wrong_sbox",
        "runtime_spn_ct_k1y_compact_histogram_true": "true",
        "runtime_spn_ct_k1y_compact_histogram_wrong_sbox": "wrong_sbox",
        "runtime_spn_ct_k1aa_virtual_slot_histogram_true": "true",
        "runtime_spn_ct_k1aa_virtual_slot_histogram_wrong_sbox": "wrong_sbox",
        "runtime_spn_ct_k1aa_virtual_slot_histogram_corrupted_linear": (
            "corrupted_linear"
        ),
        "runtime_spn_ct_k1aa_virtual_slot_histogram_none": "none",
        "runtime_spn_ct_k1ak_sbox_transition_true": "true",
        "runtime_spn_ct_k1ak_sbox_transition_wrong_sbox": "wrong_sbox",
        "runtime_spn_ct_k1ak_sbox_transition_corrupted_linear": "corrupted_linear",
        "runtime_spn_ct_k1ak_sbox_transition_none": "none",
        "runtime_spn_ct_k1as_structure_gate_true": "true",
        "runtime_spn_ct_k1av_dual_path_structure_gate_true": "true",
        "runtime_spn_ct_k1ay_component_separated_structure_gate_true": "true",
        "runtime_spn_ct_k1an_walsh_transition_true": "true",
        "runtime_spn_ct_k1an_walsh_transition_wrong_sbox": "wrong_sbox",
        "runtime_spn_ct_k1an_walsh_transition_branch_off": "true",
    }
    if name in position_histogram_models:
        descriptor_path = options.get("runtime_structure_path")
        if not isinstance(descriptor_path, str) or not descriptor_path.strip():
            raise ValueError(
                f"model {name} requires non-empty model option runtime_structure_path"
            )
        runtime_rounds = int_option(options, "runtime_rounds", 2)
        runtime_round_start = int_option(options, "runtime_round_start", 0)
        assert runtime_rounds is not None
        assert runtime_round_start is not None
        descriptor = load_runtime_spn_descriptor(
            descriptor_path,
            rounds=runtime_rounds,
            round_start=runtime_round_start,
        )
        control = position_histogram_models[name]
        runtime_structure = descriptor.structure
        apply_sboxes = True
        if control == "wrong_sbox":
            try:
                runtime_structure = runtime_structure.shuffled_sbox_assignments(
                    20260728
                )
            except ValueError as exc:
                if "identical across all cells" not in str(exc):
                    raise
                tables = torch.stack(
                    [
                        runtime_structure.sbox_tables(slot)
                        for slot in range(runtime_structure.rounds)
                    ]
                )
                input_permutation = torch.roll(torch.arange(16), shifts=1)
                runtime_structure = runtime_spn_structure_from_truth_bits(
                    runtime_structure.cell_membership,
                    runtime_structure.bit_role,
                    _sbox_truth_bits_from_tables(tables[:, :, input_permutation]),
                    runtime_structure.linear_matrices,
                )
        elif control == "corrupted_linear":
            corruption_seed = int_option(options, "topology_corruption_seed", 20260728)
            assert corruption_seed is not None
            runtime_structure = runtime_structure.corrupted(corruption_seed)
        elif control == "none":
            identity = torch.eye(runtime_structure.block_bits, dtype=torch.uint8)
            runtime_structure = runtime_spn_structure_from_truth_bits(
                runtime_structure.cell_membership,
                runtime_structure.bit_role,
                runtime_structure.sbox_truth_bits,
                identity.unsqueeze(0).repeat(runtime_structure.rounds, 1, 1),
            )
            apply_sboxes = False
        pair_embedding_dim = int_option(options, "pair_embedding_dim", 128)
        histogram_value_dim = int_option(options, "histogram_value_dim", 8)
        assert pair_embedding_dim is not None
        assert histogram_value_dim is not None
        compact_model = name.startswith(
            (
                "runtime_spn_ct_k1w_",
                "runtime_spn_ct_k1y_",
                "runtime_spn_ct_k1aa_",
                "runtime_spn_ct_k1ak_",
                "runtime_spn_ct_k1an_",
                "runtime_spn_ct_k1as_",
                "runtime_spn_ct_k1av_",
                "runtime_spn_ct_k1ay_",
            )
        )
        if name.startswith(
            (
                "runtime_spn_ct_k1ak_",
                "runtime_spn_ct_k1an_",
                "runtime_spn_ct_k1as_",
                "runtime_spn_ct_k1av_",
                "runtime_spn_ct_k1ay_",
            )
        ):
            transition_value_dim = int_option(options, "transition_value_dim", 20)
            virtual_slots = int_option(options, "virtual_projection_slots", 16)
            assert transition_value_dim is not None
            assert virtual_slots is not None
            if name.startswith("runtime_spn_ct_k1ak_") and virtual_slots != 16:
                raise ValueError("K1-AK virtual_projection_slots must remain 16")
            canonical_walsh_features = None
            transition_branch_enabled = True
            if name.startswith("runtime_spn_ct_k1an_"):
                canonical_walsh_features = int_option(
                    options,
                    "canonical_walsh_features",
                    64,
                )
                if canonical_walsh_features != 64:
                    raise ValueError("K1-AN canonical_walsh_features must remain 64")
                transition_branch_enabled = not name.endswith("_branch_off")
            if name.startswith("runtime_spn_ct_k1ay_"):
                adapter = (
                    FixedComponentSeparatedDualPathSboxTransitionResidualSpnProtocolAdapter
                )
            elif name.startswith("runtime_spn_ct_k1av_"):
                adapter = (
                    FixedDualStructureConditionedSboxTransitionResidualSpnProtocolAdapter
                )
            elif name.startswith("runtime_spn_ct_k1as_"):
                adapter = FixedStructureConditionedSboxTransitionResidualSpnProtocolAdapter
            else:
                adapter = FixedCompactSboxTransitionResidualSpnProtocolAdapter
            adapter_kwargs = dict(
                input_bits=input_bits,
                pair_bits=(
                    2 * runtime_structure.block_bits if pair_bits is None else pair_bits
                ),
                structure=runtime_structure,
                spec=SboxTransitionResidualSpnSpec(
                    hidden_dim=hidden_bits,
                    pair_embedding_dim=pair_embedding_dim,
                    transition_value_dim=transition_value_dim,
                    dropout=float(options.get("dropout", 0.0)),
                    initial_edge_gate=float(
                        options.get("residual_gate_initial_effective", 0.05)
                    ),
                    initial_transition_gate=float(
                        options.get("transition_gate_initial_effective", 0.05)
                    ),
                    virtual_projection_slots=virtual_slots,
                ),
                descriptor_name=descriptor.name,
                descriptor_path=str(descriptor.path),
                descriptor_sha256=descriptor.sha256,
                descriptor_round_start=descriptor.round_start,
                descriptor_available_rounds=descriptor.available_rounds,
                runtime_structure_mode=control,
                apply_sboxes=apply_sboxes,
                canonical_walsh_features=canonical_walsh_features,
                transition_branch_enabled=transition_branch_enabled,
            )
            if name.startswith(
                (
                    "runtime_spn_ct_k1as_",
                    "runtime_spn_ct_k1av_",
                    "runtime_spn_ct_k1ay_",
                )
            ):
                adapter_kwargs.pop("canonical_walsh_features")
                adapter_kwargs.pop("transition_branch_enabled")
                adapter_kwargs["structure_gate_hidden_dim"] = int(
                    options.get("structure_gate_hidden_dim", 12)
                )
            return adapter(**adapter_kwargs)
        adapter = (
            FixedCompactInvariantHistogramResidualSpnProtocolAdapter
            if compact_model
            else FixedPositionHistogramResidualSpnProtocolAdapter
        )
        adapter_kwargs = {
            "input_bits": input_bits,
            "pair_bits": (
                2 * runtime_structure.block_bits if pair_bits is None else pair_bits
            ),
            "structure": runtime_structure,
            "spec": PositionHistogramResidualSpnSpec(
                hidden_dim=hidden_bits,
                pair_embedding_dim=pair_embedding_dim,
                histogram_value_dim=histogram_value_dim,
                dropout=float(options.get("dropout", 0.0)),
                initial_edge_gate=float(
                    options.get("residual_gate_initial_effective", 0.05)
                ),
                initial_histogram_gate=float(
                    options.get("histogram_gate_initial_effective", 0.05)
                ),
            ),
            "descriptor_name": descriptor.name,
            "descriptor_path": str(descriptor.path),
            "descriptor_sha256": descriptor.sha256,
            "descriptor_round_start": descriptor.round_start,
            "descriptor_available_rounds": descriptor.available_rounds,
            "runtime_structure_mode": control,
            "apply_sboxes": apply_sboxes,
        }
        if adapter is FixedPositionHistogramResidualSpnProtocolAdapter:
            adapter_kwargs["invariant_cells"] = control == "invariant"
        elif name.startswith("runtime_spn_ct_k1aa_"):
            virtual_slots = int_option(options, "virtual_projection_slots", 16)
            if virtual_slots != 16:
                raise ValueError("K1-AA virtual_projection_slots must remain 16")
            adapter_kwargs["virtual_projection_slots"] = virtual_slots
        model = adapter(
            **adapter_kwargs,
        )
        if name.startswith("runtime_spn_ct_k1y_"):
            multiplier = float(options.get("histogram_projection_lr_multiplier", 0.0))
            if multiplier != 16.0:
                raise ValueError(
                    "K1-Y histogram_projection_lr_multiplier must remain 16.0"
                )
            parameter_name = "backbone.histogram_projection.0.weight"
            model.optimizer_parameter_lr_multipliers = {parameter_name: multiplier}
            model.histogram_projection_lr_multiplier = multiplier
            model.histogram_projection_lr_parameter = parameter_name
        return model
    exact_composition_models = {
        "runtime_spn_ct_k1n_exact_composition_true": "true",
        "runtime_spn_ct_k1n_exact_composition_wrong_sbox": "wrong_sbox",
        "runtime_spn_ct_k1n_exact_composition_reversed_linear": "reversed_linear",
        "runtime_spn_ct_k1n_exact_composition_corrupted_linear": "corrupted_linear",
        "runtime_spn_ct_k1n_exact_composition_no_sbox": "no_sbox",
        "runtime_spn_ct_k1n_exact_composition_none": "none",
    }
    if name in exact_composition_models:
        descriptor_path = options.get("runtime_structure_path")
        if not isinstance(descriptor_path, str) or not descriptor_path.strip():
            raise ValueError(
                f"model {name} requires non-empty model option runtime_structure_path"
            )
        runtime_rounds = int_option(options, "runtime_rounds", 2)
        runtime_round_start = int_option(options, "runtime_round_start", 0)
        assert runtime_rounds is not None
        assert runtime_round_start is not None
        descriptor = load_runtime_spn_descriptor(
            descriptor_path,
            rounds=runtime_rounds,
            round_start=runtime_round_start,
        )
        control = exact_composition_models[name]
        runtime_structure = descriptor.structure
        apply_sboxes = True
        if control == "wrong_sbox":
            try:
                runtime_structure = runtime_structure.shuffled_sbox_assignments(
                    20260728
                )
            except ValueError as exc:
                if "identical across all cells" not in str(exc):
                    raise
                tables = torch.stack(
                    [
                        runtime_structure.sbox_tables(slot)
                        for slot in range(runtime_structure.rounds)
                    ]
                )
                input_permutation = torch.roll(torch.arange(16), shifts=1)
                runtime_structure = runtime_spn_structure_from_truth_bits(
                    runtime_structure.cell_membership,
                    runtime_structure.bit_role,
                    _sbox_truth_bits_from_tables(tables[:, :, input_permutation]),
                    runtime_structure.linear_matrices,
                )
        elif control == "reversed_linear":
            runtime_structure = runtime_spn_structure_from_truth_bits(
                runtime_structure.cell_membership,
                runtime_structure.bit_role,
                runtime_structure.sbox_truth_bits,
                runtime_structure.linear_matrices.flip(0),
            )
        elif control == "corrupted_linear":
            corruption_seed = int_option(options, "topology_corruption_seed", 20260728)
            assert corruption_seed is not None
            runtime_structure = runtime_structure.corrupted(corruption_seed)
        elif control == "no_sbox":
            apply_sboxes = False
        elif control == "none":
            identity = torch.eye(runtime_structure.block_bits, dtype=torch.uint8)
            runtime_structure = runtime_spn_structure_from_truth_bits(
                runtime_structure.cell_membership,
                runtime_structure.bit_role,
                runtime_structure.sbox_truth_bits,
                identity.unsqueeze(0).repeat(runtime_structure.rounds, 1, 1),
            )
            apply_sboxes = False
        pair_embedding_dim = int_option(options, "pair_embedding_dim", 128)
        assert pair_embedding_dim is not None
        return FixedExactOperatorCompositionSpnProtocolAdapter(
            input_bits=input_bits,
            pair_bits=(
                2 * runtime_structure.block_bits if pair_bits is None else pair_bits
            ),
            structure=runtime_structure,
            spec=TopologyEdgeResidualSpnSpec(
                hidden_dim=hidden_bits,
                pair_embedding_dim=pair_embedding_dim,
                dropout=float(options.get("dropout", 0.0)),
                initial_effective_gate=float(
                    options.get("residual_gate_initial_effective", 0.05)
                ),
            ),
            descriptor_name=descriptor.name,
            descriptor_path=str(descriptor.path),
            descriptor_sha256=descriptor.sha256,
            descriptor_round_start=descriptor.round_start,
            descriptor_available_rounds=descriptor.available_rounds,
            runtime_structure_mode=control,
            apply_sboxes=apply_sboxes,
        )
    edge_residual_models = {
        "runtime_spn_ct_k1k_edge_residual_true": "true",
        "runtime_spn_ct_k1k_edge_residual_reversed": "reversed",
        "runtime_spn_ct_k1k_edge_residual_corrupted": "corrupted",
        "runtime_spn_ct_k1k_edge_residual_none": "none",
    }
    if name in edge_residual_models:
        descriptor_path = options.get("runtime_structure_path")
        if not isinstance(descriptor_path, str) or not descriptor_path.strip():
            raise ValueError(
                f"model {name} requires non-empty model option runtime_structure_path"
            )
        runtime_rounds = int_option(options, "runtime_rounds", 2)
        runtime_round_start = int_option(options, "runtime_round_start", 0)
        assert runtime_rounds is not None
        assert runtime_round_start is not None
        descriptor = load_runtime_spn_descriptor(
            descriptor_path,
            rounds=runtime_rounds,
            round_start=runtime_round_start,
        )
        control = edge_residual_models[name]
        runtime_structure = descriptor.structure
        if control == "reversed":
            runtime_structure = runtime_structure.rotate_transitions()
        elif control == "corrupted":
            corruption_seed = int_option(options, "topology_corruption_seed", 20260728)
            assert corruption_seed is not None
            runtime_structure = runtime_structure.corrupted(corruption_seed)
        elif control == "none":
            identity = torch.eye(runtime_structure.block_bits, dtype=torch.uint8)
            runtime_structure = runtime_spn_structure_from_truth_bits(
                runtime_structure.cell_membership,
                runtime_structure.bit_role,
                runtime_structure.sbox_truth_bits,
                identity.unsqueeze(0).repeat(runtime_structure.rounds, 1, 1),
            )
        pair_embedding_dim = int_option(options, "pair_embedding_dim", 128)
        assert pair_embedding_dim is not None
        return FixedTopologyEdgeResidualSpnProtocolAdapter(
            input_bits=input_bits,
            pair_bits=(
                2 * runtime_structure.block_bits if pair_bits is None else pair_bits
            ),
            structure=runtime_structure,
            spec=TopologyEdgeResidualSpnSpec(
                hidden_dim=hidden_bits,
                pair_embedding_dim=pair_embedding_dim,
                dropout=float(options.get("dropout", 0.0)),
                initial_effective_gate=float(
                    options.get("residual_gate_initial_effective", 0.0)
                ),
            ),
            descriptor_name=descriptor.name,
            descriptor_path=str(descriptor.path),
            descriptor_sha256=descriptor.sha256,
            descriptor_round_start=descriptor.round_start,
            descriptor_available_rounds=descriptor.available_rounds,
            runtime_structure_mode=control,
            runtime_structure_window_control=control,
        )
    boolean_view_models = {
        "runtime_spn_ct_k1i_boolean_view_true": "true",
        "runtime_spn_ct_k1i_boolean_view_reversed": "reversed",
        "runtime_spn_ct_k1i_boolean_view_corrupted": "corrupted",
        "runtime_spn_ct_k1i_boolean_view_none": "none",
    }
    if name in boolean_view_models:
        descriptor_path = options.get("runtime_structure_path")
        if not isinstance(descriptor_path, str) or not descriptor_path.strip():
            raise ValueError(
                f"model {name} requires non-empty model option runtime_structure_path"
            )
        runtime_rounds = int_option(options, "runtime_rounds", 2)
        runtime_round_start = int_option(options, "runtime_round_start", 0)
        assert runtime_rounds is not None
        assert runtime_round_start is not None
        descriptor = load_runtime_spn_descriptor(
            descriptor_path,
            rounds=runtime_rounds,
            round_start=runtime_round_start,
        )
        control = boolean_view_models[name]
        runtime_structure = descriptor.structure
        if control == "reversed":
            runtime_structure = runtime_structure.rotate_transitions()
        elif control == "corrupted":
            corruption_seed = int_option(options, "topology_corruption_seed", 20260728)
            assert corruption_seed is not None
            runtime_structure = runtime_structure.corrupted(corruption_seed)
        elif control == "none":
            identity = torch.eye(runtime_structure.block_bits, dtype=torch.uint8)
            runtime_structure = runtime_spn_structure_from_truth_bits(
                runtime_structure.cell_membership,
                runtime_structure.bit_role,
                runtime_structure.sbox_truth_bits,
                identity.unsqueeze(0).repeat(runtime_structure.rounds, 1, 1),
            )
        pair_embedding_dim = int_option(options, "pair_embedding_dim", 128)
        assert pair_embedding_dim is not None
        return FixedGf2BooleanViewSpnProtocolAdapter(
            input_bits=input_bits,
            pair_bits=(
                2 * runtime_structure.block_bits if pair_bits is None else pair_bits
            ),
            structure=runtime_structure,
            spec=Gf2BooleanViewSpnSpec(
                hidden_dim=hidden_bits,
                pair_embedding_dim=pair_embedding_dim,
                dropout=float(options.get("dropout", 0.0)),
            ),
            descriptor_name=descriptor.name,
            descriptor_path=str(descriptor.path),
            descriptor_sha256=descriptor.sha256,
            descriptor_round_start=descriptor.round_start,
            descriptor_available_rounds=descriptor.available_rounds,
            runtime_structure_mode=control,
            runtime_structure_window_control=control,
        )
    operator_tied_models = {
        "runtime_spn_ct_k1h_operator_tied_true": "true",
        "runtime_spn_ct_k1h_operator_tied_reversed": "reversed",
        "runtime_spn_ct_k1h_operator_tied_corrupted": "corrupted",
        "runtime_spn_ct_k1h_operator_tied_none": "none",
    }
    if name in operator_tied_models:
        descriptor_path = options.get("runtime_structure_path")
        if not isinstance(descriptor_path, str) or not descriptor_path.strip():
            raise ValueError(
                f"model {name} requires non-empty model option runtime_structure_path"
            )
        runtime_rounds = int_option(options, "runtime_rounds", 2)
        runtime_round_start = int_option(options, "runtime_round_start", 0)
        assert runtime_rounds is not None
        assert runtime_round_start is not None
        descriptor = load_runtime_spn_descriptor(
            descriptor_path,
            rounds=runtime_rounds,
            round_start=runtime_round_start,
        )
        control = operator_tied_models[name]
        runtime_structure = descriptor.structure
        if control == "reversed":
            runtime_structure = runtime_structure.rotate_transitions()
        elif control == "corrupted":
            corruption_seed = int_option(options, "topology_corruption_seed", 20260728)
            assert corruption_seed is not None
            runtime_structure = runtime_structure.corrupted(corruption_seed)
        elif control == "none":
            identity = torch.eye(runtime_structure.block_bits, dtype=torch.uint8)
            runtime_structure = runtime_spn_structure_from_truth_bits(
                runtime_structure.cell_membership,
                runtime_structure.bit_role,
                runtime_structure.sbox_truth_bits,
                identity.unsqueeze(0).repeat(runtime_structure.rounds, 1, 1),
            )
        pair_embedding_dim = int_option(options, "pair_embedding_dim", 128)
        assert pair_embedding_dim is not None
        return FixedOperatorTiedLatentSpnProtocolAdapter(
            input_bits=input_bits,
            pair_bits=(
                2 * runtime_structure.block_bits if pair_bits is None else pair_bits
            ),
            structure=runtime_structure,
            spec=OperatorTiedLatentSpnSpec(
                hidden_dim=hidden_bits,
                pair_embedding_dim=pair_embedding_dim,
                dropout=float(options.get("dropout", 0.0)),
            ),
            descriptor_name=descriptor.name,
            descriptor_path=str(descriptor.path),
            descriptor_sha256=descriptor.sha256,
            descriptor_round_start=descriptor.round_start,
            descriptor_available_rounds=descriptor.available_rounds,
            runtime_structure_mode=control,
            runtime_structure_window_control=control,
        )
    cell_path_hypergraph_models = {
        "runtime_spn_ct_k1f_hypergraph_true": ("true", "true", False, "true"),
        "runtime_spn_ct_k1f_hypergraph_corrupted": (
            "true",
            "true",
            True,
            "corrupted",
        ),
        "runtime_spn_ct_k1f_hypergraph_independent": (
            "independent",
            "true",
            False,
            "independent",
        ),
        "runtime_spn_ct_k1f_hypergraph_incidence_shuffled": (
            "true",
            "shuffled",
            False,
            "true",
        ),
    }
    if name in cell_path_hypergraph_models:
        descriptor_path = options.get("runtime_structure_path")
        if not isinstance(descriptor_path, str) or not descriptor_path.strip():
            raise ValueError(
                f"model {name} requires non-empty model option runtime_structure_path"
            )
        runtime_rounds = int_option(options, "runtime_rounds", 2)
        assert runtime_rounds is not None
        runtime_round_start = int_option(options, "runtime_round_start", 0)
        assert runtime_round_start is not None
        descriptor = load_runtime_spn_descriptor(
            descriptor_path,
            rounds=runtime_rounds,
            round_start=runtime_round_start,
        )
        relation_mode, incidence_mode, corrupt, structure_mode = (
            cell_path_hypergraph_models[name]
        )
        runtime_structure = descriptor.structure
        if corrupt:
            corruption_seed = int_option(options, "topology_corruption_seed", 20260727)
            assert corruption_seed is not None
            runtime_structure = runtime_structure.corrupted(corruption_seed)
        runtime_structure, window_control = _apply_runtime_structure_window_control(
            runtime_structure,
            options,
        )
        processor_steps = int_option(options, "processor_steps", 2)
        pair_embedding_dim = int_option(options, "pair_embedding_dim", hidden_bits * 2)
        assert processor_steps is not None
        assert pair_embedding_dim is not None
        return FixedCellPathHypergraphSpnProtocolAdapter(
            input_bits=input_bits,
            pair_bits=(
                2 * runtime_structure.block_bits if pair_bits is None else pair_bits
            ),
            structure=runtime_structure,
            relation_mode=relation_mode,
            incidence_mode=incidence_mode,
            spec=CellPathHypergraphSpnSpec(
                hidden_dim=hidden_bits,
                pair_embedding_dim=pair_embedding_dim,
                processor_steps=processor_steps,
                dropout=float(options.get("dropout", 0.0)),
            ),
            descriptor_name=descriptor.name,
            descriptor_path=str(descriptor.path),
            descriptor_sha256=descriptor.sha256,
            descriptor_round_start=descriptor.round_start,
            descriptor_available_rounds=descriptor.available_rounds,
            runtime_structure_mode=structure_mode,
            runtime_structure_window_control=window_control,
        )
    relative_path_models = {
        "runtime_spn_ct_k1d_relative_path_true": ("true", False, "true"),
        "runtime_spn_ct_k1d_relative_path_corrupted": (
            "true",
            True,
            "corrupted",
        ),
        "runtime_spn_ct_k1d_relative_path_independent": (
            "independent",
            False,
            "independent",
        ),
    }
    if name in relative_path_models:
        descriptor_path = options.get("runtime_structure_path")
        if not isinstance(descriptor_path, str) or not descriptor_path.strip():
            raise ValueError(
                f"model {name} requires non-empty model option runtime_structure_path"
            )
        runtime_rounds = int_option(options, "runtime_rounds", 2)
        assert runtime_rounds is not None
        runtime_round_start = int_option(options, "runtime_round_start", 0)
        assert runtime_round_start is not None
        descriptor = load_runtime_spn_descriptor(
            descriptor_path,
            rounds=runtime_rounds,
            round_start=runtime_round_start,
        )
        relation_mode, corrupt, structure_mode = relative_path_models[name]
        runtime_structure = descriptor.structure
        if corrupt:
            corruption_seed = int_option(options, "topology_corruption_seed", 20260727)
            assert corruption_seed is not None
            runtime_structure = runtime_structure.corrupted(corruption_seed)
        runtime_structure, window_control = _apply_runtime_structure_window_control(
            runtime_structure,
            options,
        )
        processor_steps = int_option(options, "processor_steps", 2)
        pair_embedding_dim = int_option(options, "pair_embedding_dim", hidden_bits * 2)
        assert processor_steps is not None
        assert pair_embedding_dim is not None
        return FixedRelativePathSpnProtocolAdapter(
            input_bits=input_bits,
            pair_bits=(
                2 * runtime_structure.block_bits if pair_bits is None else pair_bits
            ),
            structure=runtime_structure,
            relation_mode=relation_mode,
            spec=RelativePathSpnSpec(
                hidden_dim=hidden_bits,
                pair_embedding_dim=pair_embedding_dim,
                processor_steps=processor_steps,
                dropout=float(options.get("dropout", 0.0)),
            ),
            descriptor_name=descriptor.name,
            descriptor_path=str(descriptor.path),
            descriptor_sha256=descriptor.sha256,
            descriptor_round_start=descriptor.round_start,
            descriptor_available_rounds=descriptor.available_rounds,
            runtime_structure_mode=structure_mode,
            runtime_structure_window_control=window_control,
        )
    canonical_transition_models = {
        "runtime_spn_ct_k1_canonical_true": (
            "true",
            False,
            "true",
            "edge_invariant",
        ),
        "runtime_spn_ct_k1_canonical_corrupted": (
            "true",
            True,
            "corrupted",
            "edge_invariant",
        ),
        "runtime_spn_ct_k1_canonical_independent": (
            "independent",
            False,
            "independent",
            "edge_invariant",
        ),
        "runtime_spn_ct_k1b_endpoint_true": (
            "true",
            False,
            "true",
            "native_cell_role",
        ),
        "runtime_spn_ct_k1b_endpoint_corrupted": (
            "true",
            True,
            "corrupted",
            "native_cell_role",
        ),
        "runtime_spn_ct_k1b_endpoint_independent": (
            "independent",
            False,
            "independent",
            "native_cell_role",
        ),
    }
    if name in canonical_transition_models:
        descriptor_path = options.get("runtime_structure_path")
        if not isinstance(descriptor_path, str) or not descriptor_path.strip():
            raise ValueError(
                f"model {name} requires non-empty model option runtime_structure_path"
            )
        runtime_rounds = int_option(options, "runtime_rounds", 2)
        assert runtime_rounds is not None
        runtime_round_start = int_option(options, "runtime_round_start", 0)
        assert runtime_round_start is not None
        descriptor = load_runtime_spn_descriptor(
            descriptor_path,
            rounds=runtime_rounds,
            round_start=runtime_round_start,
        )
        relation_mode, corrupt, structure_mode, endpoint_identity_mode = (
            canonical_transition_models[name]
        )
        runtime_structure = descriptor.structure
        if corrupt:
            corruption_seed = int_option(options, "topology_corruption_seed", 20260727)
            assert corruption_seed is not None
            runtime_structure = runtime_structure.corrupted(corruption_seed)
        runtime_structure, window_control = _apply_runtime_structure_window_control(
            runtime_structure,
            options,
        )
        processor_steps = int_option(options, "processor_steps", 2)
        pair_embedding_dim = int_option(options, "pair_embedding_dim", hidden_bits * 2)
        temporal_hidden_dim = int_option(options, "temporal_hidden_dim", 76)
        assert processor_steps is not None
        assert pair_embedding_dim is not None
        assert temporal_hidden_dim is not None
        return FixedCanonicalTransitionSpnProtocolAdapter(
            input_bits=input_bits,
            pair_bits=(
                2 * runtime_structure.block_bits if pair_bits is None else pair_bits
            ),
            structure=runtime_structure,
            relation_mode=relation_mode,
            spec=CanonicalTransitionSpnSpec(
                hidden_dim=hidden_bits,
                pair_embedding_dim=pair_embedding_dim,
                processor_steps=processor_steps,
                temporal_hidden_dim=temporal_hidden_dim,
                dropout=float(options.get("dropout", 0.0)),
                endpoint_identity_mode=endpoint_identity_mode,
            ),
            descriptor_name=descriptor.name,
            descriptor_path=str(descriptor.path),
            descriptor_sha256=descriptor.sha256,
            descriptor_round_start=descriptor.round_start,
            descriptor_available_rounds=descriptor.available_rounds,
            runtime_structure_mode=structure_mode,
            runtime_structure_window_control=window_control,
            canonical_schedule_control=str(
                options.get("canonical_schedule_control", "ordered")
            ),
        )
    external_runtime_models = {
        "runtime_spn_e4_equivariant_true": ("true", False, False, "true"),
        "runtime_spn_e4_equivariant_corrupted": (
            "true",
            True,
            False,
            "corrupted",
        ),
        "runtime_spn_e4_equivariant_sbox_shuffled": (
            "true",
            False,
            True,
            "sbox_shuffled",
        ),
        "runtime_spn_e4_equivariant_independent": (
            "independent",
            False,
            False,
            "independent",
        ),
        "runtime_spn_e5_gated_residual_true": ("true", False, False, "true"),
        "runtime_spn_e5_gated_residual_corrupted": (
            "true",
            True,
            False,
            "corrupted",
        ),
        "runtime_spn_e5_gated_residual_independent": (
            "independent",
            False,
            False,
            "independent",
        ),
    }
    if name in external_runtime_models:
        descriptor_path = options.get("runtime_structure_path")
        if not isinstance(descriptor_path, str) or not descriptor_path.strip():
            raise ValueError(
                f"model {name} requires non-empty model option runtime_structure_path"
            )
        processor_steps = int_option(options, "processor_steps", 2)
        assert processor_steps is not None
        runtime_rounds = int_option(options, "runtime_rounds", processor_steps)
        assert runtime_rounds is not None
        if runtime_rounds <= 0:
            raise ValueError("runtime_rounds must be positive")
        runtime_round_start = int_option(options, "runtime_round_start", 0)
        assert runtime_round_start is not None
        descriptor = load_runtime_spn_descriptor(
            descriptor_path,
            rounds=runtime_rounds,
            round_start=runtime_round_start,
        )
        relation_mode, corrupt, shuffle_sboxes, structure_mode = (
            external_runtime_models[name]
        )
        runtime_structure = descriptor.structure
        if corrupt:
            corruption_seed = int_option(options, "topology_corruption_seed", 20260724)
            assert corruption_seed is not None
            runtime_structure = runtime_structure.corrupted(corruption_seed)
        if shuffle_sboxes:
            shuffle_seed = int_option(options, "sbox_assignment_shuffle_seed", 20260724)
            assert shuffle_seed is not None
            runtime_structure = runtime_structure.shuffled_sbox_assignments(
                shuffle_seed
            )
        runtime_structure, window_control = _apply_runtime_structure_window_control(
            runtime_structure,
            options,
        )
        pair_embedding_dim = int_option(options, "pair_embedding_dim", hidden_bits * 2)
        assert pair_embedding_dim is not None
        aggregation_mode = (
            "e5_gated_residual"
            if name.startswith("runtime_spn_e5_gated_residual_")
            else "e4_equivariant"
        )
        return FixedRuntimeSpnProtocolAdapter(
            input_bits=input_bits,
            pair_bits=(
                2 * runtime_structure.block_bits if pair_bits is None else pair_bits
            ),
            structure=runtime_structure,
            relation_mode=relation_mode,
            spec=RuntimeParameterizedSpnSpec(
                hidden_dim=hidden_bits,
                pair_embedding_dim=pair_embedding_dim,
                processor_steps=processor_steps,
                dropout=float(options.get("dropout", 0.0)),
                sbox_context_scale=float(options.get("sbox_context_scale", 1.0)),
                sbox_context_mode=str(options.get("sbox_context_mode", "early_add")),
                cell_input_mode=str(options.get("cell_input_mode", "difference_only")),
                round_window_mode=str(
                    options.get("round_window_mode", "last_transition")
                ),
            ),
            aggregation_mode=aggregation_mode,
            descriptor_name=descriptor.name,
            descriptor_path=str(descriptor.path),
            descriptor_sha256=descriptor.sha256,
            descriptor_round_start=descriptor.round_start,
            descriptor_available_rounds=descriptor.available_rounds,
            runtime_structure_mode=structure_mode,
            runtime_structure_window_control=window_control,
        )
    runtime_models = {
        "present_runtime_spn_true": (present_runtime_structure, "true", False),
        "present_runtime_spn_corrupted": (
            present_runtime_structure,
            "true",
            True,
        ),
        "present_runtime_spn_independent": (
            present_runtime_structure,
            "independent",
            False,
        ),
        "present_runtime_e4_equivariant_true": (
            present_runtime_structure,
            "true",
            False,
        ),
        "present_runtime_e4_equivariant_corrupted": (
            present_runtime_structure,
            "true",
            True,
        ),
        "present_runtime_e4_equivariant_independent": (
            present_runtime_structure,
            "independent",
            False,
        ),
        "gift64_runtime_spn_true": (gift64_runtime_structure, "true", False),
        "gift64_runtime_spn_corrupted": (
            gift64_runtime_structure,
            "true",
            True,
        ),
        "gift64_runtime_spn_independent": (
            gift64_runtime_structure,
            "independent",
            False,
        ),
        "gift64_runtime_cell_token_true": (
            gift64_runtime_structure,
            "true",
            False,
        ),
        "gift64_runtime_cell_token_corrupted": (
            gift64_runtime_structure,
            "true",
            True,
        ),
        "gift64_runtime_e4_equivariant_true": (
            gift64_runtime_structure,
            "true",
            False,
        ),
        "gift64_runtime_e4_equivariant_corrupted": (
            gift64_runtime_structure,
            "true",
            True,
        ),
        "gift64_runtime_e4_equivariant_independent": (
            gift64_runtime_structure,
            "independent",
            False,
        ),
        "skinny64_runtime_e4_equivariant_true": (
            skinny64_runtime_structure,
            "true",
            False,
        ),
        "skinny64_runtime_e4_equivariant_corrupted": (
            skinny64_runtime_structure,
            "true",
            True,
        ),
        "skinny64_runtime_e4_equivariant_independent": (
            skinny64_runtime_structure,
            "independent",
            False,
        ),
    }
    if name in runtime_models:
        structure_factory, relation_mode, corrupt = runtime_models[name]
        processor_steps = int_option(options, "processor_steps", 2) or 2
        runtime_rounds = int_option(options, "runtime_rounds", processor_steps)
        assert runtime_rounds is not None
        if runtime_rounds <= 0:
            raise ValueError("runtime_rounds must be positive")
        runtime_structure = structure_factory(runtime_rounds)
        if corrupt:
            runtime_structure = runtime_structure.corrupted()
        runtime_structure, window_control = _apply_runtime_structure_window_control(
            runtime_structure,
            options,
        )
        return FixedRuntimeSpnProtocolAdapter(
            input_bits=input_bits,
            pair_bits=128 if pair_bits is None else pair_bits,
            structure=runtime_structure,
            relation_mode=relation_mode,
            spec=RuntimeParameterizedSpnSpec(
                hidden_dim=hidden_bits,
                pair_embedding_dim=(
                    int_option(options, "pair_embedding_dim", hidden_bits * 2)
                    or hidden_bits * 2
                ),
                processor_steps=processor_steps,
                dropout=float(options.get("dropout", 0.0)),
                sbox_context_scale=float(options.get("sbox_context_scale", 1.0)),
                sbox_context_mode=str(options.get("sbox_context_mode", "early_add")),
                cell_input_mode=str(options.get("cell_input_mode", "difference_only")),
                round_window_mode=str(
                    options.get("round_window_mode", "last_transition")
                ),
            ),
            aggregation_mode=(
                "e4_equivariant"
                if "runtime_e4_equivariant" in name
                else "cell_pair"
                if "runtime_cell_token" in name
                else "bit_pair"
            ),
            runtime_structure_window_control=window_control,
        )
    cross_spn_typed_models = {
        "present_cross_spn_typed_cell_true": PresentCrossSpnTypedCellTrueDistinguisher,
        "present_cross_spn_typed_cell_shuffled": PresentCrossSpnTypedCellShuffledDistinguisher,
        "present_cross_spn_typed_cell_raw": PresentCrossSpnTypedCellRawDistinguisher,
        "gift_cross_spn_typed_cell_true": GiftCrossSpnTypedCellTrueDistinguisher,
        "gift_cross_spn_typed_cell_no_position": GiftCrossSpnTypedCellNoPositionDistinguisher,
        "gift_cross_spn_typed_cell_shared_view_encoder": GiftCrossSpnTypedCellSharedViewEncoderDistinguisher,
        "gift_cross_spn_typed_cell_equivariant_mixer": GiftCrossSpnTypedCellEquivariantMixerDistinguisher,
        "gift_cross_spn_typed_cell_shuffled": GiftCrossSpnTypedCellShuffledDistinguisher,
        "gift_cross_spn_typed_cell_raw": GiftCrossSpnTypedCellRawDistinguisher,
        "gift_cross_spn_typed_cell_true_from_present_true": GiftCrossSpnTypedCellTrueFromPresentTrueDistinguisher,
        "gift_cross_spn_typed_cell_true_from_present_true_s0": GiftCrossSpnTypedCellTrueFromPresentTrueDistinguisher,
        "gift_cross_spn_typed_cell_true_from_present_true_s1": GiftCrossSpnTypedCellTrueFromPresentTrueDistinguisher,
        "gift_cross_spn_typed_cell_true_from_present_shuffled": GiftCrossSpnTypedCellTrueFromPresentShuffledDistinguisher,
        "gift_cross_spn_typed_cell_shuffled_from_present_true": GiftCrossSpnTypedCellShuffledFromPresentTrueDistinguisher,
        "present_cross_spn_typed_cell_e5_off": PresentCrossSpnTypedCellE5OffDistinguisher,
        "present_cross_spn_typed_cell_e5_true_shuffled": PresentCrossSpnTypedCellE5TrueShuffledDistinguisher,
        "present_cross_spn_typed_cell_e5_shuffled_placebo": PresentCrossSpnTypedCellE5ShuffledPlaceboDistinguisher,
        "gift_cross_spn_typed_cell_e5_scratch": GiftCrossSpnTypedCellE5ScratchDistinguisher,
        "gift_cross_spn_typed_cell_e5_from_present_off": GiftCrossSpnTypedCellE5FromPresentOffDistinguisher,
        "gift_cross_spn_typed_cell_e5_from_present_true_shuffled": GiftCrossSpnTypedCellE5FromPresentTrueShuffledDistinguisher,
        "gift_cross_spn_typed_cell_e5_from_present_shuffled_placebo": GiftCrossSpnTypedCellE5FromPresentShuffledPlaceboDistinguisher,
        "present_cross_spn_typed_cell_e6_off": PresentCrossSpnTypedCellE6OffDistinguisher,
        "present_cross_spn_typed_cell_e6_functional_margin": PresentCrossSpnTypedCellE6FunctionalMarginDistinguisher,
        "present_cross_spn_typed_cell_e6_shuffled_placebo": PresentCrossSpnTypedCellE6ShuffledPlaceboDistinguisher,
        "gift_cross_spn_typed_cell_e6_scratch": GiftCrossSpnTypedCellE6ScratchDistinguisher,
        "gift_cross_spn_typed_cell_e6_from_present_off": GiftCrossSpnTypedCellE6FromPresentOffDistinguisher,
        "gift_cross_spn_typed_cell_e6_from_present_functional_margin": GiftCrossSpnTypedCellE6FromPresentFunctionalMarginDistinguisher,
        "gift_cross_spn_typed_cell_e6_from_present_shuffled_placebo": GiftCrossSpnTypedCellE6FromPresentShuffledPlaceboDistinguisher,
    }
    if name in cross_spn_typed_models:
        return cross_spn_typed_models[name](
            input_bits=input_bits,
            pair_bits=128 if pair_bits is None else pair_bits,
            base_channels=hidden_bits,
            token_dim=int_option(options, "token_dim"),
            mixer_depth=int_option(options, "mixer_depth", 2),
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2),
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            pooling=str(options.get("pooling", "attention_mean_max")),
            dropout=float(options.get("dropout", 0.0)),
            position_mode=str(
                options.get(
                    "position_mode",
                    "zero"
                    if name
                    in {
                        "gift_cross_spn_typed_cell_no_position",
                        "gift_cross_spn_typed_cell_shared_view_encoder",
                        "gift_cross_spn_typed_cell_equivariant_mixer",
                    }
                    else "learned",
                )
            ),
            view_encoder_mode=str(
                options.get(
                    "view_encoder_mode",
                    "shared_current"
                    if name == "gift_cross_spn_typed_cell_shared_view_encoder"
                    or name == "gift_cross_spn_typed_cell_equivariant_mixer"
                    else "separate",
                )
            ),
            cell_mixer_mode=str(
                options.get(
                    "cell_mixer_mode",
                    "equivariant"
                    if name == "gift_cross_spn_typed_cell_equivariant_mixer"
                    else "fixed",
                )
            ),
            topology_auxiliary_scale=float(
                options.get("topology_auxiliary_scale", 0.1)
            ),
            topology_functional_margin=float(
                options.get("topology_functional_margin", 0.01)
            ),
        )
    if name == "gift64_sun_style_lstm_pairset":
        return Gift64SunStyleLstmPairSetDistinguisher(
            input_bits=input_bits,
            pair_bits=128 if pair_bits is None else pair_bits,
            hidden_bits=int_option(options, "lstm_hidden_bits", 128) or 128,
            classifier_bits=int_option(options, "classifier_bits", 128) or 128,
        )
    if name == "gift64_gohr_style_resnet_pairset":
        return Gift64GohrStyleResNetPairSetDistinguisher(
            input_bits=input_bits,
            pair_bits=128 if pair_bits is None else pair_bits,
            channels=int_option(options, "resnet_channels", 64) or 64,
            blocks=int_option(options, "resnet_blocks", 7) or 7,
            classifier_bits=int_option(options, "classifier_bits", 128) or 128,
        )
    if name == "gift_cross_spn_aligned_token_mixer_raw_anchor":
        return GiftAlignedTokenMixerRawInputDistinguisher(
            input_bits=input_bits,
            pair_bits=128 if pair_bits is None else pair_bits,
            base_channels=hidden_bits,
            token_dim=int_option(options, "token_dim"),
            mixer_depth=int_option(options, "mixer_depth", 1),
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2),
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            dropout=float(options.get("dropout", 0.0)),
            top_k=int_option(options, "top_k", 2),
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "present_zhang_wang_keras_mcnd":
        return PresentZhangWangKerasMCNDDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            blocks=int_option(options, "blocks", 5) or 5,
            activation=str(options.get("activation", "relu")),
            dropout=float(options.get("dropout", 0.0)),
            initial_kernel_sizes=int_tuple_option(
                options, "initial_kernel_sizes", (1, 2, 4)
            ),
            residual_kernel_size=int_option(options, "residual_kernel_size", 3) or 3,
        )
    if name == "spn_zhang_wang_mcnd_adapter":
        return SpnZhangWangMCNDAdapterDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            blocks=int_option(options, "blocks", 5) or 5,
            activation=str(options.get("activation", "relu")),
            dropout=float(options.get("dropout", 0.0)),
            initial_kernel_sizes=int_tuple_option(
                options, "initial_kernel_sizes", (1, 2, 4)
            ),
            residual_kernel_size=int_option(options, "residual_kernel_size", 3) or 3,
            cell_bits=int_option(options, "cell_bits", 4) or 4,
        )
    if name == "spn_liu_case3_conv2d_adapter":
        return SpnLiuCase3Conv2DAdapterDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            cell_bits=int_option(options, "cell_bits", 4) or 4,
            conv_depth=int_option(options, "conv_depth", 3),
            kernel_size=int_option(options, "kernel_size", 3),
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "batchnorm2d")),
            dropout=float(options.get("dropout", 0.0)),
        )
    if name == "present_nibble_paligned_mcnd":
        return PresentNibblePAlignedMCNDDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            blocks=int_option(options, "blocks", 5) or 5,
            spn_token_dim=int_option(options, "spn_token_dim"),
            spn_mixer_depth=int_option(options, "spn_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            initial_kernel_sizes=int_tuple_option(
                options, "initial_kernel_sizes", (1, 2, 4)
            ),
            residual_kernel_size=int_option(options, "residual_kernel_size", 3) or 3,
        )
    if name == "present_nibble_paligned_spn_only":
        return PresentNibblePAlignedSpnOnlyDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            spn_token_dim=int_option(options, "spn_token_dim"),
            spn_mixer_depth=int_option(options, "spn_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
        )
    if name == "present_nibble_delta_only_spn_only":
        return PresentNibbleDeltaOnlySpnOnlyDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            spn_token_dim=int_option(options, "spn_token_dim"),
            spn_mixer_depth=int_option(options, "spn_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
        )
    if name == "present_nibble_invp_only_spn_only":
        return PresentNibbleInvPOnlySpnOnlyDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            spn_token_dim=int_option(options, "spn_token_dim"),
            spn_mixer_depth=int_option(options, "spn_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
        )
    state_matrix_conv2d_models = {
        "present_nibble_invp_state_matrix_conv2d_spn_only": (
            PresentNibbleInvPStateMatrixConv2DSpnOnlyDistinguisher
        ),
        "present_nibble_shuffled_p_state_matrix_conv2d_spn_only": (
            PresentNibbleShuffledPStateMatrixConv2DSpnOnlyDistinguisher
        ),
        "present_nibble_delta_state_matrix_conv2d_spn_only": (
            PresentNibbleDeltaStateMatrixConv2DSpnOnlyDistinguisher
        ),
    }
    if name in state_matrix_conv2d_models:
        return state_matrix_conv2d_models[name](
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            conv_depth=int_option(options, "conv_depth", 3),
            kernel_size=int_option(options, "kernel_size", 3),
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "batchnorm2d")),
            dropout=float(options.get("dropout", 0.0)),
        )
    topology_residual_models = {
        "present_nibble_invp_topology_residual_spn_only": (
            PresentNibbleInvPTopologyResidualSpnOnlyDistinguisher
        ),
        "present_nibble_shuffled_p_topology_residual_spn_only": (
            PresentNibbleShuffledPTopologyResidualSpnOnlyDistinguisher
        ),
        "present_nibble_delta_topology_residual_spn_only": (
            PresentNibbleDeltaTopologyResidualSpnOnlyDistinguisher
        ),
    }
    if name in topology_residual_models:
        return topology_residual_models[name](
            input_bits=input_bits,
            pair_bits=128 if pair_bits is None else pair_bits,
            base_channels=hidden_bits,
            spn_token_dim=int_option(options, "spn_token_dim"),
            spn_mixer_depth=int_option(options, "spn_mixer_depth", 2),
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2),
            local_channels=int_option(options, "local_channels", 16),
            local_depth=int_option(options, "local_depth", 1),
            local_kernel_size=int_option(options, "local_kernel_size", 3),
            local_residual_scale_init=float(
                options.get("local_residual_scale_init", 0.1)
            ),
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            local_norm=str(options.get("local_norm", "batchnorm2d")),
            dropout=float(options.get("dropout", 0.0)),
        )
    case3_topology_residual_models = {
        "present_nibble_case3_invp_topology_residual_spn_only": (
            PresentNibbleCase3InvPTopologyResidualSpnOnlyDistinguisher
        ),
        "present_nibble_case3_shuffled_p_topology_residual_spn_only": (
            PresentNibbleCase3ShuffledPTopologyResidualSpnOnlyDistinguisher
        ),
        "present_nibble_case3_raw_topology_residual_spn_only": (
            PresentNibbleCase3RawTopologyResidualSpnOnlyDistinguisher
        ),
    }
    if name in case3_topology_residual_models:
        return case3_topology_residual_models[name](
            input_bits=input_bits,
            pair_bits=128 if pair_bits is None else pair_bits,
            base_channels=hidden_bits,
            spn_token_dim=int_option(options, "spn_token_dim"),
            spn_mixer_depth=int_option(options, "spn_mixer_depth", 2),
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2),
            local_channels=int_option(options, "local_channels", 16),
            local_depth=int_option(options, "local_depth", 1),
            local_kernel_size=int_option(options, "local_kernel_size", 3),
            local_residual_scale_init=float(
                options.get("local_residual_scale_init", 0.1)
            ),
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            local_norm=str(options.get("local_norm", "batchnorm2d")),
            dropout=float(options.get("dropout", 0.0)),
        )
    same_input_dbitnet_models = {
        "present_invp_dbitnet2023": PresentInvPDBitNet2023Distinguisher,
        "present_shuffled_p_dbitnet2023": PresentShuffledPDBitNet2023Distinguisher,
        "present_raw_delta_dbitnet2023": PresentRawDeltaDBitNet2023Distinguisher,
    }
    if name in same_input_dbitnet_models:
        return same_input_dbitnet_models[name](
            input_bits=input_bits,
            pair_bits=128 if pair_bits is None else pair_bits,
        )
    if name == "present_nibble_invp_active_aux_spn_only":
        return PresentNibbleInvPActiveAuxSpnOnlyDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            spn_token_dim=int_option(options, "spn_token_dim"),
            spn_mixer_depth=int_option(options, "spn_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
        )
    if name == "present_nibble_invp_sbox_prior_gate":
        return PresentNibbleInvPSboxPriorGateDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            prior_token_dim=int_option(options, "prior_token_dim"),
            prior_mixer_depth=int_option(options, "prior_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            gate_scale=float(options.get("gate_scale", 0.25)),
        )
    if name == "present_nibble_invp_no_ddt_gate":
        return PresentNibbleInvPNoDDTGateDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            prior_token_dim=int_option(options, "prior_token_dim"),
            prior_mixer_depth=int_option(options, "prior_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            gate_scale=float(options.get("gate_scale", 0.25)),
        )
    if name == "present_nibble_invp_shuffled_sbox_prior_gate":
        return PresentNibbleInvPShuffledSboxPriorGateDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            prior_token_dim=int_option(options, "prior_token_dim"),
            prior_mixer_depth=int_option(options, "prior_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            gate_scale=float(options.get("gate_scale", 0.25)),
        )
    if name == "present_nibble_invp_pair_consistency_spn_only":
        return PresentNibbleInvPPairConsistencySpnOnlyDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            spn_token_dim=int_option(options, "spn_token_dim"),
            spn_mixer_depth=int_option(options, "spn_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "present_nibble_invp_pair_mixer_consistency_spn_only":
        return PresentNibbleInvPPairMixerConsistencySpnOnlyDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            spn_token_dim=int_option(options, "spn_token_dim"),
            spn_mixer_depth=int_option(options, "spn_mixer_depth", 2) or 2,
            pair_mixer_depth=int_option(options, "pair_mixer_depth", 1) or 1,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "present_nibble_shuffled_paligned_spn_only":
        return PresentNibbleShuffledPAlignedSpnOnlyDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            spn_token_dim=int_option(options, "spn_token_dim"),
            spn_mixer_depth=int_option(options, "spn_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
        )
    if name == "present_nibble_paligned_gated_mcnd":
        return PresentNibblePAlignedGatedMCNDDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            blocks=int_option(options, "blocks", 5) or 5,
            spn_token_dim=int_option(options, "spn_token_dim"),
            spn_mixer_depth=int_option(options, "spn_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            initial_kernel_sizes=int_tuple_option(
                options, "initial_kernel_sizes", (1, 2, 4)
            ),
            residual_kernel_size=int_option(options, "residual_kernel_size", 3) or 3,
            gate_scale=float(options.get("gate_scale", 0.25)),
        )
    if name == "present_nibble_shuffled_paligned_gated_mcnd":
        return PresentNibbleShuffledPAlignedGatedMCNDDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            blocks=int_option(options, "blocks", 5) or 5,
            spn_token_dim=int_option(options, "spn_token_dim"),
            spn_mixer_depth=int_option(options, "spn_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            initial_kernel_sizes=int_tuple_option(
                options, "initial_kernel_sizes", (1, 2, 4)
            ),
            residual_kernel_size=int_option(options, "residual_kernel_size", 3) or 3,
            gate_scale=float(options.get("gate_scale", 0.25)),
        )
    if name == "present_nibble_paligned_transition":
        return PresentNibblePAlignedTransitionDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            spn_token_dim=int_option(options, "spn_token_dim"),
            spn_mixer_depth=int_option(options, "spn_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "present_nibble_paligned_transition_residual":
        return PresentNibblePAlignedTransitionResidualDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            transition_token_dim=int_option(options, "transition_token_dim"),
            transition_mixer_depth=int_option(options, "transition_mixer_depth", 2)
            or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "present_nibble_shuffled_transition_residual":
        return PresentNibbleShuffledTransitionResidualDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            transition_token_dim=int_option(options, "transition_token_dim"),
            transition_mixer_depth=int_option(options, "transition_mixer_depth", 2)
            or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "present_nibble_ddt_graph":
        return PresentNibbleDDTGraphDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            ddt_token_dim=int_option(options, "ddt_token_dim"),
            ddt_mixer_depth=int_option(options, "ddt_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "present_nibble_no_ddt_graph":
        return PresentNibbleNoDDTGraphDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            ddt_token_dim=int_option(options, "ddt_token_dim"),
            ddt_mixer_depth=int_option(options, "ddt_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "present_nibble_shuffled_ddt_graph":
        return PresentNibbleShuffledDDTGraphDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            ddt_token_dim=int_option(options, "ddt_token_dim"),
            ddt_mixer_depth=int_option(options, "ddt_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "present_nibble_invp_p_layer_graph_spn_only":
        return PresentNibbleInvPPLayerGraphSpnOnlyDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            graph_token_dim=int_option(options, "graph_token_dim"),
            graph_mixer_depth=int_option(options, "graph_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "present_nibble_invp_shuffled_p_layer_graph_spn_only":
        return PresentNibbleInvPShuffledPLayerGraphSpnOnlyDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            graph_token_dim=int_option(options, "graph_token_dim"),
            graph_mixer_depth=int_option(options, "graph_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "relu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "present_inception_mcnd":
        return PresentInceptionMCNDDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            branches=int_option(options, "branches"),
            blocks=int_option(options, "blocks", 3) or 3,
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "batchnorm1d")),
            pooling=str(options.get("pooling", "attention_mean_max")),
            dropout=float(options.get("dropout", 0.0)),
            kernel_sizes=int_tuple_option(options, "kernel_sizes", (1, 3, 5)),
        )
    if name == "present_inception_mcnd_matrix":
        return PresentInceptionMCNDMatrixDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            branches=int_option(options, "branches"),
            blocks=int_option(options, "blocks", 3) or 3,
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "batchnorm2d")),
            pooling=str(options.get("pooling", "attention_mean_max")),
            dropout=float(options.get("dropout", 0.0)),
            kernel_sizes=tuple(
                matrix_kernel_size_option(item)
                for item in options.get("kernel_sizes", [[1, 1], [1, 2], [2, 4]])
            ),
            cell_bits=int_option(options, "cell_bits", 4) or 4,
        )
    if name == "present_inception_mcnd_global_matrix":
        return PresentInceptionMCNDGlobalMatrixDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            branches=int_option(options, "branches"),
            blocks=int_option(options, "blocks", 3) or 3,
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "batchnorm2d")),
            dropout=float(options.get("dropout", 0.0)),
            kernel_sizes=tuple(
                matrix_kernel_size_option(item)
                for item in options.get("kernel_sizes", [[1, 1], [1, 2], [2, 4]])
            ),
            cell_bits=int_option(options, "cell_bits", 4) or 4,
        )
    if name == "present_inception_mcnd_pair_stack_matrix":
        return PresentInceptionMCNDPairStackMatrixDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            branches=int_option(options, "branches"),
            blocks=int_option(options, "blocks", 3) or 3,
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "batchnorm2d")),
            dropout=float(options.get("dropout", 0.0)),
            kernel_sizes=tuple(
                matrix_kernel_size_option(item)
                for item in options.get(
                    "kernel_sizes", [[1, 1], [1, 2], [2, 4], [4, 4]]
                )
            ),
            cell_bits=int_option(options, "cell_bits", 4) or 4,
        )
    if name == "spn_pairset_dbitnet_v2":
        return SpnCellPairSetDBitNetDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 192,
            base_channels=hidden_bits,
        )
    if name == "spn_nibble_conv_pairset":
        return SpnNibbleConvPairSetDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 192,
            base_channels=hidden_bits,
            nibble_embed_dim=int_option(options, "nibble_embed_dim"),
            conv_depth=int_option(options, "conv_depth", 3),
            kernel_size=int_option(options, "kernel_size", 3),
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            pooling=str(options.get("pooling", "attention_mean_max")),
            dropout=float(options.get("dropout", 0.0)),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "spn_token_mixer_pairset":
        return SpnTokenMixerPairSetDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 192,
            base_channels=hidden_bits,
            token_dim=int_option(options, "token_dim"),
            mixer_depth=int_option(options, "mixer_depth", 3),
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2),
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            pooling=str(options.get("pooling", "attention_mean_max")),
            dropout=float(options.get("dropout", 0.0)),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "present_trail_mixer_pairset":
        return PresentTrailMixerPairSetDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 768,
            base_channels=hidden_bits,
            token_dim=int_option(options, "token_dim"),
            mixer_depth=int_option(options, "mixer_depth", 3) or 3,
            role_mixer_depth=int_option(options, "role_mixer_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            dropout=float(options.get("dropout", 0.0)),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name in {
        "present_matrix_trail_hybrid_pairset",
        "present_matrix_trail_hybrid_pairset_invp",
        "present_matrix_trail_hybrid_pairset_invp_sinv",
    }:
        return PresentMatrixTrailHybridPairSetDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 768,
            base_channels=hidden_bits,
            token_dim=int_option(options, "token_dim"),
            mixer_depth=int_option(options, "mixer_depth", 3) or 3,
            role_mixer_depth=int_option(options, "role_mixer_depth", 2) or 2,
            matrix_depth=int_option(options, "matrix_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            dropout=float(options.get("dropout", 0.0)),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
        )
    if name == "present_pairset_stats_hybrid":
        return PresentPairSetStatsHybridDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 2496,
            base_channels=hidden_bits,
            token_dim=int_option(options, "token_dim"),
            mixer_depth=int_option(options, "mixer_depth", 2) or 2,
            role_mixer_depth=int_option(options, "role_mixer_depth", 1) or 1,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            stats_hidden_bits=int_option(options, "stats_hidden_bits"),
        )
    if name == "present_pairset_histogram_hybrid":
        return PresentPairSetHistogramHybridDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 2496,
            base_channels=hidden_bits,
            token_dim=int_option(options, "token_dim"),
            mixer_depth=int_option(options, "mixer_depth", 2) or 2,
            role_mixer_depth=int_option(options, "role_mixer_depth", 1) or 1,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            histogram_hidden_bits=int_option(options, "histogram_hidden_bits"),
        )
    if name == "present_pairset_global_stats_hybrid":
        return PresentPairSetGlobalStatsHybridDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 2496,
            base_channels=hidden_bits,
            token_dim=int_option(options, "token_dim"),
            mixer_depth=int_option(options, "mixer_depth", 2) or 2,
            role_mixer_depth=int_option(options, "role_mixer_depth", 1) or 1,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            global_hidden_bits=int_option(options, "global_hidden_bits"),
        )
    if name == "present_pairset_global_stats":
        return PresentPairSetGlobalStatsDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 2496,
            base_channels=hidden_bits,
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            global_hidden_bits=int_option(options, "global_hidden_bits"),
        )
    if name == "present_active_cell_graph_pairset":
        return PresentActiveCellGraphPairSetDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 320,
            base_channels=hidden_bits,
            token_dim=int_option(options, "token_dim"),
            graph_depth=int_option(options, "graph_depth", 2) or 2,
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2) or 2,
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            dropout=float(options.get("dropout", 0.0)),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
            metadata_bits=int_option(options, "metadata_bits", 0) or 0,
            graph_mode=str(options.get("graph_mode", "true")),
            edge_mode=str(options.get("edge_mode", "active_only")),
            cross_pair_consistency=str(options.get("cross_pair_consistency", "none")),
            active_metadata_fusion=str(options.get("active_metadata_fusion", "direct")),
            topology_auxiliary_scale=float(
                options.get("topology_auxiliary_scale", 0.0)
            ),
            topology_contrast_fusion=str(
                options.get("topology_contrast_fusion", "none")
            ),
            active_relative_summary=str(options.get("active_relative_summary", "none")),
            active_relative_contrast_fusion=str(
                options.get("active_relative_contrast_fusion", "none")
            ),
        )
    if name == "present_trail_position_stats_pairset":
        trail_depth = int_option(options, "trail_depth", 4)
        trail_words_per_depth = int_option(options, "trail_words_per_depth", 9)
        return PresentTrailPositionStatsPairSetDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 2496,
            base_channels=hidden_bits,
            trail_depth=4 if trail_depth is None else trail_depth,
            trail_words_per_depth=9
            if trail_words_per_depth is None
            else trail_words_per_depth,
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            dropout=float(options.get("dropout", 0.0)),
            stats_hidden_bits=int_option(options, "stats_hidden_bits"),
            metadata_bits=int_option(options, "metadata_bits", 0) or 0,
            active_conditioning=str(options.get("active_conditioning", "none")),
            trail_position_control=str(options.get("trail_position_control", "none")),
            trail_normalization=str(options.get("trail_normalization", "none")),
            trail_fusion=str(options.get("trail_fusion", "concat")),
            trail_gate=str(options.get("trail_gate", "vector")),
            trail_auxiliary_scale=float(options.get("trail_auxiliary_scale", 0.25)),
        )
    if name == "present_p_layer_mixer_pairset":
        return PresentPLayerMixerPairSetDistinguisher(
            input_bits=input_bits,
            pair_bits=pair_bits or 128,
            base_channels=hidden_bits,
            token_dim=int_option(options, "token_dim"),
            mixer_depth=int_option(options, "mixer_depth", 3),
            token_mlp_ratio=int_option(options, "token_mlp_ratio", 2),
            activation=str(options.get("activation", "gelu")),
            norm=str(options.get("norm", "layernorm")),
            pooling=str(options.get("pooling", "topk_logsumexp")),
            dropout=float(options.get("dropout", 0.0)),
            top_k=int_option(options, "top_k", 4) or 4,
            lse_temperature=float(options.get("lse_temperature", 1.0)),
            metadata_bits=int_option(options, "metadata_bits", 0) or 0,
            active_conditioning=str(options.get("active_conditioning", "none")),
            p_topology=str(options.get("p_topology", "true")),
        )
    return None


def _sbox_truth_bits_from_tables(tables: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(4, dtype=torch.long)
    return (((tables[..., None] >> shifts) & 1).reshape(*tables.shape[:2], 64)).to(
        torch.uint8
    )
