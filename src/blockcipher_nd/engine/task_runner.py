from __future__ import annotations

import argparse
from typing import Any

from blockcipher_nd.engine.checkpoint_initialization import build_initialized_task_model
from blockcipher_nd.engine.final_evaluation import run_final_evaluation
from blockcipher_nd.engine.modeling import configure_structure_aware_model
from blockcipher_nd.engine.pretraining import (
    dataset_size_metadata,
    resolve_optimizer_state_transition,
    run_optional_pretraining,
)
from blockcipher_nd.engine.progress import progress_callback
from blockcipher_nd.engine.results import build_task_result
from blockcipher_nd.engine.task_config import build_training_config, target_epochs
from blockcipher_nd.engine.task_inputs import prepare_task_inputs
from blockcipher_nd.training import OptimizerSession, train_binary_classifier


def run_task(
    task: dict[str, Any],
    args: argparse.Namespace,
    *,
    progress_path: str | None = None,
    index: int | None = None,
    total: int | None = None,
) -> dict[str, Any]:
    inputs = prepare_task_inputs(
        task,
        args,
        progress_path=progress_path,
        index=index,
        total=total,
    )

    model, initialization = build_initialized_task_model(
        task=task,
        args=args,
        model_key=inputs.model_key,
        input_bits=inputs.train_dataset.features.shape[1],
        pair_bits=inputs.pair_bits,
        structure=inputs.train_cipher.structure,
        progress_path=progress_path,
        index=index,
        total=total,
    )
    optimizer_state_transition = resolve_optimizer_state_transition(task, args)
    optimizer_session = (
        OptimizerSession()
        if optimizer_state_transition == "carry_across_stages"
        else None
    )
    pretrain_result = run_optional_pretraining(
        model,
        task,
        args,
        pair_bits=inputs.pair_bits,
        progress_path=progress_path,
        index=index,
        total=total,
        optimizer_session=optimizer_session,
    )
    configure_structure_aware_model(model, task["cipher_key"], task["rounds"])
    training_result = train_binary_classifier(
        model,
        inputs.train_dataset,
        inputs.validation_dataset,
        build_training_config(
            task,
            args,
            epochs=target_epochs(task, args),
            seed=task["seed"],
        ),
        progress_callback=progress_callback(
            progress_path,
            "training",
            task,
            index=index,
            total=total,
        ),
        optimizer_session=optimizer_session,
    )
    training_result.metadata["optimizer_state_transition"] = optimizer_state_transition
    training_result.metadata.update(
        dataset_size_metadata(inputs.train_dataset, inputs.validation_dataset)
    )
    final_evaluation = run_final_evaluation(
        model,
        task,
        args,
        cipher=inputs.final_test_cipher,
        final_test_key=inputs.final_test_key,
        progress_path=progress_path,
        index=index,
        total=total,
    )
    return build_task_result(
        task=task,
        args=args,
        train_cipher=inputs.train_cipher,
        validation_cipher=inputs.validation_cipher,
        train_key=inputs.train_key,
        validation_key=inputs.validation_key,
        model=model,
        model_key=inputs.model_key,
        pair_bits=inputs.pair_bits,
        train_dataset=inputs.train_dataset,
        validation_dataset=inputs.validation_dataset,
        training_result=training_result,
        pretrain_result=pretrain_result,
        final_evaluation=final_evaluation,
        initialization=initialization,
    )
