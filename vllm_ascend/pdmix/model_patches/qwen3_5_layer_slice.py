# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from typing import Optional, Set

logger = logging.getLogger(__name__)

_patched_classes: Set[type] = set()


def apply_patch() -> None:
    """
    Apply layer slicing patches for Qwen3.5 models.
    This adds forward_edge_cloud_segment method to the model classes.
    """
    try:
        from vllm.model_executor.models.qwen3_5 import (
            Qwen3_5Model,
            Qwen3_5ForCausalLM,
            Qwen3_5ForConditionalGeneration,
            Qwen3_5MoeForCausalLM,
            Qwen3_5MoeForConditionalGeneration,
        )
    except ImportError:
        logger.debug("Qwen3.5 models not available, skipping qwen3_5 layer slice patch")
        return

    # Patch Qwen3_5Model
    if Qwen3_5Model not in _patched_classes:
        _patch_qwen3_5_model(Qwen3_5Model)
        _patched_classes.add(Qwen3_5Model)

    # Patch Qwen3_5ForCausalLM
    if Qwen3_5ForCausalLM not in _patched_classes:
        _patch_qwen3_5_for_causal_lm(Qwen3_5ForCausalLM)
        _patched_classes.add(Qwen3_5ForCausalLM)

    # Patch Qwen3_5ForConditionalGeneration
    if Qwen3_5ForConditionalGeneration not in _patched_classes:
        _patch_qwen3_5_for_conditional_generation(Qwen3_5ForConditionalGeneration)
        _patched_classes.add(Qwen3_5ForConditionalGeneration)

    # Patch Qwen3_5MoeForCausalLM
    if Qwen3_5MoeForCausalLM not in _patched_classes:
        _patch_qwen3_5_for_causal_lm(Qwen3_5MoeForCausalLM)
        _patched_classes.add(Qwen3_5MoeForCausalLM)

    # Patch Qwen3_5MoeForConditionalGeneration
    if Qwen3_5MoeForConditionalGeneration not in _patched_classes:
        _patch_qwen3_5_for_conditional_generation(Qwen3_5MoeForConditionalGeneration)
        _patched_classes.add(Qwen3_5MoeForConditionalGeneration)


def _patch_qwen3_5_model(model_cls: type) -> None:
    """
    Patch Qwen3_5Model with layer slicing support.
    """
    original_forward = model_cls.forward

    def forward_edge_cloud_segment(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        layer_slice_start: Optional[int] = None,
        layer_slice_end: Optional[int] = None,
        layer_slice_return_intermediate: bool = False,
    ):
        """
        Forward pass with layer slicing support.
        """
        return original_forward(
            self,
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            layer_slice_start=layer_slice_start,
            layer_slice_end=layer_slice_end,
            layer_slice_return_intermediate=layer_slice_return_intermediate,
        )

    # Add the method to the class
    model_cls.forward_edge_cloud_segment = forward_edge_cloud_segment


def _patch_qwen3_5_for_causal_lm(model_cls: type) -> None:
    """
    Patch Qwen3_5ForCausalLM/Qwen3_5MoeForCausalLM with layer slicing support.
    """
    original_forward = model_cls.forward

    def forward_edge_cloud_segment(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        layer_slice_start: Optional[int] = None,
        layer_slice_end: Optional[int] = None,
        layer_slice_return_intermediate: bool = False,
    ):
        """
        Forward pass with layer slicing support.
        """
        return original_forward(
            self,
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            layer_slice_start=layer_slice_start,
            layer_slice_end=layer_slice_end,
            layer_slice_return_intermediate=layer_slice_return_intermediate,
        )

    # Add the method to the class
    model_cls.forward_edge_cloud_segment = forward_edge_cloud_segment


def _patch_qwen3_5_for_conditional_generation(model_cls: type) -> None:
    """
    Patch Qwen3_5ForConditionalGeneration with layer slicing support.
    """
    original_forward = model_cls.forward

    def forward_edge_cloud_segment(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        layer_slice_start: Optional[int] = None,
        layer_slice_end: Optional[int] = None,
        layer_slice_return_intermediate: bool = False,
        **kwargs,
    ):
        """
        Forward pass with layer slicing support.
        """
        return original_forward(
            self,
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            layer_slice_start=layer_slice_start,
            layer_slice_end=layer_slice_end,
            layer_slice_return_intermediate=layer_slice_return_intermediate,
            **kwargs,
        )

    # Add the method to the class
    model_cls.forward_edge_cloud_segment = forward_edge_cloud_segment
