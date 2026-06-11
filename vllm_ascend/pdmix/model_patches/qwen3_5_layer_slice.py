# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from typing import Optional, Set

logger = logging.getLogger(__name__)

_patched_classes: Set[type] = set()


def apply_patch() -> None:
    """
    Apply layer slicing patches for Qwen3.5 models.
    This function is idempotent - it can be called multiple times safely.
    """
    try:
        from vllm.model_executor.models.qwen3_5 import (
            Qwen3_5ForCausalLM,
            Qwen3_5ForConditionalGeneration,
        )
    except ImportError:
        logger.debug("Qwen3.5 models not available, skipping qwen3_5 layer slice patch")
        return

    # Patch Qwen3_5ForCausalLM
    if Qwen3_5ForCausalLM not in _patched_classes:
        _patch_qwen3_5_for_causal_lm(Qwen3_5ForCausalLM)
        _patched_classes.add(Qwen3_5ForCausalLM)

    # Patch Qwen3_5ForConditionalGeneration
    if Qwen3_5ForConditionalGeneration not in _patched_classes:
        _patch_qwen3_5_for_conditional_generation(Qwen3_5ForConditionalGeneration)
        _patched_classes.add(Qwen3_5ForConditionalGeneration)


def _patch_qwen3_5_for_causal_lm(model_cls: type) -> None:
    """
    Patch Qwen3_5ForCausalLM with layer slicing support.
    """
    original_forward = model_cls.forward

    def patched_forward(
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
        Forward pass that passes layer_slice parameters to self.model.
        """
        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            layer_slice_start=layer_slice_start,
            layer_slice_end=layer_slice_end,
            layer_slice_return_intermediate=layer_slice_return_intermediate,
        )
        return hidden_states

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
        Forward pass with layer slicing support for edge-cloud.
        """
        return patched_forward(
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

    # Patch the model
    model_cls.forward = patched_forward
    model_cls.forward_edge_cloud_segment = forward_edge_cloud_segment


def _patch_qwen3_5_for_conditional_generation(model_cls: type) -> None:
    """
    Patch Qwen3_5ForConditionalGeneration with layer slicing support.
    """
    original_forward = model_cls.forward

    def patched_forward(
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
        Forward pass that passes layer_slice parameters to language_model.model.
        """
        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            layer_slice_start=layer_slice_start,
            layer_slice_end=layer_slice_end,
            layer_slice_return_intermediate=layer_slice_return_intermediate,
        )
        return hidden_states

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
        Forward pass with layer slicing support for edge-cloud.
        """
        return patched_forward(
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

    # Patch the model
    model_cls.forward = patched_forward
    model_cls.forward_edge_cloud_segment = forward_edge_cloud_segment
