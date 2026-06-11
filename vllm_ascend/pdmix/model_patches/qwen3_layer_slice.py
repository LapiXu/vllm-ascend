# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from typing import Optional, Set

logger = logging.getLogger(__name__)

_patched_classes: Set[type] = set()


def apply_patch() -> None:
    """
    Apply layer slicing patches for Qwen3 models.
    This adds forward_edge_cloud_segment method to the model classes.
    """
    try:
        from vllm.model_executor.models.qwen3 import Qwen3Model, Qwen3ForCausalLM
    except ImportError:
        logger.debug("Qwen3 models not available, skipping qwen3 layer slice patch")
        return

    # Patch Qwen3Model
    if Qwen3Model not in _patched_classes:
        _patch_qwen3_model(Qwen3Model)
        _patched_classes.add(Qwen3Model)

    # Patch Qwen3ForCausalLM
    if Qwen3ForCausalLM not in _patched_classes:
        _patch_qwen3_for_causal_lm(Qwen3ForCausalLM)
        _patched_classes.add(Qwen3ForCausalLM)


def _patch_qwen3_model(model_cls: type) -> None:
    """
    Patch Qwen3Model with layer slicing support.
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


def _patch_qwen3_for_causal_lm(model_cls: type) -> None:
    """
    Patch Qwen3ForCausalLM with layer slicing support.
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
