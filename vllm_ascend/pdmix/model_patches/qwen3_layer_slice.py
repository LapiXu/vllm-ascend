# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from typing import Optional, Set

logger = logging.getLogger(__name__)

_patched_classes: Set[type] = set()


def apply_patch() -> None:
    """
    Apply layer slicing patches for Qwen3 models.
    This function is idempotent - it can be called multiple times safely.

    Note: Qwen3Model inherits from Qwen2Model, so we only need to patch
    Qwen3ForCausalLM to pass layer_slice parameters to the already-patched
    Qwen2Model.forward method.
    """
    try:
        from vllm.model_executor.models.qwen3 import Qwen3Model, Qwen3ForCausalLM
    except ImportError:
        logger.debug("Qwen3 models not available, skipping qwen3 layer slice patch")
        return

    # Patch Qwen3ForCausalLM - Qwen3Model inherits from Qwen2Model which should already be patched
    if Qwen3ForCausalLM not in _patched_classes:
        _patch_qwen3_for_causal_lm(Qwen3ForCausalLM)
        _patched_classes.add(Qwen3ForCausalLM)


def _patch_qwen3_for_causal_lm(model_cls: type) -> None:
    """
    Patch Qwen3ForCausalLM with layer slicing support.
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
    ):
        """
        Forward pass that passes layer_slice parameters to self.model.forward.
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
        )

    # Patch the model
    model_cls.forward = patched_forward
    model_cls.forward_edge_cloud_segment = forward_edge_cloud_segment
