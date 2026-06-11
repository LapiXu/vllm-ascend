# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from itertools import islice
from typing import Optional, Set, TYPE_CHECKING

logger = logging.getLogger(__name__)

_patched_classes: Set[type] = set()


def apply_patch() -> None:
    """
    Apply layer slicing patches for Qwen2 models.
    This function is idempotent - it can be called multiple times safely.
    """
    try:
        from vllm.model_executor.models.qwen2 import Qwen2Model, Qwen2ForCausalLM
        from vllm.sequence import IntermediateTensors
        from vllm.distributed import get_pp_group
    except ImportError:
        logger.debug("Qwen2 models not available, skipping qwen2 layer slice patch")
        return

    # Patch Qwen2Model
    if Qwen2Model not in _patched_classes:
        _patch_qwen2_model(Qwen2Model, IntermediateTensors, get_pp_group, islice)
        _patched_classes.add(Qwen2Model)

    # Patch Qwen2ForCausalLM
    if Qwen2ForCausalLM not in _patched_classes:
        _patch_qwen2_for_causal_lm(Qwen2ForCausalLM)
        _patched_classes.add(Qwen2ForCausalLM)


def _patch_qwen2_model(model_cls: type, IntermediateTensors: type, get_pp_group,
                       islice) -> None:
    """
    Patch Qwen2Model with layer slicing support.
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
        Forward pass with layer slicing support.
        Implements true layer_slice logic based on dest reference.
        """
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # Determine the layer range to execute.
        exec_start = (
            self.start_layer + layer_slice_start
            if layer_slice_start is not None
            else self.start_layer
        )
        exec_end = (
            self.start_layer + layer_slice_end
            if layer_slice_end is not None
            else self.end_layer
        )

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for idx, layer in enumerate(
            islice(self.layers, exec_start - self.start_layer, exec_end - self.start_layer)
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank or layer_slice_return_intermediate:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states

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
        Calls the patched forward with layer_slice parameters.
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


def _patch_qwen2_for_causal_lm(model_cls: type) -> None:
    """
    Patch Qwen2ForCausalLM with layer slicing support.
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
