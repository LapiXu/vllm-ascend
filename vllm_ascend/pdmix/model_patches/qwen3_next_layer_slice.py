# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from itertools import islice
from typing import Optional, Set

logger = logging.getLogger(__name__)

_patched_classes: Set[type] = set()


def apply_patch() -> None:
    """
    Apply layer slicing patches for Qwen3Next models.
    This function is idempotent - it can be called multiple times safely.
    """
    try:
        from vllm.model_executor.models.qwen3_next import (
            Qwen3NextModel,
            Qwen3NextForCausalLM,
        )
        from vllm.sequence import IntermediateTensors
        from vllm.distributed import get_pp_group
    except ImportError:
        logger.debug(
            "Qwen3Next models not available, skipping qwen3_next layer slice patch"
        )
        return

    # Patch Qwen3NextModel
    if Qwen3NextModel not in _patched_classes:
        _patch_qwen3_next_model(Qwen3NextModel, IntermediateTensors, get_pp_group, islice)
        _patched_classes.add(Qwen3NextModel)

    # Patch Qwen3NextForCausalLM
    if Qwen3NextForCausalLM not in _patched_classes:
        _patch_qwen3_next_for_causal_lm(Qwen3NextForCausalLM)
        _patched_classes.add(Qwen3NextForCausalLM)


def _patch_qwen3_next_model(model_cls: type, IntermediateTensors: type, get_pp_group,
                             islice) -> None:
    """
    Patch Qwen3NextModel with layer slicing support.
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
        for layer_idx, layer in enumerate(
            islice(self.layers, exec_start - self.start_layer, exec_end - self.start_layer),
            start=exec_start,
        ):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank or layer_slice_return_intermediate:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
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


def _patch_qwen3_next_for_causal_lm(model_cls: type) -> None:
    """
    Patch Qwen3NextForCausalLM with layer slicing support.
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
