# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from typing import Optional, Set

logger = logging.getLogger(__name__)

_patched: Set[type] = set()


def apply_patch() -> None:
    """
    Apply compatibility patches for Qwen model configurations.
    This ensures that Qwen3.5/Qwen3.5-MoE config classes have necessary
    attribute delegations.
    """
    # Try to import config classes
    try:
        from vllm.transformers_utils.configs.qwen3_5 import (
            Qwen3_5Config,
            Qwen3_5TextConfig,
        )
    except ImportError:
        logger.debug("Qwen3.5 configs not available, skipping config patches")
        Qwen3_5Config = None
        Qwen3_5TextConfig = None

    try:
        from vllm.transformers_utils.configs.qwen3_5_moe import (
            Qwen3_5MoeConfig,
            Qwen3_5MoeTextConfig,
        )
    except ImportError:
        logger.debug("Qwen3.5-MoE configs not available, skipping config patches")
        Qwen3_5MoeConfig = None
        Qwen3_5MoeTextConfig = None

    # Patch Qwen3_5Config if needed
    if Qwen3_5Config is not None and Qwen3_5Config not in _patched:
        _patch_qwen_config(Qwen3_5Config)
        _patched.add(Qwen3_5Config)

    # Patch Qwen3_5MoeConfig if needed
    if Qwen3_5MoeConfig is not None and Qwen3_5MoeConfig not in _patched:
        _patch_qwen_config(Qwen3_5MoeConfig)
        _patched.add(Qwen3_5MoeConfig)

    # Patch text configs if needed
    if Qwen3_5TextConfig is not None and Qwen3_5TextConfig not in _patched:
        _patch_qwen_text_config(Qwen3_5TextConfig)
        _patched.add(Qwen3_5TextConfig)

    if Qwen3_5MoeTextConfig is not None and Qwen3_5MoeTextConfig not in _patched:
        _patch_qwen_text_config(Qwen3_5MoeTextConfig)
        _patched.add(Qwen3_5MoeTextConfig)


def _patch_qwen_config(config_cls: type) -> None:
    """
    Patch Qwen multimodal config class with additional property delegations.
    """
    # Check and add properties that might be missing
    properties_to_add = [
        ("intermediate_size", "intermediate_size"),
        ("rms_norm_eps", "rms_norm_eps"),
        ("head_dim", "head_dim"),
        ("max_position_embeddings", "max_position_embeddings"),
        ("hidden_act", "hidden_act"),
        ("rope_parameters", "rope_parameters"),
        ("moe_intermediate_size", "moe_intermediate_size"),
        ("shared_expert_intermediate_size", "shared_expert_intermediate_size"),
        ("num_experts_per_tok", "num_experts_per_tok"),
        ("num_experts", "num_experts"),
        ("intermediate_size", "intermediate_size"),
        ("linear_conv_kernel_dim", "linear_conv_kernel_dim"),
        ("linear_key_head_dim", "linear_key_head_dim"),
        ("linear_value_head_dim", "linear_value_head_dim"),
        ("linear_num_key_heads", "linear_num_key_heads"),
        ("linear_num_value_heads", "linear_num_value_heads"),
    ]

    for prop_name, attr_name in properties_to_add:
        if not hasattr(config_cls, prop_name):
            _create_delegated_property(config_cls, prop_name, attr_name)


def _patch_qwen_text_config(config_cls: type) -> None:
    """
    Patch Qwen text config class with any necessary compatibility fixes.
    """
    # Currently just a placeholder for potential future text config patches
    pass


def _create_delegated_property(
    config_cls: type,
    prop_name: str,
    attr_name: str,
    default: Optional[object] = None,
) -> None:
    """
    Create a property that delegates to self.text_config.<attr_name>.
    """

    def getter(self):
        try:
            return getattr(self.text_config, attr_name)
        except AttributeError:
            return default

    getter.__name__ = prop_name
    setattr(config_cls, prop_name, property(getter))
