# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging

logger = logging.getLogger(__name__)

_applied = False


def apply_model_patches() -> None:
    """
    Apply all Qwen model patches.
    This function is idempotent - it can be called multiple times safely.
    """
    global _applied
    if _applied:
        logger.debug("Model patches already applied, skipping")
        return

    try:
        from . import qwen_config_compat
        from . import qwen2_layer_slice
        from . import qwen3_layer_slice
        from . import qwen3_5_layer_slice
        from . import qwen3_next_layer_slice

        qwen_config_compat.apply_patch()
        qwen2_layer_slice.apply_patch()
        qwen3_layer_slice.apply_patch()
        qwen3_5_layer_slice.apply_patch()
        qwen3_next_layer_slice.apply_patch()

        _applied = True
        logger.debug("Model patches applied successfully")
    except Exception as e:
        logger.warning(f"Failed to apply model patches: {e}", exc_info=True)
        # Don't re-raise - allow the system to continue
