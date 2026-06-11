# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging

logger = logging.getLogger(__name__)


def apply_model_patches() -> None:
    """
    Apply all Qwen model patches.
    This function is idempotent - it can be called multiple times safely.
    """
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

    logger.debug("Model patches applied successfully")
