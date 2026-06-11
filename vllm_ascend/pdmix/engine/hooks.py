# SPDX-License-Identifier: Apache-2.0


def apply_engine_patches(*, is_global_patch: bool) -> None:
    # NOTE: EngineCore creation is currently handled by the Ascend multiproc executor patch
    #       in vllm_ascend/patch/platform/patch_multiproc_executor.py.
    #
    # This module is kept as a centralized registration point for future PDMix engine
    # patches that cannot be expressed through platform hooks.
    if is_global_patch:
        return None
    return None
