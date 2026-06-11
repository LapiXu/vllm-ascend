# SPDX-License-Identifier: Apache-2.0

from vllm_ascend.pdmix.model_patches import apply_model_patches


def apply_pdmix_patches(*, is_global_patch: bool = False) -> None:
    if not is_global_patch:
        apply_model_patches()
