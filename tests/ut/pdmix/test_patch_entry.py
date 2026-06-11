# SPDX-License-Identifier: Apache-2.0


def test_apply_pdmix_patches_accepts_global_and_worker_modes():
    from vllm_ascend.pdmix import apply_pdmix_patches

    apply_pdmix_patches(is_global_patch=True)
    apply_pdmix_patches(is_global_patch=False)
