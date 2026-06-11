# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
import unittest
from unittest.mock import patch, MagicMock


class TestModelPatches(unittest.TestCase):
    """Test the model patches module."""

    def setUp(self):
        """Reset module state before each test."""
        # Clear imports related to our patch modules
        modules_to_remove = [
            "vllm_ascend.pdmix.model_patches",
            "vllm_ascend.pdmix.model_patches.__init__",
            "vllm_ascend.pdmix.model_patches.qwen_config_compat",
            "vllm_ascend.pdmix.model_patches.qwen2_layer_slice",
            "vllm_ascend.pdmix.model_patches.qwen3_layer_slice",
            "vllm_ascend.pdmix.model_patches.qwen3_5_layer_slice",
            "vllm_ascend.pdmix.model_patches.qwen3_next_layer_slice",
        ]

        for mod_name in modules_to_remove:
            if mod_name in sys.modules:
                del sys.modules[mod_name]

    def test_import_succeeds(self):
        """Test that the model_patches module can be imported."""
        from vllm_ascend.pdmix.model_patches import apply_model_patches

        self.assertIsNotNone(apply_model_patches)

    def test_apply_model_patches_idempotent(self):
        """Test that apply_model_patches can be called multiple times."""
        from vllm_ascend.pdmix.model_patches import apply_model_patches

        # First call
        apply_model_patches()
        # Second call - should not raise
        apply_model_patches()
        # Third call - should not raise
        apply_model_patches()

    def test_individual_patch_modules_import(self):
        """Test that individual patch modules can be imported."""
        from vllm_ascend.pdmix.model_patches import qwen_config_compat
        from vllm_ascend.pdmix.model_patches import qwen2_layer_slice
        from vllm_ascend.pdmix.model_patches import qwen3_layer_slice
        from vllm_ascend.pdmix.model_patches import qwen3_5_layer_slice
        from vllm_ascend.pdmix.model_patches import qwen3_next_layer_slice

        self.assertTrue(hasattr(qwen_config_compat, "apply_patch"))
        self.assertTrue(hasattr(qwen2_layer_slice, "apply_patch"))
        self.assertTrue(hasattr(qwen3_layer_slice, "apply_patch"))
        self.assertTrue(hasattr(qwen3_5_layer_slice, "apply_patch"))
        self.assertTrue(hasattr(qwen3_next_layer_slice, "apply_patch"))

    def test_individual_patch_apply_safe(self):
        """Test that individual patch apply functions are safe."""
        from vllm_ascend.pdmix.model_patches import qwen_config_compat
        from vllm_ascend.pdmix.model_patches import qwen2_layer_slice
        from vllm_ascend.pdmix.model_patches import qwen3_layer_slice
        from vllm_ascend.pdmix.model_patches import qwen3_5_layer_slice
        from vllm_ascend.pdmix.model_patches import qwen3_next_layer_slice

        # Should not raise even if models are not available
        qwen_config_compat.apply_patch()
        qwen2_layer_slice.apply_patch()
        qwen3_layer_slice.apply_patch()
        qwen3_5_layer_slice.apply_patch()
        qwen3_next_layer_slice.apply_patch()

        # Can call again
        qwen_config_compat.apply_patch()
        qwen2_layer_slice.apply_patch()

    @patch("vllm_ascend.pdmix.model_patches.qwen2_layer_slice")
    @patch("vllm_ascend.pdmix.model_patches.qwen3_layer_slice")
    @patch("vllm_ascend.pdmix.model_patches.qwen3_5_layer_slice")
    @patch("vllm_ascend.pdmix.model_patches.qwen3_next_layer_slice")
    @patch("vllm_ascend.pdmix.model_patches.qwen_config_compat")
    def test_apply_model_patches_calls_individual_patches(
        self,
        mock_qwen_config,
        mock_qwen2,
        mock_qwen3,
        mock_qwen3_5,
        mock_qwen3_next,
    ):
        """Test that apply_model_patches calls individual patches."""
        from vllm_ascend.pdmix.model_patches import apply_model_patches

        apply_model_patches()

        mock_qwen_config.apply_patch.assert_called_once()
        mock_qwen2.apply_patch.assert_called_once()
        mock_qwen3.apply_patch.assert_called_once()
        mock_qwen3_5.apply_patch.assert_called_once()
        mock_qwen3_next.apply_patch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
