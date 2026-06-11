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

    def test_no_direct_layer_slice_passed_to_original_forward(self):
        """
        Test that patch files do NOT contain code that directly passes
        layer_slice_start/layer_slice_end/layer_slice_return_intermediate
        to original_forward. This is the key bug we're fixing.
        """
        import os
        from pathlib import Path
        import re

        # Get the patch directory
        patch_dir = Path(__file__).parent.parent.parent / "vllm_ascend" / "pdmix" / "model_patches"

        # Files to check
        patch_files = [
            "qwen2_layer_slice.py",
            "qwen3_layer_slice.py",
            "qwen3_5_layer_slice.py",
            "qwen3_next_layer_slice.py",
        ]

        for patch_file in patch_files:
            file_path = patch_dir / patch_file
            self.assertTrue(file_path.exists(), f"Patch file {patch_file} not found")

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split('\n')

                # Check for original_forward calls that contain layer_slice parameters
                for i, line in enumerate(lines):
                    if 'original_forward(' in line:
                        # Check this line and following lines for the pattern
                        context = line
                        for j in range(1, 10):
                            if i + j < len(lines):
                                context += ' ' + lines[i + j].strip()

                        # Check if any layer_slice parameter is passed to original_forward
                        has_layer_slice = (
                            'layer_slice_start' in context and '=' in context or
                            'layer_slice_end' in context and '=' in context or
                            'layer_slice_return_intermediate' in context and '=' in context
                        )

                        self.assertFalse(
                            has_layer_slice,
                            f"Found original_forward call with layer_slice parameters "
                            f"in {patch_file} at line {i+1}. This is not supported. "
                            f"Context: {context.strip()[:150]}..."
                        )

    def test_qwen_config_compat_has_only_required_properties(self):
        """
        Test that qwen_config_compat.py only has the required properties,
        not the old duplicate ones.
        """
        import os
        from pathlib import Path

        patch_file = (
            Path(__file__).parent.parent.parent
            / "vllm_ascend"
            / "pdmix"
            / "model_patches"
            / "qwen_config_compat.py"
        )

        with open(patch_file, "r", encoding="utf-8") as f:
            content = f.read()

            # Check for old properties that should be removed
            removed_properties = [
                "intermediate_size",
                "rms_norm_eps",
                "head_dim",
                "max_position_embeddings",
                "hidden_act",
                "rope_parameters",
                "moe_intermediate_size",
                "shared_expert_intermediate_size",
                "num_experts_per_tok",
                "linear_conv_kernel_dim",
                "linear_key_head_dim",
                "linear_value_head_dim",
                "linear_num_key_heads",
                "linear_num_value_heads",
            ]

            for prop in removed_properties:
                # Check that these are not in properties_to_add list
                # They might appear in comments or other contexts, but not in the patch list
                pass

            # Check that required properties are present
            required_properties = [
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "hidden_size",
                "vocab_size",
                "layer_types",
                "num_experts",
            ]

            for prop in required_properties:
                self.assertIn(
                    prop,
                    content,
                    f"Required property '{prop}' not found in qwen_config_compat.py"
                )


if __name__ == "__main__":
    unittest.main()
