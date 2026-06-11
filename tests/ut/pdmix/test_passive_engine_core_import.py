# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that PassiveEngineCoreProc can be imported from the pdmix package."""


def test_import_passive_engine_core_proc():
    """Test that PassiveEngineCoreProc can be imported."""
    from vllm_ascend.pdmix.engine.passive_engine_core import PassiveEngineCoreProc
    assert PassiveEngineCoreProc is not None


def test_import_from_engine_module():
    """Test that PassiveEngineCoreProc can be imported from engine module."""
    from vllm_ascend.pdmix.engine import PassiveEngineCoreProc
    assert PassiveEngineCoreProc is not None
