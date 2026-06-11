# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
INVENTORY = REPO_ROOT / "docs" / "pdmix_patch_inventory.md"


def test_patch_inventory_exists():
    assert INVENTORY.exists()


def test_patch_inventory_records_pdmix_patch_modules():
    text = INVENTORY.read_text(encoding="utf-8")
    for required in (
        "qwen2_layer_slice.py",
        "qwen3_layer_slice.py",
        "qwen3_5_layer_slice.py",
        "qwen3_next_layer_slice.py",
        "qwen_config_compat.py",
        "patch_multiproc_executor.py",
        "patch_distributed.py",
    ):
        assert required in text
