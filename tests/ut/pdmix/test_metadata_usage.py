# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_production_code_reads_pdmix_batch_type_via_metadata_helper():
    forbidden = (
        "scheduler_output.batch_type",
        "head_state.scheduler_output.batch_type",
        "so.batch_type",
    )
    offenders: list[str] = []

    for path in (REPO_ROOT / "vllm_ascend").rglob("*.py"):
        if path.relative_to(REPO_ROOT).as_posix() == "vllm_ascend/pdmix/sched/output.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for needle in forbidden:
            if needle in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} contains {needle}")

    assert offenders == []
