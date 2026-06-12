# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

from vllm.v1.core.sched.output import BatchType, HiddenChannelType


@dataclass(frozen=True)
class PDMixSchedulerMetadata:
    batch_type: BatchType = BatchType.PD_MIX
    head_token: str | None = None
    hidden_channel: HiddenChannelType | None = None


def set_pdmix_metadata(scheduler_output: Any,
                       metadata: PDMixSchedulerMetadata) -> None:
    setattr(scheduler_output, "batch_type", metadata.batch_type)
    setattr(scheduler_output, "head_token", metadata.head_token)
    setattr(scheduler_output, "hidden_channel", metadata.hidden_channel)


def get_pdmix_metadata(scheduler_output: Any) -> PDMixSchedulerMetadata:
    return PDMixSchedulerMetadata(
        batch_type=getattr(scheduler_output, "batch_type", BatchType.PD_MIX),
        head_token=getattr(scheduler_output, "head_token", None),
        hidden_channel=getattr(scheduler_output, "hidden_channel", None),
    )
