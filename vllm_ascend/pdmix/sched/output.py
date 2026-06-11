# SPDX-License-Identifier: Apache-2.0

import enum
from dataclasses import dataclass
from typing import Any


class HiddenChannelType(enum.Enum):
    PREFILL_1 = "PREFILL_1"
    PREFILL_2 = "PREFILL_2"
    DECODE = "DECODE"


class BatchType(enum.Enum):
    PD_MIX = "PD_MIX"
    PURE_PREFILL = "PURE_PREFILL"
    PURE_DECODE = "PURE_DECODE"
    EMPTY = "EMPTY"
    PREFILL_FIRST = "PREFILL_FIRST"
    PREFILL_LAST = "PREFILL_LAST"
    DECODE_FIRST = "DECODE_FIRST"
    DECODE_LAST = "DECODE_LAST"


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
