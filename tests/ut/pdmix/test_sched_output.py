# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from vllm_ascend.pdmix.sched.output import (
    BatchType,
    HiddenChannelType,
    PDMixSchedulerMetadata,
    get_pdmix_metadata,
    set_pdmix_metadata,
)


@dataclass
class DummySchedulerOutput:
    pass


def test_batch_type_values_match_original_vllm_patch():
    assert BatchType.PD_MIX.value == "PD_MIX"
    assert BatchType.PURE_PREFILL.value == "PURE_PREFILL"
    assert BatchType.PURE_DECODE.value == "PURE_DECODE"
    assert BatchType.EMPTY.value == "EMPTY"
    assert BatchType.PREFILL_FIRST.value == "PREFILL_FIRST"
    assert BatchType.PREFILL_LAST.value == "PREFILL_LAST"
    assert BatchType.DECODE_FIRST.value == "DECODE_FIRST"
    assert BatchType.DECODE_LAST.value == "DECODE_LAST"


def test_hidden_channel_values_match_original_vllm_patch():
    assert HiddenChannelType.PREFILL_1.value == "PREFILL_1"
    assert HiddenChannelType.PREFILL_2.value == "PREFILL_2"
    assert HiddenChannelType.DECODE.value == "DECODE"


def test_metadata_can_be_attached_without_modifying_vllm_class_definition():
    output = DummySchedulerOutput()
    metadata = PDMixSchedulerMetadata(
        batch_type=BatchType.PREFILL_FIRST,
        head_token="head-1",
        hidden_channel=HiddenChannelType.PREFILL_1,
    )

    set_pdmix_metadata(output, metadata)

    assert get_pdmix_metadata(output) == metadata
    assert getattr(output, "batch_type") == BatchType.PREFILL_FIRST
    assert getattr(output, "head_token") == "head-1"
    assert getattr(output, "hidden_channel") == HiddenChannelType.PREFILL_1
