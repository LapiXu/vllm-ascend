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


def test_batch_type_values_match_vllm_patch():
    assert BatchType.PD_MIX.value == "pd_mix"
    assert BatchType.PURE_PREFILL.value == "pure_prefill"
    assert BatchType.PURE_DECODE.value == "pure_decode"
    assert BatchType.EMPTY.value == "empty"
    assert BatchType.PREFILL_FIRST.value == "prefill_first"
    assert BatchType.PREFILL_LAST.value == "prefill_last"
    assert BatchType.DECODE_FIRST.value == "decode_first"
    assert BatchType.DECODE_LAST.value == "decode_last"


def test_hidden_channel_values_match_vllm_patch():
    assert HiddenChannelType.PREFILL_1.value == "prefill_1"
    assert HiddenChannelType.PREFILL_2.value == "prefill_2"
    assert HiddenChannelType.DECODE.value == "decode"


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


def test_get_pdmix_metadata_returns_defaults_when_not_set():
    output = DummySchedulerOutput()
    metadata = get_pdmix_metadata(output)

    assert metadata.batch_type == BatchType.PD_MIX
    assert metadata.head_token is None
    assert metadata.hidden_channel is None
