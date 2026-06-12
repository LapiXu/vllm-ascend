# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser
from unittest.mock import patch

from vllm_ascend.platform import NPUPlatform


def test_pdmix_legacy_cli_args_are_registered_by_platform_hook():
    parser = ArgumentParser()

    with (
        patch("vllm_ascend.utils.adapt_patch"),
        patch("vllm_ascend.platform.config_deprecated_logging"),
    ):
        NPUPlatform.pre_register_and_update(parser)

    args = parser.parse_args([
        "--cloud-addr",
        "76.76.26.16",
        "--enable-pd-separation",
        "--pd-prefill-inflight-limit",
        "2",
        "--pd-scheduling-policy",
        "prefill_first",
    ])

    assert args.cloud_addr == "76.76.26.16"
    assert args.enable_pd_separation is True
    assert args.pd_prefill_inflight_limit == 2
    assert args.pd_scheduling_policy == "prefill_first"


def test_pdmix_common_misspellings_are_accepted_for_compatibility():
    parser = ArgumentParser()

    with (
        patch("vllm_ascend.utils.adapt_patch"),
        patch("vllm_ascend.platform.config_deprecated_logging"),
    ):
        NPUPlatform.pre_register_and_update(parser)

    args = parser.parse_args([
        "--enabled-pd-separation",
        "--pd-prefill-inlight-limit",
        "2",
    ])

    assert args.enable_pd_separation is True
    assert args.pd_prefill_inflight_limit == 2
