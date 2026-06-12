# SPDX-License-Identifier: Apache-2.0

import argparse
from typing import Any

_PATCHED = False


def register_pdmix_cli_args(parser: argparse.ArgumentParser) -> None:
    if parser._option_string_actions.get("--cloud-addr") is None:
        parser.add_argument(
            "--cloud-addr",
            default=None,
            help="PDMix compatibility: cloud node address for edge-cloud mode.",
        )
    if parser._option_string_actions.get("--enable-pd-separation") is None:
        parser.add_argument(
            "--enable-pd-separation",
            "--enabled-pd-separation",
            dest="enable_pd_separation",
            action="store_true",
            default=False,
            help="PDMix compatibility: enable prefill/decode separation.",
        )
    if parser._option_string_actions.get("--pd-scheduling-policy") is None:
        parser.add_argument(
            "--pd-scheduling-policy",
            default="prefill_first",
            choices=("prefill_first", "decode_first", "strict_alternation"),
            help="PDMix compatibility: scheduling policy for PD separation.",
        )
    if parser._option_string_actions.get("--pd-prefill-inflight-limit") is None:
        parser.add_argument(
            "--pd-prefill-inflight-limit",
            "--pd-prefill-inlight-limit",
            dest="pd_prefill_inflight_limit",
            type=int,
            default=1,
            help="PDMix compatibility: max in-flight PREFILL_FIRST batches.",
        )


def apply_engine_args_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return

    from vllm.engine.arg_utils import EngineArgs

    original_from_cli_args = EngineArgs.from_cli_args
    original_create_engine_config = EngineArgs.create_engine_config

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        engine_args = original_from_cli_args(args)
        for name in (
            "cloud_addr",
            "enable_pd_separation",
            "pd_scheduling_policy",
            "pd_prefill_inflight_limit",
        ):
            if hasattr(args, name):
                setattr(engine_args, name, getattr(args, name))
        return engine_args

    def create_engine_config(self, *args: Any, **kwargs: Any):
        config = original_create_engine_config(self, *args, **kwargs)
        parallel_config = config.parallel_config
        scheduler_config = config.scheduler_config

        if hasattr(self, "cloud_addr"):
            setattr(parallel_config, "cloud_addr", self.cloud_addr)
        if hasattr(self, "enable_pd_separation"):
            setattr(parallel_config, "enable_pd_separation", self.enable_pd_separation)
            setattr(scheduler_config, "enable_pd_separation", self.enable_pd_separation)
        if hasattr(self, "pd_scheduling_policy"):
            setattr(scheduler_config, "pd_scheduling_policy", self.pd_scheduling_policy)
        if hasattr(self, "pd_prefill_inflight_limit"):
            setattr(scheduler_config, "pd_prefill_inflight_limit", self.pd_prefill_inflight_limit)
        return config

    EngineArgs.from_cli_args = from_cli_args
    EngineArgs.create_engine_config = create_engine_config
    _PATCHED = True
