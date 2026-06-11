# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_ascend.pdmix.engine.zmq_channel import PPSchedulerZmqChannel
    from vllm_ascend.pdmix.engine.passive_engine_core import PassiveEngineCoreProc

__all__ = ["PPSchedulerZmqChannel", "PassiveEngineCoreProc"]


def __getattr__(name: str):
    if name == "PPSchedulerZmqChannel":
        from vllm_ascend.pdmix.engine.zmq_channel import PPSchedulerZmqChannel
        return PPSchedulerZmqChannel
    if name == "PassiveEngineCoreProc":
        from vllm_ascend.pdmix.engine.passive_engine_core import PassiveEngineCoreProc
        return PassiveEngineCoreProc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
