# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_ascend.pdmix.engine.zmq_channel import PPSchedulerZmqChannel
    from vllm_ascend.pdmix.engine.passive_engine_core import PassiveEngineCoreProc
else:
    try:
        from vllm_ascend.pdmix.engine.zmq_channel import PPSchedulerZmqChannel
    except ImportError:
        PPSchedulerZmqChannel = None
    from vllm_ascend.pdmix.engine.passive_engine_core import PassiveEngineCoreProc

__all__ = ["PPSchedulerZmqChannel", "PassiveEngineCoreProc"]
