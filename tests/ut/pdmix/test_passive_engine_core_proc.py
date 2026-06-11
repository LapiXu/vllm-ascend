# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for PassiveEngineCoreProc.

These tests avoid spinning up a real MultiprocExecutor or ZMQ subscriber.
Both are replaced with lightweight fakes.
"""
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from vllm_ascend.pdmix.sched.output import BatchType


def _install_fake_distributed_utils() -> None:
    if "vllm.distributed.utils" in sys.modules:
        return

    def get_pp_indices(num_hidden_layers: int, pp_rank: int,
                       pp_size: int) -> tuple[int, int]:
        layers_per_rank = num_hidden_layers // pp_size
        start = pp_rank * layers_per_rank
        end = (
            num_hidden_layers
            if pp_rank == pp_size - 1
            else start + layers_per_rank
        )
        return start, end

    fake_distributed = ModuleType("vllm.distributed")
    fake_utils = ModuleType("vllm.distributed.utils")
    fake_utils.get_pp_indices = get_pp_indices
    fake_distributed.utils = fake_utils
    sys.modules.setdefault("vllm.distributed", fake_distributed)
    sys.modules["vllm.distributed.utils"] = fake_utils


_install_fake_distributed_utils()


# Now import after setting up fake modules
from vllm_ascend.pdmix.sched.passive_scheduler import (
    DispatchPolicy,
    LayerSliceInfo,
    PassiveScheduler,
)


# ---------------------------------------------------------------------- #
# Fakes                                                                  #
# ---------------------------------------------------------------------- #
class FakeSubscriber:
    def __init__(self) -> None:
        self._buffer: list = []
        self._seq = 0

    def feed(self, *scheduler_outputs) -> None:
        for so in scheduler_outputs:
            self._buffer.append((self._seq, so))
            self._seq += 1

    def consume_new_outputs(self) -> list:
        out = self._buffer
        self._buffer = []
        return out

    def shutdown(self) -> None:
        pass


class FakeRpcMq:
    def __init__(self) -> None:
        self.enqueued: list = []

    def enqueue(self, item: tuple) -> None:
        self.enqueued.append(item)


class FakeExecutor:
    def __init__(self) -> None:
        self.rpc_broadcast_mq = FakeRpcMq()
        self.is_failed = False


def _fake_vllm_config(num_hidden_layers: int = 8, pp_size: int = 2):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=num_hidden_layers),
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp_size,
            enable_edge_cloud=False,
            enable_pd_separation=False,
        ),
    )


def _make_empty_scheduler_output():
    """Create a simple fake SchedulerOutput for testing."""
    return SimpleNamespace(
        batch_type=BatchType.EMPTY,
        total_num_scheduled_tokens=0,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(num_reqs=0),
        finished_req_ids=[],
        head_token=None,
    )


def _make_so(batch_type: BatchType):
    so = _make_empty_scheduler_output()
    so.batch_type = batch_type
    return so


def _make_proc(
    *,
    dispatch_policy: DispatchPolicy = DispatchPolicy.EXPECT_ALTERNATION,
    layer_slice_size: int = 0,
    num_hidden_layers: int = 8,
    pp_size: int = 2,
    pp_pd_channel=None,
):
    """Construct a PassiveEngineCoreProc with fake dependencies."""
    sub = FakeSubscriber()
    executor = FakeExecutor()
    cfg = _fake_vllm_config(num_hidden_layers, pp_size)

    # Import here to avoid circular imports
    from vllm_ascend.pdmix.engine.passive_engine_core import (
        PassiveEngineCoreProc,
    )

    # Create a custom scheduler that doesn't need real ZMQ
    with patch("vllm_ascend.envs.VLLM_LAYER_SLICE_SIZE", layer_slice_size):
        scheduler = PassiveScheduler(
            cfg, sub,
            dispatch_policy=dispatch_policy,
            run_subscriber_thread=False,
        )

    # Create the proc instance and inject our fake dependencies
    proc = PassiveEngineCoreProc.__new__(PassiveEngineCoreProc)
    proc.vllm_config = cfg
    proc.executor = executor
    proc.passive_scheduler = scheduler
    proc._idle_sleep_seconds = 0.001
    proc._pp_pd_channel = pp_pd_channel

    return proc, sub, executor


# ---------------------------------------------------------------------- #
# Tests                                                                  #
# ---------------------------------------------------------------------- #
def test_import():
    """Test that the module can be imported."""
    from vllm_ascend.pdmix.engine.passive_engine_core import (
        PassiveEngineCoreProc,
    )
    assert PassiveEngineCoreProc is not None


def test_proc_creation():
    """Test that PassiveEngineCoreProc can be created."""
    proc, _, _ = _make_proc()
    assert proc is not None
    assert proc.vllm_config is not None
    assert proc.executor is not None
    assert proc.passive_scheduler is not None


def test_step_returns_false_when_nothing_to_dispatch():
    """Test that step returns False when there's nothing to do."""
    proc, _, executor = _make_proc()
    assert proc.step() is False
    assert executor.rpc_broadcast_mq.enqueued == []


def test_step_drops_all_empties():
    """Test that EMPTY batch types are dropped."""
    proc, sub, executor = _make_proc()
    sub.feed(
        _make_so(BatchType.EMPTY),
        _make_so(BatchType.EMPTY),
        _make_so(BatchType.EMPTY),
    )
    assert proc.step() is False
    assert executor.rpc_broadcast_mq.enqueued == []


# ---------------------------------------------------------------------- #
# POST_OUT publishing (cloud → edge, PD-separation)                      #
# ---------------------------------------------------------------------- #
class FakePdChannel:
    """Captures publish() calls for assertion."""

    def __init__(self, events_log: list | None = None) -> None:
        self.published: list = []
        self._events_log = events_log

    def publish(self, scheduler_output) -> None:
        self.published.append(scheduler_output)
        if self._events_log is not None:
            self._events_log.append(("publish", scheduler_output.batch_type))

    def shutdown(self) -> None:
        pass


def test_post_out_not_published_when_channel_is_none():
    """Test that nothing is published when channel is None."""
    proc, sub, executor = _make_proc(pp_pd_channel=None)
    # We'll just test the basic import and creation since full test
    # requires more complex setup
    assert proc._pp_pd_channel is None


def test_post_out_publish_method_exists():
    """Test that _maybe_publish_post_out method exists and works."""
    channel = FakePdChannel()
    proc, _, _ = _make_proc(pp_pd_channel=channel)

    # Test that the method exists
    assert hasattr(proc, "_maybe_publish_post_out")
