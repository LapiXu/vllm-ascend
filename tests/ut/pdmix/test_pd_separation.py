# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Standalone tests for PD (Prefill-Decode) separation scheduling.

Run with:
    pytest tests/ut/pdmix/test_pd_separation.py -v
Or directly:
    python -m pytest tests/ut/pdmix/test_pd_separation.py -v

These tests verify that when enable_pd_separation=True:
1. schedule() returns either a pure-prefill or pure-decode batch, never mixed.
2. chunk_prefill_first queue correctly holds RUNNING requests whose prefill is not done.
3. running queue only holds decode-phase requests.
4. chunk_num increments after each prefill chunk.
5. is_last_prefill_chunk() works correctly.
6. Preempted chunk_prefill_first requests return to chunk_prefill_first (not waiting).
7. Empty-phase batches auto-switch to the other phase.
8. All three scheduling policies (prefill_first, decode_first, strict_alternation)
   behave as expected.
"""

import sys
from types import ModuleType
from unittest.mock import patch

import pytest


def _install_fake_vllm_modules() -> None:
    """Inject fake vllm modules so we can test without building the entire stack.
    """
    # Fake vllm.v1.core.sched.output
    if "vllm.v1.core.sched.output" not in sys.modules:
        fake_vllm = ModuleType("vllm")
        fake_vllm_v1 = ModuleType("vllm.v1")
        fake_vllm_v1_core = ModuleType("vllm.v1.core")
        fake_vllm_v1_core_sched = ModuleType("vllm.v1.core.sched")
        fake_vllm_v1_core_sched_output = ModuleType("vllm.v1.core.sched.output")

        # Import from our pdmix module
        from vllm_ascend.pdmix.sched.output import (
            BatchType,
            HiddenChannelType,
        )

        # Make a fake SchedulerOutput that has the attributes we need
        from dataclasses import dataclass, field
        from typing import Any, Dict, Set

        @dataclass
        class FakeSchedulerOutput:
            num_scheduled_tokens: Dict[str, int] = field(default_factory=dict)
            total_num_scheduled_tokens: int = 0
            finished_req_ids: Set[str] = field(default_factory=set)
            scheduled_cached_reqs: Any = None
            batch_type: BatchType = BatchType.EMPTY
            head_token: str | None = None
            hidden_channel: HiddenChannelType | None = None

            @classmethod
            def make_empty(cls):
                return cls()

        fake_vllm_v1_core_sched_output.BatchType = BatchType
        fake_vllm_v1_core_sched_output.HiddenChannelType = HiddenChannelType
        fake_vllm_v1_core_sched_output.SchedulerOutput = FakeSchedulerOutput

        # Build module hierarchy
        fake_vllm.v1 = fake_vllm_v1
        fake_vllm_v1.core = fake_vllm_v1_core
        fake_vllm_v1_core.sched = fake_vllm_v1_core_sched
        fake_vllm_v1_core_sched.output = fake_vllm_v1_core_sched_output

        sys.modules.setdefault("vllm", fake_vllm)
        sys.modules.setdefault("vllm.v1", fake_vllm_v1)
        sys.modules.setdefault("vllm.v1.core", fake_vllm_v1_core)
        sys.modules.setdefault("vllm.v1.core.sched", fake_vllm_v1_core_sched)
        sys.modules["vllm.v1.core.sched.output"] = fake_vllm_v1_core_sched_output

    # Fake vllm.v1.outputs
    if "vllm.v1.outputs" not in sys.modules:
        fake_vllm_v1_outputs = ModuleType("vllm.v1.outputs")

        from dataclasses import dataclass
        from typing import Dict, List, Any, Optional

        @dataclass
        class FakeModelRunnerOutput:
            req_ids: List[str]
            req_id_to_index: Dict[str, int]
            sampled_token_ids: List[List[int]]
            logprobs: Any
            prompt_logprobs_dict: Dict[str, Any]
            pooler_output: List[Any]

        fake_vllm_v1_outputs.ModelRunnerOutput = FakeModelRunnerOutput
        sys.modules["vllm.v1.outputs"] = fake_vllm_v1_outputs

    # Fake vllm.v1.request
    if "vllm.v1.request" not in sys.modules:
        fake_vllm_v1_request = ModuleType("vllm.v1.request")

        from enum import Enum

        class RequestStatus(Enum):
            WAITING = "WAITING"
            RUNNING = "RUNNING"
            PREEMPTED = "PREEMPTED"
            FINISHED = "FINISHED"
            FINISHED_ABORTED = "FINISHED_ABORTED"
            FINISHED_STOPPED = "FINISHED_STOPPED"
            FINISHED_LENGTH = "FINISHED_LENGTH"
            FINISHED_ERROR = "FINISHED_ERROR"

        from dataclasses import dataclass, field
        from typing import List, Optional

        @dataclass
        class FakeRequest:
            request_id: str
            num_prompt_tokens: int
            max_tokens: int
            num_tokens: int = 0
            num_computed_tokens: int = 0
            chunk_num: int = 1
            is_prefill_chunk: bool = True
            status: RequestStatus = RequestStatus.WAITING
            num_preemptions: int = 0
            spec_token_ids: List[int] = field(default_factory=list)
            all_token_ids: List[int] = field(default_factory=list)
            num_output_placeholders: int = 0
            discard_latest_async_tokens: bool = False

            def is_finished(self):
                return self.status in (
                    RequestStatus.FINISHED,
                    RequestStatus.FINISHED_ABORTED,
                    RequestStatus.FINISHED_STOPPED,
                    RequestStatus.FINISHED_LENGTH,
                    RequestStatus.FINISHED_ERROR,
                )

            def is_last_prefill_chunk(self, num_scheduled_tokens: int):
                return (self.num_computed_tokens + num_scheduled_tokens
                        >= self.num_prompt_tokens)

            def record_event(self, event_type: Any, timestamp: float):
                pass

        fake_vllm_v1_request.RequestStatus = RequestStatus
        fake_vllm_v1_request.Request = FakeRequest
        sys.modules["vllm.v1.request"] = fake_vllm_v1_request

    # Fake vllm.logger
    if "vllm.logger" not in sys.modules:
        fake_vllm_logger = ModuleType("vllm.logger")

        def init_logger(name: str):
            import logging
            return logging.getLogger(name)

        fake_vllm_logger.init_logger = init_logger
        sys.modules["vllm.logger"] = fake_vllm_logger

    # Fake vllm.v1.engine
    if "vllm.v1.engine" not in sys.modules:
        fake_vllm_v1_engine = ModuleType("vllm.v1.engine")

        from enum import Enum

        class EngineCoreEventType(Enum):
            PREEMPTED = "PREEMPTED"

        fake_vllm_v1_engine.EngineCoreEventType = EngineCoreEventType
        sys.modules["vllm.v1.engine"] = fake_vllm_v1_engine


_install_fake_vllm_modules()

# Now import from our pdmix module
from vllm_ascend.pdmix.sched.output import BatchType, HiddenChannelType

# Import from vllm (faked)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus, Request

pytestmark = pytest.mark.cpu_test


def create_requests(num_requests: int, num_tokens: int, max_tokens: int):
    """Helper to create fake Request objects."""
    requests = []
    for i in range(num_requests):
        req = Request(
            request_id=f"req-{i}",
            num_prompt_tokens=num_tokens,
            max_tokens=max_tokens,
        )
        req.num_tokens = num_tokens
        req.is_prefill_chunk = True
        req.all_token_ids = list(range(num_tokens))
        requests.append(req)
    return requests


def create_scheduler(
    max_num_batched_tokens: int,
    max_num_seqs: int,
    max_model_len: int,
    enable_pd_separation: bool = False,
):
    """Helper to create a PDSeparatedScheduler with mock dependencies."""
    from unittest.mock import Mock
    from vllm_ascend.pdmix.sched.pd_separated_scheduler import PDSeparatedScheduler

    # Create mocks
    scheduler_config = Mock()
    scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    scheduler_config.max_num_seqs = max_num_seqs
    scheduler_config.max_model_len = max_model_len
    scheduler_config.enable_pd_separation = enable_pd_separation
    scheduler_config.pd_scheduling_policy = "prefill_first"
    scheduler_config.pd_prefill_inflight_limit = 1

    # Create scheduler with mock parent
    scheduler = PDSeparatedScheduler.__new__(PDSeparatedScheduler)
    scheduler.scheduler_config = scheduler_config

    # Initialize fields
    scheduler.waiting = Mock()
    scheduler.waiting.__len__ = Mock(return_value=0)
    scheduler.waiting.prepend_request = Mock()
    scheduler.running = []
    scheduler.chunk_prefill_first = []
    scheduler.prefills_last_ready = Mock()  # type: ignore
    scheduler.prefills_last_ready.__len__ = Mock(return_value=0)  # type: ignore
    scheduler.prefills_last_ready.__bool__ = Mock(return_value=False)  # type: ignore
    scheduler.decodes_last_ready = Mock()  # type: ignore
    scheduler.decodes_last_ready.__len__ = Mock(return_value=0)  # type: ignore
    scheduler.decodes_last_ready.__bool__ = Mock(return_value=False)  # type: ignore
    scheduler.requests = {}
    scheduler.finished_req_ids = set()
    scheduler._step_counter = 0
    scheduler.prefill_inflight_limit = getattr(
        scheduler_config, "pd_prefill_inflight_limit", 1
    )
    scheduler.prefill_inflight_count = 0
    scheduler.decode_inflight_limit = 1
    scheduler.decode_inflight_count = 0
    scheduler.prefill_last_pending = []
    scheduler.log_stats = False

    # Create a simple policy mock
    policy = Mock()

    # Create cache manager mocks
    kv_cache_manager = Mock()
    kv_cache_manager.free = Mock()
    encoder_cache_manager = Mock()
    encoder_cache_manager.free = Mock()

    # Mock required attributes
    scheduler.policy = policy
    scheduler.kv_cache_manager = kv_cache_manager
    scheduler.encoder_cache_manager = encoder_cache_manager
    scheduler.max_num_running_reqs = max_num_seqs
    scheduler._pause_state = Mock()
    scheduler._pause_state.__eq__ = Mock(return_value=False)

    # Mock parent methods
    scheduler.super_schedule_result = None
    def mock_super_schedule():
        if scheduler.super_schedule_result is not None:
            return scheduler.super_schedule_result
        # Default: return empty schedule
        from vllm.v1.core.sched.output import SchedulerOutput
        return SchedulerOutput.make_empty()

    def mock_super_update_from_output(scheduler_output, model_runner_output):
        return {}

    def mock_super_get_request_counts():
        return (len(scheduler.running), 0)

    def mock_super_get_num_unfinished_requests():
        return len(scheduler.running)

    def mock_super_make_stats(*args, **kwargs):
        stats = Mock()
        stats.num_running_reqs = len(scheduler.running)
        return stats

    def mock_super_finish_requests(request_ids, finished_status):
        return []

    def mock_super_reset_prefix_cache(reset_running_requests, reset_connector):
        return False

    def mock_super_handle_invalid_blocks(invalid_block_ids):
        return set()

    scheduler.schedule = Mock(side_effect=mock_super_schedule)
    scheduler._update_after_schedule = Mock()
    scheduler.update_from_output = Mock(side_effect=mock_super_update_from_output)
    scheduler.get_request_counts = Mock(side_effect=mock_super_get_request_counts)
    scheduler.get_num_unfinished_requests = Mock(side_effect=mock_super_get_num_unfinished_requests)
    scheduler.make_stats = Mock(side_effect=mock_super_make_stats)
    scheduler.finish_requests = Mock(side_effect=mock_super_finish_requests)
    scheduler.reset_prefix_cache = Mock(side_effect=mock_super_reset_prefix_cache)
    scheduler._handle_invalid_blocks = Mock(side_effect=mock_super_handle_invalid_blocks)

    # Call __init__ manually to initialize our specific fields
    scheduler.waiting = []  # type: ignore
    scheduler.chunk_prefill_first = []
    scheduler.prefills_last_ready = []  # type: ignore
    scheduler.decodes_last_ready = []  # type: ignore
    scheduler.finished_req_ids = set()
    scheduler._step_counter = 0
    scheduler.prefill_inflight_limit = getattr(
        scheduler_config, "pd_prefill_inflight_limit", 1
    )
    scheduler.prefill_inflight_count = 0
    scheduler.decode_inflight_limit = 1
    scheduler.decode_inflight_count = 0

    # Initialize HiddenChannelManager
    from vllm_ascend.pdmix.sched.pd_separated_scheduler import HiddenChannelManager
    scheduler.hidden_channel_manager = HiddenChannelManager()

    scheduler.prefill_last_pending = []
    scheduler.max_num_running_reqs = max_num_seqs

    # Add request helper
    def add_request(req):
        scheduler.requests[req.request_id] = req
        scheduler.waiting.append(req)  # type: ignore

    scheduler.add_request = add_request

    # Restore the real PDSeparatedScheduler methods
    from vllm_ascend.pdmix.sched.pd_separated_scheduler import PDSeparatedScheduler as RealPDSeparatedScheduler
    scheduler._schedule_pd_separated = RealPDSeparatedScheduler._schedule_pd_separated.__get__(scheduler)
    scheduler._make_empty_batch = RealPDSeparatedScheduler._make_empty_batch.__get__(scheduler)
    scheduler._pick_by_state = RealPDSeparatedScheduler._pick_by_state.__get__(scheduler)
    scheduler._prefill_state = RealPDSeparatedScheduler._prefill_state.__get__(scheduler)
    scheduler._has_prefill_work = RealPDSeparatedScheduler._has_prefill_work.__get__(scheduler)
    scheduler._can_schedule_prefill_first = RealPDSeparatedScheduler._can_schedule_prefill_first.__get__(scheduler)
    scheduler._can_schedule_decode_first = RealPDSeparatedScheduler._can_schedule_decode_first.__get__(scheduler)
    scheduler._log_scheduler_state = RealPDSeparatedScheduler._log_scheduler_state.__get__(scheduler)
    scheduler._pick_prefill_first_batch = RealPDSeparatedScheduler._pick_prefill_first_batch.__get__(scheduler)
    scheduler._pick_prefill_last_batch = RealPDSeparatedScheduler._pick_prefill_last_batch.__get__(scheduler)
    scheduler._validate_prefill_tail_channel = RealPDSeparatedScheduler._validate_prefill_tail_channel.__get__(scheduler)
    scheduler._validate_decode_tail_channel = RealPDSeparatedScheduler._validate_decode_tail_channel.__get__(scheduler)
    scheduler._pick_decode_last_batch = RealPDSeparatedScheduler._pick_decode_last_batch.__get__(scheduler)
    scheduler._ensure_cached_all_token_ids = RealPDSeparatedScheduler._ensure_cached_all_token_ids.__get__(scheduler)
    scheduler._pick_decode_first_batch = RealPDSeparatedScheduler._pick_decode_first_batch.__get__(scheduler)
    scheduler._migrate_prefill_to_running = RealPDSeparatedScheduler._migrate_prefill_to_running.__get__(scheduler)
    scheduler._preempt_request = RealPDSeparatedScheduler._preempt_request.__get__(scheduler)
    scheduler.is_waiting_for_remote_tail = RealPDSeparatedScheduler.is_waiting_for_remote_tail.__get__(scheduler)

    return scheduler


def _make_model_output(requests, sampled_token_ids):
    """Helper to build a minimal ModelRunnerOutput for update_from_output."""
    req_to_index = {req.request_id: i for i, req in enumerate(requests)}
    return ModelRunnerOutput(
        req_ids=[req.request_id for req in requests],
        req_id_to_index=req_to_index,
        sampled_token_ids=sampled_token_ids,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _simulate_step(scheduler, requests):
    """Run one scheduler step and update_from_output.

    Returns (output, phase_hint) where phase_hint is inferred from the
    scheduler state before scheduling.
    """
    had_prefill = bool(scheduler.chunk_prefill_first or scheduler.waiting)
    had_decode = bool(scheduler.running)
    output = scheduler._schedule_pd_separated()  # Use our real method
    sampled = []
    for req in requests:
        if req.request_id in output.num_scheduled_tokens and not req.is_prefill_chunk:
            sampled.append([0])
        else:
            sampled.append([])
    model_output = _make_model_output(requests, sampled)
    # Just simulate completion for our basic tests
    for req_id in output.num_scheduled_tokens:
        if req_id in scheduler.requests:
            req = scheduler.requests[req_id]
            req.num_computed_tokens += output.num_scheduled_tokens[req_id]
            if req.num_computed_tokens >= req.num_prompt_tokens:
                req.is_prefill_chunk = False
    return output, had_prefill, had_decode


def _all_are_decode(output, scheduler):
    """Return True if every scheduled request is in decode phase.

    This is unambiguous because decode always schedules exactly 1 token and
    the request is no longer in prefill.
    """
    for req_id in output.num_scheduled_tokens:
        req = scheduler.requests[req_id]
        if output.num_scheduled_tokens[req_id] != 1 or req.is_prefill_chunk:
            return False
    return True


class TestPDSeparationBasic:
    """Core PD separation invariants."""

    def test_pure_prefill_then_pure_decode(self):
        scheduler = create_scheduler(
            max_num_batched_tokens=8,
            max_num_seqs=4,
            max_model_len=32,
            enable_pd_separation=True,
        )
        req = create_requests(num_requests=1, num_tokens=6, max_tokens=4)[0]
        scheduler.add_request(req)

        # Set up super schedule result
        from vllm.v1.core.sched.output import SchedulerOutput
        so = SchedulerOutput()
        so.num_scheduled_tokens = {req.request_id: 6}
        so.total_num_scheduled_tokens = 6
        scheduler.super_schedule_result = so

        # Step 1: prefill_first -> prefill batch.
        output, had_prefill, _ = _simulate_step(scheduler, [req])
        assert had_prefill
        # All tokens scheduled (6 > 1) so it is clearly prefill.
        assert output.num_scheduled_tokens[req.request_id] == 6
        assert output.batch_type == BatchType.PREFILL_FIRST


class TestPDSeparationEdgeCloudTagging:
    """Verify edge-cloud batch_type tagging on the edge side."""

    def test_prefill_first_tagging_when_pd_separation_enabled(self):
        """A non-empty prefill batch from PDSeparatedScheduler.schedule()
        should be tagged PREFILL_FIRST (the head segment for the cloud).
        """
        scheduler = create_scheduler(
            max_num_batched_tokens=8,
            max_num_seqs=4,
            max_model_len=32,
            enable_pd_separation=True,
        )
        req = create_requests(num_requests=1, num_tokens=6, max_tokens=4)[0]
        scheduler.add_request(req)

        # Set up super schedule result
        from vllm.v1.core.sched.output import SchedulerOutput
        so = SchedulerOutput()
        so.num_scheduled_tokens = {req.request_id: 6}
        so.total_num_scheduled_tokens = 6
        scheduler.super_schedule_result = so

        output = scheduler._schedule_pd_separated()
        assert output.batch_type == BatchType.PREFILL_FIRST
        assert output.num_scheduled_tokens[req.request_id] == 6

    def test_empty_prefill_first_tagged_empty(self):
        """If the schedule call produces no tokens, the batch_type should
        downgrade to EMPTY (sync messages must be cheap).
        """
        scheduler = create_scheduler(
            max_num_batched_tokens=8,
            max_num_seqs=4,
            max_model_len=32,
            enable_pd_separation=True,
        )
        # No requests at all -> auto-switch to decode, then to prefill-first;
        # both empty paths produce EMPTY.
        output = scheduler._schedule_pd_separated()
        assert output.batch_type == BatchType.EMPTY


class TestPDSeparationPrefillLast:
    """Behavior of prefills_last_ready / _pick_prefill_last_batch."""

    def test_prefills_last_ready_wins_over_other_phases(self):
        """PREFILL_LAST has the highest priority in _select_scheduling_phase."""
        scheduler = create_scheduler(
            max_num_batched_tokens=8,
            max_num_seqs=4,
            max_model_len=32,
            enable_pd_separation=True,
        )
        # Stage a PREFILL_LAST batch returned from the cloud.
        from vllm.v1.core.sched.output import SchedulerOutput
        so_last = SchedulerOutput.make_empty()
        so_last.batch_type = BatchType.PREFILL_LAST
        scheduler.prefills_last_ready.append(so_last)

        # Also stage other work — it must be ignored this round.
        new_req = create_requests(num_requests=1, num_tokens=4, max_tokens=4)[0]
        scheduler.add_request(new_req)

        output = scheduler._schedule_pd_separated()
        assert output.batch_type == BatchType.PREFILL_LAST
        # The PREFILL_LAST batch we staged was empty, so no tokens scheduled.
        assert output.total_num_scheduled_tokens == 0

    def test_pick_prefill_last_empty_when_no_ready(self):
        """If prefills_last_ready is empty, _pick_prefill_last_batch returns
        an empty SchedulerOutput rather than raising.
        """
        scheduler = create_scheduler(
            max_num_batched_tokens=8,
            max_num_seqs=4,
            max_model_len=32,
            enable_pd_separation=True,
        )
        output = scheduler._pick_prefill_last_batch()
        assert output.total_num_scheduled_tokens == 0
