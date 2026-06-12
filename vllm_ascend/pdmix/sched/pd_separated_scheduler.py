# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
import time
from collections import deque
from collections.abc import Iterable
from typing import Any
from uuid import uuid4

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm_ascend.pdmix.sched.output import (
    BatchType,
    HiddenChannelType,
    PDMixSchedulerMetadata,
    get_pdmix_metadata,
    set_pdmix_metadata,
)

logger = init_logger(__name__)


class PrefillState(enum.Enum):
    """Edge-side prefill in-flight state machine.

    See Phase5 design in `PDbatch分离边云协同Phase5&7详细设计.md`.
    """
    IDLE = "idle"       # prefill_inflight_count == 0
    LOW = "low"         # prefill_inflight_count == 1
    HIGH = "high"       # prefill_inflight_count >= prefill_inflight_limit


class HiddenChannelManager:
    """Manages data-plane hidden tensor channels for edge-cloud PD separation.

    Two prefill channels (PREFILL_1 / PREFILL_2) support 2P1D; one decode
    channel (DECODE) supports single in-flight decode. Channels are allocated
    in FIFO order and freed when the tail segment completes.
    """

    def __init__(self) -> None:
        self._free_prefills: deque[HiddenChannelType] = deque([
            HiddenChannelType.PREFILL_1,
            HiddenChannelType.PREFILL_2,
        ])
        # Mapping from head_token to the allocated channel.  Only prefill
        # batches are recorded here; decode batches always use DECODE and
        # do not need a mapping.
        self._head_token_to_channel: dict[str, HiddenChannelType] = {}

    # ------------------------------------------------------------------ #
    # Prefill channel allocation / release                               #
    # ------------------------------------------------------------------ #
    def allocate_prefill(self, head_token: str) -> HiddenChannelType:
        """Allocate a free prefill channel for the batch identified by
        `head_token`. Raises if none available.
        """
        if not self._free_prefills:
            raise RuntimeError(
                "No free prefill hidden channel available"
            )
        channel = self._free_prefills.popleft()
        self._head_token_to_channel[head_token] = channel
        print(
            f"[PD-CHAN] allocate prefill channel={channel.value} "
            f"head_token={head_token}"
        )
        return channel

    def release_prefill(self, head_token: str) -> HiddenChannelType | None:
        """Release the prefill channel previously allocated for
        `head_token`. Returns the freed channel (or None if not found).
        """
        channel = self._head_token_to_channel.pop(head_token, None)
        if channel is None:
            return None
        self._free_prefills.append(channel)
        print(
            f"[PD-CHAN] release prefill channel={channel.value} "
            f"head_token={head_token}"
        )
        return channel

    def has_free_prefill(self) -> bool:
        return bool(self._free_prefills)

    # ------------------------------------------------------------------ #
    # Decode channel (always DECODE, no free-list)                        #
    # ------------------------------------------------------------------ #
    @staticmethod
    def decode_channel() -> HiddenChannelType:
        return HiddenChannelType.DECODE

    # ------------------------------------------------------------------ #
    # Introspection                                                      #
    # ------------------------------------------------------------------ #
    def get_channel(self, head_token: str) -> HiddenChannelType | None:
        return self._head_token_to_channel.get(head_token)

    @property
    def in_use_prefills(self) -> list[HiddenChannelType]:
        return list(self._head_token_to_channel.values())


class PDSeparatedScheduler:
    """Scheduler that separates prefill and decode into distinct steps.

    In edge-cloud PD-separated mode the four cardinal phases are:
      - PREFILL_FIRST  (edge head segment)
      - PREFILL_LAST   (edge tail segment, sourced from cloud-returned outputs)
      - DECODE_FIRST   (Phase 4)
      - DECODE_LAST    (Phase 4)

    This class owns the request bookkeeping for *first* segments
    (`chunk_prefill_first` + parent's `waiting` / `running`) and the
    ready queues for *last* segments (`prefills_last_ready` /
    `decodes_last_ready`), which are filled by the EngineCore from the
    POST_OUT channel before each `schedule()` call.
    """

    def __init__(self, *args, **kwargs) -> None:
        # Requests that have started their P-first segment but have not yet
        # been fully consumed (still chunking, or still in flight on cloud).
        self.chunk_prefill_first: list = []

        # SchedulerOutputs returned from cloud (POST_OUT channel) carrying
        # the metadata needed to execute the edge tail segment.
        # Populated by EngineCore.step() before calling self.schedule().
        self.prefills_last_ready: deque[SchedulerOutput] = deque()
        self.decodes_last_ready: deque[SchedulerOutput] = deque()

        self._step_counter: int = 0

        # In-flight prefill limit (head-segment batches).
        self.prefill_inflight_limit: int = 1
        self.prefill_inflight_count: int = 0
        self.decode_inflight_limit: int = 1
        self.decode_inflight_count: int = 0

        # Phase6 data-plane channel manager.  Two prefill hidden channels are
        # available for 2P1D; decode uses a dedicated fixed channel.
        self.hidden_channel_manager = HiddenChannelManager()

        # Buffer queue: requests whose P-first segment is done but P-last
        # segment has not yet returned from the cloud.  Not eligible for
        # decode scheduling until PL completes and they are moved to running.
        self.prefill_last_pending: list = []

        # Parent scheduler fields (simplified for pdmix module)
        self.waiting: list = []
        self.running: list = []
        self.requests: dict = {}
        self.finished_req_ids: set = set()
        self.max_num_running_reqs: int = 0

    def schedule(self) -> SchedulerOutput:
        return self._schedule_pd_separated()

    def _make_empty_batch(self) -> SchedulerOutput:
        scheduler_output = SchedulerOutput.make_empty()
        scheduler_output.finished_req_ids = self.finished_req_ids
        self.finished_req_ids = set()
        return scheduler_output

    def _schedule_pd_separated(self) -> SchedulerOutput:
        state = self._prefill_state()
        scheduler_output = self._pick_by_state(state)
        has_work = scheduler_output.total_num_scheduled_tokens > 0
        is_tail = get_pdmix_metadata(scheduler_output).batch_type in (
            BatchType.PREFILL_LAST,
            BatchType.DECODE_LAST,
        )
        if has_work or is_tail:
            self._log_scheduler_state(state)
        return scheduler_output

    def _pick_by_state(self, state: PrefillState) -> SchedulerOutput:
        if state == PrefillState.IDLE:
            # IDLE: P首/chunk0首 > D首 > D尾 > Empty.
            if self._can_schedule_prefill_first():
                return self._pick_prefill_first_batch()
            if self._can_schedule_decode_first():
                return self._pick_decode_first_batch()
            if self.decodes_last_ready:
                return self._pick_decode_last_batch()
            return self._make_empty_batch()

        if state == PrefillState.LOW:
            # LOW: chunk/P首(when slot available) > P尾 > D首 > D尾 > Empty.
            if self._can_schedule_prefill_first():
                return self._pick_prefill_first_batch()
            if self.prefills_last_ready:
                return self._pick_prefill_last_batch()
            if self._can_schedule_decode_first():
                return self._pick_decode_first_batch()
            if self.decodes_last_ready:
                return self._pick_decode_last_batch()
            return self._make_empty_batch()

        # HIGH: P尾 > D首 > D尾 > Empty. New P首 is forbidden.
        if self.prefills_last_ready:
            return self._pick_prefill_last_batch()
        if self._can_schedule_decode_first():
            return self._pick_decode_first_batch()
        if self.decodes_last_ready:
            return self._pick_decode_last_batch()
        return self._make_empty_batch()

    def is_waiting_for_remote_tail(self) -> bool:
        """True when local requests exist only as remote in-flight work.

        In this state the edge has no local batch to execute until POST_OUT
        returns a PREFILL_LAST/DECODE_LAST, so the EngineCore should yield
        instead of tight-loop scheduling EMPTY batches.
        """
        return bool(
            (self.prefill_inflight_count > 0 or self.decode_inflight_count > 0)
            and not self.prefills_last_ready
            and not self.decodes_last_ready
            and not self._can_schedule_prefill_first()
            and not self._can_schedule_decode_first()
        )

    def _prefill_state(self) -> PrefillState:
        if self.prefill_inflight_count <= 0:
            return PrefillState.IDLE
        if (
            self.prefill_inflight_limit > 1
            and self.prefill_inflight_count >= self.prefill_inflight_limit
        ):
            return PrefillState.HIGH
        return PrefillState.LOW

    def _has_prefill_work(self) -> bool:
        return bool(self.chunk_prefill_first or self.waiting)

    def _can_schedule_prefill_first(self) -> bool:
        return (
            self._has_prefill_work()
            and self.prefill_inflight_count < self.prefill_inflight_limit
            and self.hidden_channel_manager.has_free_prefill()
        )

    def _can_schedule_decode_first(self) -> bool:
        return bool(
            self.running
            and self.decode_inflight_count < self.decode_inflight_limit
        )

    def _log_scheduler_state(self, state: PrefillState) -> None:
        self._step_counter += 1
        print(
            f"\r\n[PD] Step{self._step_counter}, state is {state.value},    "
            f"waiting[]: {len(self.waiting)}, "
            f"chunk_prefill_first[]: {len(self.chunk_prefill_first)}, "
            f"prefill_last_pending[]: {len(self.prefill_last_pending)}, "
            f"running[]: {len(self.running)}, "
            f"prefills_last_ready[]: {len(self.prefills_last_ready)}, "
            f"decodes_last_ready[]: {len(self.decodes_last_ready)}, "
            f"prefill_inflight: {self.prefill_inflight_count}/{self.prefill_inflight_limit}, "
            f"decode_inflight: {self.decode_inflight_count}/{self.decode_inflight_limit}"
        )
        for req in self.chunk_prefill_first:
            print(
                f"[PD] chunk_prefill_first[{req.request_id}],    "
                f"num_prompt_tokens: {req.num_prompt_tokens}, "
                f"num_tokens: {req.num_tokens}, "
                f"num_computed_tokens: {req.num_computed_tokens}, "
                f"chunk_num: {req.chunk_num}"
            )
        for req in self.running:
            print(
                f"[PD] running[{req.request_id}],    "
                f"num_prompt_tokens: {req.num_prompt_tokens}, "
                f"num_tokens: {req.num_tokens}, "
                f"num_computed_tokens: {req.num_computed_tokens}, "
                f"chunk_num: {req.chunk_num}"
            )

    def _pick_prefill_first_batch(self) -> SchedulerOutput:
        # Simplified version for pdmix: just create an empty batch with tags
        # since we don't have the full parent scheduler implementation
        scheduler_output = self._make_empty_batch()

        head_token = uuid4().hex
        hidden_channel = self.hidden_channel_manager.allocate_prefill(head_token)
        set_pdmix_metadata(scheduler_output, PDMixSchedulerMetadata(
            batch_type=BatchType.PREFILL_FIRST,
            head_token=head_token,
            hidden_channel=hidden_channel,
        ))
        self.prefill_inflight_count += 1

        print(
            f"[PD] _pick_prefill_first_batch done: "
            f"chunk_prefill_first[]: {len(self.chunk_prefill_first)}, "
            f"prefill_last_pending[]: {len(self.prefill_last_pending)}, "
            f"running[]: {len(self.running)}, "
            f"prefill_inflight: {self.prefill_inflight_count}/{self.prefill_inflight_limit}"
        )

        return scheduler_output

    def _pick_prefill_last_batch(self) -> SchedulerOutput:
        """Pop one cloud-returned SchedulerOutput from prefills_last_ready.

        The cloud has already rewritten `batch_type=PREFILL_LAST` and kept
        all original KV / sampling metadata intact, so the edge worker can
        directly run segment_e + sampler on it. We also remove the involved
        requests from `chunk_prefill_first` so the parent class's
        `update_from_output` does not double-account them.
        """
        if not self.prefills_last_ready:
            return self._make_empty_batch()
        so = self.prefills_last_ready.popleft()
        assert get_pdmix_metadata(so).batch_type == BatchType.PREFILL_LAST, (
            f"prefills_last_ready expects PREFILL_LAST, got {get_pdmix_metadata(so).batch_type}"
        )
        # Drop these reqs from chunk_prefill_first. Keep them in
        # prefill_last_pending until update_from_output() moves them to running.
        last_req_ids = set(so.num_scheduled_tokens.keys())
        if last_req_ids:
            self.chunk_prefill_first = [
                req for req in self.chunk_prefill_first
                if req.request_id not in last_req_ids
            ]
        self._validate_prefill_tail_channel(so)
        print(
            f"[PD] _pick_prefill_last_batch popped {len(last_req_ids)} reqs; "
            f"remaining prefills_last_ready[]: {len(self.prefills_last_ready)}, "
            f"prefill_last_pending[]: {len(self.prefill_last_pending)}, "
            f"hidden_channel: {get_pdmix_metadata(so).hidden_channel}"
        )
        return so

    def _validate_prefill_tail_channel(self, scheduler_output: SchedulerOutput) -> None:
        token = get_pdmix_metadata(scheduler_output).head_token
        channel = get_pdmix_metadata(scheduler_output).hidden_channel
        if not token:
            raise RuntimeError("PREFILL_LAST missing head_token")
        if channel not in (HiddenChannelType.PREFILL_1, HiddenChannelType.PREFILL_2):
            raise RuntimeError(
                f"PREFILL_LAST expects a prefill hidden channel, got {channel}"
            )
        expected = self.hidden_channel_manager.get_channel(token)
        if expected != channel:
            raise RuntimeError(
                f"PREFILL_LAST hidden channel mismatch: expected {expected}, "
                f"got {channel}, head_token={token}"
            )

    def _validate_decode_tail_channel(self, scheduler_output: SchedulerOutput) -> None:
        if get_pdmix_metadata(scheduler_output).hidden_channel != HiddenChannelType.DECODE:
            raise RuntimeError(
                "DECODE_LAST expects decode hidden channel, got "
                f"{get_pdmix_metadata(scheduler_output).hidden_channel}"
            )

    def _pick_decode_last_batch(self) -> SchedulerOutput:
        if not self.decodes_last_ready:
            return self._make_empty_batch()
        so = self.decodes_last_ready.popleft()
        assert get_pdmix_metadata(so).batch_type == BatchType.DECODE_LAST, (
            f"decodes_last_ready expects DECODE_LAST, got {get_pdmix_metadata(so).batch_type}"
        )
        self._validate_decode_tail_channel(so)
        print(
            f"[PD] _pick_decode_last_batch popped "
            f"{len(so.num_scheduled_tokens)} reqs; "
            f"remaining decodes_last_ready[]: {len(self.decodes_last_ready)}"
        )
        return so

    def _ensure_cached_all_token_ids(
        self, scheduler_output: SchedulerOutput,
    ) -> None:
        """Ensure every cached decode req carries all_token_ids.

        In PD-separated mode the edge worker's persistent input_batch may not
        retain a request across the PF -> PL -> DF transitions (the tail segment
        may have been skipped by `_update_states`), yet the async-scheduling
        model runner requires `all_token_ids` to reconstruct
        `output_token_ids` for any resumed (req_index is None) request with
        `num_output_tokens > 0`.

        The base `_make_cached_request_data` only fills `all_token_ids`
        when the request was *not* scheduled in the previous step
        (`prev_step_scheduled_req_ids`).  Because PD separation interleaves
        PF / PL / DF / DL phases — and PL/DL are popped from cloud-returned
        queues without going through `super().schedule()` — the scheduler's
        `prev_step_scheduled_req_ids` can be stale or misleading, causing
        `all_token_ids` to be omitted for requests that the worker actually
        needs it for.

        This helper unconditionally back-fills `all_token_ids` for any
        cached request that has output tokens but is missing from the dict.
        The extra payload is cheap (control-plane only) and avoids the
        `KeyError` in `gpu_model_runner._update_states`.
        """
        cached_reqs = scheduler_output.scheduled_cached_reqs
        if cached_reqs is None:
            return
        for req_id, num_output_tokens in zip(
            cached_reqs.req_ids,
            cached_reqs.num_output_tokens,
        ):
            if num_output_tokens > 0 and req_id not in cached_reqs.all_token_ids:
                cached_reqs.all_token_ids[req_id] = (
                    self.requests[req_id].all_token_ids.copy()
                )

    def _pick_decode_first_batch(self) -> SchedulerOutput:
        # Simplified version for pdmix
        if not self.running:
            return self._make_empty_batch()

        scheduler_output = self._make_empty_batch()

        head_token = uuid4().hex
        hidden_channel = self.hidden_channel_manager.decode_channel()
        set_pdmix_metadata(scheduler_output, PDMixSchedulerMetadata(
            batch_type=BatchType.DECODE_FIRST,
            head_token=head_token,
            hidden_channel=hidden_channel,
        ))
        self._ensure_cached_all_token_ids(scheduler_output)
        self.decode_inflight_count += 1

        print(
            f"[PD] _pick_decode_first_batch done: "
            f"running: {len(self.running)}, "
            f"chunk_prefill_first: {len(self.chunk_prefill_first)}, "
            f"prefill_last_pending: {len(self.prefill_last_pending)}"
        )

        return scheduler_output

    def _migrate_prefill_to_running(self) -> None:
        completed = [
            req for req in self.chunk_prefill_first if not req.is_prefill_chunk
        ]
        if completed:
            print(
                f"[PD] _migrate_prefill_to_running: moving {len(completed)} "
                f"requests from chunk_prefill_first to running"
            )
        for req in completed:
            self.chunk_prefill_first.remove(req)
            self.running.append(req)

    def _preempt_request(self, request: Any, timestamp: float) -> None:
        assert request.status.value == "RUNNING", (
            "Only running requests can be preempted"
        )
        # Simplified for pdmix - just record the event
        request.status = type(request.status)("PREEMPTED")
        request.num_preemptions += 1
        if request.spec_token_ids:
            request.spec_token_ids = []
        if self.log_stats:
            request.record_event(None, timestamp)

        if request.is_prefill_chunk:
            print(
                f"[PD] _preempt_request: request {request.request_id} "
                f"stays in chunk_prefill_first "
                f"(computed={request.num_computed_tokens})"
            )
            self.chunk_prefill_first.append(request)
        else:
            print(
                f"[PD] _preempt_request: request {request.request_id} "
                f"goes back to waiting (decode or finished prefill)"
            )
            request.num_computed_tokens = 0
            self.waiting.insert(0, request)

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        was_prefill_map = {}
        for req_id in scheduler_output.num_scheduled_tokens:
            was_prefill_map[req_id] = self.requests[req_id].is_prefill_chunk

        # Parent would update here

        for req_id, num_scheduled_token in scheduler_output.num_scheduled_tokens.items():
            if was_prefill_map[req_id] and num_scheduled_token > 0:
                self.requests[req_id].chunk_num += 1
                print(
                    f"[PD] _update_after_schedule: request {req_id} "
                    f"chunk_num={self.requests[req_id].chunk_num} "
                    f"tokens={self.requests[req_id].num_tokens} "
                    f"scheduled={num_scheduled_token} "
                    f"computed={self.requests[req_id].num_computed_tokens} "
                    f"is_prefill_chunk={self.requests[req_id].is_prefill_chunk}"
                )

        self._migrate_prefill_to_running()
        self.finished_req_ids = set()

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: Any,
    ) -> dict[int, Any]:
        if get_pdmix_metadata(scheduler_output).batch_type == BatchType.PREFILL_LAST:
            if self.prefill_inflight_count > 0:
                self.prefill_inflight_count -= 1
            metadata = get_pdmix_metadata(scheduler_output)
            if metadata.head_token:
                self.hidden_channel_manager.release_prefill(
                    metadata.head_token
                )
            # Move completed requests from prefill_last_pending to running.
            completed_req_ids = set(scheduler_output.num_scheduled_tokens.keys())
            newly_running = [
                req for req in self.prefill_last_pending
                if req.request_id in completed_req_ids
            ]
            self.prefill_last_pending = [
                req for req in self.prefill_last_pending
                if req.request_id not in completed_req_ids
            ]
            self.running.extend(newly_running)
            print(
                f"[PD] update_from_output PREFILL_LAST done, "
                f"prefill_inflight: {self.prefill_inflight_count}/{self.prefill_inflight_limit}, "
                f"moved {len(newly_running)} reqs to running[], running[]: {len(self.running)}"
            )
        if get_pdmix_metadata(scheduler_output).batch_type == BatchType.DECODE_LAST:
            if self.decode_inflight_count > 0:
                self.decode_inflight_count -= 1
            print(
                f"[PD] update_from_output DECODE_LAST done, "
                f"decode_inflight: {self.decode_inflight_count}/{self.decode_inflight_limit}"
            )
        # Parent would update here
        self.chunk_prefill_first = [
            req for req in self.chunk_prefill_first if not req.is_finished()
        ]
        self.prefill_last_pending = [
            req for req in self.prefill_last_pending if not req.is_finished()
        ]
        return {}

    def get_request_counts(self) -> tuple[int, int]:
        # Parent would compute
        return (
            len(self.running)
            + len(self.chunk_prefill_first)
            + len(self.prefill_last_pending),
            len(self.waiting),
        )

    def get_num_unfinished_requests(self) -> int:
        # Simplified
        return (
            len(self.running)
            + len(self.chunk_prefill_first)
            + len(self.prefill_last_pending)
        )

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: Any
    ) -> list[tuple[str, int]]:
        # Parent would handle
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = self.requests.keys()

        to_remove = set()
        for req_id in request_ids:
            req = self.requests.get(req_id)
            if req and req.is_finished():
                to_remove.add(req)

        if to_remove:
            self.chunk_prefill_first = [
                req for req in self.chunk_prefill_first if req not in to_remove
            ]

        return []

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        if reset_running_requests:
            timestamp = time.monotonic()
            while self.chunk_prefill_first:
                request = self.chunk_prefill_first.pop()
                # Free caches (simplified)
                request.status = type(request.status)("PREEMPTED")
                request.num_computed_tokens = 0
                if request.spec_token_ids:
                    request.spec_token_ids = []
                request.num_preemptions += 1
                if self.log_stats:
                    request.record_event(None, timestamp)
                request.num_output_placeholders = 0
                request.discard_latest_async_tokens = True
                self.waiting.insert(0, request)

        return False

    def make_stats(self, *args, **kwargs):
        # Parent would create
        stats = type('Stats', (), {})()
        stats.num_running_reqs = len(self.running) + len(self.chunk_prefill_first)
        return stats

    def _handle_invalid_blocks(self, invalid_block_ids: set[int]) -> set[str]:
        # Parent would handle
        return set()
