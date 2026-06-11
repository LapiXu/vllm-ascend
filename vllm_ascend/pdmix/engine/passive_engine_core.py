# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import signal
import time
from typing import TYPE_CHECKING

import vllm_ascend.envs as envs
from vllm.logger import init_logger
from vllm.utils import set_process_title
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm_ascend.pdmix.sched.output import SchedulerOutput
    from vllm_ascend.pdmix.sched.passive_scheduler import DispatchPolicy

logger = init_logger(__name__)


class PassiveEngineCoreProc:
    """Passive EngineCore process for non-leader PP ranks.

    Mirrors the `EngineCore` / `EngineCoreProc` shape on rank0:

    - `step()` is the single-tick action: poll the ZMQ inbox, ask the
      `PassiveScheduler` for one batch, fan its slice plan out to the
      worker `rpc_broadcast_mq`.
    - `run_busy_loop()` is the long-running driver that keeps calling
      `step()` until the executor reports failure.

    Unlike rank0, there is no local scheduling decision — every batch
    comes pre-decided over ZMQ from the leader rank. The static
    :py:meth:`run_passive_engine_core` is the process entry point that
    constructs the executor + subscriber, builds an instance, and hands
    off to `run_busy_loop`.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        executor,  # MultiprocExecutor — duck-typed to avoid heavy import
        pp_subscriber: "PPSchedulerZmqSubscriber",
        dispatch_policy: "DispatchPolicy | None" = None,
        pp_pd_channel: "PPSchedulerZmqChannel | None" = None,
    ) -> None:
        from vllm_ascend.pdmix.sched.passive_scheduler import (
            DispatchPolicy,
            PassiveScheduler,
        )
        if dispatch_policy is None:
            dispatch_policy = DispatchPolicy.EXPECT_ALTERNATION
        self.vllm_config = vllm_config
        self.executor = executor
        self.passive_scheduler = PassiveScheduler(
            vllm_config, pp_subscriber, dispatch_policy=dispatch_policy
        )
        # Optional POST_OUT (cloud → edge) channel. Only set on the cloud
        # side in PD-separation mode; left None for the legacy PP path.
        self._pp_pd_channel = pp_pd_channel
        if vllm_config.parallel_config.enable_edge_cloud:
            logger.info(
                "PassiveEngineCore: edge-cloud mode enabled "
                "(enable_pd_separation=%s, pd_channel=%s)",
                vllm_config.parallel_config.enable_pd_separation,
                "on" if pp_pd_channel is not None else "off",
            )
        self._idle_sleep_seconds = 0.001

    def step(self) -> bool:
        """Single tick: poll ZMQ → pick batches → enqueue worker payloads.

        Batches are dispatched one phase at a time in the order encoded by
        the configured dispatch policy.

        Returns:
            True if at least one payload was enqueued, False if the
            scheduler had nothing to dispatch.
        """
        self.passive_scheduler.poll_and_classify()
        batch = self.passive_scheduler.schedule()
        if batch.is_empty():
            return False

        for slice_info in batch.slices:
            # PD-separation: on the cloud side, publish the rewritten
            # tail-segment SchedulerOutput on POST_OUT only when the dispatched
            # work can produce the final middle-segment hidden state.  With
            # slice-aware scheduling, early prefill slices must not wake the
            # edge tail segment because doing so can block the edge on a recv
            # and prevent it from issuing decode head work between P slices.
            if slice_info is None or slice_info.is_last_slice:
                self._maybe_publish_post_out(batch.scheduler_output)

            payload = (
                (batch.scheduler_output, slice_info)
                if slice_info is not None
                else (batch.scheduler_output,)
            )
            self.executor.rpc_broadcast_mq.enqueue(
                (b"pp_scheduler_output", payload, {}, None)
            )
        return True

    def _maybe_publish_post_out(
        self, scheduler_output: "SchedulerOutput"
    ) -> None:
        """Rewrite + publish a head-segment batch as a tail-segment one
        on the POST_OUT (cloud → edge) channel.

        Mapping (cloud-side):
            PREFILL_FIRST → PREFILL_LAST
            DECODE_FIRST  → DECODE_LAST
            anything else → dropped (legacy PP batches don't trigger return)

        Uses a shallow copy via :py:func:`dataclasses.replace` so the original
        SchedulerOutput (still about to be enqueued for the local executor)
        keeps its head-segment ``batch_type``.
        """
        if self._pp_pd_channel is None:
            return
        from dataclasses import replace
        from vllm_ascend.pdmix.sched.output import BatchType
        bt = scheduler_output.batch_type
        if bt == BatchType.PREFILL_FIRST:
            tail = replace(scheduler_output, batch_type=BatchType.PREFILL_LAST)
        elif bt == BatchType.DECODE_FIRST:
            tail = replace(scheduler_output, batch_type=BatchType.DECODE_LAST)
        else:
            return
        # Echo the head_token back so the edge can correlate the tail
        # segment with its suspended head state.
        self._pp_pd_channel.publish(tail)

    def run_busy_loop(self) -> None:
        """Drive `step()` until the executor reports failure or shutdown."""
        try:
            while not self.executor.is_failed:
                if not self.step():
                    time.sleep(self._idle_sleep_seconds)
        finally:
            self.passive_scheduler.shutdown()

    @staticmethod
    def run_passive_engine_core(
        vllm_config: "VllmConfig",
        ready_pipe,  # multiprocessing.Connection for signaling readiness
    ):
        """Entry point for the passive EngineCore process.

        Creates a MultiprocExecutor to spawn workers, optionally wires up
        a ZMQ subscriber to receive SchedulerOutputs from the leader PP
        rank, then hands off to `PassiveEngineCoreProc.run_busy_loop`.
        """
        from vllm_ascend.pdmix.engine.zmq_channel import (
            PPSchedulerZmqSubscriber,
            PPSchedulerZmqChannel,
        )

        # NOTE: We keep this import inside the method to avoid heavy import
        # when the module is loaded. In the original vLLM, this imports from
        # vllm.v1.executor.multiproc_executor.
        # For PDMix, we would need to decide if we have a custom executor.
        # For now, we'll keep it as a placeholder that would need to be
        # implemented or adapted.
        try:
            from vllm.v1.executor.multiproc_executor import MultiprocExecutor
        except ImportError:
            # In a real PDMix implementation, this would be adapted
            raise NotImplementedError(
                "PassiveEngineCoreProc requires an executor implementation"
            )

        maybe_register_config_serialize_by_value()

        # Mark this process as a non-leader PP rank running with passive
        # EngineCore, so that MultiprocExecutor and WorkerProc set up dual
        # message queues (local + cross-node).
        os.environ["VLLM_PP_NON_LEADER_ENGINE_CORE"] = "1"
        envs.disable_envs_cache()

        set_process_title("PassiveEngineCore")
        # TODO: Add tracing initialization if needed
        # maybe_init_worker_tracer(
        #     "vllm.engine_core", "engine_core", "PassiveEngineCore"
        # )
        # TODO: Add log decoration if needed
        # decorate_logs()

        pp_subscriber: PPSchedulerZmqSubscriber | None = None
        if envs.VLLM_PP_SCHEDULER_ZMQ_ADDR is not None:
            pp_subscriber = PPSchedulerZmqSubscriber(
                envs.VLLM_PP_SCHEDULER_ZMQ_ADDR
            )

        # Cloud-side PD-separation channel is constructed inside the try
        # block below (depends on `vllm_config`); declared here so the
        # `finally` clean-up can reference it unconditionally.
        pp_pd_channel: PPSchedulerZmqChannel | None = None

        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        executor = None
        try:
            executor = MultiprocExecutor(vllm_config, monitor_workers=False)

            ready_pipe.send({"status": "READY"})
            ready_pipe.close()
            ready_pipe = None

            if pp_subscriber is not None:
                executor.start_worker_monitor(inline=False)

                from vllm_ascend.pdmix.sched.passive_scheduler import (
                    DispatchPolicy,
                )
                try:
                    policy = DispatchPolicy(envs.VLLM_PP_PASSIVE_DISPATCH_POLICY)
                except ValueError:
                    logger.warning(
                        "Unknown VLLM_PP_PASSIVE_DISPATCH_POLICY=%r; "
                        "falling back to expect_alternation.",
                        envs.VLLM_PP_PASSIVE_DISPATCH_POLICY,
                    )
                    policy = DispatchPolicy.EXPECT_ALTERNATION

                # Set up edge-cloud PD-separation channel (cloud side).
                # The cloud binds POST_OUT and connects PRE_OUT via
                # master_addr (the edge's IP) so PRE_OUT connects back.
                if vllm_config.parallel_config.enable_pd_separation:
                    master_addr = vllm_config.parallel_config.master_addr
                    post_out_bind = (
                        f"tcp://*:{envs.VLLM_PP_POST_OUT_ZMQ_PORT}"
                    )
                    pre_out_connect = (
                        f"tcp://{master_addr}:"
                        f"{envs.VLLM_PP_PRE_OUT_ZMQ_PORT}"
                    )
                    pp_pd_channel = PPSchedulerZmqChannel(
                        send_endpoint=post_out_bind,
                        recv_endpoint=pre_out_connect,
                        name="pd-cloud",
                    )
                    logger.info(
                        "PD-separation cloud channel: POST_OUT=%s, "
                        "PRE_OUT=%s",
                        post_out_bind, pre_out_connect,
                    )

                proc = PassiveEngineCoreProc(
                    vllm_config, executor, pp_subscriber,
                    dispatch_policy=policy,
                    pp_pd_channel=pp_pd_channel,
                )
                proc.run_busy_loop()
            else:
                # No ZMQ subscriber, just monitor workers inline.
                executor.start_worker_monitor(inline=True)

        except SystemExit:
            logger.debug("PassiveEngineCore exiting.")
        except Exception:
            logger.exception("PassiveEngineCore encountered a fatal error.")
            raise
        finally:
            if ready_pipe is not None:
                try:
                    ready_pipe.send({"status": "FAILED"})
                except Exception:
                    pass
                ready_pipe.close()
            if pp_subscriber is not None:
                pp_subscriber.shutdown()
            if pp_pd_channel is not None:
                pp_pd_channel.shutdown()
            if executor is not None:
                executor.shutdown()
