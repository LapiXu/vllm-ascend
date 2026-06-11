#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# This file is mainly Adapted from vllm-project/vllm/vllm/v1/engine/core.py
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project.

import pickle
import queue
import threading
import time
from typing import Optional

import zmq

from vllm_ascend.pdmix.sched.output import BatchType, get_pdmix_metadata
from vllm.logger import init_logger

logger = init_logger(__name__)


class PPSchedulerZmqPublisher:
    """Publishes SchedulerOutput from pp rank0 EngineCore to pp rank1
    PassiveEngineCore via ZMQ PUSH/PULL pattern.

    Architecture: caller thread (scheduler loop) only enqueues the raw
    `SchedulerOutput` object into `_queue`. A dedicated background thread
    pulls from the queue, pickles, and sends over ZMQ. This keeps the
    scheduler step path free of pickling cost and mirrors the symmetric
    queue.Queue bridge used on the subscriber/PassiveScheduler side.
    """

    SHUTDOWN_TIMEOUT: float = 2.0

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        self._queue: queue.Queue[
            tuple[int, object] | None
        ] = queue.Queue(maxsize=1000)
        self._running = True
        self._seq = 0

        # Set up ZMQ PUSH socket
        self._ctx = zmq.Context.instance()
        self._push = self._ctx.socket(zmq.PUSH)
        self._push.set_hwm(1000)
        # Bind if wildcard (pp rank0), otherwise connect
        if "*" in endpoint or "::" in endpoint:
            self._push.bind(endpoint)
        else:
            self._push.connect(endpoint)

        logger.info("PP Scheduler ZMQ publisher started on %s", endpoint)

        # Start background publisher thread
        self._thread = threading.Thread(
            target=self._publisher_thread,
            daemon=True,
            name="pp_scheduler_zmq_pub",
        )
        self._thread.start()

    def publish(self, scheduler_output: object) -> None:
        """Queue a SchedulerOutput for publishing. Non-blocking: drops the
        message if the bridge queue is full (back-pressure protection).
        """
        if not self._running or get_pdmix_metadata(scheduler_output).batch_type is BatchType.EMPTY:
            return
        try:
            seq = self._seq
            self._seq += 1
            self._queue.put_nowait((seq, scheduler_output))
        except queue.Full:
            logger.warning(
                "PP Scheduler ZMQ publish queue full, dropping message"
            )

    def _publisher_thread(self) -> None:
        while self._running or self._queue.qsize() > 0:
            try:
                item = self._queue.get(timeout=0.1)
                if item is None:
                    break
                seq, scheduler_output = item
                try:
                    data = pickle.dumps(
                        scheduler_output, protocol=pickle.HIGHEST_PROTOCOL
                    )
                except Exception:
                    logger.exception(
                        "Failed to serialize SchedulerOutput for ZMQ"
                    )
                    continue
                seq_bytes = seq.to_bytes(8, "big")
                self._push.send_multipart((seq_bytes, data))
            except queue.Empty:
                continue
            except Exception:
                logger.exception("Error in PP scheduler ZMQ publisher thread")
                time.sleep(0.1)

    def shutdown(self) -> None:
        self._running = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=self.SHUTDOWN_TIMEOUT)
        try:
            if self._push is not None:
                self._push.close(linger=0)
        except Exception:
            pass


class PPSchedulerZmqSubscriber:
    """Receives SchedulerOutput from pp rank0 EngineCore on pp rank1
    via ZMQ PUSH/PULL pattern.

    Runs a background thread that receives SchedulerOutputs messages,
    saves them locally, and logs a summary.
    """

    SHUTDOWN_TIMEOUT: float = 2.0

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        self._running = True
        self._received_outputs: list[tuple[int, object]] = []
        self._lock = threading.Lock()

        # Set up ZMQ PULL socket
        self._ctx = zmq.Context.instance()
        self._pull = self._ctx.socket(zmq.PULL)
        self._pull.set_hwm(1000)
        self._pull.connect(endpoint)

        logger.info("PP Scheduler ZMQ subscriber connecting to %s", endpoint)

        # Start background subscriber thread
        self._thread = threading.Thread(
            target=self._subscriber_thread,
            daemon=True,
            name="pp_scheduler_zmq_sub",
        )
        self._thread.start()

    def _subscriber_thread(self) -> None:
        while self._running:
            try:
                if not self._pull.poll(timeout=100):
                    continue
                seq_bytes, data = self._pull.recv_multipart()
                seq = int.from_bytes(seq_bytes, "big")
                scheduler_output = pickle.loads(data)
                if get_pdmix_metadata(scheduler_output).batch_type is BatchType.EMPTY:
                    continue
                with self._lock:
                    self._received_outputs.append((seq, scheduler_output))
                logger.info(
                    "PP rank1 received SchedulerOutput seq=%d, "
                    "total_scheduled_tokens=%d, "
                    "new_reqs=%d, cached_reqs=%d, "
                    "finished_req_ids=%s",
                    seq,
                    scheduler_output.total_num_scheduled_tokens,
                    len(scheduler_output.scheduled_new_reqs),
                    scheduler_output.scheduled_cached_reqs.num_reqs,
                    scheduler_output.finished_req_ids,
                )
            except zmq.ZMQError:
                if self._running:
                    logger.exception("ZMQ error in PP scheduler subscriber")
            except Exception:
                if self._running:
                    logger.exception("Error in PP scheduler ZMQ subscriber thread")

    def get_latest_output(self) -> Optional[object]:
        """Return the most recently received SchedulerOutput, or None."""
        with self._lock:
            if self._received_outputs:
                return self._received_outputs[-1][1]
        return None

    def get_all_outputs(self) -> list[tuple[int, object]]:
        """Return all received (seq, SchedulerOutput) pairs."""
        with self._lock:
            return list(self._received_outputs)

    def consume_new_outputs(self) -> list[tuple[int, object]]:
        """Return and clear all new (seq, SchedulerOutput) pairs since last call."""
        with self._lock:
            outputs = self._received_outputs
            self._received_outputs = []
            return outputs

    def shutdown(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=self.SHUTDOWN_TIMEOUT)
        try:
            if self._pull is not None:
                self._pull.close(linger=0)
        except Exception:
            pass


class PPSchedulerZmqChannel:
    """Bidirectional ZMQ channel for SchedulerOutput exchange between two
    PP engines.

    A `PPSchedulerZmqChannel` owns one send side (a `PPSchedulerZmqPublisher`)
    and one receive side (a `PPSchedulerZmqSubscriber`), each backed by its
    own dedicated ZMQ PUSH / PULL socket on independent endpoints. It is the
    symmetric primitive needed by the edge-cloud PD-separation flow:

    - Edge constructs one channel with:

          send_endpoint = "tcp://*:<PRE_OUT_PORT>"          # bind, edge → cloud
          recv_endpoint = "tcp://<cloud_addr>:<POST_OUT_PORT>"   # connect

      and uses ``publish()`` to forward PREFILL_FIRST / DECODE_FIRST
      batches, and ``consume_new_outputs()`` to drain PREFILL_LAST /
      DECODE_LAST batches returned from the cloud.

    - Cloud constructs the mirror channel with:

          send_endpoint = "tcp://*:<POST_OUT_PORT>"          # bind, cloud → edge
          recv_endpoint = "tcp://<master_addr>:<PRE_OUT_PORT>"   # connect

    Both endpoints use the same PUSH/PULL + background-thread + queue.Queue
    bridge as the legacy unidirectional classes, so no scheduler-thread time
    is spent on pickling or socket I/O.

    Channel naming (``name``) is purely diagnostic; it is included in the
    log lines emitted by the underlying publisher / subscriber so the two
    edge-cloud channels can be told apart in a single combined log.
    """

    def __init__(
        self,
        send_endpoint: str,
        recv_endpoint: str,
        name: str = "pp_channel",
    ) -> None:
        self._name = name
        self._send_endpoint = send_endpoint
        self._recv_endpoint = recv_endpoint
        # Publisher binds-if-wildcard / connects-otherwise (see existing
        # `PPSchedulerZmqPublisher.__init__`); subscriber always connects.
        # The endpoints chosen by the caller therefore fully determine the
        # bind/connect roles of each side.
        self._publisher = PPSchedulerZmqPublisher(send_endpoint)
        self._subscriber = PPSchedulerZmqSubscriber(recv_endpoint)
        logger.info(
            "PPSchedulerZmqChannel[%s] up: send=%s, recv=%s",
            name,
            send_endpoint,
            recv_endpoint,
        )

    def publish(self, scheduler_output: object) -> None:
        """Queue a SchedulerOutput for the peer. Non-blocking."""
        self._publisher.publish(scheduler_output)

    def consume_new_outputs(self) -> list[tuple[int, object]]:
        """Return and clear all (seq, SchedulerOutput) pairs received since
        the last call. Suitable for use as the ``pp_subscriber`` argument
        of `PassiveScheduler`, which only relies on this method.
        """
        return self._subscriber.consume_new_outputs()

    def shutdown(self) -> None:
        self._publisher.shutdown()
        self._subscriber.shutdown()
