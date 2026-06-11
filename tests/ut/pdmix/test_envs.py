# SPDX-License-Identifier: Apache-2.0

from vllm_ascend import envs


def test_pdmix_env_defaults(monkeypatch):
    for name in (
        "VLLM_ASCEND_PDMIX_NON_LEADER_ENGINE_CORE",
        "VLLM_ASCEND_PDMIX_SCHEDULER_ZMQ_ADDR",
        "VLLM_ASCEND_PDMIX_PRE_OUT_ZMQ_PORT",
        "VLLM_ASCEND_PDMIX_POST_OUT_ZMQ_PORT",
        "VLLM_ASCEND_PDMIX_LAYER_SLICE_SIZE",
        "VLLM_ASCEND_PDMIX_PASSIVE_DISPATCH_POLICY",
    ):
        monkeypatch.delenv(name, raising=False)

    assert envs.VLLM_ASCEND_PDMIX_NON_LEADER_ENGINE_CORE is False
    assert envs.VLLM_ASCEND_PDMIX_SCHEDULER_ZMQ_ADDR == ""
    assert envs.VLLM_ASCEND_PDMIX_PRE_OUT_ZMQ_PORT == 5558
    assert envs.VLLM_ASCEND_PDMIX_POST_OUT_ZMQ_PORT == 5559
    assert envs.VLLM_ASCEND_PDMIX_LAYER_SLICE_SIZE == 0
    assert envs.VLLM_ASCEND_PDMIX_PASSIVE_DISPATCH_POLICY == "expect_alternation"


def test_pdmix_env_values(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_PDMIX_NON_LEADER_ENGINE_CORE", "1")
    monkeypatch.setenv("VLLM_ASCEND_PDMIX_SCHEDULER_ZMQ_ADDR", "tcp://127.0.0.1:5560")
    monkeypatch.setenv("VLLM_ASCEND_PDMIX_PRE_OUT_ZMQ_PORT", "6001")
    monkeypatch.setenv("VLLM_ASCEND_PDMIX_POST_OUT_ZMQ_PORT", "6002")
    monkeypatch.setenv("VLLM_ASCEND_PDMIX_LAYER_SLICE_SIZE", "4")
    monkeypatch.setenv("VLLM_ASCEND_PDMIX_PASSIVE_DISPATCH_POLICY", "prefill_first")

    assert envs.VLLM_ASCEND_PDMIX_NON_LEADER_ENGINE_CORE is True
    assert envs.VLLM_ASCEND_PDMIX_SCHEDULER_ZMQ_ADDR == "tcp://127.0.0.1:5560"
    assert envs.VLLM_ASCEND_PDMIX_PRE_OUT_ZMQ_PORT == 6001
    assert envs.VLLM_ASCEND_PDMIX_POST_OUT_ZMQ_PORT == 6002
    assert envs.VLLM_ASCEND_PDMIX_LAYER_SLICE_SIZE == 4
    assert envs.VLLM_ASCEND_PDMIX_PASSIVE_DISPATCH_POLICY == "prefill_first"
