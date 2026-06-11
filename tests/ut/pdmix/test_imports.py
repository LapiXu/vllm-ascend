# SPDX-License-Identifier: Apache-2.0


def test_pdmix_modules_are_importable():
    import vllm_ascend.pdmix as pdmix
    from vllm_ascend.pdmix.sched.output import BatchType, HiddenChannelType
    from vllm_ascend.pdmix.sched.passive_scheduler import PassiveScheduler
    from vllm_ascend.pdmix.sched.pd_separated_scheduler import PDSeparatedScheduler
    from vllm_ascend.pdmix.engine.zmq_channel import PPSchedulerZmqChannel
    from vllm_ascend.pdmix.model_patches import apply_model_patches

    assert callable(pdmix.apply_pdmix_patches)
    assert BatchType.PD_MIX.value == "PD_MIX"
    assert HiddenChannelType.DECODE.value == "DECODE"
    assert PassiveScheduler.__name__ == "PassiveScheduler"
    assert PDSeparatedScheduler.__name__ == "PDSeparatedScheduler"
    assert PPSchedulerZmqChannel.__name__ == "PPSchedulerZmqChannel"
    assert callable(apply_model_patches)
