# SPDX-License-Identifier: Apache-2.0
#
# Unit tests for the GDN pre-warmup path.
#
# Background: on NPU, the very first launch of ``chunk_scaled_dot_kkt_fwd_kernel``
# during cudagraph CAPTURE differs from subsequent REPLAY launches (up to 15%
# deviation in the KKT A tensor sum on TP1, observed on Qwen3-27B edge-cloud).
# This corrupts the ssm_state written for the first request and produces
# garbled output.  ``_warmup_prefill_kernels`` runs a dummy GDN prefill on
# every layer *before* cudagraph capture so the first real request hits the
# steady-state kernel binary.

from unittest.mock import patch

import torch

from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention


def _make_layer(
    num_k_heads: int = 8,
    num_v_heads: int = 24,
    head_k_dim: int = 128,
    head_v_dim: int = 128,
    tp_size: int = 2,
) -> AscendGatedDeltaNetAttention:
    """Build a bare AscendGatedDeltaNetAttention without calling __init__.

    Only the attributes touched by ``_warmup_prefill_kernels`` are populated.
    """
    layer = AscendGatedDeltaNetAttention.__new__(AscendGatedDeltaNetAttention)
    layer._prefill_kernels_warmed_up = False
    layer.num_k_heads = num_k_heads
    layer.num_v_heads = num_v_heads
    layer.head_k_dim = head_k_dim
    layer.head_v_dim = head_v_dim
    layer.tp_size = tp_size
    layer.prefix = "model.layers.0.self_attn"
    return layer


class _FakeState:
    """Mimic ``MambaStateDtypeCalculator.gated_delta_net_state_dtype`` return."""

    def __getitem__(self, idx):
        return torch.bfloat16


def test_warmup_runs_only_once():
    """``_warmup_prefill_kernels`` must be idempotent (single-shot)."""
    layer = _make_layer()
    layer.get_state_dtype = lambda: (torch.bfloat16,)
    call_count = {"n": 0}

    def fake_chunk_gated_delta_rule(**kwargs):
        call_count["n"] += 1
        T = kwargs["q"].shape[1]
        V = kwargs["v"].shape[-1]
        K = kwargs["q"].shape[-1]
        H_v = kwargs["v"].shape[2]
        # Return (o, final_state) — both are not used by warmup, just need
        # to be valid shapes so the warmup does not raise.
        return (
            torch.zeros(1, T, H_v, V),
            torch.zeros(1, H_v, V, K),
        )

    with patch(
        "vllm_ascend.ops.gdn.chunk_gated_delta_rule",
        side_effect=fake_chunk_gated_delta_rule,
    ):
        dummy = torch.zeros(64, 1, device="meta")
        layer._warmup_prefill_kernels(dummy, v_dim=0)
        layer._warmup_prefill_kernels(dummy, v_dim=0)
        layer._warmup_prefill_kernels(dummy, v_dim=0)

    assert call_count["n"] == 1, (
        "warmup must short-circuit on subsequent calls; "
        f"got {call_count['n']} invocations")


def test_warmup_uses_ascend_chunk_gated_delta_rule():
    """The warmup must call the vllm-ascend (not upstream) kernel.

    Verifies the kwargs match the ascend API: ``prebuilt_meta=None``,
    ``head_first=False``, ``use_qk_l2norm_in_kernel=True``.
    """
    layer = _make_layer()
    layer.get_state_dtype = lambda: (torch.bfloat16,)

    captured_kwargs = {}

    def fake_chunk_gated_delta_rule(**kwargs):
        captured_kwargs.update(kwargs)
        T = kwargs["q"].shape[1]
        V = kwargs["v"].shape[-1]
        K = kwargs["q"].shape[-1]
        H_v = kwargs["v"].shape[2]
        return (
            torch.zeros(1, T, H_v, V),
            torch.zeros(1, H_v, V, K),
        )

    with patch(
        "vllm_ascend.ops.gdn.chunk_gated_delta_rule",
        side_effect=fake_chunk_gated_delta_rule,
    ):
        layer._warmup_prefill_kernels(torch.zeros(64, 1, device="meta"),
                                      v_dim=0)

    assert captured_kwargs["head_first"] is False
    assert captured_kwargs["use_qk_l2norm_in_kernel"] is True
    assert captured_kwargs["output_final_state"] is True
    assert captured_kwargs["prebuilt_meta"] is None
    assert captured_kwargs["initial_state"] is not None
    assert captured_kwargs["cu_seqlens"] is not None


def test_warmup_uses_tp_partitioned_head_dims():
    """The warmup must respect TP partitioning of k/v heads.

    Q has ``num_k_heads // tp_size`` heads and V has
    ``num_v_heads // tp_size`` heads.
    """
    layer = _make_layer(num_k_heads=8, num_v_heads=24, tp_size=2)
    layer.get_state_dtype = lambda: (torch.bfloat16,)

    captured_kwargs = {}

    def fake_chunk_gated_delta_rule(**kwargs):
        captured_kwargs.update(kwargs)
        T = kwargs["q"].shape[1]
        V = kwargs["v"].shape[-1]
        K = kwargs["q"].shape[-1]
        H_v = kwargs["v"].shape[2]
        return (
            torch.zeros(1, T, H_v, V),
            torch.zeros(1, H_v, V, K),
        )

    with patch(
        "vllm_ascend.ops.gdn.chunk_gated_delta_rule",
        side_effect=fake_chunk_gated_delta_rule,
    ):
        layer._warmup_prefill_kernels(torch.zeros(64, 1, device="meta"),
                                      v_dim=0)

    q, k, v, g, beta = (
        captured_kwargs["q"],
        captured_kwargs["k"],
        captured_kwargs["v"],
        captured_kwargs["g"],
        captured_kwargs["beta"],
    )
    # q/k are (1, T, num_k_heads // tp_size, head_k_dim)
    assert q.shape[2] == 4, q.shape
    assert k.shape[2] == 4
    # v/g/beta are (1, T, num_v_heads // tp_size, ...)
    assert v.shape[2] == 12, v.shape
    assert g.shape[2] == 12, g.shape
    assert beta.shape[2] == 12, beta.shape


def test_warmup_v0202_delegates_to_warmup():
    """``_warmup_prefill_kernels_v0202`` must delegate to the v0.23 path."""
    layer = _make_layer()
    layer.get_state_dtype = lambda: (torch.bfloat16,)

    def fake_chunk_gated_delta_rule(**kwargs):
        T = kwargs["q"].shape[1]
        V = kwargs["v"].shape[-1]
        K = kwargs["q"].shape[-1]
        H_v = kwargs["v"].shape[2]
        return (
            torch.zeros(1, T, H_v, V),
            torch.zeros(1, H_v, V, K),
        )

    with patch(
        "vllm_ascend.ops.gdn.chunk_gated_delta_rule",
        side_effect=fake_chunk_gated_delta_rule,
    ):
        # v0.20.2 entry point should still trigger the warmup.
        layer._warmup_prefill_kernels_v0202(torch.zeros(64, 1, device="meta"))
        assert layer._prefill_kernels_warmed_up is True


def test_warmup_swallows_exceptions():
    """A warmup failure must not break profile_run."""
    layer = _make_layer()
    layer.get_state_dtype = lambda: (torch.bfloat16,)

    with patch(
        "vllm_ascend.ops.gdn.chunk_gated_delta_rule",
        side_effect=RuntimeError("simulated OOM"),
    ):
        # Should not raise.
        layer._warmup_prefill_kernels(torch.zeros(64, 1, device="meta"),
                                      v_dim=0)
    # After a failed warmup, the flag is still set so we don't retry
    # every call (we cannot recover anyway).
    assert layer._prefill_kernels_warmed_up is True
