# PDMix Validation Notes

## Verified locally

- vLLM clean diff against source baseline:
  - `diff -qr source/vllm-0.20.2_layerwise/vllm-0.20.2_layerwise goal/vllm-pdmix/.worktrees/pdmix-migration`
  - Result: no source differences.
- vLLM PDMix residue scan:
  - Checked `PDMix`, `PDSeparatedScheduler`, `PassiveScheduler`, `HiddenChannelType`, `VLLM_PP_NON_LEADER_ENGINE_CORE`, `VLLM_LAYER_SLICE_SIZE` in `vllm` and `tests`.
  - Result: no residue.
- vLLM Ascend forbidden import / old environment variable scan:
  - Checked old vLLM PDMix scheduler imports and old `VLLM_PP_*` / `VLLM_LAYER_SLICE_SIZE` names in `vllm_ascend` and `tests`.
  - Result: no residue.
- Python compile check:
  - `python -m compileall vllm_ascend/pdmix tests/ut/pdmix`
  - Result: passed.

## Not run successfully locally

- `python -m pytest tests/ut/pdmix -q`
  - Result: failed during test discovery because the local environment is missing the `regex` dependency.
  - Error: `ModuleNotFoundError: No module named 'regex'` from `vllm_ascend/utils.py` imported by `tests/ut/conftest.py`.

## Not run locally

- Multi-node edge-cloud PDMix end-to-end validation.
- Full hidden tensor data-plane validation over HCCL / NPU runtime.
- Real Qwen layer-slice execution on Ascend NPU hardware.

## Required manual validation

1. Install the normal vLLM Ascend test dependencies, including `regex`.
2. Run `python -m pytest tests/ut/pdmix -q` again.
3. Start the edge node with matching `VLLM_ASCEND_PDMIX_*` variables.
4. Start the cloud node with matching ZMQ ports and rank layout.
5. Run a Qwen PDMix prefill/decode workload.
6. Confirm `PREFILL_FIRST`, `PREFILL_LAST`, `DECODE_FIRST`, and `DECODE_LAST` logs appear.
7. Confirm generated output matches the pre-migration `dest` behavior for the same prompt and seed.
