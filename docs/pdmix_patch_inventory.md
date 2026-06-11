# PDMix Patch Inventory

This document records all patch points required for PDMix (Predictive Decoding Mixture) functionality in vLLM-Ascend.

## Patch Points

| Patch Point | File | Reason | Plugin alternative attempted | Risk | Follow-up |
|-------------|------|--------|------------------------------|------|-----------|
| Qwen2 layer slice forward | `vllm_ascend/pdmix/model_patches/qwen2_layer_slice.py` | Implement speculative decoding layer slicing for Qwen2 models | No | Medium | Monitor upstream for native support |
| Qwen3 layer slice forward | `vllm_ascend/pdmix/model_patches/qwen3_layer_slice.py` | Implement speculative decoding layer slicing for Qwen3 models | No | Medium | Monitor upstream for native support |
| Qwen3.5 layer slice forward | `vllm_ascend/pdmix/model_patches/qwen3_5_layer_slice.py` | Implement speculative decoding layer slicing for Qwen3.5 models | No | Medium | Monitor upstream for native support |
| Qwen3 Next layer slice forward | `vllm_ascend/pdmix/model_patches/qwen3_next_layer_slice.py` | Implement speculative decoding layer slicing for Qwen3 Next models | No | Medium | Monitor upstream for native support |
| Qwen config compatibility | `vllm_ascend/pdmix/model_patches/qwen_config_compat.py` | Handle Qwen model config variations across versions | No | Low | Remove when upstream configs are standardized |
| Passive engine process routing | `vllm_ascend/patch/platform/patch_multiproc_executor.py` | Enable passive engine routing for multiprocess executor | No | High | Coordinate with upstream for plugin hooks |
| Hidden channel distributed groups | `vllm_ascend/patch/worker/patch_distributed.py` | Implement hidden communication channels for distributed speculative decoding | No | High | Design generic plugin API for distributed communication |

## Notes

- All patches are currently maintained as monkey-patches within the vllm_ascend module.
- Future work should focus on upstreaming these capabilities or establishing a formal plugin interface.
- Risk assessment is based on:
  - **High**: Core functionality changes that may break during upstream updates
  - **Medium**: Model-specific changes that require version-specific maintenance
  - **Low**: Compatibility layers that can be easily removed
