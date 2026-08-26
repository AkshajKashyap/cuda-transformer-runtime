#pragma once

#include "cuda_transformer/normalized_attention_core.h"

namespace cuda_transformer {

// All weight matrices are FP32 row-major [hidden, hidden]. The caller owns
// these buffers for the duration of attention_sublayer_cuda.
struct AttentionSublayerWeights {
  const float* norm_weight = nullptr;
  const float* wq = nullptr;
  const float* wk = nullptr;
  const float* wv = nullptr;
  const float* wo = nullptr;
};

// Reuses the normalized-attention workspace and owns only the two additional
// token-major buffers needed before the final caller-owned output.
struct AttentionSublayerWorkspace {
  NormalizedAttentionCoreWorkspace normalized_attention;
  float* attention_token_major = nullptr;
  float* projected = nullptr;
  NormalizedAttentionCoreConfig config{};

  AttentionSublayerWorkspace() = default;
  AttentionSublayerWorkspace(const AttentionSublayerWorkspace&) = delete;
  AttentionSublayerWorkspace& operator=(const AttentionSublayerWorkspace&) =
      delete;
};

cudaError_t attention_sublayer_workspace_create(
    AttentionSublayerWorkspace* workspace,
    NormalizedAttentionCoreConfig config);
cudaError_t attention_sublayer_workspace_destroy(
    AttentionSublayerWorkspace* workspace);

// Enqueues normalized attention, head-major to token-major repack, Wo, and
// residual addition. `output` is caller-owned [sequence, hidden] storage; the
// supplied cuBLAS handle is caller-owned. This function does not synchronize.
cublasStatus_t attention_sublayer_cuda(cublasHandle_t handle, const float* input,
                                       AttentionSublayerWeights weights,
                                       AttentionSublayerWorkspace* workspace,
                                       float* output,
                                       cudaStream_t stream = nullptr);

}  // namespace cuda_transformer
