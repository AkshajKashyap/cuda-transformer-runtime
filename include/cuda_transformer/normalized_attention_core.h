#pragma once

#include "cuda_transformer/qkv_projection.h"

#include <cstddef>

namespace cuda_transformer {

// Batch size is fixed at one. Input is [sequence, hidden]; the workspace
// output is [heads, sequence, head_dim].
struct NormalizedAttentionCoreConfig {
  std::size_t sequence = 0;
  std::size_t hidden = 0;
  std::size_t heads = 0;
  std::size_t head_dim = 0;
  float rmsnorm_epsilon = 0.0F;
};

// Owns only the buffers needed around the existing QKV projection workspace.
// It is non-copyable because it owns CUDA allocations.
struct NormalizedAttentionCoreWorkspace {
  float* normalized = nullptr;
  float* rotated_q = nullptr;
  float* rotated_k = nullptr;
  float* probabilities = nullptr;
  float* output = nullptr;
  QkvProjectionWorkspace qkv;
  NormalizedAttentionCoreConfig config{};

  NormalizedAttentionCoreWorkspace() = default;
  NormalizedAttentionCoreWorkspace(const NormalizedAttentionCoreWorkspace&) =
      delete;
  NormalizedAttentionCoreWorkspace& operator=(
      const NormalizedAttentionCoreWorkspace&) = delete;
};

bool valid_normalized_attention_core_config(NormalizedAttentionCoreConfig config);

cudaError_t normalized_attention_core_workspace_create(
    NormalizedAttentionCoreWorkspace* workspace,
    NormalizedAttentionCoreConfig config);
cudaError_t normalized_attention_core_workspace_destroy(
    NormalizedAttentionCoreWorkspace* workspace);

// The caller owns the cuBLAS handle and all input/weight buffers. Wq, Wk, and
// Wv are row-major [hidden, hidden] matrices. Work is enqueued on `stream`;
// this function does not synchronize.
cublasStatus_t normalized_attention_core_cuda(
    cublasHandle_t handle, const float* input, const float* norm_weight,
    const float* wq, const float* wk, const float* wv,
    NormalizedAttentionCoreWorkspace* workspace,
    cudaStream_t stream = nullptr);

}  // namespace cuda_transformer
