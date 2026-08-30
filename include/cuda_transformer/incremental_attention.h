#pragma once

#include "cuda_transformer/kv_cache.h"

namespace cuda_transformer {

// Reusable FP32 scratch for one batch-1 incremental-attention token. The
// score buffer has [heads, max_sequence] capacity and is reused in-place by
// stable softmax. Copying is disabled because this type owns device buffers.
struct IncrementalAttentionWorkspace {
  float* rotated_q = nullptr;
  float* rotated_k = nullptr;
  float* scores = nullptr;
  std::size_t heads = 0;
  std::size_t head_dim = 0;
  std::size_t max_sequence = 0;

  IncrementalAttentionWorkspace() = default;
  IncrementalAttentionWorkspace(const IncrementalAttentionWorkspace&) = delete;
  IncrementalAttentionWorkspace& operator=(
      const IncrementalAttentionWorkspace&) = delete;
};

cudaError_t incremental_attention_workspace_create(
    IncrementalAttentionWorkspace* workspace, std::size_t heads,
    std::size_t head_dim, std::size_t max_sequence);
cudaError_t incremental_attention_workspace_destroy(
    IncrementalAttentionWorkspace* workspace);

// P×V helpers for the contiguous [heads, max_sequence, head_dim] V cache and
// compact [heads, length] probability rows. The serial path is retained for
// short histories; the parallel path cooperatively reduces one (head, dim).
cudaError_t incremental_probability_value_serial_cuda(
    const float* probabilities, const float* values, float* output,
    std::size_t heads, std::size_t length, std::size_t max_sequence,
    std::size_t head_dim, cudaStream_t stream = nullptr);
cudaError_t incremental_probability_value_parallel_cuda(
    const float* probabilities, const float* values, float* output,
    std::size_t heads, std::size_t length, std::size_t max_sequence,
    std::size_t head_dim, cudaStream_t stream = nullptr);

// Performs one no-allocation cached decode. The workspace capacity must match
// the supplied cache, and the function does not synchronize.
cudaError_t incremental_decode_with_workspace(
    KvCache* cache, const float* q, const float* k, const float* v,
    float* output, IncrementalAttentionWorkspace* workspace,
    cudaStream_t stream = nullptr);

// Stores K after RoPE at cache.current_length; V is stored unchanged.
// This compatibility wrapper creates temporary scratch for a single call.
cudaError_t incremental_decode(KvCache *cache, const float *q, const float *k,
                               const float *v, float *output,
                               cudaStream_t stream = nullptr);
}  // namespace cuda_transformer
