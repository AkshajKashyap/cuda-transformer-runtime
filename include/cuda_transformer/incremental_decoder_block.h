#pragma once

#include "cuda_transformer/attention_sublayer.h"
#include "cuda_transformer/incremental_attention.h"
#include "cuda_transformer/kv_cache.h"
#include "cuda_transformer/mlp_sublayer.h"
#include "cuda_transformer/qkv_projection.h"

namespace cuda_transformer {

struct IncrementalDecoderBlockConfig {
  std::size_t hidden = 0;
  std::size_t heads = 0;
  std::size_t head_dim = 0;
  std::size_t intermediate = 0;
  float attention_rmsnorm_epsilon = 0.0F;
  float mlp_rmsnorm_epsilon = 0.0F;
  // Scratch capacity for one-token attention. The external cache remains
  // caller-owned and must have this same fixed capacity.
  std::size_t max_sequence = 0;
};

// Caller-owned FP32 weights. The nested structures retain their established
// row-major matrix layouts.
struct IncrementalDecoderBlockWeights {
  AttentionSublayerWeights attention{};
  MlpSublayerWeights mlp{};
};

// Reuses one-token QKV and attention scratch plus temporary token activations.
// The KV cache remains supplied separately by the caller.
struct IncrementalDecoderBlockWorkspace {
  QkvProjectionWorkspace qkv;
  IncrementalAttentionWorkspace attention;
  float* attention_normalized = nullptr;
  float* attention_head = nullptr;
  float* attention_token = nullptr;
  float* attention_projected = nullptr;
  float* attention_residual = nullptr;
  float* mlp_normalized = nullptr;
  float* gate = nullptr;
  float* up = nullptr;
  float* activated_gate = nullptr;
  float* gated = nullptr;
  float* down = nullptr;
  IncrementalDecoderBlockConfig config{};
  IncrementalDecoderBlockConfig allocation_config{};

  IncrementalDecoderBlockWorkspace() = default;
  IncrementalDecoderBlockWorkspace(const IncrementalDecoderBlockWorkspace&) =
      delete;
  IncrementalDecoderBlockWorkspace& operator=(
      const IncrementalDecoderBlockWorkspace&) = delete;
};

bool valid_incremental_decoder_block_config(IncrementalDecoderBlockConfig config);
cudaError_t incremental_decoder_block_workspace_create(
    IncrementalDecoderBlockWorkspace* workspace,
    IncrementalDecoderBlockConfig config);
cudaError_t incremental_decoder_block_workspace_destroy(
    IncrementalDecoderBlockWorkspace* workspace);

// Enqueues one cached decoder-block token. `cache` is caller-owned and gains
// one rotated K/raw V entry on success. `output` is caller-owned [1, hidden].
// The caller owns `handle`; this function does not synchronize.
cublasStatus_t incremental_decoder_block_cuda(
    cublasHandle_t handle, const float* input,
    IncrementalDecoderBlockWeights weights, KvCache* cache,
    IncrementalDecoderBlockWorkspace* workspace, float* output,
    cudaStream_t stream = nullptr);

}  // namespace cuda_transformer
