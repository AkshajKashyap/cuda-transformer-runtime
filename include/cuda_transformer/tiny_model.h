#pragma once

#include "cuda_transformer/decoder_block.h"

#include <cstddef>

namespace cuda_transformer {

// Batch size is fixed at one. Every forward uses this fixed sequence length.
struct TinyModelConfig {
  std::size_t vocabulary_size = 0;
  std::size_t sequence = 0;
  std::size_t hidden = 0;
  std::size_t layers = 0;
  std::size_t heads = 0;
  std::size_t head_dim = 0;
  std::size_t intermediate = 0;
  float rmsnorm_epsilon = 0.0F;
};

// These pointers are non-owning. The same shape is used with host pointers by
// the CPU oracle and device pointers by the CUDA forward path.
struct TinyModelLayerWeights {
  DecoderBlockWeights decoder{};
};

struct TinyModelWeights {
  const float* token_embeddings = nullptr;  // [vocabulary_size, hidden]
  const TinyModelLayerWeights* layers = nullptr;
  const float* final_norm_weight = nullptr;  // [hidden]
  const float* lm_head_weight = nullptr;     // [hidden, vocabulary_size]
};

// Owns only reusable execution storage. Weights, input token IDs for the core
// API, and output logits always remain caller-owned.
struct TinyModelWorkspace {
  int* copied_token_ids = nullptr;
  float* activation_a = nullptr;
  float* activation_b = nullptr;
  DecoderBlockWorkspace* layer_workspaces = nullptr;
  TinyModelConfig config{};
  TinyModelConfig allocation_config{};

  TinyModelWorkspace() = default;
  TinyModelWorkspace(const TinyModelWorkspace&) = delete;
  TinyModelWorkspace& operator=(const TinyModelWorkspace&) = delete;
};

bool valid_tiny_model_config(TinyModelConfig config);
cudaError_t tiny_model_workspace_create(TinyModelWorkspace* workspace,
                                        TinyModelConfig config);
cudaError_t tiny_model_workspace_destroy(TinyModelWorkspace* workspace);

// Embedding lookup selects row token_ids[position] from the contiguous
// [vocabulary_size, hidden] table and writes [sequence, hidden]. Token IDs
// must be in [0, vocabulary_size).
void embedding_lookup_cpu(const int* token_ids, const float* embeddings,
                          float* output, std::size_t sequence,
                          std::size_t vocabulary_size, std::size_t hidden);
cudaError_t embedding_lookup_cuda(const int* device_token_ids,
                                  const float* device_embeddings,
                                  float* device_output,
                                  std::size_t sequence,
                                  std::size_t vocabulary_size,
                                  std::size_t hidden,
                                  cudaStream_t stream = nullptr);

// A clear CPU oracle assembled from the repository's validated CPU primitives.
// It returns false for invalid configuration, weights, or token IDs.
bool tiny_model_forward_cpu(const int* token_ids, TinyModelWeights weights,
                            TinyModelConfig config, float* logits);

// Core device-native API. Token IDs and logits are caller-owned device buffers:
// token_ids is [sequence] and logits is [sequence, vocabulary_size]. This
// enqueues embedding lookup, all decoder layers, final RMSNorm, and LM head;
// it does not synchronize or allocate GPU memory. Device token IDs must be
// valid because their values cannot be synchronously inspected by this API.
cublasStatus_t tiny_model_forward_cuda(cublasHandle_t handle,
                                        const int* device_token_ids,
                                        TinyModelWeights device_weights,
                                        TinyModelWorkspace* workspace,
                                        float* device_logits,
                                        cudaStream_t stream = nullptr);

// Convenience wrapper for host token IDs. It validates IDs, asynchronously
// copies them to workspace-owned device storage, then invokes the core API.
cublasStatus_t tiny_model_forward_host_tokens_cuda(
    cublasHandle_t handle, const int* host_token_ids,
    TinyModelWeights device_weights, TinyModelWorkspace* workspace,
    float* device_logits, cudaStream_t stream = nullptr);

}  // namespace cuda_transformer
