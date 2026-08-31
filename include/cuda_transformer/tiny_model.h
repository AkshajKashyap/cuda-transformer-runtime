#pragma once

#include "cuda_transformer/decoder_block.h"
#include "cuda_transformer/incremental_decoder_block.h"

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
  std::size_t layer_count = 0;
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

// Separate state for one-token decoding. `max_sequence_length` is cache
// capacity, intentionally distinct from TinyModelConfig::sequence, which
// describes a full-sequence model forward. Every layer owns a separate cache
// and incremental decoder-block workspace.
struct TinyModelIncrementalWorkspace {
  int* copied_token_id = nullptr;
  float* activation_a = nullptr;
  float* activation_b = nullptr;
  IncrementalDecoderBlockWorkspace* layer_workspaces = nullptr;
  KvCache* layer_caches = nullptr;
  TinyModelConfig model_config{};
  TinyModelConfig allocation_model_config{};
  std::size_t max_sequence_length = 0;
  std::size_t allocation_max_sequence_length = 0;

  TinyModelIncrementalWorkspace() = default;
  TinyModelIncrementalWorkspace(const TinyModelIncrementalWorkspace&) = delete;
  TinyModelIncrementalWorkspace& operator=(
      const TinyModelIncrementalWorkspace&) = delete;
};

cudaError_t tiny_model_incremental_workspace_create(
    TinyModelIncrementalWorkspace* workspace, TinyModelConfig model_config,
    std::size_t max_sequence_length);
cudaError_t tiny_model_incremental_workspace_destroy(
    TinyModelIncrementalWorkspace* workspace);

// Resets only logical cache lengths. It preserves all device allocations and
// cached bytes; later decodes overwrite positions starting at zero.
cudaError_t tiny_model_incremental_workspace_reset(
    TinyModelIncrementalWorkspace* workspace);

// Core device-native cached decode. device_token_id is [1] and device_logits
// is [1, vocabulary_size]. It preflights every layer's cache/workspace before
// launching work, then enqueues one-token embedding, all incremental blocks,
// final RMSNorm, and LM head without synchronization or allocation.
cublasStatus_t tiny_model_incremental_decode_cuda(
    cublasHandle_t handle, const int* device_token_id,
    TinyModelWeights device_weights, TinyModelIncrementalWorkspace* workspace,
    float* device_logits, cudaStream_t stream = nullptr);

// Host-token convenience wrapper. It validates the one host token, copies it
// to workspace-owned device storage, and then calls the core API.
cublasStatus_t tiny_model_incremental_decode_host_token_cuda(
    cublasHandle_t handle, int host_token_id, TinyModelWeights device_weights,
    TinyModelIncrementalWorkspace* workspace, float* device_logits,
    cudaStream_t stream = nullptr);

// Correctness-first sequential prefill. Each prefix token follows the exact
// incremental path, including final RMSNorm and LM head, so this is not an
// optimized parallel transformer prefill implementation.
cublasStatus_t tiny_model_incremental_prefill_cuda(
    cublasHandle_t handle, const int* device_token_ids,
    std::size_t prefix_length, TinyModelWeights device_weights,
    TinyModelIncrementalWorkspace* workspace, float* device_logits,
    cudaStream_t stream = nullptr);

// Selects the largest finite logit. Exact ties keep the first (lowest) token
// ID. Returns false for null, empty, non-finite, or unrepresentable inputs.
bool tiny_model_greedy_argmax_host(const float* host_logits,
                                   std::size_t vocabulary_size,
                                   int* selected_token_id);

// Host-orchestrated greedy generation from host prompt IDs. For a nonzero
// max_new_tokens, this first logically resets the supplied incremental
// workspace, then processes the prompt and returns max_new_tokens selected
// IDs in host_generated_token_ids. device_logits is caller-owned [1, vocab]
// storage. CPU argmax requires a stream synchronization after each logits
// copy; the lower-level device-native decode API remains asynchronous.
//
// The final returned generated ID is not decoded: it has no successor logits
// to produce. Therefore, on success each layer's cache length is
// prompt_length + max_new_tokens - 1. A zero-token request succeeds without
// resetting or otherwise mutating the workspace.
cublasStatus_t tiny_model_generate_greedy_cuda(
    cublasHandle_t handle, const int* host_prompt_token_ids,
    std::size_t prompt_length, TinyModelWeights device_weights,
    TinyModelIncrementalWorkspace* workspace, std::size_t max_new_tokens,
    int* host_generated_token_ids, float* device_logits,
    cudaStream_t stream = nullptr);

}  // namespace cuda_transformer
