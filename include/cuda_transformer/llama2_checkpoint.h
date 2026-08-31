#pragma once

#include "cuda_transformer/tiny_model.h"

#include <cstddef>
#include <string>
#include <vector>

namespace cuda_transformer {

// Legacy llama2.c FP32 checkpoints begin with seven native little-endian
// int32_t values: dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size,
// and seq_len. A negative vocab_size means an untied classifier follows the
// legacy RoPE frequency arrays; a positive value means tied embeddings.
struct Llama2CheckpointConfig {
  std::size_t dim = 0;
  std::size_t hidden_dim = 0;
  std::size_t n_layers = 0;
  std::size_t n_heads = 0;
  std::size_t n_kv_heads = 0;
  std::size_t vocabulary_size = 0;
  std::size_t max_sequence_length = 0;
  bool shared_classifier = false;
};

enum class Llama2CheckpointStatus {
  kSuccess,
  kInvalidArgument,
  kIoError,
  kInvalidFormat,
  kUnsupportedArchitecture,
  kMemoryAllocationFailure,
  kCudaError,
};

const char* llama2_checkpoint_status_string(Llama2CheckpointStatus status);

// Owns adapted host weights for the CPU oracle and device allocations for the
// CUDA runtime. The two TinyModelWeights views remain non-owning, consistent
// with the low-level model API. Copying is disabled to keep device ownership
// unambiguous.
class Llama2CheckpointModel {
 public:
  Llama2CheckpointModel() = default;
  ~Llama2CheckpointModel();
  Llama2CheckpointModel(const Llama2CheckpointModel&) = delete;
  Llama2CheckpointModel& operator=(const Llama2CheckpointModel&) = delete;

  // Releases every device allocation and clears both host/device views.
  // Destruction also performs this cleanup, but explicit reset makes cleanup
  // testable at a controlled point.
  cudaError_t reset();

  Llama2CheckpointConfig checkpoint_config{};
  TinyModelWeights host_weights{};
  TinyModelWeights device_weights{};

 private:
  friend Llama2CheckpointStatus llama2_checkpoint_load(
      const char*, Llama2CheckpointModel*, std::string*);

  static void bind_host_views_(Llama2CheckpointModel* model);
  static void bind_device_views_(Llama2CheckpointModel* model);
  static Llama2CheckpointStatus upload_weights_(Llama2CheckpointModel* model,
                                                std::string* error_message);

  std::vector<float> host_embeddings_;
  std::vector<float> host_attention_norms_;
  std::vector<float> host_wq_;
  std::vector<float> host_wk_;
  std::vector<float> host_wv_;
  std::vector<float> host_wo_;
  std::vector<float> host_mlp_norms_;
  std::vector<float> host_w_gate_;
  std::vector<float> host_w_up_;
  std::vector<float> host_w_down_;
  std::vector<float> host_final_norm_;
  std::vector<float> host_lm_head_;
  std::vector<TinyModelLayerWeights> host_layers_;
  std::vector<TinyModelLayerWeights> device_layers_;

  float* device_embeddings_ = nullptr;
  float* device_attention_norms_ = nullptr;
  float* device_wq_ = nullptr;
  float* device_wk_ = nullptr;
  float* device_wv_ = nullptr;
  float* device_wo_ = nullptr;
  float* device_mlp_norms_ = nullptr;
  float* device_w_gate_ = nullptr;
  float* device_w_up_ = nullptr;
  float* device_w_down_ = nullptr;
  float* device_final_norm_ = nullptr;
  float* device_lm_head_ = nullptr;
};

// Parses, validates, adapts, and uploads a legacy llama2.c FP32 checkpoint.
// Error detail is optional but, when supplied, describes malformed, unsupported,
// I/O, allocation, or CUDA failures. Existing model contents are released only
// after the incoming header and exact byte layout have passed validation.
Llama2CheckpointStatus llama2_checkpoint_load(
    const char* path, Llama2CheckpointModel* model,
    std::string* error_message = nullptr);

// Creates a TinyModelConfig for a selected full-forward sequence no longer
// than the checkpoint's maximum context. llama2.c uses RMSNorm epsilon 1e-5.
Llama2CheckpointStatus llama2_checkpoint_tiny_model_config(
    const Llama2CheckpointConfig& checkpoint_config, std::size_t sequence,
    TinyModelConfig* model_config);

}  // namespace cuda_transformer
