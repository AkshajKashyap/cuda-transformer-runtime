#pragma once

#include "cuda_transformer/attention_sublayer.h"
#include "cuda_transformer/mlp_sublayer.h"

namespace cuda_transformer {

// Reuses the independently validated attention and MLP configuration types.
struct DecoderBlockConfig {
  NormalizedAttentionCoreConfig attention{};
  MlpSublayerConfig mlp{};
};

// All weights are caller-owned FP32 buffers described by their sublayers.
struct DecoderBlockWeights {
  AttentionSublayerWeights attention{};
  MlpSublayerWeights mlp{};
};

// Embeds sublayer workspaces and owns only their connecting attention output.
// The final [sequence, hidden] output remains caller-owned.
struct DecoderBlockWorkspace {
  AttentionSublayerWorkspace attention;
  MlpSublayerWorkspace mlp;
  float* attention_output = nullptr;
  DecoderBlockConfig config{};
  DecoderBlockConfig allocation_config{};

  DecoderBlockWorkspace() = default;
  DecoderBlockWorkspace(const DecoderBlockWorkspace&) = delete;
  DecoderBlockWorkspace& operator=(const DecoderBlockWorkspace&) = delete;
};

bool valid_decoder_block_config(DecoderBlockConfig config);
cudaError_t decoder_block_workspace_create(DecoderBlockWorkspace* workspace,
                                           DecoderBlockConfig config);
cudaError_t decoder_block_workspace_destroy(DecoderBlockWorkspace* workspace);

// Enqueues attention followed by MLP. The MLP receives attention_output, so
// its residual source is the attention sublayer result rather than `input`.
// The caller owns `handle` and final [sequence, hidden] `output`; no sync.
cublasStatus_t decoder_block_cuda(cublasHandle_t handle, const float* input,
                                  DecoderBlockWeights weights,
                                  DecoderBlockWorkspace* workspace,
                                  float* output,
                                  cudaStream_t stream = nullptr);

}  // namespace cuda_transformer
