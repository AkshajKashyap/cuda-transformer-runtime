#pragma once

#include "cuda_transformer/gemm.h"

#include <cstddef>

namespace cuda_transformer {

struct MlpSublayerConfig {
  std::size_t sequence = 0;
  std::size_t hidden = 0;
  std::size_t intermediate = 0;
  float rmsnorm_epsilon = 0.0F;
};

// The caller owns all FP32 weights. Matrices are row-major: gate/up are
// [hidden, intermediate] and down is [intermediate, hidden].
struct MlpSublayerWeights {
  const float* norm_weight = nullptr;
  const float* w_gate = nullptr;
  const float* w_up = nullptr;
  const float* w_down = nullptr;
};

// Owns reusable CUDA intermediates. Output remains caller-owned.
struct MlpSublayerWorkspace {
  float* normalized = nullptr;
  float* gate = nullptr;
  float* up = nullptr;
  float* activated_gate = nullptr;
  float* gated = nullptr;
  float* down = nullptr;
  MlpSublayerConfig config{};
  // Records the dimensions used for allocation so execution can reject a
  // workspace whose public configuration metadata has been changed.
  MlpSublayerConfig allocation_config{};

  MlpSublayerWorkspace() = default;
  MlpSublayerWorkspace(const MlpSublayerWorkspace&) = delete;
  MlpSublayerWorkspace& operator=(const MlpSublayerWorkspace&) = delete;
};

bool valid_mlp_sublayer_config(MlpSublayerConfig config);
cudaError_t mlp_sublayer_workspace_create(MlpSublayerWorkspace* workspace,
                                          MlpSublayerConfig config);
cudaError_t mlp_sublayer_workspace_destroy(MlpSublayerWorkspace* workspace);

// Enqueues RMSNorm, SwiGLU projections/activation, down projection, and a
// residual from the original input into caller-owned [sequence, hidden]
// output. The caller owns the cuBLAS handle; this function does not sync.
cublasStatus_t mlp_sublayer_cuda(cublasHandle_t handle, const float* input,
                                 MlpSublayerWeights weights,
                                 MlpSublayerWorkspace* workspace, float* output,
                                 cudaStream_t stream = nullptr);

}  // namespace cuda_transformer
