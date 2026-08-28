#include "cuda_transformer/mlp_sublayer.h"

#include "cuda_transformer/transformer_primitives.h"

#include <cmath>

namespace cuda_transformer {
namespace {

void clear(MlpSublayerWorkspace* workspace) {
  workspace->normalized = nullptr;
  workspace->gate = nullptr;
  workspace->up = nullptr;
  workspace->activated_gate = nullptr;
  workspace->gated = nullptr;
  workspace->down = nullptr;
  workspace->config = {};
  workspace->allocation_config = {};
}

bool configs_match(MlpSublayerConfig left, MlpSublayerConfig right) {
  return left.sequence == right.sequence && left.hidden == right.hidden &&
         left.intermediate == right.intermediate &&
         left.rmsnorm_epsilon == right.rmsnorm_epsilon;
}

bool workspace_is_empty(const MlpSublayerWorkspace* workspace) {
  return workspace != nullptr && workspace->normalized == nullptr &&
         workspace->gate == nullptr && workspace->up == nullptr &&
         workspace->activated_gate == nullptr && workspace->gated == nullptr &&
         workspace->down == nullptr;
}

bool workspace_matches(const MlpSublayerWorkspace* workspace) {
  return workspace != nullptr && valid_mlp_sublayer_config(workspace->config) &&
         configs_match(workspace->config, workspace->allocation_config) &&
         workspace->normalized != nullptr && workspace->gate != nullptr &&
         workspace->up != nullptr && workspace->activated_gate != nullptr &&
         workspace->gated != nullptr && workspace->down != nullptr;
}

cudaError_t first_error(cudaError_t first, cudaError_t next) {
  return first == cudaSuccess ? next : first;
}

}  // namespace

bool valid_mlp_sublayer_config(MlpSublayerConfig config) {
  return config.sequence != 0 && config.hidden != 0 &&
         config.intermediate != 0 && std::isfinite(config.rmsnorm_epsilon) &&
         config.rmsnorm_epsilon > 0.0F;
}

cudaError_t mlp_sublayer_workspace_create(MlpSublayerWorkspace* workspace,
                                          MlpSublayerConfig config) {
  if (!workspace || !workspace_is_empty(workspace) ||
      !valid_mlp_sublayer_config(config))
    return cudaErrorInvalidValue;

  const std::size_t activation_bytes =
      config.sequence * config.hidden * sizeof(float);
  const std::size_t intermediate_bytes =
      config.sequence * config.intermediate * sizeof(float);
  cudaError_t error = cudaMalloc(&workspace->normalized, activation_bytes);
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->gate, intermediate_bytes);
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->up, intermediate_bytes);
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->activated_gate, intermediate_bytes);
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->gated, intermediate_bytes);
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->down, activation_bytes);
  if (error != cudaSuccess)
    mlp_sublayer_workspace_destroy(workspace);
  else {
    workspace->config = config;
    workspace->allocation_config = config;
  }
  return error;
}

cudaError_t mlp_sublayer_workspace_destroy(MlpSublayerWorkspace* workspace) {
  if (!workspace)
    return cudaErrorInvalidValue;

  cudaError_t error = cudaSuccess;
  for (float* pointer : {workspace->normalized, workspace->gate, workspace->up,
                         workspace->activated_gate, workspace->gated,
                         workspace->down}) {
    if (pointer != nullptr)
      error = first_error(error, cudaFree(pointer));
  }
  clear(workspace);
  return error;
}

cublasStatus_t mlp_sublayer_cuda(cublasHandle_t handle, const float* input,
                                 MlpSublayerWeights weights,
                                 MlpSublayerWorkspace* workspace, float* output,
                                 cudaStream_t stream) {
  if (!handle || !input || !output || !weights.norm_weight || !weights.w_gate ||
      !weights.w_up || !weights.w_down || !workspace_matches(workspace))
    return CUBLAS_STATUS_INVALID_VALUE;

  const MlpSublayerConfig config = workspace->config;
  cudaError_t cuda_error = rmsnorm_cuda(
      input, weights.norm_weight, workspace->normalized, config.sequence,
      config.hidden, config.rmsnorm_epsilon, stream);
  if (cuda_error != cudaSuccess)
    return CUBLAS_STATUS_EXECUTION_FAILED;

  cublasStatus_t status = linear_cublas_row_major(
      handle, workspace->normalized, weights.w_gate, workspace->gate,
      config.sequence, config.hidden, config.intermediate, stream);
  if (status != CUBLAS_STATUS_SUCCESS)
    return status;
  status = linear_cublas_row_major(handle, workspace->normalized, weights.w_up,
                                   workspace->up, config.sequence, config.hidden,
                                   config.intermediate, stream);
  if (status != CUBLAS_STATUS_SUCCESS)
    return status;

  cuda_error = silu_cuda(workspace->gate, workspace->activated_gate,
                         config.sequence * config.intermediate, stream);
  if (cuda_error == cudaSuccess)
    cuda_error = multiply_cuda(workspace->activated_gate, workspace->up,
                               workspace->gated,
                               config.sequence * config.intermediate, stream);
  if (cuda_error != cudaSuccess)
    return CUBLAS_STATUS_EXECUTION_FAILED;

  status = linear_cublas_row_major(handle, workspace->gated, weights.w_down,
                                   workspace->down, config.sequence,
                                   config.intermediate, config.hidden, stream);
  if (status != CUBLAS_STATUS_SUCCESS)
    return status;

  cuda_error = residual_add_cuda(input, workspace->down, output,
                                 config.sequence * config.hidden, stream);
  return cuda_error == cudaSuccess ? CUBLAS_STATUS_SUCCESS
                                   : CUBLAS_STATUS_EXECUTION_FAILED;
}

}  // namespace cuda_transformer
