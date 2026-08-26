#include "cuda_transformer/attention_sublayer.h"

#include "cuda_transformer/gemm.h"
#include "cuda_transformer/qkv_layout.h"
#include "cuda_transformer/transformer_primitives.h"

namespace cuda_transformer {
namespace {

void clear(AttentionSublayerWorkspace* workspace) {
  workspace->attention_token_major = nullptr;
  workspace->projected = nullptr;
  workspace->config = {};
}

bool configs_match(NormalizedAttentionCoreConfig left,
                   NormalizedAttentionCoreConfig right) {
  return left.sequence == right.sequence && left.hidden == right.hidden &&
         left.heads == right.heads && left.head_dim == right.head_dim &&
         left.rmsnorm_epsilon == right.rmsnorm_epsilon;
}

bool workspace_matches(const AttentionSublayerWorkspace* workspace) {
  return workspace != nullptr &&
         valid_normalized_attention_core_config(workspace->config) &&
         workspace->attention_token_major != nullptr &&
         workspace->projected != nullptr &&
         configs_match(workspace->config, workspace->normalized_attention.config);
}

bool workspace_is_empty(const AttentionSublayerWorkspace* workspace) {
  if (workspace == nullptr)
    return false;
  const auto& core = workspace->normalized_attention;
  return workspace->attention_token_major == nullptr &&
         workspace->projected == nullptr && core.normalized == nullptr &&
         core.rotated_q == nullptr && core.rotated_k == nullptr &&
         core.probabilities == nullptr && core.output == nullptr &&
         core.qkv.q_token == nullptr && core.qkv.k_token == nullptr &&
         core.qkv.v_token == nullptr && core.qkv.q_head == nullptr &&
         core.qkv.k_head == nullptr && core.qkv.v_head == nullptr;
}

cudaError_t first_error(cudaError_t first, cudaError_t next) {
  return first == cudaSuccess ? next : first;
}

}  // namespace

cudaError_t attention_sublayer_workspace_create(
    AttentionSublayerWorkspace* workspace,
    NormalizedAttentionCoreConfig config) {
  if (!workspace || !workspace_is_empty(workspace) ||
      !valid_normalized_attention_core_config(config))
    return cudaErrorInvalidValue;

  cudaError_t error = normalized_attention_core_workspace_create(
      &workspace->normalized_attention, config);
  const std::size_t bytes = config.sequence * config.hidden * sizeof(float);
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->attention_token_major, bytes);
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->projected, bytes);
  if (error != cudaSuccess)
    attention_sublayer_workspace_destroy(workspace);
  else
    workspace->config = config;
  return error;
}

cudaError_t attention_sublayer_workspace_destroy(
    AttentionSublayerWorkspace* workspace) {
  if (!workspace)
    return cudaErrorInvalidValue;

  cudaError_t error = cudaSuccess;
  if (workspace->attention_token_major != nullptr)
    error = cudaFree(workspace->attention_token_major);
  if (workspace->projected != nullptr)
    error = first_error(error, cudaFree(workspace->projected));
  error = first_error(
      error, normalized_attention_core_workspace_destroy(&workspace->normalized_attention));
  clear(workspace);
  return error;
}

cublasStatus_t attention_sublayer_cuda(cublasHandle_t handle, const float* input,
                                       AttentionSublayerWeights weights,
                                       AttentionSublayerWorkspace* workspace,
                                       float* output, cudaStream_t stream) {
  if (!handle || !input || !output || !weights.norm_weight || !weights.wq ||
      !weights.wk || !weights.wv || !weights.wo || !workspace_matches(workspace))
    return CUBLAS_STATUS_INVALID_VALUE;

  const NormalizedAttentionCoreConfig config = workspace->config;
  cublasStatus_t status = normalized_attention_core_cuda(
      handle, input, weights.norm_weight, weights.wq, weights.wk, weights.wv,
      &workspace->normalized_attention, stream);
  if (status != CUBLAS_STATUS_SUCCESS)
    return status;

  cudaError_t cuda_error = head_major_to_token_major_cuda(
      workspace->normalized_attention.output, workspace->attention_token_major,
      config.sequence, config.heads, config.head_dim, stream);
  if (cuda_error != cudaSuccess)
    return CUBLAS_STATUS_EXECUTION_FAILED;

  status = linear_cublas_row_major(handle, workspace->attention_token_major,
                                   weights.wo, workspace->projected,
                                   config.sequence, config.hidden, config.hidden,
                                   stream);
  if (status != CUBLAS_STATUS_SUCCESS)
    return status;

  cuda_error = residual_add_cuda(input, workspace->projected, output,
                                 config.sequence * config.hidden, stream);
  return cuda_error == cudaSuccess ? CUBLAS_STATUS_SUCCESS
                                   : CUBLAS_STATUS_EXECUTION_FAILED;
}

}  // namespace cuda_transformer
