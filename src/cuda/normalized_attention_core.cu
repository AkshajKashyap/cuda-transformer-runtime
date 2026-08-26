#include "cuda_transformer/normalized_attention_core.h"

#include "cuda_transformer/attention.h"
#include "cuda_transformer/transformer_primitives.h"

#include <cmath>

namespace cuda_transformer {
namespace {

void clear(NormalizedAttentionCoreWorkspace* workspace) {
  workspace->normalized = nullptr;
  workspace->rotated_q = nullptr;
  workspace->rotated_k = nullptr;
  workspace->probabilities = nullptr;
  workspace->output = nullptr;
  workspace->config = {};
}

bool qkv_matches(const QkvProjectionWorkspace& workspace,
                 NormalizedAttentionCoreConfig config) {
  const QkvProjectionConfig expected{config.sequence, config.hidden,
                                     config.heads, config.head_dim};
  const QkvProjectionConfig actual = workspace.config;
  return valid_qkv_projection_config(actual) &&
         actual.sequence == expected.sequence && actual.hidden == expected.hidden &&
         actual.heads == expected.heads && actual.head_dim == expected.head_dim &&
         workspace.q_token != nullptr && workspace.k_token != nullptr &&
         workspace.v_token != nullptr && workspace.q_head != nullptr &&
         workspace.k_head != nullptr && workspace.v_head != nullptr;
}

bool workspace_matches(const NormalizedAttentionCoreWorkspace* workspace) {
  return workspace != nullptr &&
         valid_normalized_attention_core_config(workspace->config) &&
         workspace->normalized != nullptr && workspace->rotated_q != nullptr &&
         workspace->rotated_k != nullptr && workspace->probabilities != nullptr &&
         workspace->output != nullptr && qkv_matches(workspace->qkv, workspace->config);
}

bool workspace_is_empty(const NormalizedAttentionCoreWorkspace* workspace) {
  return workspace != nullptr && workspace->normalized == nullptr &&
         workspace->rotated_q == nullptr && workspace->rotated_k == nullptr &&
         workspace->probabilities == nullptr && workspace->output == nullptr &&
         workspace->qkv.q_token == nullptr && workspace->qkv.k_token == nullptr &&
         workspace->qkv.v_token == nullptr && workspace->qkv.q_head == nullptr &&
         workspace->qkv.k_head == nullptr && workspace->qkv.v_head == nullptr;
}

cudaError_t first_error(cudaError_t first, cudaError_t next) {
  return first == cudaSuccess ? next : first;
}

}  // namespace

bool valid_normalized_attention_core_config(NormalizedAttentionCoreConfig config) {
  return config.sequence != 0 && config.hidden != 0 && config.heads != 0 &&
         config.head_dim != 0 && config.head_dim % 2 == 0 &&
         std::isfinite(config.rmsnorm_epsilon) && config.rmsnorm_epsilon > 0.0F &&
         config.hidden % config.heads == 0 &&
         config.hidden / config.heads == config.head_dim;
}

cudaError_t normalized_attention_core_workspace_create(
    NormalizedAttentionCoreWorkspace* workspace,
    NormalizedAttentionCoreConfig config) {
  if (!workspace || !workspace_is_empty(workspace) ||
      !valid_normalized_attention_core_config(config))
    return cudaErrorInvalidValue;

  const QkvProjectionConfig qkv_config{config.sequence, config.hidden,
                                       config.heads, config.head_dim};
  cudaError_t error = qkv_projection_workspace_create(&workspace->qkv, qkv_config);
  const std::size_t activation_bytes =
      config.sequence * config.hidden * sizeof(float);
  const std::size_t probability_bytes =
      config.heads * config.sequence * config.sequence * sizeof(float);
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->normalized, activation_bytes);
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->rotated_q, activation_bytes);
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->rotated_k, activation_bytes);
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->probabilities, probability_bytes);
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->output, activation_bytes);
  if (error != cudaSuccess)
    normalized_attention_core_workspace_destroy(workspace);
  else
    workspace->config = config;
  return error;
}

cudaError_t normalized_attention_core_workspace_destroy(
    NormalizedAttentionCoreWorkspace* workspace) {
  if (!workspace)
    return cudaErrorInvalidValue;

  cudaError_t error = cudaSuccess;
  for (float* pointer : {workspace->normalized, workspace->rotated_q,
                         workspace->rotated_k, workspace->probabilities,
                         workspace->output}) {
    if (pointer != nullptr)
      error = first_error(error, cudaFree(pointer));
  }
  error = first_error(error, qkv_projection_workspace_destroy(&workspace->qkv));
  clear(workspace);
  return error;
}

cublasStatus_t normalized_attention_core_cuda(
    cublasHandle_t handle, const float* input, const float* norm_weight,
    const float* wq, const float* wk, const float* wv,
    NormalizedAttentionCoreWorkspace* workspace, cudaStream_t stream) {
  if (!handle || !input || !norm_weight || !wq || !wk || !wv ||
      !workspace_matches(workspace))
    return CUBLAS_STATUS_INVALID_VALUE;

  const NormalizedAttentionCoreConfig config = workspace->config;
  cudaError_t cuda_error = rmsnorm_cuda(
      input, norm_weight, workspace->normalized, config.sequence, config.hidden,
      config.rmsnorm_epsilon, stream);
  if (cuda_error != cudaSuccess)
    return CUBLAS_STATUS_EXECUTION_FAILED;

  cublasStatus_t cublas_error = project_qkv_cuda(
      handle, workspace->normalized, wq, wk, wv, &workspace->qkv, stream);
  if (cublas_error != CUBLAS_STATUS_SUCCESS)
    return cublas_error;

  const AttentionShape attention{1, config.heads, config.sequence,
                                 config.head_dim};
  cuda_error = rope_cuda(workspace->qkv.q_head, workspace->rotated_q,
                         attention, stream);
  if (cuda_error == cudaSuccess)
    cuda_error = rope_cuda(workspace->qkv.k_head, workspace->rotated_k,
                           attention, stream);
  if (cuda_error == cudaSuccess)
    cuda_error = attention_scores_cuda(workspace->rotated_q, workspace->rotated_k,
                                       workspace->probabilities, attention, stream);
  if (cuda_error == cudaSuccess)
    cuda_error = softmax_cuda(workspace->probabilities, workspace->probabilities,
                              config.heads * config.sequence, config.sequence,
                              stream);
  if (cuda_error == cudaSuccess)
    cuda_error = attention_values_cuda(workspace->probabilities,
                                       workspace->qkv.v_head, workspace->output,
                                       attention, stream);
  return cuda_error == cudaSuccess ? CUBLAS_STATUS_SUCCESS
                                   : CUBLAS_STATUS_EXECUTION_FAILED;
}

}  // namespace cuda_transformer
