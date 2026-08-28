#include "cuda_transformer/incremental_decoder_block.h"

#include "cuda_transformer/gemm.h"
#include "cuda_transformer/incremental_attention.h"
#include "cuda_transformer/qkv_layout.h"
#include "cuda_transformer/transformer_primitives.h"

#include <cmath>

namespace cuda_transformer {
namespace {

void clear(IncrementalDecoderBlockWorkspace* workspace) {
  workspace->attention_normalized = nullptr;
  workspace->attention_head = nullptr;
  workspace->attention_token = nullptr;
  workspace->attention_projected = nullptr;
  workspace->attention_residual = nullptr;
  workspace->mlp_normalized = nullptr;
  workspace->gate = nullptr;
  workspace->up = nullptr;
  workspace->activated_gate = nullptr;
  workspace->gated = nullptr;
  workspace->down = nullptr;
  workspace->config = {};
  workspace->allocation_config = {};
}

bool configs_match(IncrementalDecoderBlockConfig left,
                   IncrementalDecoderBlockConfig right) {
  return left.hidden == right.hidden && left.heads == right.heads &&
         left.head_dim == right.head_dim &&
         left.intermediate == right.intermediate &&
         left.attention_rmsnorm_epsilon == right.attention_rmsnorm_epsilon &&
         left.mlp_rmsnorm_epsilon == right.mlp_rmsnorm_epsilon;
}

bool workspace_is_empty(const IncrementalDecoderBlockWorkspace* workspace) {
  return workspace != nullptr && workspace->qkv.q_token == nullptr &&
         workspace->qkv.k_token == nullptr && workspace->qkv.v_token == nullptr &&
         workspace->qkv.q_head == nullptr && workspace->qkv.k_head == nullptr &&
         workspace->qkv.v_head == nullptr &&
         workspace->attention_normalized == nullptr &&
         workspace->attention_head == nullptr && workspace->attention_token == nullptr &&
         workspace->attention_projected == nullptr &&
         workspace->attention_residual == nullptr &&
         workspace->mlp_normalized == nullptr && workspace->gate == nullptr &&
         workspace->up == nullptr && workspace->activated_gate == nullptr &&
         workspace->gated == nullptr && workspace->down == nullptr;
}

bool qkv_matches(const QkvProjectionWorkspace& workspace,
                 IncrementalDecoderBlockConfig config) {
  const QkvProjectionConfig expected{1, config.hidden, config.heads,
                                     config.head_dim};
  const QkvProjectionConfig actual = workspace.config;
  return valid_qkv_projection_config(actual) &&
         actual.sequence == expected.sequence && actual.hidden == expected.hidden &&
         actual.heads == expected.heads && actual.head_dim == expected.head_dim &&
         workspace.q_token != nullptr && workspace.k_token != nullptr &&
         workspace.v_token != nullptr && workspace.q_head != nullptr &&
         workspace.k_head != nullptr && workspace.v_head != nullptr;
}

bool workspace_matches(const IncrementalDecoderBlockWorkspace* workspace) {
  return workspace != nullptr &&
         valid_incremental_decoder_block_config(workspace->config) &&
         configs_match(workspace->config, workspace->allocation_config) &&
         qkv_matches(workspace->qkv, workspace->config) &&
         workspace->attention_normalized != nullptr &&
         workspace->attention_head != nullptr && workspace->attention_token != nullptr &&
         workspace->attention_projected != nullptr &&
         workspace->attention_residual != nullptr &&
         workspace->mlp_normalized != nullptr && workspace->gate != nullptr &&
         workspace->up != nullptr && workspace->activated_gate != nullptr &&
         workspace->gated != nullptr && workspace->down != nullptr;
}

bool cache_matches(const KvCache* cache, IncrementalDecoderBlockConfig config) {
  return cache != nullptr && cache->keys != nullptr && cache->values != nullptr &&
         cache->batch == 1 && cache->heads == config.heads &&
         cache->head_dim == config.head_dim && cache->max_sequence != 0 &&
         cache->current_length < cache->max_sequence;
}

bool valid_weights(IncrementalDecoderBlockWeights weights) {
  return weights.attention.norm_weight && weights.attention.wq &&
         weights.attention.wk && weights.attention.wv && weights.attention.wo &&
         weights.mlp.norm_weight && weights.mlp.w_gate && weights.mlp.w_up &&
         weights.mlp.w_down;
}

cudaError_t first_error(cudaError_t first, cudaError_t next) {
  return first == cudaSuccess ? next : first;
}

}  // namespace

bool valid_incremental_decoder_block_config(IncrementalDecoderBlockConfig config) {
  return config.hidden != 0 && config.heads != 0 && config.head_dim != 0 &&
         config.intermediate != 0 && config.head_dim % 2 == 0 &&
         config.hidden % config.heads == 0 &&
         config.hidden / config.heads == config.head_dim &&
         std::isfinite(config.attention_rmsnorm_epsilon) &&
         config.attention_rmsnorm_epsilon > 0.0F &&
         std::isfinite(config.mlp_rmsnorm_epsilon) &&
         config.mlp_rmsnorm_epsilon > 0.0F;
}

cudaError_t incremental_decoder_block_workspace_create(
    IncrementalDecoderBlockWorkspace* workspace,
    IncrementalDecoderBlockConfig config) {
  if (!workspace || !workspace_is_empty(workspace) ||
      !valid_incremental_decoder_block_config(config))
    return cudaErrorInvalidValue;

  const std::size_t hidden_bytes = config.hidden * sizeof(float);
  const std::size_t intermediate_bytes = config.intermediate * sizeof(float);
  cudaError_t error = qkv_projection_workspace_create(
      &workspace->qkv, {1, config.hidden, config.heads, config.head_dim});
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->attention_normalized, hidden_bytes);
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->attention_head, hidden_bytes);
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->attention_token, hidden_bytes);
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->attention_projected, hidden_bytes);
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->attention_residual, hidden_bytes);
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->mlp_normalized, hidden_bytes);
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->gate, intermediate_bytes);
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->up, intermediate_bytes);
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->activated_gate, intermediate_bytes);
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->gated, intermediate_bytes);
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->down, hidden_bytes);
  if (error != cudaSuccess)
    incremental_decoder_block_workspace_destroy(workspace);
  else {
    workspace->config = config;
    workspace->allocation_config = config;
  }
  return error;
}

cudaError_t incremental_decoder_block_workspace_destroy(
    IncrementalDecoderBlockWorkspace* workspace) {
  if (!workspace)
    return cudaErrorInvalidValue;

  cudaError_t error = cudaSuccess;
  for (float* pointer : {workspace->attention_normalized,
                         workspace->attention_head, workspace->attention_token,
                         workspace->attention_projected,
                         workspace->attention_residual,
                         workspace->mlp_normalized, workspace->gate,
                         workspace->up, workspace->activated_gate,
                         workspace->gated, workspace->down}) {
    if (pointer != nullptr)
      error = first_error(error, cudaFree(pointer));
  }
  error = first_error(error, qkv_projection_workspace_destroy(&workspace->qkv));
  clear(workspace);
  return error;
}

cublasStatus_t incremental_decoder_block_cuda(
    cublasHandle_t handle, const float* input,
    IncrementalDecoderBlockWeights weights, KvCache* cache,
    IncrementalDecoderBlockWorkspace* workspace, float* output,
    cudaStream_t stream) {
  if (!handle || !input || !output || !valid_weights(weights) ||
      !cache_matches(cache, workspace ? workspace->config
                                      : IncrementalDecoderBlockConfig{}) ||
      !workspace_matches(workspace))
    return CUBLAS_STATUS_INVALID_VALUE;

  const IncrementalDecoderBlockConfig config = workspace->config;
  cudaError_t cuda_error = rmsnorm_cuda(
      input, weights.attention.norm_weight, workspace->attention_normalized, 1,
      config.hidden, config.attention_rmsnorm_epsilon, stream);
  if (cuda_error != cudaSuccess)
    return CUBLAS_STATUS_EXECUTION_FAILED;

  cublasStatus_t status = project_qkv_cuda(
      handle, workspace->attention_normalized, weights.attention.wq,
      weights.attention.wk, weights.attention.wv, &workspace->qkv, stream);
  if (status != CUBLAS_STATUS_SUCCESS)
    return status;
  cuda_error = incremental_decode(cache, workspace->qkv.q_head,
                                  workspace->qkv.k_head, workspace->qkv.v_head,
                                  workspace->attention_head, stream);
  if (cuda_error != cudaSuccess)
    return CUBLAS_STATUS_EXECUTION_FAILED;
  cuda_error = head_major_to_token_major_cuda(
      workspace->attention_head, workspace->attention_token, 1, config.heads,
      config.head_dim, stream);
  if (cuda_error != cudaSuccess)
    return CUBLAS_STATUS_EXECUTION_FAILED;
  status = linear_cublas_row_major(
      handle, workspace->attention_token, weights.attention.wo,
      workspace->attention_projected, 1, config.hidden, config.hidden, stream);
  if (status != CUBLAS_STATUS_SUCCESS)
    return status;
  cuda_error = residual_add_cuda(input, workspace->attention_projected,
                                 workspace->attention_residual, config.hidden,
                                 stream);
  if (cuda_error != cudaSuccess)
    return CUBLAS_STATUS_EXECUTION_FAILED;

  cuda_error = rmsnorm_cuda(workspace->attention_residual, weights.mlp.norm_weight,
                            workspace->mlp_normalized, 1, config.hidden,
                            config.mlp_rmsnorm_epsilon, stream);
  if (cuda_error != cudaSuccess)
    return CUBLAS_STATUS_EXECUTION_FAILED;
  status = linear_cublas_row_major(
      handle, workspace->mlp_normalized, weights.mlp.w_gate, workspace->gate, 1,
      config.hidden, config.intermediate, stream);
  if (status != CUBLAS_STATUS_SUCCESS)
    return status;
  status = linear_cublas_row_major(handle, workspace->mlp_normalized,
                                   weights.mlp.w_up, workspace->up, 1,
                                   config.hidden, config.intermediate, stream);
  if (status != CUBLAS_STATUS_SUCCESS)
    return status;
  cuda_error = silu_cuda(workspace->gate, workspace->activated_gate,
                         config.intermediate, stream);
  if (cuda_error == cudaSuccess)
    cuda_error = multiply_cuda(workspace->activated_gate, workspace->up,
                               workspace->gated, config.intermediate, stream);
  if (cuda_error != cudaSuccess)
    return CUBLAS_STATUS_EXECUTION_FAILED;
  status = linear_cublas_row_major(handle, workspace->gated, weights.mlp.w_down,
                                   workspace->down, 1, config.intermediate,
                                   config.hidden, stream);
  if (status != CUBLAS_STATUS_SUCCESS)
    return status;
  cuda_error = residual_add_cuda(workspace->attention_residual, workspace->down,
                                 output, config.hidden, stream);
  return cuda_error == cudaSuccess ? CUBLAS_STATUS_SUCCESS
                                   : CUBLAS_STATUS_EXECUTION_FAILED;
}

}  // namespace cuda_transformer
