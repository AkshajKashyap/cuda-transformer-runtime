#include "cuda_transformer/decoder_block.h"

namespace cuda_transformer {
namespace {

bool attention_configs_match(NormalizedAttentionCoreConfig left,
                             NormalizedAttentionCoreConfig right) {
  return left.sequence == right.sequence && left.hidden == right.hidden &&
         left.heads == right.heads && left.head_dim == right.head_dim &&
         left.rmsnorm_epsilon == right.rmsnorm_epsilon;
}

bool mlp_configs_match(MlpSublayerConfig left, MlpSublayerConfig right) {
  return left.sequence == right.sequence && left.hidden == right.hidden &&
         left.intermediate == right.intermediate &&
         left.rmsnorm_epsilon == right.rmsnorm_epsilon;
}

bool configs_match(DecoderBlockConfig left, DecoderBlockConfig right) {
  return attention_configs_match(left.attention, right.attention) &&
         mlp_configs_match(left.mlp, right.mlp);
}

void clear(DecoderBlockWorkspace* workspace) {
  workspace->attention_output = nullptr;
  workspace->config = {};
  workspace->allocation_config = {};
}

bool workspace_is_empty(const DecoderBlockWorkspace* workspace) {
  return workspace != nullptr && workspace->attention_output == nullptr &&
         workspace->attention.attention_token_major == nullptr &&
         workspace->attention.projected == nullptr &&
         workspace->mlp.normalized == nullptr && workspace->mlp.gate == nullptr &&
         workspace->mlp.up == nullptr && workspace->mlp.activated_gate == nullptr &&
         workspace->mlp.gated == nullptr && workspace->mlp.down == nullptr;
}

bool workspace_matches(const DecoderBlockWorkspace* workspace) {
  return workspace != nullptr && valid_decoder_block_config(workspace->config) &&
         configs_match(workspace->config, workspace->allocation_config) &&
         attention_configs_match(workspace->config.attention,
                                 workspace->attention.config) &&
         mlp_configs_match(workspace->config.mlp, workspace->mlp.config) &&
         workspace->attention_output != nullptr;
}

cudaError_t first_error(cudaError_t first, cudaError_t next) {
  return first == cudaSuccess ? next : first;
}

bool valid_weights(DecoderBlockWeights weights) {
  return weights.attention.norm_weight && weights.attention.wq &&
         weights.attention.wk && weights.attention.wv && weights.attention.wo &&
         weights.mlp.norm_weight && weights.mlp.w_gate && weights.mlp.w_up &&
         weights.mlp.w_down;
}

}  // namespace

bool valid_decoder_block_config(DecoderBlockConfig config) {
  return valid_normalized_attention_core_config(config.attention) &&
         valid_mlp_sublayer_config(config.mlp) &&
         config.attention.sequence == config.mlp.sequence &&
         config.attention.hidden == config.mlp.hidden;
}

cudaError_t decoder_block_workspace_create(DecoderBlockWorkspace* workspace,
                                           DecoderBlockConfig config) {
  if (!workspace || !workspace_is_empty(workspace) ||
      !valid_decoder_block_config(config))
    return cudaErrorInvalidValue;

  cudaError_t error = attention_sublayer_workspace_create(&workspace->attention,
                                                           config.attention);
  if (error == cudaSuccess)
    error = mlp_sublayer_workspace_create(&workspace->mlp, config.mlp);
  if (error == cudaSuccess) {
    const std::size_t bytes =
        config.attention.sequence * config.attention.hidden * sizeof(float);
    error = cudaMalloc(&workspace->attention_output, bytes);
  }
  if (error != cudaSuccess)
    decoder_block_workspace_destroy(workspace);
  else {
    workspace->config = config;
    workspace->allocation_config = config;
  }
  return error;
}

cudaError_t decoder_block_workspace_destroy(DecoderBlockWorkspace* workspace) {
  if (!workspace)
    return cudaErrorInvalidValue;

  cudaError_t error = cudaSuccess;
  if (workspace->attention_output != nullptr)
    error = cudaFree(workspace->attention_output);
  error = first_error(
      error, attention_sublayer_workspace_destroy(&workspace->attention));
  error = first_error(error, mlp_sublayer_workspace_destroy(&workspace->mlp));
  clear(workspace);
  return error;
}

cublasStatus_t decoder_block_cuda(cublasHandle_t handle, const float* input,
                                  DecoderBlockWeights weights,
                                  DecoderBlockWorkspace* workspace, float* output,
                                  cudaStream_t stream) {
  if (!handle || !input || !output || !valid_weights(weights) ||
      !workspace_matches(workspace))
    return CUBLAS_STATUS_INVALID_VALUE;

  cublasStatus_t status = attention_sublayer_cuda(
      handle, input, weights.attention, &workspace->attention,
      workspace->attention_output, stream);
  if (status != CUBLAS_STATUS_SUCCESS)
    return status;
  return mlp_sublayer_cuda(handle, workspace->attention_output, weights.mlp,
                           &workspace->mlp, output, stream);
}

}  // namespace cuda_transformer
