#include "cuda_transformer/tiny_model.h"

#include "cuda_transformer/attention.h"
#include "cuda_transformer/gemm.h"
#include "cuda_transformer/qkv_layout.h"
#include "cuda_transformer/transformer_primitives.h"

#include <cmath>
#include <new>
#include <vector>

namespace cuda_transformer {
namespace {

constexpr int kEmbeddingThreads = 256;

__global__ void embedding_lookup_kernel(const int* token_ids,
                                        const float* embeddings, float* output,
                                        std::size_t sequence,
                                        std::size_t hidden) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = sequence * hidden;
  if (index >= count)
    return;
  const std::size_t position = index / hidden;
  const std::size_t dimension = index % hidden;
  const int token_id = token_ids[position];
  output[index] = embeddings[static_cast<std::size_t>(token_id) * hidden +
                             dimension];
}

void clear(TinyModelWorkspace* workspace) {
  workspace->copied_token_ids = nullptr;
  workspace->activation_a = nullptr;
  workspace->activation_b = nullptr;
  workspace->layer_workspaces = nullptr;
  workspace->config = {};
  workspace->allocation_config = {};
}

bool configs_match(TinyModelConfig left, TinyModelConfig right) {
  return left.vocabulary_size == right.vocabulary_size &&
         left.sequence == right.sequence && left.hidden == right.hidden &&
         left.layers == right.layers && left.heads == right.heads &&
         left.head_dim == right.head_dim &&
         left.intermediate == right.intermediate &&
         left.rmsnorm_epsilon == right.rmsnorm_epsilon;
}

DecoderBlockConfig layer_config(TinyModelConfig config) {
  return {{config.sequence, config.hidden, config.heads, config.head_dim,
           config.rmsnorm_epsilon},
          {config.sequence, config.hidden, config.intermediate,
           config.rmsnorm_epsilon}};
}

bool workspace_is_empty(const TinyModelWorkspace* workspace) {
  return workspace != nullptr && workspace->copied_token_ids == nullptr &&
         workspace->activation_a == nullptr && workspace->activation_b == nullptr &&
         workspace->layer_workspaces == nullptr;
}

bool valid_layer_weights(DecoderBlockWeights weights) {
  return weights.attention.norm_weight != nullptr && weights.attention.wq != nullptr &&
         weights.attention.wk != nullptr && weights.attention.wv != nullptr &&
         weights.attention.wo != nullptr && weights.mlp.norm_weight != nullptr &&
         weights.mlp.w_gate != nullptr && weights.mlp.w_up != nullptr &&
         weights.mlp.w_down != nullptr;
}

bool valid_weights(TinyModelWeights weights, TinyModelConfig config) {
  if (weights.token_embeddings == nullptr || weights.layers == nullptr ||
      weights.final_norm_weight == nullptr || weights.lm_head_weight == nullptr)
    return false;
  for (std::size_t layer = 0; layer < config.layers; ++layer)
    if (!valid_layer_weights(weights.layers[layer].decoder))
      return false;
  return true;
}

bool workspace_matches(const TinyModelWorkspace* workspace) {
  if (workspace == nullptr || !valid_tiny_model_config(workspace->config) ||
      !configs_match(workspace->config, workspace->allocation_config) ||
      workspace->copied_token_ids == nullptr || workspace->activation_a == nullptr ||
      workspace->activation_b == nullptr || workspace->layer_workspaces == nullptr)
    return false;
  const DecoderBlockConfig expected = layer_config(workspace->config);
  for (std::size_t layer = 0; layer < workspace->config.layers; ++layer) {
    const DecoderBlockWorkspace& block = workspace->layer_workspaces[layer];
    if (block.config.attention.sequence != expected.attention.sequence ||
        block.config.attention.hidden != expected.attention.hidden ||
        block.config.attention.heads != expected.attention.heads ||
        block.config.attention.head_dim != expected.attention.head_dim ||
        block.config.attention.rmsnorm_epsilon != expected.attention.rmsnorm_epsilon ||
        block.config.mlp.sequence != expected.mlp.sequence ||
        block.config.mlp.hidden != expected.mlp.hidden ||
        block.config.mlp.intermediate != expected.mlp.intermediate ||
        block.config.mlp.rmsnorm_epsilon != expected.mlp.rmsnorm_epsilon)
      return false;
  }
  return true;
}

bool valid_token_ids(const int* token_ids, TinyModelConfig config) {
  if (token_ids == nullptr)
    return false;
  for (std::size_t position = 0; position < config.sequence; ++position)
    if (token_ids[position] < 0 ||
        static_cast<std::size_t>(token_ids[position]) >= config.vocabulary_size)
      return false;
  return true;
}

cudaError_t first_error(cudaError_t first, cudaError_t next) {
  return first == cudaSuccess ? next : first;
}

void decoder_block_cpu(const float* input, DecoderBlockWeights weights,
                       TinyModelConfig config, float* output) {
  const std::size_t activation_count = config.sequence * config.hidden;
  const std::size_t probability_count =
      config.heads * config.sequence * config.sequence;
  const std::size_t intermediate_count = config.sequence * config.intermediate;
  std::vector<float> normalized(activation_count), q_token(activation_count),
      k_token(activation_count), v_token(activation_count),
      q_head(activation_count), k_head(activation_count), v_head(activation_count),
      rotated_q(activation_count), probabilities(probability_count),
      attention_head(activation_count), attention_token(activation_count),
      projected(activation_count), attention_residual(activation_count),
      mlp_normalized(activation_count), gate(intermediate_count),
      up(intermediate_count), activated_gate(intermediate_count),
      gated(intermediate_count), down(activation_count);

  rmsnorm_cpu(input, weights.attention.norm_weight, normalized.data(),
              config.sequence, config.hidden, config.rmsnorm_epsilon);
  gemm_cpu(normalized.data(), weights.attention.wq, q_token.data(),
           config.sequence, config.hidden, config.hidden);
  gemm_cpu(normalized.data(), weights.attention.wk, k_token.data(),
           config.sequence, config.hidden, config.hidden);
  gemm_cpu(normalized.data(), weights.attention.wv, v_token.data(),
           config.sequence, config.hidden, config.hidden);
  token_major_to_head_major_cpu(q_token.data(), q_head.data(), config.sequence,
                                config.heads, config.head_dim);
  token_major_to_head_major_cpu(k_token.data(), k_head.data(), config.sequence,
                                config.heads, config.head_dim);
  token_major_to_head_major_cpu(v_token.data(), v_head.data(), config.sequence,
                                config.heads, config.head_dim);
  const AttentionShape attention_shape{1, config.heads, config.sequence,
                                       config.head_dim};
  attention_cpu(q_head.data(), k_head.data(), v_head.data(), rotated_q.data(),
                probabilities.data(), attention_head.data(), attention_shape);
  head_major_to_token_major_cpu(attention_head.data(), attention_token.data(),
                                config.sequence, config.heads, config.head_dim);
  gemm_cpu(attention_token.data(), weights.attention.wo, projected.data(),
           config.sequence, config.hidden, config.hidden);
  residual_add_cpu(input, projected.data(), attention_residual.data(),
                   activation_count);

  rmsnorm_cpu(attention_residual.data(), weights.mlp.norm_weight,
              mlp_normalized.data(), config.sequence, config.hidden,
              config.rmsnorm_epsilon);
  gemm_cpu(mlp_normalized.data(), weights.mlp.w_gate, gate.data(),
           config.sequence, config.hidden, config.intermediate);
  gemm_cpu(mlp_normalized.data(), weights.mlp.w_up, up.data(), config.sequence,
           config.hidden, config.intermediate);
  silu_cpu(gate.data(), activated_gate.data(), intermediate_count);
  multiply_cpu(activated_gate.data(), up.data(), gated.data(), intermediate_count);
  gemm_cpu(gated.data(), weights.mlp.w_down, down.data(), config.sequence,
           config.intermediate, config.hidden);
  residual_add_cpu(attention_residual.data(), down.data(), output,
                   activation_count);
}

}  // namespace

bool valid_tiny_model_config(TinyModelConfig config) {
  return config.vocabulary_size != 0 && config.sequence != 0 &&
         config.hidden != 0 && config.layers != 0 && config.heads != 0 &&
         config.head_dim != 0 && config.intermediate != 0 &&
         config.head_dim % 2 == 0 && config.hidden == config.heads * config.head_dim &&
         std::isfinite(config.rmsnorm_epsilon) && config.rmsnorm_epsilon > 0.0F;
}

cudaError_t tiny_model_workspace_create(TinyModelWorkspace* workspace,
                                        TinyModelConfig config) {
  if (workspace == nullptr || !workspace_is_empty(workspace) ||
      !valid_tiny_model_config(config))
    return cudaErrorInvalidValue;

  workspace->layer_workspaces =
      new (std::nothrow) DecoderBlockWorkspace[config.layers];
  if (workspace->layer_workspaces == nullptr)
    return cudaErrorMemoryAllocation;
  workspace->allocation_config = config;

  const std::size_t token_bytes = config.sequence * sizeof(int);
  const std::size_t activation_bytes =
      config.sequence * config.hidden * sizeof(float);
  cudaError_t error = cudaMalloc(&workspace->copied_token_ids, token_bytes);
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->activation_a, activation_bytes);
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->activation_b, activation_bytes);
  for (std::size_t layer = 0; error == cudaSuccess && layer < config.layers;
       ++layer)
    error = decoder_block_workspace_create(&workspace->layer_workspaces[layer],
                                           layer_config(config));
  if (error != cudaSuccess) {
    tiny_model_workspace_destroy(workspace);
  } else {
    workspace->config = config;
  }
  return error;
}

cudaError_t tiny_model_workspace_destroy(TinyModelWorkspace* workspace) {
  if (workspace == nullptr)
    return cudaErrorInvalidValue;

  cudaError_t error = cudaSuccess;
  const std::size_t layers = workspace->allocation_config.layers;
  if (workspace->layer_workspaces != nullptr) {
    for (std::size_t layer = 0; layer < layers; ++layer)
      error = first_error(error,
                          decoder_block_workspace_destroy(
                              &workspace->layer_workspaces[layer]));
    delete[] workspace->layer_workspaces;
  }
  if (workspace->copied_token_ids != nullptr)
    error = first_error(error, cudaFree(workspace->copied_token_ids));
  if (workspace->activation_a != nullptr)
    error = first_error(error, cudaFree(workspace->activation_a));
  if (workspace->activation_b != nullptr)
    error = first_error(error, cudaFree(workspace->activation_b));
  clear(workspace);
  return error;
}

void embedding_lookup_cpu(const int* token_ids, const float* embeddings,
                          float* output, std::size_t sequence,
                          std::size_t vocabulary_size, std::size_t hidden) {
  (void)vocabulary_size;
  for (std::size_t position = 0; position < sequence; ++position)
    for (std::size_t dimension = 0; dimension < hidden; ++dimension)
      output[position * hidden + dimension] =
          embeddings[static_cast<std::size_t>(token_ids[position]) * hidden +
                     dimension];
}

cudaError_t embedding_lookup_cuda(const int* device_token_ids,
                                  const float* device_embeddings,
                                  float* device_output,
                                  std::size_t sequence,
                                  std::size_t vocabulary_size,
                                  std::size_t hidden, cudaStream_t stream) {
  if (device_token_ids == nullptr || device_embeddings == nullptr ||
      device_output == nullptr || sequence == 0 || vocabulary_size == 0 ||
      hidden == 0)
    return cudaErrorInvalidValue;
  const std::size_t count = sequence * hidden;
  embedding_lookup_kernel<<<static_cast<unsigned int>(
                               (count + kEmbeddingThreads - 1) /
                               kEmbeddingThreads),
                           kEmbeddingThreads, 0, stream>>>(
      device_token_ids, device_embeddings, device_output, sequence, hidden);
  return cudaGetLastError();
}

bool tiny_model_forward_cpu(const int* token_ids, TinyModelWeights weights,
                            TinyModelConfig config, float* logits) {
  if (!valid_tiny_model_config(config) || !valid_weights(weights, config) ||
      !valid_token_ids(token_ids, config) || logits == nullptr)
    return false;

  const std::size_t activation_count = config.sequence * config.hidden;
  std::vector<float> activation_a(activation_count), activation_b(activation_count);
  embedding_lookup_cpu(token_ids, weights.token_embeddings, activation_a.data(),
                       config.sequence, config.vocabulary_size, config.hidden);
  const float* current = activation_a.data();
  float* next = activation_b.data();
  for (std::size_t layer = 0; layer < config.layers; ++layer) {
    decoder_block_cpu(current, weights.layers[layer].decoder, config, next);
    current = next;
    next = next == activation_a.data() ? activation_b.data() : activation_a.data();
  }
  rmsnorm_cpu(current, weights.final_norm_weight, next, config.sequence,
              config.hidden, config.rmsnorm_epsilon);
  gemm_cpu(next, weights.lm_head_weight, logits, config.sequence, config.hidden,
           config.vocabulary_size);
  return true;
}

cublasStatus_t tiny_model_forward_cuda(cublasHandle_t handle,
                                        const int* device_token_ids,
                                        TinyModelWeights device_weights,
                                        TinyModelWorkspace* workspace,
                                        float* device_logits,
                                        cudaStream_t stream) {
  if (handle == nullptr || device_token_ids == nullptr || device_logits == nullptr ||
      !workspace_matches(workspace) ||
      !valid_weights(device_weights, workspace->config))
    return CUBLAS_STATUS_INVALID_VALUE;

  const TinyModelConfig config = workspace->config;
  cudaError_t cuda_error = embedding_lookup_cuda(
      device_token_ids, device_weights.token_embeddings, workspace->activation_a,
      config.sequence, config.vocabulary_size, config.hidden, stream);
  if (cuda_error != cudaSuccess)
    return CUBLAS_STATUS_EXECUTION_FAILED;

  const float* current = workspace->activation_a;
  float* next = workspace->activation_b;
  for (std::size_t layer = 0; layer < config.layers; ++layer) {
    const cublasStatus_t status = decoder_block_cuda(
        handle, current, device_weights.layers[layer].decoder,
        &workspace->layer_workspaces[layer], next, stream);
    if (status != CUBLAS_STATUS_SUCCESS)
      return status;
    current = next;
    next = next == workspace->activation_a ? workspace->activation_b
                                            : workspace->activation_a;
  }
  cuda_error = rmsnorm_cuda(current, device_weights.final_norm_weight, next,
                             config.sequence, config.hidden,
                             config.rmsnorm_epsilon, stream);
  if (cuda_error != cudaSuccess)
    return CUBLAS_STATUS_EXECUTION_FAILED;
  return linear_cublas_row_major(handle, next, device_weights.lm_head_weight,
                                 device_logits, config.sequence, config.hidden,
                                 config.vocabulary_size, stream);
}

cublasStatus_t tiny_model_forward_host_tokens_cuda(
    cublasHandle_t handle, const int* host_token_ids,
    TinyModelWeights device_weights, TinyModelWorkspace* workspace,
    float* device_logits, cudaStream_t stream) {
  if (handle == nullptr || device_logits == nullptr || !workspace_matches(workspace) ||
      !valid_weights(device_weights, workspace->config) ||
      !valid_token_ids(host_token_ids, workspace->config))
    return CUBLAS_STATUS_INVALID_VALUE;
  const cudaError_t error = cudaMemcpyAsync(
      workspace->copied_token_ids, host_token_ids,
      workspace->config.sequence * sizeof(int), cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess)
    return CUBLAS_STATUS_EXECUTION_FAILED;
  return tiny_model_forward_cuda(handle, workspace->copied_token_ids,
                                 device_weights, workspace, device_logits, stream);
}

}  // namespace cuda_transformer
