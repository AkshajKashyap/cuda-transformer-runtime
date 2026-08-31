#include "cuda_transformer/tiny_model.h"

#include "cuda_transformer/attention.h"
#include "cuda_transformer/gemm.h"
#include "cuda_transformer/qkv_layout.h"
#include "cuda_transformer/transformer_primitives.h"

#include <cmath>
#include <limits>
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
      weights.layer_count != config.layers || weights.final_norm_weight == nullptr ||
      weights.lm_head_weight == nullptr)
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

void clear_incremental(TinyModelIncrementalWorkspace* workspace) {
  workspace->copied_token_id = nullptr;
  workspace->activation_a = nullptr;
  workspace->activation_b = nullptr;
  workspace->layer_workspaces = nullptr;
  workspace->layer_caches = nullptr;
  workspace->model_config = {};
  workspace->allocation_model_config = {};
  workspace->max_sequence_length = 0;
  workspace->allocation_max_sequence_length = 0;
}

IncrementalDecoderBlockConfig incremental_layer_config(
    TinyModelConfig config, std::size_t max_sequence_length) {
  return {config.hidden, config.heads, config.head_dim, config.intermediate,
          config.rmsnorm_epsilon, config.rmsnorm_epsilon, max_sequence_length};
}

bool incremental_workspace_is_empty(
    const TinyModelIncrementalWorkspace* workspace) {
  return workspace != nullptr && workspace->copied_token_id == nullptr &&
         workspace->activation_a == nullptr && workspace->activation_b == nullptr &&
         workspace->layer_workspaces == nullptr && workspace->layer_caches == nullptr;
}

bool incremental_block_workspace_matches(
    const IncrementalDecoderBlockWorkspace& workspace,
    IncrementalDecoderBlockConfig expected) {
  const IncrementalDecoderBlockConfig config = workspace.config;
  const IncrementalDecoderBlockConfig allocation = workspace.allocation_config;
  return config.hidden == expected.hidden && config.heads == expected.heads &&
         config.head_dim == expected.head_dim &&
         config.intermediate == expected.intermediate &&
         config.attention_rmsnorm_epsilon == expected.attention_rmsnorm_epsilon &&
         config.mlp_rmsnorm_epsilon == expected.mlp_rmsnorm_epsilon &&
         config.max_sequence == expected.max_sequence &&
         allocation.hidden == expected.hidden && allocation.heads == expected.heads &&
         allocation.head_dim == expected.head_dim &&
         allocation.intermediate == expected.intermediate &&
         allocation.attention_rmsnorm_epsilon == expected.attention_rmsnorm_epsilon &&
         allocation.mlp_rmsnorm_epsilon == expected.mlp_rmsnorm_epsilon &&
         allocation.max_sequence == expected.max_sequence &&
         workspace.qkv.q_token != nullptr && workspace.qkv.k_token != nullptr &&
         workspace.qkv.v_token != nullptr && workspace.qkv.q_head != nullptr &&
         workspace.qkv.k_head != nullptr && workspace.qkv.v_head != nullptr &&
         workspace.qkv.config.sequence == 1 &&
         workspace.qkv.config.hidden == expected.hidden &&
         workspace.qkv.config.heads == expected.heads &&
         workspace.qkv.config.head_dim == expected.head_dim &&
         workspace.attention.rotated_q != nullptr &&
         workspace.attention.rotated_k != nullptr && workspace.attention.scores != nullptr &&
         workspace.attention.heads == expected.heads &&
         workspace.attention.head_dim == expected.head_dim &&
         workspace.attention.max_sequence == expected.max_sequence &&
         workspace.attention_normalized != nullptr && workspace.attention_head != nullptr &&
         workspace.attention_token != nullptr && workspace.attention_projected != nullptr &&
         workspace.attention_residual != nullptr && workspace.mlp_normalized != nullptr &&
         workspace.gate != nullptr && workspace.up != nullptr &&
         workspace.activated_gate != nullptr && workspace.gated != nullptr &&
         workspace.down != nullptr;
}

bool cache_matches(const KvCache& cache, TinyModelConfig config,
                   std::size_t max_sequence_length) {
  return cache.keys != nullptr && cache.values != nullptr && cache.batch == 1 &&
         cache.heads == config.heads && cache.head_dim == config.head_dim &&
         cache.max_sequence == max_sequence_length &&
         cache.current_length <= max_sequence_length;
}

bool incremental_workspace_structure_matches(
    const TinyModelIncrementalWorkspace* workspace) {
  if (workspace == nullptr || !valid_tiny_model_config(workspace->model_config) ||
      !configs_match(workspace->model_config, workspace->allocation_model_config) ||
      workspace->max_sequence_length == 0 ||
      workspace->max_sequence_length != workspace->allocation_max_sequence_length ||
      workspace->copied_token_id == nullptr || workspace->activation_a == nullptr ||
      workspace->activation_b == nullptr || workspace->layer_workspaces == nullptr ||
      workspace->layer_caches == nullptr)
    return false;
  const IncrementalDecoderBlockConfig expected = incremental_layer_config(
      workspace->model_config, workspace->max_sequence_length);
  for (std::size_t layer = 0; layer < workspace->model_config.layers; ++layer)
    if (!incremental_block_workspace_matches(workspace->layer_workspaces[layer],
                                             expected) ||
        !cache_matches(workspace->layer_caches[layer], workspace->model_config,
                       workspace->max_sequence_length))
      return false;
  return true;
}

bool incremental_decode_preflight(const TinyModelIncrementalWorkspace* workspace,
                                  TinyModelWeights weights,
                                  std::size_t* current_length) {
  if (!incremental_workspace_structure_matches(workspace) ||
      !valid_weights(weights, workspace->model_config))
    return false;
  const std::size_t length = workspace->layer_caches[0].current_length;
  if (length >= workspace->max_sequence_length)
    return false;
  for (std::size_t layer = 1; layer < workspace->model_config.layers; ++layer)
    if (workspace->layer_caches[layer].current_length != length ||
        workspace->layer_caches[layer].current_length >=
            workspace->max_sequence_length)
      return false;
  *current_length = length;
  return true;
}

bool valid_token_id(int token_id, TinyModelConfig config) {
  return token_id >= 0 &&
         static_cast<std::size_t>(token_id) < config.vocabulary_size;
}

bool valid_prompt_token_ids(const int* token_ids, std::size_t count,
                            TinyModelConfig config) {
  if (token_ids == nullptr || count == 0)
    return false;
  for (std::size_t position = 0; position < count; ++position)
    if (!valid_token_id(token_ids[position], config))
      return false;
  return true;
}

bool generation_preflight(const int* host_prompt_token_ids,
                          std::size_t prompt_length,
                          TinyModelWeights weights,
                          const TinyModelIncrementalWorkspace* workspace,
                          std::size_t max_new_tokens,
                          const int* host_generated_token_ids,
                          const float* device_logits) {
  if (!incremental_workspace_structure_matches(workspace) ||
      !valid_weights(weights, workspace->model_config) ||
      !valid_prompt_token_ids(host_prompt_token_ids, prompt_length,
                              workspace->model_config))
    return false;
  if (max_new_tokens == 0)
    return true;
  if (host_generated_token_ids == nullptr || device_logits == nullptr)
    return false;

  // The prompt is always decoded. Each generated token except the final one
  // is decoded to obtain logits for its successor.
  const std::size_t generated_decodes = max_new_tokens - 1;
  if (prompt_length > std::numeric_limits<std::size_t>::max() -
                          generated_decodes)
    return false;
  return prompt_length + generated_decodes <= workspace->max_sequence_length;
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

cudaError_t tiny_model_incremental_workspace_create(
    TinyModelIncrementalWorkspace* workspace, TinyModelConfig model_config,
    std::size_t max_sequence_length) {
  if (workspace == nullptr || !incremental_workspace_is_empty(workspace) ||
      !valid_tiny_model_config(model_config) || max_sequence_length == 0)
    return cudaErrorInvalidValue;

  workspace->layer_workspaces =
      new (std::nothrow) IncrementalDecoderBlockWorkspace[model_config.layers];
  workspace->layer_caches = new (std::nothrow) KvCache[model_config.layers];
  if (workspace->layer_workspaces == nullptr || workspace->layer_caches == nullptr) {
    delete[] workspace->layer_workspaces;
    delete[] workspace->layer_caches;
    clear_incremental(workspace);
    return cudaErrorMemoryAllocation;
  }
  workspace->allocation_model_config = model_config;
  workspace->allocation_max_sequence_length = max_sequence_length;

  const std::size_t token_bytes = sizeof(int);
  const std::size_t activation_bytes = model_config.hidden * sizeof(float);
  cudaError_t error = cudaMalloc(&workspace->copied_token_id, token_bytes);
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->activation_a, activation_bytes);
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->activation_b, activation_bytes);
  const IncrementalDecoderBlockConfig block_config = incremental_layer_config(
      model_config, max_sequence_length);
  for (std::size_t layer = 0; error == cudaSuccess && layer < model_config.layers;
       ++layer) {
    error = kv_cache_create(&workspace->layer_caches[layer], 1, model_config.heads,
                            max_sequence_length, model_config.head_dim);
    if (error == cudaSuccess)
      error = incremental_decoder_block_workspace_create(
          &workspace->layer_workspaces[layer], block_config);
  }
  if (error != cudaSuccess) {
    tiny_model_incremental_workspace_destroy(workspace);
  } else {
    workspace->model_config = model_config;
    workspace->max_sequence_length = max_sequence_length;
  }
  return error;
}

cudaError_t tiny_model_incremental_workspace_destroy(
    TinyModelIncrementalWorkspace* workspace) {
  if (workspace == nullptr)
    return cudaErrorInvalidValue;

  cudaError_t error = cudaSuccess;
  const std::size_t layers = workspace->allocation_model_config.layers;
  if (workspace->layer_workspaces != nullptr) {
    for (std::size_t layer = 0; layer < layers; ++layer)
      error = first_error(error, incremental_decoder_block_workspace_destroy(
                                     &workspace->layer_workspaces[layer]));
    delete[] workspace->layer_workspaces;
  }
  if (workspace->layer_caches != nullptr) {
    for (std::size_t layer = 0; layer < layers; ++layer)
      error = first_error(error, kv_cache_destroy(&workspace->layer_caches[layer]));
    delete[] workspace->layer_caches;
  }
  if (workspace->copied_token_id != nullptr)
    error = first_error(error, cudaFree(workspace->copied_token_id));
  if (workspace->activation_a != nullptr)
    error = first_error(error, cudaFree(workspace->activation_a));
  if (workspace->activation_b != nullptr)
    error = first_error(error, cudaFree(workspace->activation_b));
  clear_incremental(workspace);
  return error;
}

cudaError_t tiny_model_incremental_workspace_reset(
    TinyModelIncrementalWorkspace* workspace) {
  if (!incremental_workspace_structure_matches(workspace))
    return cudaErrorInvalidValue;
  for (std::size_t layer = 0; layer < workspace->model_config.layers; ++layer)
    kv_cache_reset(&workspace->layer_caches[layer]);
  return cudaSuccess;
}

cublasStatus_t tiny_model_incremental_decode_cuda(
    cublasHandle_t handle, const int* device_token_id,
    TinyModelWeights device_weights, TinyModelIncrementalWorkspace* workspace,
    float* device_logits, cudaStream_t stream) {
  std::size_t current_length = 0;
  if (handle == nullptr || device_token_id == nullptr || device_logits == nullptr ||
      !incremental_decode_preflight(workspace, device_weights, &current_length))
    return CUBLAS_STATUS_INVALID_VALUE;
  (void)current_length;

  const TinyModelConfig config = workspace->model_config;
  cudaError_t cuda_error = embedding_lookup_cuda(
      device_token_id, device_weights.token_embeddings, workspace->activation_a, 1,
      config.vocabulary_size, config.hidden, stream);
  if (cuda_error != cudaSuccess)
    return CUBLAS_STATUS_EXECUTION_FAILED;

  const float* current = workspace->activation_a;
  float* next = workspace->activation_b;
  for (std::size_t layer = 0; layer < config.layers; ++layer) {
    const IncrementalDecoderBlockWeights layer_weights{
        device_weights.layers[layer].decoder.attention,
        device_weights.layers[layer].decoder.mlp};
    const cublasStatus_t status = incremental_decoder_block_cuda(
        handle, current, layer_weights,
        &workspace->layer_caches[layer], &workspace->layer_workspaces[layer], next,
        stream);
    if (status != CUBLAS_STATUS_SUCCESS)
      return status;
    current = next;
    next = next == workspace->activation_a ? workspace->activation_b
                                            : workspace->activation_a;
  }
  cuda_error = rmsnorm_cuda(current, device_weights.final_norm_weight, next, 1,
                             config.hidden, config.rmsnorm_epsilon, stream);
  if (cuda_error != cudaSuccess)
    return CUBLAS_STATUS_EXECUTION_FAILED;
  return linear_cublas_row_major(handle, next, device_weights.lm_head_weight,
                                 device_logits, 1, config.hidden,
                                 config.vocabulary_size, stream);
}

cublasStatus_t tiny_model_incremental_decode_host_token_cuda(
    cublasHandle_t handle, int host_token_id, TinyModelWeights device_weights,
    TinyModelIncrementalWorkspace* workspace, float* device_logits,
    cudaStream_t stream) {
  std::size_t current_length = 0;
  if (handle == nullptr || device_logits == nullptr ||
      !incremental_decode_preflight(workspace, device_weights, &current_length) ||
      !valid_token_id(host_token_id, workspace->model_config))
    return CUBLAS_STATUS_INVALID_VALUE;
  const cudaError_t error = cudaMemcpyAsync(
      workspace->copied_token_id, &host_token_id, sizeof(int),
      cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess)
    return CUBLAS_STATUS_EXECUTION_FAILED;
  return tiny_model_incremental_decode_cuda(handle, workspace->copied_token_id,
                                            device_weights, workspace,
                                            device_logits, stream);
}

cublasStatus_t tiny_model_incremental_prefill_cuda(
    cublasHandle_t handle, const int* device_token_ids,
    std::size_t prefix_length, TinyModelWeights device_weights,
    TinyModelIncrementalWorkspace* workspace, float* device_logits,
    cudaStream_t stream) {
  std::size_t current_length = 0;
  if (handle == nullptr || device_logits == nullptr ||
      !incremental_decode_preflight(workspace, device_weights, &current_length) ||
      (prefix_length != 0 && device_token_ids == nullptr) ||
      prefix_length > workspace->max_sequence_length - current_length)
    return CUBLAS_STATUS_INVALID_VALUE;
  for (std::size_t position = 0; position < prefix_length; ++position) {
    const cublasStatus_t status = tiny_model_incremental_decode_cuda(
        handle, device_token_ids + position, device_weights, workspace,
        device_logits, stream);
    if (status != CUBLAS_STATUS_SUCCESS)
      return status;
  }
  return CUBLAS_STATUS_SUCCESS;
}

bool tiny_model_greedy_argmax_host(const float* host_logits,
                                   std::size_t vocabulary_size,
                                   int* selected_token_id) {
  if (host_logits == nullptr || selected_token_id == nullptr ||
      vocabulary_size == 0 ||
      vocabulary_size >
          static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      !std::isfinite(host_logits[0]))
    return false;

  std::size_t best = 0;
  for (std::size_t token = 1; token < vocabulary_size; ++token) {
    if (!std::isfinite(host_logits[token]))
      return false;
    // Strictly greater preserves the lower ID when values are tied exactly.
    if (host_logits[token] > host_logits[best])
      best = token;
  }
  *selected_token_id = static_cast<int>(best);
  return true;
}

cublasStatus_t tiny_model_generate_greedy_cuda(
    cublasHandle_t handle, const int* host_prompt_token_ids,
    std::size_t prompt_length, TinyModelWeights device_weights,
    TinyModelIncrementalWorkspace* workspace, std::size_t max_new_tokens,
    int* host_generated_token_ids, float* device_logits, cudaStream_t stream) {
  if (handle == nullptr ||
      !generation_preflight(host_prompt_token_ids, prompt_length,
                            device_weights, workspace, max_new_tokens,
                            host_generated_token_ids, device_logits))
    return CUBLAS_STATUS_INVALID_VALUE;

  // A zero-token request is intentionally a no-op: it does not discard a
  // caller's existing cache history or require output storage.
  if (max_new_tokens == 0)
    return CUBLAS_STATUS_SUCCESS;

  cudaError_t cuda_error = tiny_model_incremental_workspace_reset(workspace);
  if (cuda_error != cudaSuccess)
    return CUBLAS_STATUS_EXECUTION_FAILED;

  std::vector<float> host_logits(workspace->model_config.vocabulary_size);
  for (std::size_t position = 0; position < prompt_length; ++position) {
    const cublasStatus_t status = tiny_model_incremental_decode_host_token_cuda(
        handle, host_prompt_token_ids[position], device_weights, workspace,
        device_logits, stream);
    if (status != CUBLAS_STATUS_SUCCESS)
      return status;
  }

  cuda_error = cudaMemcpyAsync(host_logits.data(), device_logits,
                               host_logits.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream);
  if (cuda_error != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess)
    return CUBLAS_STATUS_EXECUTION_FAILED;

  for (std::size_t generated = 0; generated < max_new_tokens; ++generated) {
    int next_token = -1;
    if (!tiny_model_greedy_argmax_host(host_logits.data(), host_logits.size(),
                                       &next_token))
      return CUBLAS_STATUS_EXECUTION_FAILED;
    host_generated_token_ids[generated] = next_token;

    // The final selected token has no successor to score, so deliberately do
    // not decode it or consume an unnecessary cache position.
    if (generated + 1 == max_new_tokens)
      break;
    const cublasStatus_t status = tiny_model_incremental_decode_host_token_cuda(
        handle, next_token, device_weights, workspace, device_logits, stream);
    if (status != CUBLAS_STATUS_SUCCESS)
      return status;
    cuda_error = cudaMemcpyAsync(host_logits.data(), device_logits,
                                 host_logits.size() * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream);
    if (cuda_error != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess)
      return CUBLAS_STATUS_EXECUTION_FAILED;
  }
  return CUBLAS_STATUS_SUCCESS;
}

}  // namespace cuda_transformer
