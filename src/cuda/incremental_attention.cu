#include "cuda_transformer/incremental_attention.h"

#include "cuda_transformer/transformer_primitives.h"

#include <cmath>
#include <limits>

namespace cuda_transformer {
namespace {

constexpr int kThreads = 256;

__global__ void rotate(const float* input, float* output, std::size_t heads,
                       std::size_t head_dim, std::size_t position) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= heads * head_dim)
    return;
  const std::size_t pair = (index % head_dim) / 2;
  const std::size_t base =
      index - (index % head_dim) + (index % head_dim & ~std::size_t(1));
  const float even = input[base];
  const float odd = input[base + 1];
  const float theta =
      powf(10000.0F, -static_cast<float>(2 * pair) / head_dim) * position;
  const float cosine = cosf(theta);
  const float sine = sinf(theta);
  output[index] = (index % head_dim & 1) ? even * sine + odd * cosine
                                          : even * cosine - odd * sine;
}

__global__ void scores(const float* query, const float* keys, float* output,
                       std::size_t heads, std::size_t length,
                       std::size_t max_sequence, std::size_t head_dim) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= heads * length)
    return;
  const std::size_t head = index / length;
  const std::size_t position = index % length;
  float sum = 0.0F;
  for (std::size_t dimension = 0; dimension < head_dim; ++dimension)
    sum += query[head * head_dim + dimension] *
           keys[(head * max_sequence + position) * head_dim + dimension];
  output[index] = sum / sqrtf(static_cast<float>(head_dim));
}

__global__ void probability_value(const float* probabilities, const float* values,
                                  float* output, std::size_t heads,
                                  std::size_t length,
                                  std::size_t max_sequence,
                                  std::size_t head_dim) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= heads * head_dim)
    return;
  const std::size_t head = index / head_dim;
  const std::size_t dimension = index % head_dim;
  float sum = 0.0F;
  for (std::size_t position = 0; position < length; ++position)
    sum += probabilities[head * length + position] *
           values[(head * max_sequence + position) * head_dim + dimension];
  output[index] = sum;
}

bool multiply_fits(std::size_t left, std::size_t right) {
  return left != 0 && right <= std::numeric_limits<std::size_t>::max() / left;
}

bool valid_workspace_shape(std::size_t heads, std::size_t head_dim,
                           std::size_t max_sequence) {
  return heads != 0 && head_dim != 0 && max_sequence != 0 &&
         head_dim % 2 == 0 && multiply_fits(heads, head_dim) &&
         multiply_fits(heads, max_sequence) &&
         multiply_fits(heads * head_dim, sizeof(float)) &&
         multiply_fits(heads * max_sequence, sizeof(float));
}

bool workspace_is_empty(const IncrementalAttentionWorkspace* workspace) {
  return workspace != nullptr && workspace->rotated_q == nullptr &&
         workspace->rotated_k == nullptr && workspace->scores == nullptr &&
         workspace->heads == 0 && workspace->head_dim == 0 &&
         workspace->max_sequence == 0;
}

bool workspace_matches(const IncrementalAttentionWorkspace* workspace,
                       const KvCache* cache) {
  return workspace != nullptr && cache != nullptr &&
         workspace->rotated_q != nullptr && workspace->rotated_k != nullptr &&
         workspace->scores != nullptr &&
         valid_workspace_shape(workspace->heads, workspace->head_dim,
                               workspace->max_sequence) &&
         workspace->heads == cache->heads &&
         workspace->head_dim == cache->head_dim &&
         workspace->max_sequence == cache->max_sequence;
}

bool valid_cache(const KvCache* cache) {
  return cache != nullptr && cache->batch == 1 && cache->keys != nullptr &&
         cache->values != nullptr && cache->heads != 0 &&
         cache->head_dim != 0 && cache->max_sequence != 0 &&
         cache->head_dim % 2 == 0 && cache->current_length < cache->max_sequence;
}

void clear(IncrementalAttentionWorkspace* workspace) {
  workspace->rotated_q = nullptr;
  workspace->rotated_k = nullptr;
  workspace->scores = nullptr;
  workspace->heads = 0;
  workspace->head_dim = 0;
  workspace->max_sequence = 0;
}

cudaError_t first_error(cudaError_t first, cudaError_t next) {
  return first == cudaSuccess ? next : first;
}

}  // namespace

cudaError_t incremental_attention_workspace_create(
    IncrementalAttentionWorkspace* workspace, std::size_t heads,
    std::size_t head_dim, std::size_t max_sequence) {
  if (!workspace || !workspace_is_empty(workspace) ||
      !valid_workspace_shape(heads, head_dim, max_sequence))
    return cudaErrorInvalidValue;

  const std::size_t token_count = heads * head_dim;
  const std::size_t score_count = heads * max_sequence;
  cudaError_t error = cudaMalloc(&workspace->rotated_q,
                                 token_count * sizeof(float));
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->rotated_k, token_count * sizeof(float));
  if (error == cudaSuccess)
    error = cudaMalloc(&workspace->scores, score_count * sizeof(float));
  if (error != cudaSuccess) {
    incremental_attention_workspace_destroy(workspace);
    return error;
  }
  workspace->heads = heads;
  workspace->head_dim = head_dim;
  workspace->max_sequence = max_sequence;
  return cudaSuccess;
}

cudaError_t incremental_attention_workspace_destroy(
    IncrementalAttentionWorkspace* workspace) {
  if (!workspace)
    return cudaErrorInvalidValue;
  cudaError_t error = cudaSuccess;
  for (float* pointer : {workspace->rotated_q, workspace->rotated_k,
                         workspace->scores}) {
    if (pointer != nullptr)
      error = first_error(error, cudaFree(pointer));
  }
  clear(workspace);
  return error;
}

cudaError_t incremental_decode_with_workspace(
    KvCache* cache, const float* q, const float* k, const float* v,
    float* output, IncrementalAttentionWorkspace* workspace,
    cudaStream_t stream) {
  if (!valid_cache(cache) || !q || !k || !v || !output ||
      !workspace_matches(workspace, cache))
    return cudaErrorInvalidValue;

  const std::size_t token_count = cache->heads * cache->head_dim;
  const std::size_t position = cache->current_length;
  rotate<<<(token_count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      q, workspace->rotated_q, cache->heads, cache->head_dim, position);
  cudaError_t error = cudaGetLastError();
  if (error == cudaSuccess) {
    rotate<<<(token_count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        k, workspace->rotated_k, cache->heads, cache->head_dim, position);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess)
    error = kv_cache_append(cache, workspace->rotated_k, v, stream);

  const std::size_t length = cache->current_length;
  if (error == cudaSuccess) {
    scores<<<(cache->heads * length + kThreads - 1) / kThreads, kThreads, 0,
             stream>>>(workspace->rotated_q, cache->keys, workspace->scores,
                       cache->heads, length, cache->max_sequence,
                       cache->head_dim);
    error = cudaGetLastError();
  }
  if (error == cudaSuccess)
    error = softmax_cuda(workspace->scores, workspace->scores, cache->heads,
                         length, stream);
  if (error == cudaSuccess) {
    probability_value<<<(token_count + kThreads - 1) / kThreads, kThreads, 0,
                        stream>>>(workspace->scores, cache->values, output,
                                  cache->heads, length, cache->max_sequence,
                                  cache->head_dim);
    error = cudaGetLastError();
  }
  return error;
}

cudaError_t incremental_decode(KvCache* cache, const float* q, const float* k,
                               const float* v, float* output,
                               cudaStream_t stream) {
  if (!valid_cache(cache) || !q || !k || !v || !output)
    return cudaErrorInvalidValue;

  IncrementalAttentionWorkspace workspace;
  cudaError_t error = incremental_attention_workspace_create(
      &workspace, cache->heads, cache->head_dim, cache->max_sequence);
  if (error == cudaSuccess)
    error = incremental_decode_with_workspace(cache, q, k, v, output,
                                              &workspace, stream);
  const cudaError_t destroy_error =
      incremental_attention_workspace_destroy(&workspace);
  return error == cudaSuccess ? destroy_error : error;
}

}  // namespace cuda_transformer
