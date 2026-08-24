#include "cuda_transformer/kv_cache.h"

#include <limits>

namespace cuda_transformer {
namespace {
constexpr int kThreads = 256;

bool valid(const KvCache *c) {
  return c != nullptr && c->keys != nullptr && c->values != nullptr &&
         c->batch != 0 && c->heads != 0 && c->max_sequence != 0 &&
         c->head_dim != 0;
}

void clear(KvCache *c) {
  c->keys = nullptr;
  c->values = nullptr;
  c->batch = 0;
  c->heads = 0;
  c->max_sequence = 0;
  c->head_dim = 0;
  c->current_length = 0;
}

__global__ void prefill_kernel(const float *input_keys,
                               const float *input_values, float *cache_keys,
                               float *cache_values, std::size_t batch,
                               std::size_t heads, std::size_t sequence,
                               std::size_t max_sequence, std::size_t head_dim) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = batch * heads * sequence * head_dim;
  if (index >= count)
    return;
  const std::size_t d = index % head_dim;
  const std::size_t t = index / head_dim;
  const std::size_t s = t % sequence;
  const std::size_t bh = t / sequence;
  const std::size_t destination = ((bh * max_sequence + s) * head_dim) + d;
  cache_keys[destination] = input_keys[index];
  cache_values[destination] = input_values[index];
}

__global__ void append_kernel(const float *input_key, const float *input_value,
                              float *cache_keys, float *cache_values,
                              std::size_t batch, std::size_t heads,
                              std::size_t position, std::size_t max_sequence,
                              std::size_t head_dim) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = batch * heads * head_dim;
  if (index >= count)
    return;
  const std::size_t d = index % head_dim;
  const std::size_t bh = index / head_dim;
  const std::size_t destination =
      ((bh * max_sequence + position) * head_dim) + d;
  cache_keys[destination] = input_key[index];
  cache_values[destination] = input_value[index];
}

bool allocation_size(std::size_t b, std::size_t h, std::size_t s, std::size_t d,
                     std::size_t *bytes) {
  if (b == 0 || h == 0 || s == 0 || d == 0)
    return false;
  const std::size_t limit =
      std::numeric_limits<std::size_t>::max() / sizeof(float);
  if (b > limit / h || b * h > limit / s || b * h * s > limit / d)
    return false;
  *bytes = b * h * s * d * sizeof(float);
  return true;
}
} // namespace

cudaError_t kv_cache_create(KvCache *c, std::size_t b, std::size_t h,
                            std::size_t s, std::size_t d) {
  std::size_t bytes = 0;
  if (c == nullptr || !allocation_size(b, h, s, d, &bytes))
    return cudaErrorInvalidValue;
  clear(c);
  cudaError_t error = cudaMalloc(&c->keys, bytes);
  if (error != cudaSuccess)
    return error;
  error = cudaMalloc(&c->values, bytes);
  if (error != cudaSuccess) {
    cudaFree(c->keys);
    c->keys = nullptr;
    return error;
  }
  c->batch = b;
  c->heads = h;
  c->max_sequence = s;
  c->head_dim = d;
  return cudaSuccess;
}

cudaError_t kv_cache_destroy(KvCache *c) {
  if (c == nullptr)
    return cudaErrorInvalidValue;
  cudaError_t first = c->keys ? cudaFree(c->keys) : cudaSuccess;
  cudaError_t second = c->values ? cudaFree(c->values) : cudaSuccess;
  clear(c);
  return first != cudaSuccess ? first : second;
}

cudaError_t kv_cache_prefill(KvCache *c, const float *k, const float *v,
                             std::size_t n, cudaStream_t stream) {
  if (!valid(c) || k == nullptr || v == nullptr || n == 0 ||
      n > c->max_sequence)
    return cudaErrorInvalidValue;
  const std::size_t count = c->batch * c->heads * n * c->head_dim;
  prefill_kernel<<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      k, v, c->keys, c->values, c->batch, c->heads, n, c->max_sequence,
      c->head_dim);
  const cudaError_t error = cudaGetLastError();
  if (error == cudaSuccess)
    c->current_length = n;
  return error;
}

cudaError_t kv_cache_append(KvCache *c, const float *k, const float *v,
                            cudaStream_t stream) {
  if (!valid(c) || k == nullptr || v == nullptr ||
      c->current_length >= c->max_sequence)
    return cudaErrorInvalidValue;
  const std::size_t count = c->batch * c->heads * c->head_dim;
  append_kernel<<<(count + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      k, v, c->keys, c->values, c->batch, c->heads, c->current_length,
      c->max_sequence, c->head_dim);
  const cudaError_t error = cudaGetLastError();
  if (error == cudaSuccess)
    ++c->current_length;
  return error;
}

void kv_cache_reset(KvCache *c) {
  if (c != nullptr)
    c->current_length = 0;
}
} // namespace cuda_transformer
