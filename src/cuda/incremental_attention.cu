#include "cuda_transformer/incremental_attention.h"
#include "cuda_transformer/transformer_primitives.h"
#include <cmath>
namespace cuda_transformer {
namespace {
constexpr int T = 256;
__global__ void rotate(const float *x, float *y, size_t h, size_t d, size_t p) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= h * d)
    return;
  size_t pair = (i % d) / 2, base = i - (i % d) + (i % d & ~size_t(1));
  float a = x[base], b = x[base + 1],
        th = powf(10000.f, -float(2 * pair) / d) * p, c = cosf(th),
        s = sinf(th);
  y[i] = (i % d & 1) ? a * s + b * c : a * c - b * s;
}
__global__ void scores(const float *q, const float *k, float *s, size_t h,
                       size_t len, size_t max, size_t d) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= h * len)
    return;
  size_t head = i / len, p = i % len;
  float z = 0;
  for (size_t j = 0; j < d; j++)
    z += q[head * d + j] * k[(head * max + p) * d + j];
  s[i] = z / sqrtf(float(d));
}
__global__ void pv(const float *p, const float *v, float *out, size_t h,
                   size_t len, size_t max, size_t d) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= h * d)
    return;
  size_t head = i / d, j = i % d;
  float z = 0;
  for (size_t k = 0; k < len; k++)
    z += p[head * len + k] * v[(head * max + k) * d + j];
  out[i] = z;
}
} // namespace
cudaError_t incremental_decode(KvCache *c, const float *q, const float *k,
                               const float *v, float *out, cudaStream_t st) {
  if (!c || c->batch != 1 || !c->keys || !q || !k || !v || !out ||
      (c->head_dim % 2) != 0 || c->current_length >= c->max_sequence)
    return cudaErrorInvalidValue;
  size_t n = c->heads * c->head_dim;
  float *rq = nullptr, *rk = nullptr, *s = nullptr;
  cudaError_t e = cudaMalloc(&rq, n * 4);
  if (e != cudaSuccess)
    return e;
  e = cudaMalloc(&rk, n * 4);
  if (e != cudaSuccess) {
    cudaFree(rq);
    return e;
  }
  e = cudaMalloc(&s, c->heads * (c->current_length + 1) * 4);
  if (e != cudaSuccess) {
    cudaFree(rq);
    cudaFree(rk);
    return e;
  }
  size_t pos = c->current_length;
  rotate<<<(n + T - 1) / T, T, 0, st>>>(q, rq, c->heads, c->head_dim, pos);
  if ((e = cudaGetLastError()) == cudaSuccess) {
    rotate<<<(n + T - 1) / T, T, 0, st>>>(k, rk, c->heads, c->head_dim, pos);
    e = cudaGetLastError();
  }
  if (e == cudaSuccess)
    e = kv_cache_append(c, rk, v, st);
  size_t len = c->current_length;
  if (e == cudaSuccess) {
    scores<<<(c->heads * len + T - 1) / T, T, 0, st>>>(
        rq, c->keys, s, c->heads, len, c->max_sequence, c->head_dim);
    e = cudaGetLastError();
  }
  if (e == cudaSuccess)
    e = softmax_cuda(s, s, c->heads, len, st);
  if (e == cudaSuccess) {
    pv<<<(n + T - 1) / T, T, 0, st>>>(s, c->values, out, c->heads, len,
                                      c->max_sequence, c->head_dim);
    e = cudaGetLastError();
  }
  cudaFree(rq);
  cudaFree(rk);
  cudaFree(s);
  return e;
}
} // namespace cuda_transformer
