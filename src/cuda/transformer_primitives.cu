#include "cuda_transformer/transformer_primitives.h"
#include <cmath>
namespace cuda_transformer {
namespace {
int blocks(std::size_t n) {
  return static_cast<int>((n + kPrimitiveThreads - 1) / kPrimitiveThreads);
}
__device__ float reduce_sum(float value, float *shared) {
  const int t = threadIdx.x;
  shared[t] = value;
  __syncthreads();
  for (int s = blockDim.x / 2; s; s /= 2) {
    if (t < s)
      shared[t] += shared[t + s];
    __syncthreads();
  }
  return shared[0];
}
__device__ float reduce_max(float value, float *shared) {
  const int t = threadIdx.x;
  shared[t] = value;
  __syncthreads();
  for (int s = blockDim.x / 2; s; s /= 2) {
    if (t < s)
      shared[t] = fmaxf(shared[t], shared[t + s]);
    __syncthreads();
  }
  return shared[0];
}
__global__ void rms_k(const float *x, const float *w, float *y,
                      std::size_t hidden, float eps) {
  extern __shared__ float sh[];
  const auto r = blockIdx.x, t = threadIdx.x;
  float s = 0;
  for (std::size_t i = t; i < hidden; i += blockDim.x) {
    float v = x[r * hidden + i];
    s += v * v;
  }
  float rms = rsqrtf(reduce_sum(s, sh) / hidden + eps);
  for (std::size_t i = t; i < hidden; i += blockDim.x)
    y[r * hidden + i] = x[r * hidden + i] * rms * w[i];
}
__global__ void soft_k(const float *x, float *y, std::size_t width) {
  extern __shared__ float sh[];
  const auto r = blockIdx.x, t = threadIdx.x;
  float m = -INFINITY;
  for (std::size_t i = t; i < width; i += blockDim.x)
    m = fmaxf(m, x[r * width + i]);
  m = reduce_max(m, sh);
  float s = 0;
  for (std::size_t i = t; i < width; i += blockDim.x) {
    float e = expf(x[r * width + i] - m);
    y[r * width + i] = e;
    s += e;
  }
  float total = reduce_sum(s, sh);
  for (std::size_t i = t; i < width; i += blockDim.x)
    y[r * width + i] /= total;
}
__global__ void silu_k(const float *x, float *y, std::size_t n) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = x[i] / (1.0f + expf(-x[i]));
}
__global__ void mul_k(const float *a, const float *b, float *y, std::size_t n) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a[i] * b[i];
}
__global__ void add_k(const float *a, const float *b, float *y, std::size_t n) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a[i] + b[i];
}
} // namespace
void rmsnorm_cpu(const float *x, const float *w, float *y, std::size_t rows,
                 std::size_t h, float eps) {
  for (std::size_t r = 0; r < rows; r++) {
    float s = 0;
    for (std::size_t i = 0; i < h; i++)
      s += x[r * h + i] * x[r * h + i];
    float q = 1 / std::sqrt(s / h + eps);
    for (std::size_t i = 0; i < h; i++)
      y[r * h + i] = x[r * h + i] * q * w[i];
  }
}
void softmax_cpu(const float *x, float *y, std::size_t rows, std::size_t n) {
  for (std::size_t r = 0; r < rows; r++) {
    float m = -INFINITY;
    for (std::size_t i = 0; i < n; i++)
      m = fmaxf(m, x[r * n + i]);
    float s = 0;
    for (std::size_t i = 0; i < n; i++) {
      y[r * n + i] = std::exp(x[r * n + i] - m);
      s += y[r * n + i];
    }
    for (std::size_t i = 0; i < n; i++)
      y[r * n + i] /= s;
  }
}
void silu_cpu(const float *x, float *y, std::size_t n) {
  for (std::size_t i = 0; i < n; i++)
    y[i] = x[i] / (1 + std::exp(-x[i]));
}
void multiply_cpu(const float *a, const float *b, float *y, std::size_t n) {
  for (std::size_t i = 0; i < n; i++)
    y[i] = a[i] * b[i];
}
void residual_add_cpu(const float *a, const float *b, float *y, std::size_t n) {
  for (std::size_t i = 0; i < n; i++)
    y[i] = a[i] + b[i];
}
cudaError_t rmsnorm_cuda(const float *x, const float *w, float *y,
                         std::size_t rows, std::size_t h, float eps,
                         cudaStream_t st) {
  if (!rows || !h || eps <= 0)
    return cudaErrorInvalidValue;
  rms_k<<<rows, kPrimitiveThreads, kPrimitiveThreads * sizeof(float), st>>>(
      x, w, y, h, eps);
  return cudaGetLastError();
}
cudaError_t softmax_cuda(const float *x, float *y, std::size_t rows,
                         std::size_t n, cudaStream_t st) {
  if (!rows || !n)
    return cudaErrorInvalidValue;
  soft_k<<<rows, kPrimitiveThreads, kPrimitiveThreads * sizeof(float), st>>>(
      x, y, n);
  return cudaGetLastError();
}
cudaError_t silu_cuda(const float *x, float *y, std::size_t n,
                      cudaStream_t st) {
  if (!n)
    return cudaSuccess;
  silu_k<<<blocks(n), kPrimitiveThreads, 0, st>>>(x, y, n);
  return cudaGetLastError();
}
cudaError_t multiply_cuda(const float *a, const float *b, float *y,
                          std::size_t n, cudaStream_t st) {
  if (!n)
    return cudaSuccess;
  mul_k<<<blocks(n), kPrimitiveThreads, 0, st>>>(a, b, y, n);
  return cudaGetLastError();
}
cudaError_t residual_add_cuda(const float *a, const float *b, float *y,
                              std::size_t n, cudaStream_t st) {
  if (!n)
    return cudaSuccess;
  add_k<<<blocks(n), kPrimitiveThreads, 0, st>>>(a, b, y, n);
  return cudaGetLastError();
}
} // namespace cuda_transformer
