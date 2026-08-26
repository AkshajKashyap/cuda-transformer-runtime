#include "cuda_transformer/qkv_layout.h"

namespace cuda_transformer {
namespace {
constexpr int kThreads = 256;
__global__ void forward(const float *in, float *out, size_t sequence,
                        size_t heads, size_t dim) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x,
         total = sequence * heads * dim;
  if (i >= total)
    return;
  size_t d = i % dim, t = i / dim, head = t % heads, token = t / heads;
  out[(head * sequence + token) * dim + d] = in[i];
}
__global__ void reverse(const float *in, float *out, size_t sequence,
                        size_t heads, size_t dim) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x,
         total = sequence * heads * dim;
  if (i >= total)
    return;
  size_t d = i % dim, t = i / dim, head = t % heads, token = t / heads;
  out[i] = in[(head * sequence + token) * dim + d];
}
bool valid(size_t s, size_t h, size_t d) { return s && h && d; }
} // namespace
void token_major_to_head_major_cpu(const float *in, float *out, size_t s,
                                   size_t h, size_t d) {
  for (size_t t = 0; t < s; t++)
    for (size_t head = 0; head < h; head++)
      for (size_t x = 0; x < d; x++)
        out[(head * s + t) * d + x] = in[(t * h + head) * d + x];
}
void head_major_to_token_major_cpu(const float *in, float *out, size_t s,
                                   size_t h, size_t d) {
  for (size_t t = 0; t < s; t++)
    for (size_t head = 0; head < h; head++)
      for (size_t x = 0; x < d; x++)
        out[(t * h + head) * d + x] = in[(head * s + t) * d + x];
}
cudaError_t token_major_to_head_major_cuda(const float *in, float *out,
                                           size_t s, size_t h, size_t d,
                                           cudaStream_t st) {
  if (!in || !out || !valid(s, h, d))
    return cudaErrorInvalidValue;
  size_t n = s * h * d;
  forward<<<(n + kThreads - 1) / kThreads, kThreads, 0, st>>>(in, out, s, h, d);
  return cudaGetLastError();
}
cudaError_t head_major_to_token_major_cuda(const float *in, float *out,
                                           size_t s, size_t h, size_t d,
                                           cudaStream_t st) {
  if (!in || !out || !valid(s, h, d))
    return cudaErrorInvalidValue;
  size_t n = s * h * d;
  reverse<<<(n + kThreads - 1) / kThreads, kThreads, 0, st>>>(in, out, s, h, d);
  return cudaGetLastError();
}
} // namespace cuda_transformer
