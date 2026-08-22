#include "cuda_transformer/attention.h"
#include "cuda_transformer/transformer_primitives.h"
#include <cmath>
namespace cuda_transformer {
namespace {
constexpr int T = 256;
__host__ __device__ std::size_t xidx(AttentionShape s, std::size_t b,
                                     std::size_t h, std::size_t q,
                                     std::size_t d) {
  return (((b * s.heads + h) * s.sequence + q) * s.head_dim + d);
}
__global__ void rope(const float *x, float *y, AttentionShape s) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  auto pairs = s.batch * s.heads * s.sequence * (s.head_dim / 2);
  if (i >= pairs)
    return;
  auto p = i % (s.head_dim / 2), z = i / (s.head_dim / 2), q = z % s.sequence,
       z2 = z / s.sequence, h = z2 % s.heads, b = z2 / s.heads;
  float theta = powf(10000.f, -float(2 * p) / s.head_dim) * q, c = cosf(theta),
        sn = sinf(theta);
  auto o = xidx(s, b, h, q, 2 * p);
  y[o] = x[o] * c - x[o + 1] * sn;
  y[o + 1] = x[o] * sn + x[o + 1] * c;
}
__global__ void scores(const float *q, const float *k, float *l,
                       AttentionShape s) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  auto total = s.batch * s.heads * s.sequence * s.sequence;
  if (i >= total)
    return;
  auto key = i % s.sequence, z = i / s.sequence, query = z % s.sequence,
       z2 = z / s.sequence, h = z2 % s.heads, b = z2 / s.heads;
  if (key > query) {
    l[i] = -1.0e20F;
    return;
  }
  float sum = 0;
  for (size_t d = 0; d < s.head_dim; d++)
    sum += q[xidx(s, b, h, query, d)] * k[xidx(s, b, h, key, d)];
  l[i] = sum / sqrtf(float(s.head_dim));
}
__global__ void values(const float *p, const float *v, float *out,
                       AttentionShape s) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  auto total = s.batch * s.heads * s.sequence * s.head_dim;
  if (i >= total)
    return;
  auto d = i % s.head_dim, z = i / s.head_dim, q = z % s.sequence,
       z2 = z / s.sequence, h = z2 % s.heads, b = z2 / s.heads;
  float sum = 0;
  auto base = ((b * s.heads + h) * s.sequence + q) * s.sequence;
  for (size_t k = 0; k < s.sequence; k++)
    sum += p[base + k] * v[xidx(s, b, h, k, d)];
  out[i] = sum;
}
} // namespace
bool valid_attention_shape(AttentionShape s) {
  return s.batch && s.heads && s.sequence && s.head_dim && !(s.head_dim % 2);
}
void attention_cpu(const float *q, const float *k, const float *v, float *qr,
                   float *logits, float *out, AttentionShape s) {
  auto n = s.batch * s.heads * s.sequence * s.head_dim;
  for (size_t i = 0; i < n; i += 2) {
    size_t d = i % s.head_dim, p = d / 2, pos = (i / s.head_dim) % s.sequence;
    float th = std::pow(10000.f, -float(2 * p) / s.head_dim) * pos,
          c = std::cos(th), sn = std::sin(th);
    qr[i] = q[i] * c - q[i + 1] * sn;
    qr[i + 1] = q[i] * sn + q[i + 1] * c;
  }
  float *kr = new float[n];
  for (size_t i = 0; i < n; i += 2) {
    size_t d = i % s.head_dim, p = d / 2, pos = (i / s.head_dim) % s.sequence;
    float th = std::pow(10000.f, -float(2 * p) / s.head_dim) * pos,
          c = std::cos(th), sn = std::sin(th);
    kr[i] = k[i] * c - k[i + 1] * sn;
    kr[i + 1] = k[i] * sn + k[i + 1] * c;
  }
  for (size_t b = 0; b < s.batch; b++)
    for (size_t h = 0; h < s.heads; h++)
      for (size_t qpos = 0; qpos < s.sequence; qpos++) {
        float m = -INFINITY;
        for (size_t kp = 0; kp < s.sequence; kp++) {
          float z = -1.0e20F;
          if (kp <= qpos) {
            z = 0;
            for (size_t d = 0; d < s.head_dim; d++)
              z += qr[xidx(s, b, h, qpos, d)] * kr[xidx(s, b, h, kp, d)];
            z /= std::sqrt(float(s.head_dim));
          }
          logits[((b * s.heads + h) * s.sequence + qpos) * s.sequence + kp] = z;
          m = fmaxf(m, z);
        }
        float sum = 0;
        for (size_t kp = 0; kp < s.sequence; kp++) {
          float &e =
              logits[((b * s.heads + h) * s.sequence + qpos) * s.sequence + kp];
          e = std::exp(e - m);
          sum += e;
        }
        for (size_t kp = 0; kp < s.sequence; kp++)
          logits[((b * s.heads + h) * s.sequence + qpos) * s.sequence + kp] /=
              sum;
        for (size_t d = 0; d < s.head_dim; d++) {
          float z = 0;
          for (size_t kp = 0; kp < s.sequence; kp++)
            z += logits[((b * s.heads + h) * s.sequence + qpos) * s.sequence +
                        kp] *
                 v[xidx(s, b, h, kp, d)];
          out[xidx(s, b, h, qpos, d)] = z;
        }
      }
  delete[] kr;
}
cudaError_t rope_cuda(const float *x, float *y, AttentionShape s,
                      cudaStream_t st) {
  if (!valid_attention_shape(s))
    return cudaErrorInvalidValue;
  auto n = s.batch * s.heads * s.sequence * s.head_dim / 2;
  rope<<<(n + T - 1) / T, T, 0, st>>>(x, y, s);
  return cudaGetLastError();
}
cudaError_t attention_scores_cuda(const float *q, const float *k, float *l,
                                  AttentionShape s, cudaStream_t st) {
  if (!valid_attention_shape(s))
    return cudaErrorInvalidValue;
  auto n = s.batch * s.heads * s.sequence * s.sequence;
  scores<<<(n + T - 1) / T, T, 0, st>>>(q, k, l, s);
  return cudaGetLastError();
}
cudaError_t attention_values_cuda(const float *p, const float *v, float *out,
                                  AttentionShape s, cudaStream_t st) {
  if (!valid_attention_shape(s))
    return cudaErrorInvalidValue;
  auto n = s.batch * s.heads * s.sequence * s.head_dim;
  values<<<(n + T - 1) / T, T, 0, st>>>(p, v, out, s);
  return cudaGetLastError();
}
} // namespace cuda_transformer
