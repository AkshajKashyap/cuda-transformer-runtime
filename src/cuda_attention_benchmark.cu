#include "cuda_transformer/attention.h"
#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/transformer_primitives.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
struct Shape {
  std::size_t heads, sequence, head_dim;
};
int launches(const Shape &s) {
  return s.sequence <= 64 ? 200 : (s.sequence <= 128 ? 50 : 10);
}
template <class Launch>
bool time_stage(const char *name, Launch launch, int count) {
  constexpr int warmups = 10, batches = 9;
  for (int i = 0; i < warmups; i++)
    if (!launch())
      return false;
  if (!CTR_CUDA_CHECK(cudaDeviceSynchronize()))
    return false;
  cudaEvent_t start = nullptr, stop = nullptr;
  if (!CTR_CUDA_CHECK(cudaEventCreate(&start)) ||
      !CTR_CUDA_CHECK(cudaEventCreate(&stop)))
    return false;
  std::vector<float> samples;
  samples.reserve(batches);
  for (int b = 0; b < batches; b++) {
    if (!CTR_CUDA_CHECK(cudaEventRecord(start)))
      return false;
    for (int i = 0; i < count; i++)
      if (!launch())
        return false;
    if (!CTR_CUDA_CHECK(cudaEventRecord(stop)) ||
        !CTR_CUDA_CHECK(cudaEventSynchronize(stop)))
      return false;
    float ms = 0;
    if (!CTR_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)))
      return false;
    samples.push_back(ms / count);
  }
  std::sort(samples.begin(), samples.end());
  std::printf("  %-10s %.5f ms median (%d batches x %d launches)\n", name,
              samples[batches / 2], batches, count);
  return CTR_CUDA_CHECK(cudaEventDestroy(start)) &&
         CTR_CUDA_CHECK(cudaEventDestroy(stop));
}
bool run(Shape x) {
  cuda_transformer::AttentionShape s{1, x.heads, x.sequence, x.head_dim};
  size_t e = s.heads * s.sequence * s.head_dim,
         p = s.heads * s.sequence * s.sequence;
  std::vector<float> host(e, 0.125f);
  float *q = nullptr, *k = nullptr, *v = nullptr, *qr = nullptr, *kr = nullptr,
        *scores = nullptr, *out = nullptr;
  auto clean = [&](bool ok) {
    for (auto z : {q, k, v, qr, kr, scores, out})
      if (z && !CTR_CUDA_CHECK(cudaFree(z)))
        ok = false;
    return ok;
  };
  if (!CTR_CUDA_CHECK(cudaMalloc(&q, e * 4)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&k, e * 4)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&v, e * 4)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&qr, e * 4)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&kr, e * 4)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&scores, p * 4)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&out, e * 4)) ||
      !CTR_CUDA_CHECK(
          cudaMemcpy(q, host.data(), e * 4, cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(
          cudaMemcpy(k, host.data(), e * 4, cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(
          cudaMemcpy(v, host.data(), e * 4, cudaMemcpyHostToDevice)))
    return clean(false);
  int n = launches(x);
  std::printf("heads=%zu seq=%zu dim=%zu; scores/probabilities each %.2f MiB\n",
              x.heads, x.sequence, x.head_dim, p * 4.0 / (1024 * 1024));
  bool ok = time_stage(
      "RoPE",
      [&] {
        return CTR_CUDA_CHECK(cuda_transformer::rope_cuda(q, qr, s)) &&
               CTR_CUDA_CHECK(cuda_transformer::rope_cuda(k, kr, s));
      },
      n);
  if (ok)
    ok = time_stage(
        "QK^T",
        [&] {
          return CTR_CUDA_CHECK(
              cuda_transformer::attention_scores_cuda(qr, kr, scores, s));
        },
        n);
  if (ok)
    ok = time_stage(
        "softmax",
        [&] {
          return CTR_CUDA_CHECK(cuda_transformer::softmax_cuda(
              scores, scores, s.heads * s.sequence, s.sequence));
        },
        n);
  if (ok)
    ok = time_stage(
        "P x V",
        [&] {
          return CTR_CUDA_CHECK(
              cuda_transformer::attention_values_cuda(scores, v, out, s));
        },
        n);
  if (ok)
    ok = time_stage(
        "full staged",
        [&] {
          return CTR_CUDA_CHECK(cuda_transformer::rope_cuda(q, qr, s)) &&
                 CTR_CUDA_CHECK(cuda_transformer::rope_cuda(k, kr, s)) &&
                 CTR_CUDA_CHECK(cuda_transformer::attention_scores_cuda(
                     qr, kr, scores, s)) &&
                 CTR_CUDA_CHECK(cuda_transformer::softmax_cuda(
                     scores, scores, s.heads * s.sequence, s.sequence)) &&
                 CTR_CUDA_CHECK(cuda_transformer::attention_values_cuda(
                     scores, v, out, s));
        },
        n);
  return clean(ok);
}
} // namespace
int main() {
  for (auto s : std::array<Shape, 4>{
           {{4, 32, 64}, {4, 64, 64}, {8, 128, 64}, {8, 256, 64}}})
    if (!run(s))
      return EXIT_FAILURE;
}
