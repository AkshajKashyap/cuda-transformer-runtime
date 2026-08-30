#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/transformer_primitives.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <utility>
#include <vector>
namespace {
struct Timing {
  float median_ms;
  float average_ms;
};
template <class F> bool timeit(const char *name, F f, Timing *timing = nullptr) {
  constexpr int warm = 10, batches = 9, iters = 200;
  for (int i = 0; i < warm; i++)
    if (!f())
      return false;
  if (!CTR_CUDA_CHECK(cudaDeviceSynchronize()))
    return false;
  cudaEvent_t a = nullptr, b = nullptr;
  if (!CTR_CUDA_CHECK(cudaEventCreate(&a)) ||
      !CTR_CUDA_CHECK(cudaEventCreate(&b)))
    return false;
  std::vector<float> v;
  for (int q = 0; q < batches; q++) {
    if (!CTR_CUDA_CHECK(cudaEventRecord(a)))
      return false;
    for (int i = 0; i < iters; i++)
      if (!f())
        return false;
    if (!CTR_CUDA_CHECK(cudaEventRecord(b)) ||
        !CTR_CUDA_CHECK(cudaEventSynchronize(b)))
      return false;
    float ms;
    if (!CTR_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b)))
      return false;
    v.push_back(ms / iters);
  }
  std::sort(v.begin(), v.end());
  const float average =
      std::accumulate(v.begin(), v.end(), 0.0f) / static_cast<float>(batches);
  if (timing) {
    *timing = {v[v.size() / 2], average};
  } else {
    std::printf("%-12s median kernel-only latency: %.5f ms (%d batches x %d)\n",
                name, v[v.size() / 2], batches, iters);
  }
  return CTR_CUDA_CHECK(cudaEventDestroy(a)) &&
         CTR_CUDA_CHECK(cudaEventDestroy(b));
}
bool benchmark_layernorm(size_t rows, size_t hidden) {
  const size_t count = rows * hidden;
  std::vector<float> x(count, 1.0f), gamma(hidden, 1.0f), beta(hidden, 0.0f);
  float *dx = nullptr, *dgamma = nullptr, *dbeta = nullptr, *dy = nullptr;
  auto cleanup = [&] {
    bool ok = true;
    if (dx && !CTR_CUDA_CHECK(cudaFree(dx)))
      ok = false;
    if (dgamma && !CTR_CUDA_CHECK(cudaFree(dgamma)))
      ok = false;
    if (dbeta && !CTR_CUDA_CHECK(cudaFree(dbeta)))
      ok = false;
    if (dy && !CTR_CUDA_CHECK(cudaFree(dy)))
      ok = false;
    return ok;
  };
  if (!CTR_CUDA_CHECK(cudaMalloc(&dx, count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&dgamma, hidden * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&dbeta, hidden * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&dy, count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMemcpy(dx, x.data(), count * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(dgamma, gamma.data(), hidden * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(dbeta, beta.data(), hidden * sizeof(float),
                                 cudaMemcpyHostToDevice))) {
    cleanup();
    return false;
  }
  Timing timing{};
  const bool ok = timeit(
      "layernorm", [&] {
        return CTR_CUDA_CHECK(cuda_transformer::layernorm_cuda(
            dx, dgamma, dbeta, dy, rows, hidden, 1e-5f));
      },
      &timing);
  if (!ok) {
    cleanup();
    return false;
  }
  // This is logical input + gamma + beta + output traffic. Gamma and beta can
  // be cache-resident, so it is an effective throughput rather than DRAM BW.
  const float logical_gbps =
      4.0f * static_cast<float>(count) * sizeof(float) / timing.median_ms /
      1.0e6f;
  std::printf(
      "layernorm rows=%zu hidden=%zu median: %.5f ms average: %.5f ms "
      "effective logical throughput: %.2f GB/s (9 batches x 200)\n",
      rows, hidden, timing.median_ms, timing.average_ms, logical_gbps);
  return cleanup();
}
} // namespace
int main() {
  constexpr size_t rows = 32, h = 2048, n = rows * h;
  std::vector<float> x(n, 1), w(h, 1), z(n);
  float *dx = nullptr, *dw = nullptr, *dy = nullptr;
  auto fail = [&] {
    if (dx)
      CTR_CUDA_CHECK(cudaFree(dx));
    if (dw)
      CTR_CUDA_CHECK(cudaFree(dw));
    if (dy)
      CTR_CUDA_CHECK(cudaFree(dy));
    return EXIT_FAILURE;
  };
  if (!CTR_CUDA_CHECK(cudaMalloc(&dx, n * 4)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&dw, h * 4)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&dy, n * 4)) ||
      !CTR_CUDA_CHECK(
          cudaMemcpy(dx, x.data(), n * 4, cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(dw, w.data(), h * 4, cudaMemcpyHostToDevice)))
    return fail();
  bool ok =
      timeit("rmsnorm",
             [&] {
               return CTR_CUDA_CHECK(
                   cuda_transformer::rmsnorm_cuda(dx, dw, dy, rows, h, 1e-5f));
             }) &&
      timeit("softmax",
             [&] {
               return CTR_CUDA_CHECK(
                   cuda_transformer::softmax_cuda(dx, dy, rows, h));
             }) &&
      timeit("silu",
             [&] {
               return CTR_CUDA_CHECK(cuda_transformer::silu_cuda(dx, dy, n));
             }) &&
      timeit("multiply",
             [&] {
               return CTR_CUDA_CHECK(
                   cuda_transformer::multiply_cuda(dx, dy, dy, n));
             }) &&
      timeit("residual", [&] {
        return CTR_CUDA_CHECK(
            cuda_transformer::residual_add_cuda(dx, dy, dy, n));
      });
  if (!ok)
    return fail();
  if (!CTR_CUDA_CHECK(cudaFree(dx)) || !CTR_CUDA_CHECK(cudaFree(dw)) ||
      !CTR_CUDA_CHECK(cudaFree(dy)))
    return EXIT_FAILURE;
  for (const auto shape : std::vector<std::pair<size_t, size_t>>{
           {32, 128}, {32, 256}, {16, 512}, {16, 768}, {8, 1024}})
    if (!benchmark_layernorm(shape.first, shape.second))
      return EXIT_FAILURE;
}
