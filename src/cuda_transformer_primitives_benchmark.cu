#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/transformer_primitives.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
template <class F> bool timeit(const char *name, F f) {
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
  std::printf("%-12s median kernel-only latency: %.5f ms (%d batches x %d)\n",
              name, v[v.size() / 2], batches, iters);
  return CTR_CUDA_CHECK(cudaEventDestroy(a)) &&
         CTR_CUDA_CHECK(cudaEventDestroy(b));
}
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
}
