#include "cuda_transformer/cublas_check.h"
#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/gemm.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
struct Shape {
  std::size_t rows, in_dim, out_dim;
};
bool close(float a, float b) {
  return std::fabs(a - b) <= 1.0e-4F + 2.0e-6F * fmaxf(1.0F, std::fabs(b));
}
bool run(cublasHandle_t handle, Shape s) {
  std::vector<float> x(s.rows * s.in_dim), w(s.in_dim * s.out_dim),
      ref(s.rows * s.out_dim), got(ref.size());
  for (size_t i = 0; i < x.size(); ++i)
    x[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.0625F;
  for (size_t i = 0; i < w.size(); ++i)
    w[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.0625F;
  cuda_transformer::gemm_cpu(x.data(), w.data(), ref.data(), s.rows, s.in_dim,
                             s.out_dim);
  float *dx = nullptr, *dw = nullptr, *dy = nullptr;
  auto clean = [&](bool ok) {
    for (auto p : {dx, dw, dy})
      if (p && !CTR_CUDA_CHECK(cudaFree(p)))
        ok = false;
    return ok;
  };
  if (!CTR_CUDA_CHECK(cudaMalloc(&dx, x.size() * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&dw, w.size() * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&dy, got.size() * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMemcpy(dx, x.data(), x.size() * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(dw, w.data(), w.size() * sizeof(float),
                                 cudaMemcpyHostToDevice)))
    return clean(false);
  bool ok =
      CTR_CUBLAS_CHECK(cuda_transformer::linear_cublas_row_major(
          handle, dx, dw, dy, s.rows, s.in_dim, s.out_dim)) &&
      CTR_CUDA_CHECK(cudaDeviceSynchronize()) &&
      CTR_CUDA_CHECK(cudaMemcpy(got.data(), dy, got.size() * sizeof(float),
                                cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < got.size() && ok; ++i)
    if (!close(got[i], ref[i])) {
      std::fprintf(stderr, "linear mismatch %zux%zux%zu at %zu: %.8g vs %.8g\n",
                   s.rows, s.in_dim, s.out_dim, i, got[i], ref[i]);
      ok = false;
    }
  return clean(ok);
}
} // namespace
int main() {
  cublasHandle_t h = nullptr;
  if (!CTR_CUBLAS_CHECK(cublasCreate(&h)))
    return EXIT_FAILURE;
  bool ok = true;
  for (auto s : std::array<Shape, 8>{{{1, 1, 1},
                                      {1, 8, 16},
                                      {4, 16, 16},
                                      {7, 32, 24},
                                      {13, 17, 29},
                                      {32, 256, 256},
                                      {32, 256, 512},
                                      {32, 512, 256}}})
    if (!run(h, s)) {
      ok = false;
      break;
    }
  if (!CTR_CUBLAS_CHECK(cublasDestroy(h)))
    ok = false;
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
