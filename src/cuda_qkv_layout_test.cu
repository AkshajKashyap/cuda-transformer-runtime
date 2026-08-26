#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/qkv_layout.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>
namespace {
struct Shape {
  size_t s, h, d;
};
bool run(Shape x) {
  size_t n = x.s * x.h * x.d;
  std::vector<float> in(n), expected(n), got(n), round(n);
  for (size_t t = 0; t < x.s; t++)
    for (size_t h = 0; h < x.h; h++)
      for (size_t d = 0; d < x.d; d++)
        in[(t * x.h + h) * x.d + d] = float(10000 * t + 100 * h + d);
  cuda_transformer::token_major_to_head_major_cpu(in.data(), expected.data(),
                                                  x.s, x.h, x.d);
  bool raw_diff = false;
  for (size_t i = 0; i < n; i++)
    raw_diff |= in[i] != expected[i];
  if (x.s > 1 && x.h > 1 && !raw_diff) {
    std::fprintf(stderr, "raw layout reinterpretation unexpectedly matched\n");
    return false;
  }
  float *di = nullptr, *dh = nullptr, *do_ = nullptr;
  auto clean = [&](bool ok) {
    for (auto p : {di, dh, do_})
      if (p && !CTR_CUDA_CHECK(cudaFree(p)))
        ok = false;
    return ok;
  };
  if (!CTR_CUDA_CHECK(cudaMalloc(&di, n * 4)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&dh, n * 4)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&do_, n * 4)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(di, in.data(), n * 4, cudaMemcpyHostToDevice)))
    return clean(false);
  bool ok = CTR_CUDA_CHECK(cuda_transformer::token_major_to_head_major_cuda(
                di, dh, x.s, x.h, x.d)) &&
            CTR_CUDA_CHECK(cuda_transformer::head_major_to_token_major_cuda(
                dh, do_, x.s, x.h, x.d)) &&
            CTR_CUDA_CHECK(cudaDeviceSynchronize()) &&
            CTR_CUDA_CHECK(
                cudaMemcpy(got.data(), dh, n * 4, cudaMemcpyDeviceToHost)) &&
            CTR_CUDA_CHECK(
                cudaMemcpy(round.data(), do_, n * 4, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < n && ok; i++)
    if (got[i] != expected[i] || round[i] != in[i]) {
      std::fprintf(stderr, "layout mismatch shape %zux%zux%zu index %zu\n", x.s,
                   x.h, x.d, i);
      ok = false;
    }
  return clean(ok);
}
} // namespace
int main() {
  for (auto x : std::array<Shape, 5>{
           {{1, 1, 2}, {4, 2, 8}, {7, 4, 8}, {32, 4, 64}, {5, 3, 6}}})
    if (!run(x))
      return EXIT_FAILURE;
  std::puts("QKV layout tests passed.");
}
