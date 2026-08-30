#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/transformer_primitives.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
namespace {
bool close(float a, float b) {
  return std::fabs(a - b) <= 1e-5f + 2e-5f * fmaxf(1.f, std::fabs(b));
}
bool free3(bool ok, float *a, float *b, float *c) {
  if (a && !CTR_CUDA_CHECK(cudaFree(a)))
    ok = false;
  if (b && !CTR_CUDA_CHECK(cudaFree(b)))
    ok = false;
  if (c && !CTR_CUDA_CHECK(cudaFree(c)))
    ok = false;
  return ok;
}
bool check(const char *n, const std::vector<float> &a,
           const std::vector<float> &b) {
  if (a.size() != b.size()) {
    std::fprintf(stderr, "%s size mismatch: %zu %zu\n", n, a.size(), b.size());
    return false;
  }
  for (size_t i = 0; i < a.size(); i++)
    if (!close(a[i], b[i])) {
      std::fprintf(stderr, "%s mismatch at %zu: %.8g %.8g\n", n, i, b[i], a[i]);
      return false;
    }
  return true;
}
bool free4(bool ok, float *a, float *b, float *c, float *d) {
  if (a && !CTR_CUDA_CHECK(cudaFree(a)))
    ok = false;
  if (b && !CTR_CUDA_CHECK(cudaFree(b)))
    ok = false;
  if (c && !CTR_CUDA_CHECK(cudaFree(c)))
    ok = false;
  if (d && !CTR_CUDA_CHECK(cudaFree(d)))
    ok = false;
  return ok;
}
bool layernorm_test(const char *name, const std::vector<float> &x,
                    const std::vector<float> &gamma,
                    const std::vector<float> &beta, size_t rows, size_t width,
                    float epsilon, const std::vector<float> *expected = nullptr) {
  std::vector<float> reference(x.size()), got(x.size());
  cuda_transformer::layernorm_cpu(x.data(), gamma.data(), beta.data(),
                                  reference.data(), rows, width, epsilon);
  if (expected && !check("layernorm cpu", reference, *expected))
    return false;

  float *dx = nullptr, *dgamma = nullptr, *dbeta = nullptr, *dy = nullptr;
  const size_t x_bytes = x.size() * sizeof(float);
  const size_t parameter_bytes = width * sizeof(float);
  if (!CTR_CUDA_CHECK(cudaMalloc(&dx, x_bytes)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&dgamma, parameter_bytes)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&dbeta, parameter_bytes)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&dy, x_bytes)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(dx, x.data(), x_bytes, cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(dgamma, gamma.data(), parameter_bytes,
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(dbeta, beta.data(), parameter_bytes,
                                 cudaMemcpyHostToDevice)))
    return free4(false, dx, dgamma, dbeta, dy);

  const bool ok =
      CTR_CUDA_CHECK(cuda_transformer::layernorm_cuda(dx, dgamma, dbeta, dy,
                                                       rows, width, epsilon)) &&
      CTR_CUDA_CHECK(cudaDeviceSynchronize()) &&
      CTR_CUDA_CHECK(cudaMemcpy(got.data(), dy, x_bytes, cudaMemcpyDeviceToHost)) &&
      check(name, got, reference);
  return free4(ok, dx, dgamma, dbeta, dy);
}
bool layernorm_shape_test(size_t rows, size_t width, float epsilon) {
  std::vector<float> x(rows * width), gamma(width), beta(width);
  unsigned int state = 0xC0FFEEu;
  auto next = [&state] {
    state = state * 1664525u + 1013904223u;
    return (static_cast<float>((state >> 8) & 0xFFFFu) / 32768.0f) - 1.0f;
  };
  for (float &value : x)
    value = 3.0f * next();
  for (size_t i = 0; i < width; i++) {
    gamma[i] = 0.5f + 1.5f * (0.5f * (next() + 1.0f));
    beta[i] = 0.75f * next();
  }
  return layernorm_test("layernorm", x, gamma, beta, rows, width, epsilon);
}
bool row_test(size_t rows, size_t width) {
  std::vector<float> x(rows * width), w(width), ref(rows * width),
      got(rows * width);
  for (size_t i = 0; i < x.size(); i++)
    x[i] = (int(i % 29) - 14) * .125f;
  for (size_t i = 0; i < w.size(); i++)
    w[i] = .5f + (i % 7) * .125f;
  if (width > 3)
    x[3] = 80.f;
  if (width > 4)
    x[4] = -80.f;
  float *dx = nullptr, *dw = nullptr, *dy = nullptr;
  size_t xb = x.size() * 4, wb = w.size() * 4;
  if (!CTR_CUDA_CHECK(cudaMalloc(&dx, xb)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&dw, wb)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&dy, xb)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(dx, x.data(), xb, cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(dw, w.data(), wb, cudaMemcpyHostToDevice)))
    return free3(false, dx, dw, dy);
  cuda_transformer::rmsnorm_cpu(x.data(), w.data(), ref.data(), rows, width,
                                1e-5f);
  bool ok =
      CTR_CUDA_CHECK(
          cuda_transformer::rmsnorm_cuda(dx, dw, dy, rows, width, 1e-5f)) &&
      CTR_CUDA_CHECK(cudaDeviceSynchronize()) &&
      CTR_CUDA_CHECK(cudaMemcpy(got.data(), dy, xb, cudaMemcpyDeviceToHost)) &&
      check("rmsnorm", got, ref);
  cuda_transformer::softmax_cpu(x.data(), ref.data(), rows, width);
  ok = ok &&
       CTR_CUDA_CHECK(cuda_transformer::softmax_cuda(dx, dy, rows, width)) &&
       CTR_CUDA_CHECK(cudaDeviceSynchronize()) &&
       CTR_CUDA_CHECK(cudaMemcpy(got.data(), dy, xb, cudaMemcpyDeviceToHost)) &&
       check("softmax", got, ref);
  for (size_t r = 0; r < rows && ok; r++) {
    float s = 0;
    for (size_t i = 0; i < width; i++)
      s += got[r * width + i];
    if (!close(s, 1.f)) {
      std::fprintf(stderr, "softmax row sum mismatch\n");
      ok = false;
    }
  }
  return free3(ok, dx, dw, dy);
}
bool elem_test(size_t n) {
  std::vector<float> a(n), b(n), ref(n), got(n);
  for (size_t i = 0; i < n; i++) {
    a[i] = (int(i % 31) - 15) * .125f;
    b[i] = (int(i % 17) - 8) * .25f;
  }
  float *da = nullptr, *db = nullptr, *dy = nullptr;
  size_t bytes = n * 4;
  if (!CTR_CUDA_CHECK(cudaMalloc(&da, bytes)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&db, bytes)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&dy, bytes)))
    return free3(false, da, db, dy);
  if (!CTR_CUDA_CHECK(
          cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice)))
    return free3(false, da, db, dy);
  cuda_transformer::silu_cpu(a.data(), ref.data(), n);
  bool ok = CTR_CUDA_CHECK(cuda_transformer::silu_cuda(da, dy, n)) &&
            CTR_CUDA_CHECK(cudaDeviceSynchronize()) &&
            CTR_CUDA_CHECK(
                cudaMemcpy(got.data(), dy, bytes, cudaMemcpyDeviceToHost)) &&
            check("silu", got, ref);
  cuda_transformer::multiply_cpu(a.data(), b.data(), ref.data(), n);
  ok = ok && CTR_CUDA_CHECK(cuda_transformer::multiply_cuda(da, db, dy, n)) &&
       CTR_CUDA_CHECK(cudaDeviceSynchronize()) &&
       CTR_CUDA_CHECK(
           cudaMemcpy(got.data(), dy, bytes, cudaMemcpyDeviceToHost)) &&
       check("multiply", got, ref);
  cuda_transformer::residual_add_cpu(a.data(), b.data(), ref.data(), n);
  ok = ok &&
       CTR_CUDA_CHECK(cuda_transformer::residual_add_cuda(da, db, dy, n)) &&
       CTR_CUDA_CHECK(cudaDeviceSynchronize()) &&
       CTR_CUDA_CHECK(
           cudaMemcpy(got.data(), dy, bytes, cudaMemcpyDeviceToHost)) &&
       check("residual", got, ref);
  return free3(ok, da, db, dy);
}
} // namespace
int main() {
  if (cuda_transformer::layernorm_cuda(nullptr, nullptr, nullptr, nullptr, 1,
                                       1, 1e-5f) != cudaErrorInvalidValue ||
      cuda_transformer::layernorm_cuda(nullptr, nullptr, nullptr, nullptr, 0,
                                       1, 1e-5f) != cudaErrorInvalidValue) {
    std::fputs("layernorm invalid-argument validation failed\n", stderr);
    return EXIT_FAILURE;
  }
  // x = [1, 3, 5, 7] has mean 4 and variance 5, so this case is easy to
  // audit independently of the CPU reference.
  const float inverse_stddev = 1.0f / std::sqrt(5.0f + 1e-5f);
  const std::vector<float> hand_x{1.0f, 3.0f, 5.0f, 7.0f};
  const std::vector<float> hand_gamma{1.0f, 0.5f, 2.0f, -1.0f};
  const std::vector<float> hand_beta{0.0f, 1.0f, -1.0f, 0.25f};
  const std::vector<float> hand_expected{
      -3.0f * inverse_stddev,
      1.0f - 0.5f * inverse_stddev,
      -1.0f + 2.0f * inverse_stddev,
      0.25f - 3.0f * inverse_stddev};
  if (!layernorm_test("layernorm hand", hand_x, hand_gamma, hand_beta, 1, 4,
                      1e-5f, &hand_expected))
    return EXIT_FAILURE;
  if (!layernorm_shape_test(2, 7, 0.25f))
    return EXIT_FAILURE;
  for (auto s : std::array<std::array<size_t, 2>, 6>{
           {{{2, 7}}, {{3, 128}}, {{4, 256}}, {{5, 512}}, {{3, 768}},
            {{2, 1024}}}})
    if (!layernorm_shape_test(s[0], s[1], 1e-5f))
      return EXIT_FAILURE;
  for (auto s : std::array<std::array<size_t, 2>, 7>{
           {{{1, 1}}, {{1, 17}}, {{2, 256}}, {{3, 257}}, {{32, 768}},
            {{1, 2048}}, {{3, 1025}}}})
    if (!row_test(s[0], s[1]))
      return EXIT_FAILURE;
  for (auto n : std::array<size_t, 3>{1, 257, 4099})
    if (!elem_test(n))
      return EXIT_FAILURE;
  std::puts("Transformer primitives tests passed.");
}
