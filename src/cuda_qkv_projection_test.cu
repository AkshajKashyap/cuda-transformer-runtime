#include "cuda_transformer/cublas_check.h"
#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/qkv_layout.h"
#include "cuda_transformer/qkv_projection.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
namespace {
struct Shape {
  size_t s, h, d;
};
bool check(const char *n, const char *layout, const std::vector<float> &a,
           const std::vector<float> &b, Shape x) {
  for (size_t i = 0; i < a.size(); i++) {
    float tol = 1e-4f + 2e-6f * fmaxf(1.f, std::fabs(b[i]));
    if (std::fabs(a[i] - b[i]) > tol) {
      std::fprintf(stderr,
                   "%s %s mismatch %zux%zux%zu index %zu gpu %.8g cpu %.8g "
                   "error %.8g tolerance %.8g\n",
                   n, layout, x.s, x.h, x.d, i, a[i], b[i],
                   std::fabs(a[i] - b[i]), tol);
      return false;
    }
  }
  return true;
}
bool run(cublasHandle_t handle, Shape x) {
  cuda_transformer::QkvProjectionConfig c{x.s, x.h * x.d, x.h, x.d};
  cuda_transformer::QkvProjectionWorkspace w;
  if (!CTR_CUDA_CHECK(cuda_transformer::qkv_projection_workspace_create(&w, c)))
    return false;
  size_t in = x.s * c.hidden, weight = c.hidden * c.hidden;
  std::vector<float> X(in), Wq(weight), Wk(weight), Wv(weight), qt(in), kt(in),
      vt(in), qh(in), kh(in), vh(in), got(in);
  for (size_t i = 0; i < in; i++)
    X[i] = (int(i % 17) - 8) * .0625f;
  for (size_t i = 0; i < weight; i++) {
    Wq[i] = (int(i % 19) - 9) * .03125f;
    Wk[i] = (int((i + 3) % 23) - 11) * .03125f;
    Wv[i] = (int((i + 7) % 29) - 14) * .03125f;
  }
  cuda_transformer::gemm_cpu(X.data(), Wq.data(), qt.data(), x.s, c.hidden,
                             c.hidden);
  cuda_transformer::gemm_cpu(X.data(), Wk.data(), kt.data(), x.s, c.hidden,
                             c.hidden);
  cuda_transformer::gemm_cpu(X.data(), Wv.data(), vt.data(), x.s, c.hidden,
                             c.hidden);
  cuda_transformer::token_major_to_head_major_cpu(qt.data(), qh.data(), x.s,
                                                  x.h, x.d);
  cuda_transformer::token_major_to_head_major_cpu(kt.data(), kh.data(), x.s,
                                                  x.h, x.d);
  cuda_transformer::token_major_to_head_major_cpu(vt.data(), vh.data(), x.s,
                                                  x.h, x.d);
  float *dx = nullptr, *dq = nullptr, *dk = nullptr, *dv = nullptr;
  auto clean = [&](bool ok) {
    for (auto p : {dx, dq, dk, dv})
      if (p && !CTR_CUDA_CHECK(cudaFree(p)))
        ok = false;
    return CTR_CUDA_CHECK(
               cuda_transformer::qkv_projection_workspace_destroy(&w)) &&
           ok;
  };
  if (!CTR_CUDA_CHECK(cudaMalloc(&dx, in * 4)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&dq, weight * 4)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&dk, weight * 4)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&dv, weight * 4)) ||
      !CTR_CUDA_CHECK(
          cudaMemcpy(dx, X.data(), in * 4, cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(
          cudaMemcpy(dq, Wq.data(), weight * 4, cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(
          cudaMemcpy(dk, Wk.data(), weight * 4, cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(
          cudaMemcpy(dv, Wv.data(), weight * 4, cudaMemcpyHostToDevice)))
    return clean(false);
  bool ok = CTR_CUBLAS_CHECK(cuda_transformer::project_qkv_cuda(handle, dx, dq,
                                                                dk, dv, &w)) &&
            CTR_CUDA_CHECK(cudaDeviceSynchronize());
  for (auto pair :
       std::array<std::array<const char *, 2>, 6>{{{{"Q", "token"}},
                                                   {{"K", "token"}},
                                                   {{"V", "token"}},
                                                   {{"Q", "head"}},
                                                   {{"K", "head"}},
                                                   {{"V", "head"}}}}) {
    float *src = pair[1][0] == 't' ? (pair[0][0] == 'Q'   ? w.q_token
                                      : pair[0][0] == 'K' ? w.k_token
                                                          : w.v_token)
                                   : (pair[0][0] == 'Q'   ? w.q_head
                                      : pair[0][0] == 'K' ? w.k_head
                                                          : w.v_head);
    const auto &ref = pair[1][0] == 't' ? (pair[0][0] == 'Q'   ? qt
                                           : pair[0][0] == 'K' ? kt
                                                               : vt)
                                        : (pair[0][0] == 'Q'   ? qh
                                           : pair[0][0] == 'K' ? kh
                                                               : vh);
    ok = ok &&
         CTR_CUDA_CHECK(
             cudaMemcpy(got.data(), src, in * 4, cudaMemcpyDeviceToHost)) &&
         check(pair[0], pair[1], got, ref, x);
  }
  return clean(ok);
}
} // namespace
int main() {
  cuda_transformer::QkvProjectionWorkspace w;
  if (cuda_transformer::qkv_projection_workspace_create(&w, {0, 8, 1, 8}) !=
          cudaErrorInvalidValue ||
      cuda_transformer::qkv_projection_workspace_create(&w, {4, 15, 2, 8}) !=
          cudaErrorInvalidValue)
    return EXIT_FAILURE;
  cublasHandle_t h = nullptr;
  if (!CTR_CUBLAS_CHECK(cublasCreate(&h)))
    return EXIT_FAILURE;
  bool ok = true;
  for (auto x :
       std::array<Shape, 4>{{{1, 1, 8}, {4, 2, 8}, {7, 4, 8}, {32, 4, 64}}})
    if (!run(h, x)) {
      ok = false;
      break;
    }
  if (!CTR_CUBLAS_CHECK(cublasDestroy(h)))
    ok = false;
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
