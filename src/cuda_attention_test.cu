#include "cuda_transformer/attention.h"
#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/transformer_primitives.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
namespace {
bool near(float a, float b) {
  return std::fabs(a - b) < 2e-4f + 5e-5f * fmaxf(1.f, std::fabs(b));
}
bool run(cuda_transformer::AttentionShape s) {
  size_t x = s.batch * s.heads * s.sequence * s.head_dim,
         p = s.batch * s.heads * s.sequence * s.sequence;
  std::vector<float> q(x), k(x), v(x), qr(x), prob(p), out(x), gr(x), gp(p),
      go(x);
  for (size_t i = 0; i < x; i++) {
    q[i] = (int(i % 17) - 8) * .1f;
    k[i] = (int(i % 13) - 6) * .1f;
    v[i] = (int(i % 11) - 5) * .125f;
  }
  cuda_transformer::attention_cpu(q.data(), k.data(), v.data(), qr.data(),
                                  prob.data(), out.data(), s);
  float *dq = 0, *dk = 0, *dv = 0, *dqr = 0, *dkr = 0, *dp = 0, *do_ = 0;
  auto bye = [&](bool ok) {
    for (auto z : {dq, dk, dv, dqr, dkr, dp, do_})
      if (z && !CTR_CUDA_CHECK(cudaFree(z)))
        ok = false;
    return ok;
  };
  if (!CTR_CUDA_CHECK(cudaMalloc(&dq, x * 4)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&dk, x * 4)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&dv, x * 4)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&dqr, x * 4)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&dkr, x * 4)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&dp, p * 4)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&do_, x * 4)))
    return bye(false);
  if (!CTR_CUDA_CHECK(
          cudaMemcpy(dq, q.data(), x * 4, cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(
          cudaMemcpy(dk, k.data(), x * 4, cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(dv, v.data(), x * 4, cudaMemcpyHostToDevice)))
    return bye(false);
  bool ok =
      CTR_CUDA_CHECK(cuda_transformer::rope_cuda(dq, dqr, s)) &&
      CTR_CUDA_CHECK(cuda_transformer::rope_cuda(dk, dkr, s)) &&
      CTR_CUDA_CHECK(
          cuda_transformer::attention_scores_cuda(dqr, dkr, dp, s)) &&
      CTR_CUDA_CHECK(cuda_transformer::softmax_cuda(
          dp, dp, s.batch * s.heads * s.sequence, s.sequence)) &&
      CTR_CUDA_CHECK(cuda_transformer::attention_values_cuda(dp, dv, do_, s)) &&
      CTR_CUDA_CHECK(cudaDeviceSynchronize()) &&
      CTR_CUDA_CHECK(
          cudaMemcpy(gr.data(), dqr, x * 4, cudaMemcpyDeviceToHost)) &&
      CTR_CUDA_CHECK(
          cudaMemcpy(gp.data(), dp, p * 4, cudaMemcpyDeviceToHost)) &&
      CTR_CUDA_CHECK(cudaMemcpy(go.data(), do_, x * 4, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < x && ok; i++)
    ok = near(gr[i], qr[i]) && near(go[i], out[i]) && std::isfinite(go[i]);
  for (size_t r = 0; r < s.batch * s.heads * s.sequence && ok; r++) {
    float sum = 0;
    for (size_t j = 0; j < s.sequence; j++) {
      sum += gp[r * s.sequence + j];
      if (j > r % s.sequence && std::fabs(gp[r * s.sequence + j]) > 1e-6f)
        ok = false;
      ok = ok && near(gp[r * s.sequence + j], prob[r * s.sequence + j]) &&
           std::isfinite(gp[r * s.sequence + j]);
    }
    ok = near(sum, 1.f);
  }
  return bye(ok);
}
} // namespace
int main() {
  for (auto s : std::array<cuda_transformer::AttentionShape, 5>{{
           {1, 1, 1, 2}, {1, 1, 4, 4}, {1, 2, 5, 8}, {1, 2, 7, 16},
           {1, 4, 32, 64}}})
    if (!run(s))
      return EXIT_FAILURE;
  std::puts("Attention tests passed.");
}
