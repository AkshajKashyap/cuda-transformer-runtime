#include "cuda_transformer/attention.h"
#include "cuda_transformer/cublas_check.h"
#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/gemm.h"
#include "cuda_transformer/normalized_attention_core.h"
#include "cuda_transformer/qkv_layout.h"
#include "cuda_transformer/transformer_primitives.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr float kEpsilon = 1.0e-5F;
constexpr float kAbsoluteTolerance = 2.0e-4F;
constexpr float kRelativeTolerance = 5.0e-5F;

struct Shape {
  std::size_t sequence;
  std::size_t hidden;
  std::size_t heads;
  std::size_t head_dim;
};

bool check_output(const std::vector<float>& gpu, const std::vector<float>& cpu,
                  Shape shape) {
  for (std::size_t i = 0; i < gpu.size(); ++i) {
    const float tolerance =
        kAbsoluteTolerance + kRelativeTolerance * fmaxf(1.0F, std::fabs(cpu[i]));
    if (!std::isfinite(gpu[i]) || !std::isfinite(cpu[i]) ||
        std::fabs(gpu[i] - cpu[i]) > tolerance) {
      std::fprintf(stderr,
                   "output mismatch: seq=%zu hidden=%zu heads=%zu head_dim=%zu "
                   "index=%zu gpu=%.8g cpu=%.8g error=%.8g tolerance=%.8g\n",
                   shape.sequence, shape.hidden, shape.heads, shape.head_dim, i,
                   gpu[i], cpu[i], std::fabs(gpu[i] - cpu[i]), tolerance);
      return false;
    }
  }
  return true;
}

bool invalid_config_tests() {
  using cuda_transformer::NormalizedAttentionCoreConfig;
  using cuda_transformer::NormalizedAttentionCoreWorkspace;
  using cuda_transformer::normalized_attention_core_workspace_create;
  using cuda_transformer::valid_normalized_attention_core_config;

  for (const NormalizedAttentionCoreConfig config :
       std::array<NormalizedAttentionCoreConfig, 5>{{
           {0, 8, 1, 8, kEpsilon}, {4, 15, 2, 8, kEpsilon},
           {4, 16, 2, 7, kEpsilon}, {4, 16, 2, 8, 0.0F},
           {4, 16, 2, 8, -kEpsilon},
       }}) {
    NormalizedAttentionCoreWorkspace workspace;
    if (valid_normalized_attention_core_config(config) ||
        normalized_attention_core_workspace_create(&workspace, config) !=
            cudaErrorInvalidValue) {
      std::fprintf(stderr, "invalid normalized-attention configuration accepted\n");
      return false;
    }
  }
  return true;
}

bool run(cublasHandle_t handle, Shape shape) {
  using namespace cuda_transformer;

  const NormalizedAttentionCoreConfig config{shape.sequence, shape.hidden,
                                             shape.heads, shape.head_dim,
                                             kEpsilon};
  const AttentionShape attention{1, shape.heads, shape.sequence, shape.head_dim};
  const std::size_t activation_count = shape.sequence * shape.hidden;
  const std::size_t weight_count = shape.hidden * shape.hidden;

  std::vector<float> input(activation_count), norm_weight(shape.hidden);
  std::vector<float> wq(weight_count), wk(weight_count), wv(weight_count);
  for (std::size_t i = 0; i < activation_count; ++i)
    input[i] = static_cast<float>(static_cast<int>((i * 5 + 3) % 37) - 18) /
               32.0F;
  for (std::size_t i = 0; i < shape.hidden; ++i)
    norm_weight[i] = 0.75F + static_cast<float>(i % 13) / 32.0F;
  for (std::size_t i = 0; i < weight_count; ++i) {
    wq[i] = static_cast<float>(static_cast<int>((i * 3 + 1) % 41) - 20) / 64.0F;
    wk[i] = static_cast<float>(static_cast<int>((i * 5 + 7) % 43) - 21) / 64.0F;
    wv[i] = static_cast<float>(static_cast<int>((i * 7 + 11) % 47) - 23) / 64.0F;
  }

  std::vector<float> normalized(activation_count), q_token(activation_count),
      k_token(activation_count), v_token(activation_count),
      q_head(activation_count), k_head(activation_count), v_head(activation_count),
      rotated_q(activation_count), probabilities(shape.heads * shape.sequence *
                                                  shape.sequence),
      output_cpu(activation_count), output_gpu(activation_count);
  rmsnorm_cpu(input.data(), norm_weight.data(), normalized.data(), shape.sequence,
              shape.hidden, kEpsilon);
  gemm_cpu(normalized.data(), wq.data(), q_token.data(), shape.sequence,
           shape.hidden, shape.hidden);
  gemm_cpu(normalized.data(), wk.data(), k_token.data(), shape.sequence,
           shape.hidden, shape.hidden);
  gemm_cpu(normalized.data(), wv.data(), v_token.data(), shape.sequence,
           shape.hidden, shape.hidden);
  token_major_to_head_major_cpu(q_token.data(), q_head.data(), shape.sequence,
                                shape.heads, shape.head_dim);
  token_major_to_head_major_cpu(k_token.data(), k_head.data(), shape.sequence,
                                shape.heads, shape.head_dim);
  token_major_to_head_major_cpu(v_token.data(), v_head.data(), shape.sequence,
                                shape.heads, shape.head_dim);
  attention_cpu(q_head.data(), k_head.data(), v_head.data(), rotated_q.data(),
                probabilities.data(), output_cpu.data(), attention);

  float *d_input = nullptr, *d_norm_weight = nullptr, *d_wq = nullptr,
        *d_wk = nullptr, *d_wv = nullptr;
  NormalizedAttentionCoreWorkspace workspace;
  auto cleanup = [&](bool ok) {
    for (float* pointer : {d_input, d_norm_weight, d_wq, d_wk, d_wv}) {
      if (pointer != nullptr && !CTR_CUDA_CHECK(cudaFree(pointer)))
        ok = false;
    }
    if (!CTR_CUDA_CHECK(normalized_attention_core_workspace_destroy(&workspace)))
      ok = false;
    return ok;
  };

  if (!CTR_CUDA_CHECK(normalized_attention_core_workspace_create(&workspace,
                                                                   config)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_input, activation_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_norm_weight, shape.hidden * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_wq, weight_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_wk, weight_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_wv, weight_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_input, input.data(),
                                 activation_count * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_norm_weight, norm_weight.data(),
                                 shape.hidden * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_wq, wq.data(), weight_count * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_wk, wk.data(), weight_count * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_wv, wv.data(), weight_count * sizeof(float),
                                 cudaMemcpyHostToDevice)))
    return cleanup(false);

  const bool ok =
      CTR_CUBLAS_CHECK(normalized_attention_core_cuda(
          handle, d_input, d_norm_weight, d_wq, d_wk, d_wv, &workspace)) &&
      CTR_CUDA_CHECK(cudaDeviceSynchronize()) &&
      CTR_CUDA_CHECK(cudaMemcpy(output_gpu.data(), workspace.output,
                                 activation_count * sizeof(float),
                                 cudaMemcpyDeviceToHost)) &&
      check_output(output_gpu, output_cpu, shape);
  return cleanup(ok);
}

}  // namespace

int main() {
  if (!invalid_config_tests())
    return EXIT_FAILURE;

  cublasHandle_t handle = nullptr;
  if (!CTR_CUBLAS_CHECK(cublasCreate(&handle)))
    return EXIT_FAILURE;
  bool ok = true;
  for (const Shape shape : std::array<Shape, 4>{{
           {1, 8, 1, 8}, {4, 16, 2, 8}, {7, 32, 4, 8}, {32, 256, 4, 64},
       }}) {
    if (!run(handle, shape)) {
      ok = false;
      break;
    }
  }
  if (!CTR_CUBLAS_CHECK(cublasDestroy(handle)))
    ok = false;
  if (ok)
    std::puts("Normalized attention core tests passed.");
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
