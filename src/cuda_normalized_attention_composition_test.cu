#include "cuda_transformer/attention.h"
#include "cuda_transformer/cublas_check.h"
#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/gemm.h"
#include "cuda_transformer/qkv_layout.h"
#include "cuda_transformer/qkv_projection.h"
#include "cuda_transformer/transformer_primitives.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr float kRmsnormAbsoluteTolerance = 1.0e-5f;
constexpr float kRmsnormRelativeTolerance = 2.0e-5f;
constexpr float kProjectionAbsoluteTolerance = 1.0e-4f;
constexpr float kProjectionRelativeTolerance = 2.0e-6f;
constexpr float kAttentionAbsoluteTolerance = 2.0e-4f;
constexpr float kAttentionRelativeTolerance = 5.0e-5f;
constexpr float kRmsnormEpsilon = 1.0e-5f;

struct Shape {
  std::size_t sequence;
  std::size_t hidden;
  std::size_t heads;
  std::size_t head_dim;
};

bool check_values(const char* stage, const std::vector<float>& gpu,
                  const std::vector<float>& cpu, Shape shape, float absolute,
                  float relative) {
  for (std::size_t i = 0; i < gpu.size(); ++i) {
    const float tolerance =
        absolute + relative * fmaxf(1.0f, std::fabs(cpu[i]));
    if (!std::isfinite(gpu[i]) || !std::isfinite(cpu[i]) ||
        std::fabs(gpu[i] - cpu[i]) > tolerance) {
      std::fprintf(stderr,
                   "%s mismatch: seq=%zu hidden=%zu heads=%zu head_dim=%zu "
                   "index=%zu gpu=%.8g cpu=%.8g error=%.8g tolerance=%.8g\n",
                   stage, shape.sequence, shape.hidden, shape.heads,
                   shape.head_dim, i, gpu[i], cpu[i],
                   std::fabs(gpu[i] - cpu[i]), tolerance);
      return false;
    }
  }
  return true;
}

bool run(cublasHandle_t handle, Shape shape) {
  using namespace cuda_transformer;

  const QkvProjectionConfig projection{shape.sequence, shape.hidden,
                                       shape.heads, shape.head_dim};
  const AttentionShape attention{1, shape.heads, shape.sequence,
                                 shape.head_dim};
  const std::size_t activation_count = shape.sequence * shape.hidden;
  const std::size_t weight_count = shape.hidden * shape.hidden;
  const std::size_t probability_count =
      shape.heads * shape.sequence * shape.sequence;

  std::vector<float> input(activation_count), rms_weight(shape.hidden);
  std::vector<float> wq(weight_count), wk(weight_count), wv(weight_count);
  for (std::size_t i = 0; i < activation_count; ++i)
    input[i] = static_cast<float>(static_cast<int>((i * 5 + 3) % 37) - 18) /
               32.0f;
  for (std::size_t i = 0; i < shape.hidden; ++i)
    rms_weight[i] = 0.75f + static_cast<float>(i % 13) / 32.0f;
  for (std::size_t i = 0; i < weight_count; ++i) {
    wq[i] = static_cast<float>(static_cast<int>((i * 3 + 1) % 41) - 20) /
            64.0f;
    wk[i] = static_cast<float>(static_cast<int>((i * 5 + 7) % 43) - 21) /
            64.0f;
    wv[i] = static_cast<float>(static_cast<int>((i * 7 + 11) % 47) - 23) /
            64.0f;
  }

  std::vector<float> normalized_cpu(activation_count);
  std::vector<float> q_token(activation_count), k_token(activation_count),
      v_token(activation_count), q_head(activation_count),
      k_head(activation_count), v_head(activation_count),
      rotated_q_cpu(activation_count), probabilities_cpu(probability_count),
      output_cpu(activation_count), received(activation_count);
  rmsnorm_cpu(input.data(), rms_weight.data(), normalized_cpu.data(),
              shape.sequence, shape.hidden, kRmsnormEpsilon);
  gemm_cpu(normalized_cpu.data(), wq.data(), q_token.data(), shape.sequence,
           shape.hidden, shape.hidden);
  gemm_cpu(normalized_cpu.data(), wk.data(), k_token.data(), shape.sequence,
           shape.hidden, shape.hidden);
  gemm_cpu(normalized_cpu.data(), wv.data(), v_token.data(), shape.sequence,
           shape.hidden, shape.hidden);
  token_major_to_head_major_cpu(q_token.data(), q_head.data(), shape.sequence,
                                shape.heads, shape.head_dim);
  token_major_to_head_major_cpu(k_token.data(), k_head.data(), shape.sequence,
                                shape.heads, shape.head_dim);
  token_major_to_head_major_cpu(v_token.data(), v_head.data(), shape.sequence,
                                shape.heads, shape.head_dim);
  attention_cpu(q_head.data(), k_head.data(), v_head.data(),
                rotated_q_cpu.data(), probabilities_cpu.data(),
                output_cpu.data(), attention);

  QkvProjectionWorkspace workspace;
  float *d_input = nullptr, *d_rms_weight = nullptr, *d_normalized = nullptr,
        *d_wq = nullptr, *d_wk = nullptr, *d_wv = nullptr,
        *d_rotated_q = nullptr, *d_rotated_k = nullptr, *d_probabilities = nullptr,
        *d_output = nullptr;
  auto cleanup = [&](bool ok) {
    for (float* pointer : {d_input, d_rms_weight, d_normalized, d_wq, d_wk,
                           d_wv, d_rotated_q, d_rotated_k, d_probabilities,
                           d_output}) {
      if (pointer != nullptr && !CTR_CUDA_CHECK(cudaFree(pointer)))
        ok = false;
    }
    if (workspace.q_token != nullptr &&
        !CTR_CUDA_CHECK(qkv_projection_workspace_destroy(&workspace)))
      ok = false;
    return ok;
  };

  if (!CTR_CUDA_CHECK(qkv_projection_workspace_create(&workspace, projection)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_input, activation_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(
          cudaMalloc(&d_rms_weight, shape.hidden * sizeof(float))) ||
      !CTR_CUDA_CHECK(
          cudaMalloc(&d_normalized, activation_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_wq, weight_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_wk, weight_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_wv, weight_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(
          cudaMalloc(&d_rotated_q, activation_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(
          cudaMalloc(&d_rotated_k, activation_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(
          cudaMalloc(&d_probabilities, probability_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_output, activation_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_input, input.data(),
                                 activation_count * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_rms_weight, rms_weight.data(),
                                 shape.hidden * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_wq, wq.data(), weight_count * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_wk, wk.data(), weight_count * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_wv, wv.data(), weight_count * sizeof(float),
                                 cudaMemcpyHostToDevice)))
    return cleanup(false);

  bool ok =
      CTR_CUDA_CHECK(rmsnorm_cuda(d_input, d_rms_weight, d_normalized,
                                  shape.sequence, shape.hidden,
                                  kRmsnormEpsilon)) &&
      CTR_CUBLAS_CHECK(project_qkv_cuda(handle, d_normalized, d_wq, d_wk, d_wv,
                                         &workspace)) &&
      CTR_CUDA_CHECK(rope_cuda(workspace.q_head, d_rotated_q, attention)) &&
      CTR_CUDA_CHECK(rope_cuda(workspace.k_head, d_rotated_k, attention)) &&
      CTR_CUDA_CHECK(attention_scores_cuda(d_rotated_q, d_rotated_k,
                                           d_probabilities, attention)) &&
      CTR_CUDA_CHECK(softmax_cuda(d_probabilities, d_probabilities,
                                  shape.heads * shape.sequence,
                                  shape.sequence)) &&
      CTR_CUDA_CHECK(attention_values_cuda(d_probabilities, workspace.v_head,
                                           d_output, attention)) &&
      CTR_CUDA_CHECK(cudaDeviceSynchronize());

  const std::array<std::pair<const char*, float*>, 5> device_values{{
      {"RMSNorm", d_normalized},
      {"Q projection, head-major", workspace.q_head},
      {"K projection, head-major", workspace.k_head},
      {"V projection, head-major", workspace.v_head},
      {"Causal attention output", d_output},
  }};
  const std::array<const std::vector<float>*, 5> cpu_values{{
      &normalized_cpu, &q_head, &k_head, &v_head, &output_cpu,
  }};
  for (std::size_t i = 0; i < device_values.size() && ok; ++i) {
    const float absolute = i == 0 ? kRmsnormAbsoluteTolerance
                           : i < 4 ? kProjectionAbsoluteTolerance
                                   : kAttentionAbsoluteTolerance;
    const float relative = i == 0 ? kRmsnormRelativeTolerance
                           : i < 4 ? kProjectionRelativeTolerance
                                   : kAttentionRelativeTolerance;
    ok = CTR_CUDA_CHECK(cudaMemcpy(received.data(), device_values[i].second,
                                   activation_count * sizeof(float),
                                   cudaMemcpyDeviceToHost)) &&
         check_values(device_values[i].first, received, *cpu_values[i], shape,
                      absolute, relative);
  }
  return cleanup(ok);
}

}  // namespace

int main() {
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
    std::puts("Normalized attention composition tests passed.");
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
