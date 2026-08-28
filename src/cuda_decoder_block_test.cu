#include "cuda_transformer/attention.h"
#include "cuda_transformer/cublas_check.h"
#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/decoder_block.h"
#include "cuda_transformer/gemm.h"
#include "cuda_transformer/qkv_layout.h"
#include "cuda_transformer/transformer_primitives.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

constexpr float kEpsilon = 1.0e-5F;
constexpr float kAttentionAbsoluteTolerance = 4.0e-4F;
constexpr float kAttentionRelativeTolerance = 1.0e-4F;
// MLP down receives the already rounded attention residual output.
constexpr float kDownAbsoluteTolerance = 5.0e-4F;
constexpr float kDownRelativeTolerance = 1.0e-4F;
// Two complete FP32 sublayers and their residual additions are composed here.
constexpr float kOutputAbsoluteTolerance = 8.0e-4F;
constexpr float kOutputRelativeTolerance = 1.5e-4F;

struct Shape {
  std::size_t sequence;
  std::size_t hidden;
  std::size_t heads;
  std::size_t head_dim;
  std::size_t intermediate;
};

bool check_values(const char* stage, const std::vector<float>& gpu,
                  const std::vector<float>& cpu, Shape shape, float absolute,
                  float relative) {
  if (gpu.size() != cpu.size()) {
    std::fprintf(stderr,
                 "%s size mismatch: seq=%zu hidden=%zu heads=%zu head_dim=%zu "
                 "intermediate=%zu gpu_count=%zu cpu_count=%zu\n",
                 stage, shape.sequence, shape.hidden, shape.heads,
                 shape.head_dim, shape.intermediate, gpu.size(), cpu.size());
    return false;
  }
  for (std::size_t i = 0; i < gpu.size(); ++i) {
    const float tolerance = absolute + relative * fmaxf(1.0F, std::fabs(cpu[i]));
    if (!std::isfinite(gpu[i]) || !std::isfinite(cpu[i]) ||
        std::fabs(gpu[i] - cpu[i]) > tolerance) {
      std::fprintf(stderr,
                   "%s mismatch: seq=%zu hidden=%zu heads=%zu head_dim=%zu "
                   "intermediate=%zu index=%zu gpu=%.8g cpu=%.8g error=%.8g "
                   "tolerance=%.8g\n",
                   stage, shape.sequence, shape.hidden, shape.heads,
                   shape.head_dim, shape.intermediate, i, gpu[i], cpu[i],
                   std::fabs(gpu[i] - cpu[i]), tolerance);
      return false;
    }
  }
  return true;
}

bool invalid_config_tests() {
  using cuda_transformer::DecoderBlockConfig;
  using cuda_transformer::DecoderBlockWorkspace;
  using cuda_transformer::MlpSublayerConfig;
  using cuda_transformer::NormalizedAttentionCoreConfig;
  using cuda_transformer::decoder_block_workspace_create;
  using cuda_transformer::valid_decoder_block_config;

  const DecoderBlockConfig valid{{1, 8, 1, 8, kEpsilon},
                                 {1, 8, 16, kEpsilon}};
  auto invalid = std::array<DecoderBlockConfig, 8>{
      valid, valid, valid, valid, valid, valid, valid, valid};
  invalid[0].attention.sequence = 0;
  invalid[1].attention.hidden = 15;
  invalid[2].attention.head_dim = 7;
  invalid[3].attention.rmsnorm_epsilon = 0.0F;
  invalid[4].mlp.intermediate = 0;
  invalid[5].mlp.rmsnorm_epsilon = -kEpsilon;
  invalid[6].mlp.rmsnorm_epsilon = std::numeric_limits<float>::infinity();
  invalid[7].mlp.sequence = 2;
  for (const DecoderBlockConfig config : invalid) {
    DecoderBlockWorkspace workspace;
    if (valid_decoder_block_config(config) ||
        decoder_block_workspace_create(&workspace, config) !=
            cudaErrorInvalidValue) {
      std::fprintf(stderr, "invalid decoder-block configuration accepted\n");
      return false;
    }
  }
  return true;
}

bool run(cublasHandle_t handle, Shape shape) {
  using namespace cuda_transformer;

  const DecoderBlockConfig config{{shape.sequence, shape.hidden, shape.heads,
                                   shape.head_dim, kEpsilon},
                                  {shape.sequence, shape.hidden,
                                   shape.intermediate, kEpsilon}};
  const AttentionShape attention{1, shape.heads, shape.sequence, shape.head_dim};
  const std::size_t activation_count = shape.sequence * shape.hidden;
  const std::size_t square_weight_count = shape.hidden * shape.hidden;
  const std::size_t gate_weight_count = shape.hidden * shape.intermediate;
  const std::size_t down_weight_count = shape.intermediate * shape.hidden;
  const std::size_t intermediate_count = shape.sequence * shape.intermediate;

  std::vector<float> input(activation_count), attention_norm(shape.hidden),
      mlp_norm(shape.hidden), wq(square_weight_count), wk(square_weight_count),
      wv(square_weight_count), wo(square_weight_count),
      w_gate(gate_weight_count), w_up(gate_weight_count),
      w_down(down_weight_count);
  for (std::size_t i = 0; i < activation_count; ++i)
    input[i] = static_cast<float>(static_cast<int>((i * 5 + 3) % 37) - 18) /
               32.0F;
  for (std::size_t i = 0; i < shape.hidden; ++i) {
    attention_norm[i] = 0.75F + static_cast<float>(i % 13) / 32.0F;
    mlp_norm[i] = 0.60F + static_cast<float>((i * 3 + 1) % 17) / 40.0F;
  }
  for (std::size_t i = 0; i < square_weight_count; ++i) {
    wq[i] = static_cast<float>(static_cast<int>((i * 3 + 1) % 41) - 20) / 64.0F;
    wk[i] = static_cast<float>(static_cast<int>((i * 5 + 7) % 43) - 21) / 64.0F;
    wv[i] = static_cast<float>(static_cast<int>((i * 7 + 11) % 47) - 23) / 64.0F;
    wo[i] = static_cast<float>(static_cast<int>((i * 11 + 13) % 53) - 26) / 80.0F;
  }
  for (std::size_t i = 0; i < gate_weight_count; ++i) {
    w_gate[i] =
        static_cast<float>(static_cast<int>((i * 13 + 5) % 59) - 29) / 96.0F;
    w_up[i] =
        static_cast<float>(static_cast<int>((i * 17 + 9) % 61) - 30) / 96.0F;
  }
  for (std::size_t i = 0; i < down_weight_count; ++i)
    w_down[i] =
        static_cast<float>(static_cast<int>((i * 19 + 7) % 67) - 33) / 112.0F;

  std::vector<float> normalized_attention(activation_count),
      q_token(activation_count), k_token(activation_count), v_token(activation_count),
      q_head(activation_count), k_head(activation_count), v_head(activation_count),
      rotated_q(activation_count), probabilities(shape.heads * shape.sequence *
                                                  shape.sequence),
      attention_head(activation_count), attention_token(activation_count),
      attention_projected(activation_count), attention_output(activation_count),
      normalized_mlp(activation_count), gate(intermediate_count),
      up(intermediate_count), activated_gate(intermediate_count),
      gated(intermediate_count), down(activation_count), output_cpu(activation_count),
      received;
  rmsnorm_cpu(input.data(), attention_norm.data(), normalized_attention.data(),
              shape.sequence, shape.hidden, kEpsilon);
  gemm_cpu(normalized_attention.data(), wq.data(), q_token.data(),
           shape.sequence, shape.hidden, shape.hidden);
  gemm_cpu(normalized_attention.data(), wk.data(), k_token.data(),
           shape.sequence, shape.hidden, shape.hidden);
  gemm_cpu(normalized_attention.data(), wv.data(), v_token.data(),
           shape.sequence, shape.hidden, shape.hidden);
  token_major_to_head_major_cpu(q_token.data(), q_head.data(), shape.sequence,
                                shape.heads, shape.head_dim);
  token_major_to_head_major_cpu(k_token.data(), k_head.data(), shape.sequence,
                                shape.heads, shape.head_dim);
  token_major_to_head_major_cpu(v_token.data(), v_head.data(), shape.sequence,
                                shape.heads, shape.head_dim);
  attention_cpu(q_head.data(), k_head.data(), v_head.data(), rotated_q.data(),
                probabilities.data(), attention_head.data(), attention);
  head_major_to_token_major_cpu(attention_head.data(), attention_token.data(),
                                shape.sequence, shape.heads, shape.head_dim);
  gemm_cpu(attention_token.data(), wo.data(), attention_projected.data(),
           shape.sequence, shape.hidden, shape.hidden);
  residual_add_cpu(input.data(), attention_projected.data(),
                   attention_output.data(), activation_count);
  rmsnorm_cpu(attention_output.data(), mlp_norm.data(), normalized_mlp.data(),
              shape.sequence, shape.hidden, kEpsilon);
  gemm_cpu(normalized_mlp.data(), w_gate.data(), gate.data(), shape.sequence,
           shape.hidden, shape.intermediate);
  gemm_cpu(normalized_mlp.data(), w_up.data(), up.data(), shape.sequence,
           shape.hidden, shape.intermediate);
  silu_cpu(gate.data(), activated_gate.data(), intermediate_count);
  multiply_cpu(activated_gate.data(), up.data(), gated.data(), intermediate_count);
  gemm_cpu(gated.data(), w_down.data(), down.data(), shape.sequence,
           shape.intermediate, shape.hidden);
  residual_add_cpu(attention_output.data(), down.data(), output_cpu.data(),
                   activation_count);

  bool attention_changed = false;
  for (std::size_t i = 0; i < activation_count; ++i)
    attention_changed = attention_changed ||
                        std::fabs(attention_output[i] - input[i]) > 1.0e-4F;
  if (!attention_changed) {
    std::fprintf(stderr, "residual-chain guard: attention output equals input\n");
    return false;
  }

  float *d_input = nullptr, *d_attention_norm = nullptr, *d_mlp_norm = nullptr,
        *d_wq = nullptr, *d_wk = nullptr, *d_wv = nullptr, *d_wo = nullptr,
        *d_w_gate = nullptr, *d_w_up = nullptr, *d_w_down = nullptr,
        *d_output = nullptr;
  DecoderBlockWorkspace workspace;
  auto cleanup = [&](bool ok) {
    for (float* pointer : {d_input, d_attention_norm, d_mlp_norm, d_wq, d_wk,
                           d_wv, d_wo, d_w_gate, d_w_up, d_w_down, d_output}) {
      if (pointer != nullptr && !CTR_CUDA_CHECK(cudaFree(pointer)))
        ok = false;
    }
    if (!CTR_CUDA_CHECK(decoder_block_workspace_destroy(&workspace)))
      ok = false;
    return ok;
  };

  if (!CTR_CUDA_CHECK(decoder_block_workspace_create(&workspace, config)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_input, activation_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(
          cudaMalloc(&d_attention_norm, shape.hidden * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_mlp_norm, shape.hidden * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_wq, square_weight_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_wk, square_weight_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_wv, square_weight_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_wo, square_weight_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(
          cudaMalloc(&d_w_gate, gate_weight_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_w_up, gate_weight_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(
          cudaMalloc(&d_w_down, down_weight_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_output, activation_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_input, input.data(),
                                 activation_count * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_attention_norm, attention_norm.data(),
                                 shape.hidden * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_mlp_norm, mlp_norm.data(),
                                 shape.hidden * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_wq, wq.data(), square_weight_count * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_wk, wk.data(), square_weight_count * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_wv, wv.data(), square_weight_count * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_wo, wo.data(), square_weight_count * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_w_gate, w_gate.data(),
                                 gate_weight_count * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_w_up, w_up.data(),
                                 gate_weight_count * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_w_down, w_down.data(),
                                 down_weight_count * sizeof(float),
                                 cudaMemcpyHostToDevice)))
    return cleanup(false);

  const DecoderBlockWeights weights{
      {d_attention_norm, d_wq, d_wk, d_wv, d_wo},
      {d_mlp_norm, d_w_gate, d_w_up, d_w_down},
  };
  bool ok =
      decoder_block_cuda(handle, nullptr, weights, &workspace, d_output) ==
          CUBLAS_STATUS_INVALID_VALUE &&
      decoder_block_cuda(handle, d_input,
                         {{nullptr, d_wq, d_wk, d_wv, d_wo},
                          {d_mlp_norm, d_w_gate, d_w_up, d_w_down}},
                         &workspace, d_output) == CUBLAS_STATUS_INVALID_VALUE &&
      decoder_block_cuda(handle, d_input, weights, &workspace, nullptr) ==
          CUBLAS_STATUS_INVALID_VALUE;
  const DecoderBlockConfig saved_config = workspace.config;
  ++workspace.config.mlp.intermediate;
  ok = ok && decoder_block_cuda(handle, d_input, weights, &workspace, d_output) ==
                 CUBLAS_STATUS_INVALID_VALUE;
  workspace.config = saved_config;

  ok = ok && CTR_CUBLAS_CHECK(
                 decoder_block_cuda(handle, d_input, weights, &workspace,
                                    d_output)) &&
       CTR_CUDA_CHECK(cudaDeviceSynchronize());
  struct DeviceStage {
    const char* name;
    float* device;
    const std::vector<float>* cpu;
    std::size_t count;
    float absolute;
    float relative;
  };
  const std::array<DeviceStage, 3> stages{{
      {"attention sublayer output", workspace.attention_output, &attention_output,
       activation_count,
       kAttentionAbsoluteTolerance, kAttentionRelativeTolerance},
      {"MLP down projection", workspace.mlp.down, &down, activation_count,
       kDownAbsoluteTolerance,
       kDownRelativeTolerance},
      {"decoder block output", d_output, &output_cpu, activation_count,
       kOutputAbsoluteTolerance,
       kOutputRelativeTolerance},
  }};
  for (const DeviceStage& stage : stages) {
    if (!ok)
      break;
    received.resize(stage.count);
    ok = CTR_CUDA_CHECK(cudaMemcpy(received.data(), stage.device,
                                   stage.count * sizeof(float),
                                   cudaMemcpyDeviceToHost)) &&
         check_values(stage.name, received, *stage.cpu, shape, stage.absolute,
                      stage.relative);
  }
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
           {1, 8, 1, 8, 16}, {4, 16, 2, 8, 32}, {7, 32, 4, 8, 48},
           {32, 256, 4, 64, 512},
       }}) {
    if (!run(handle, shape)) {
      ok = false;
      break;
    }
  }
  if (!CTR_CUBLAS_CHECK(cublasDestroy(handle)))
    ok = false;
  if (ok)
    std::puts("Decoder block tests passed.");
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
