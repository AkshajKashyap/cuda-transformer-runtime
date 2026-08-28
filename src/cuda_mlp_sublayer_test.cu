#include "cuda_transformer/cublas_check.h"
#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/gemm.h"
#include "cuda_transformer/mlp_sublayer.h"
#include "cuda_transformer/transformer_primitives.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

constexpr float kEpsilon = 1.0e-5F;
constexpr float kPrimitiveAbsoluteTolerance = 1.0e-5F;
constexpr float kPrimitiveRelativeTolerance = 2.0e-5F;
constexpr float kLinearAbsoluteTolerance = 1.0e-4F;
constexpr float kLinearRelativeTolerance = 2.0e-6F;
constexpr float kDownAbsoluteTolerance = 3.0e-4F;
constexpr float kDownRelativeTolerance = 6.0e-5F;
// The residual adds one FP32 operation to the already composed down result.
constexpr float kOutputAbsoluteTolerance = 3.5e-4F;
constexpr float kOutputRelativeTolerance = 7.0e-5F;

struct Shape {
  std::size_t sequence;
  std::size_t hidden;
  std::size_t intermediate;
};

bool check_values(const char* stage, const std::vector<float>& gpu,
                  const std::vector<float>& cpu, std::size_t count, Shape shape,
                  float absolute, float relative) {
  for (std::size_t i = 0; i < count; ++i) {
    const float tolerance = absolute + relative * fmaxf(1.0F, std::fabs(cpu[i]));
    if (!std::isfinite(gpu[i]) || !std::isfinite(cpu[i]) ||
        std::fabs(gpu[i] - cpu[i]) > tolerance) {
      std::fprintf(stderr,
                   "%s mismatch: seq=%zu hidden=%zu intermediate=%zu index=%zu "
                   "gpu=%.8g cpu=%.8g error=%.8g tolerance=%.8g\n",
                   stage, shape.sequence, shape.hidden, shape.intermediate, i,
                   gpu[i], cpu[i], std::fabs(gpu[i] - cpu[i]), tolerance);
      return false;
    }
  }
  return true;
}

bool invalid_config_tests() {
  using cuda_transformer::MlpSublayerConfig;
  using cuda_transformer::MlpSublayerWorkspace;
  using cuda_transformer::mlp_sublayer_workspace_create;
  using cuda_transformer::valid_mlp_sublayer_config;

  for (const MlpSublayerConfig config : std::array<MlpSublayerConfig, 6>{{
           {0, 8, 16, kEpsilon}, {1, 0, 16, kEpsilon},
           {1, 8, 0, kEpsilon}, {1, 8, 16, 0.0F},
           {1, 8, 16, -kEpsilon},
           {1, 8, 16, std::numeric_limits<float>::infinity()},
       }}) {
    MlpSublayerWorkspace workspace;
    if (valid_mlp_sublayer_config(config) ||
        mlp_sublayer_workspace_create(&workspace, config) !=
            cudaErrorInvalidValue) {
      std::fprintf(stderr, "invalid MLP sublayer configuration accepted\n");
      return false;
    }
  }
  return true;
}

bool run(cublasHandle_t handle, Shape shape) {
  using namespace cuda_transformer;

  const MlpSublayerConfig config{shape.sequence, shape.hidden,
                                 shape.intermediate, kEpsilon};
  const std::size_t activation_count = shape.sequence * shape.hidden;
  const std::size_t intermediate_count = shape.sequence * shape.intermediate;
  const std::size_t gate_weight_count = shape.hidden * shape.intermediate;
  const std::size_t down_weight_count = shape.intermediate * shape.hidden;

  std::vector<float> input(activation_count), norm_weight(shape.hidden);
  std::vector<float> w_gate(gate_weight_count), w_up(gate_weight_count),
      w_down(down_weight_count);
  for (std::size_t i = 0; i < activation_count; ++i)
    input[i] = static_cast<float>(static_cast<int>((i * 5 + 3) % 37) - 18) /
               32.0F;
  for (std::size_t i = 0; i < shape.hidden; ++i)
    norm_weight[i] = 0.75F + static_cast<float>(i % 13) / 32.0F;
  for (std::size_t i = 0; i < gate_weight_count; ++i) {
    w_gate[i] =
        static_cast<float>(static_cast<int>((i * 3 + 1) % 41) - 20) / 64.0F;
    w_up[i] =
        static_cast<float>(static_cast<int>((i * 7 + 9) % 47) - 23) / 64.0F;
  }
  for (std::size_t i = 0; i < down_weight_count; ++i)
    w_down[i] =
        static_cast<float>(static_cast<int>((i * 11 + 13) % 53) - 26) / 80.0F;

  std::vector<float> normalized_cpu(activation_count), gate_cpu(intermediate_count),
      up_cpu(intermediate_count), activated_gate_cpu(intermediate_count),
      gated_cpu(intermediate_count), down_cpu(activation_count),
      output_cpu(activation_count), received(std::max(activation_count,
                                                       intermediate_count));
  rmsnorm_cpu(input.data(), norm_weight.data(), normalized_cpu.data(),
              shape.sequence, shape.hidden, kEpsilon);
  gemm_cpu(normalized_cpu.data(), w_gate.data(), gate_cpu.data(), shape.sequence,
           shape.hidden, shape.intermediate);
  gemm_cpu(normalized_cpu.data(), w_up.data(), up_cpu.data(), shape.sequence,
           shape.hidden, shape.intermediate);
  silu_cpu(gate_cpu.data(), activated_gate_cpu.data(), intermediate_count);
  multiply_cpu(activated_gate_cpu.data(), up_cpu.data(), gated_cpu.data(),
               intermediate_count);
  gemm_cpu(gated_cpu.data(), w_down.data(), down_cpu.data(), shape.sequence,
           shape.intermediate, shape.hidden);
  residual_add_cpu(input.data(), down_cpu.data(), output_cpu.data(),
                   activation_count);

  float *d_input = nullptr, *d_norm_weight = nullptr, *d_w_gate = nullptr,
        *d_w_up = nullptr, *d_w_down = nullptr, *d_output = nullptr;
  MlpSublayerWorkspace workspace;
  auto cleanup = [&](bool ok) {
    for (float* pointer : {d_input, d_norm_weight, d_w_gate, d_w_up, d_w_down,
                           d_output}) {
      if (pointer != nullptr && !CTR_CUDA_CHECK(cudaFree(pointer)))
        ok = false;
    }
    if (!CTR_CUDA_CHECK(mlp_sublayer_workspace_destroy(&workspace)))
      ok = false;
    return ok;
  };

  if (!CTR_CUDA_CHECK(mlp_sublayer_workspace_create(&workspace, config)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_input, activation_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_norm_weight, shape.hidden * sizeof(float))) ||
      !CTR_CUDA_CHECK(
          cudaMalloc(&d_w_gate, gate_weight_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_w_up, gate_weight_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(
          cudaMalloc(&d_w_down, down_weight_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_output, activation_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_input, input.data(),
                                 activation_count * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_norm_weight, norm_weight.data(),
                                 shape.hidden * sizeof(float),
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

  const MlpSublayerWeights weights{d_norm_weight, d_w_gate, d_w_up, d_w_down};
  bool ok =
      mlp_sublayer_cuda(handle, nullptr, weights, &workspace, d_output) ==
          CUBLAS_STATUS_INVALID_VALUE &&
      mlp_sublayer_cuda(handle, d_input,
                        {nullptr, d_w_gate, d_w_up, d_w_down}, &workspace,
                        d_output) == CUBLAS_STATUS_INVALID_VALUE &&
      mlp_sublayer_cuda(handle, d_input, weights, &workspace, nullptr) ==
          CUBLAS_STATUS_INVALID_VALUE;
  const MlpSublayerConfig saved_config = workspace.config;
  ++workspace.config.intermediate;
  ok = ok && mlp_sublayer_cuda(handle, d_input, weights, &workspace, d_output) ==
                 CUBLAS_STATUS_INVALID_VALUE;
  workspace.config = saved_config;

  ok = ok && CTR_CUBLAS_CHECK(
                 mlp_sublayer_cuda(handle, d_input, weights, &workspace,
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
  const std::array<DeviceStage, 7> stages{{
      {"RMSNorm", workspace.normalized, &normalized_cpu, activation_count,
       kPrimitiveAbsoluteTolerance, kPrimitiveRelativeTolerance},
      {"gate projection", workspace.gate, &gate_cpu, intermediate_count,
       kLinearAbsoluteTolerance, kLinearRelativeTolerance},
      {"up projection", workspace.up, &up_cpu, intermediate_count,
       kLinearAbsoluteTolerance, kLinearRelativeTolerance},
      {"SiLU(gate)", workspace.activated_gate, &activated_gate_cpu,
       intermediate_count, kPrimitiveAbsoluteTolerance,
       kPrimitiveRelativeTolerance},
      {"SiLU(gate) * up", workspace.gated, &gated_cpu, intermediate_count,
       kPrimitiveAbsoluteTolerance, kPrimitiveRelativeTolerance},
      {"down projection", workspace.down, &down_cpu, activation_count,
       kDownAbsoluteTolerance, kDownRelativeTolerance},
      {"MLP sublayer output", d_output, &output_cpu, activation_count,
       kOutputAbsoluteTolerance, kOutputRelativeTolerance},
  }};
  for (const DeviceStage& stage : stages) {
    if (!ok)
      break;
    ok = CTR_CUDA_CHECK(cudaMemcpy(received.data(), stage.device,
                                   stage.count * sizeof(float),
                                   cudaMemcpyDeviceToHost)) &&
         check_values(stage.name, received, *stage.cpu, stage.count, shape,
                      stage.absolute, stage.relative);
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
           {1, 8, 16}, {4, 16, 32}, {7, 32, 48}, {32, 256, 512},
       }}) {
    if (!run(handle, shape)) {
      ok = false;
      break;
    }
  }
  if (!CTR_CUBLAS_CHECK(cublasDestroy(handle)))
    ok = false;
  if (ok)
    std::puts("MLP sublayer tests passed.");
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
