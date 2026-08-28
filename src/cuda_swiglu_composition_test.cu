#include "cuda_transformer/cublas_check.h"
#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/gemm.h"
#include "cuda_transformer/transformer_primitives.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr float kEpsilon = 1.0e-5F;
constexpr float kPrimitiveAbsoluteTolerance = 1.0e-5F;
constexpr float kPrimitiveRelativeTolerance = 2.0e-5F;
constexpr float kLinearAbsoluteTolerance = 1.0e-4F;
constexpr float kLinearRelativeTolerance = 2.0e-6F;
// The final projection accumulates intermediate SwiGLU rounding through a
// second FP32 GEMM, so it has a modestly wider composed tolerance.
constexpr float kDownAbsoluteTolerance = 3.0e-4F;
constexpr float kDownRelativeTolerance = 6.0e-5F;

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

bool run(cublasHandle_t handle, Shape shape) {
  using namespace cuda_transformer;

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
      received(std::max(activation_count, intermediate_count));
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

  float *d_input = nullptr, *d_norm_weight = nullptr, *d_w_gate = nullptr,
        *d_w_up = nullptr, *d_w_down = nullptr, *d_normalized = nullptr,
        *d_gate = nullptr, *d_up = nullptr, *d_activated_gate = nullptr,
        *d_gated = nullptr, *d_down = nullptr;
  auto cleanup = [&](bool ok) {
    for (float* pointer : {d_input, d_norm_weight, d_w_gate, d_w_up, d_w_down,
                           d_normalized, d_gate, d_up, d_activated_gate, d_gated,
                           d_down}) {
      if (pointer != nullptr && !CTR_CUDA_CHECK(cudaFree(pointer)))
        ok = false;
    }
    return ok;
  };

  if (!CTR_CUDA_CHECK(cudaMalloc(&d_input, activation_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_norm_weight, shape.hidden * sizeof(float))) ||
      !CTR_CUDA_CHECK(
          cudaMalloc(&d_w_gate, gate_weight_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_w_up, gate_weight_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(
          cudaMalloc(&d_w_down, down_weight_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(
          cudaMalloc(&d_normalized, activation_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(
          cudaMalloc(&d_gate, intermediate_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_up, intermediate_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(
          cudaMalloc(&d_activated_gate, intermediate_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_gated, intermediate_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_down, activation_count * sizeof(float))) ||
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

  bool ok =
      CTR_CUDA_CHECK(rmsnorm_cuda(d_input, d_norm_weight, d_normalized,
                                  shape.sequence, shape.hidden, kEpsilon)) &&
      CTR_CUBLAS_CHECK(linear_cublas_row_major(
          handle, d_normalized, d_w_gate, d_gate, shape.sequence, shape.hidden,
          shape.intermediate)) &&
      CTR_CUBLAS_CHECK(linear_cublas_row_major(
          handle, d_normalized, d_w_up, d_up, shape.sequence, shape.hidden,
          shape.intermediate)) &&
      CTR_CUDA_CHECK(silu_cuda(d_gate, d_activated_gate, intermediate_count)) &&
      CTR_CUDA_CHECK(
          multiply_cuda(d_activated_gate, d_up, d_gated, intermediate_count)) &&
      CTR_CUBLAS_CHECK(linear_cublas_row_major(
          handle, d_gated, d_w_down, d_down, shape.sequence, shape.intermediate,
          shape.hidden)) &&
      CTR_CUDA_CHECK(cudaDeviceSynchronize());

  struct DeviceStage {
    const char* name;
    float* device;
    const std::vector<float>* cpu;
    std::size_t count;
    float absolute;
    float relative;
  };
  const std::array<DeviceStage, 6> stages{{
      {"RMSNorm", d_normalized, &normalized_cpu, activation_count,
       kPrimitiveAbsoluteTolerance, kPrimitiveRelativeTolerance},
      {"gate projection", d_gate, &gate_cpu, intermediate_count,
       kLinearAbsoluteTolerance, kLinearRelativeTolerance},
      {"up projection", d_up, &up_cpu, intermediate_count,
       kLinearAbsoluteTolerance, kLinearRelativeTolerance},
      {"SiLU(gate)", d_activated_gate, &activated_gate_cpu, intermediate_count,
       kPrimitiveAbsoluteTolerance, kPrimitiveRelativeTolerance},
      {"SiLU(gate) * up", d_gated, &gated_cpu, intermediate_count,
       kPrimitiveAbsoluteTolerance, kPrimitiveRelativeTolerance},
      {"down projection", d_down, &down_cpu, activation_count,
       kDownAbsoluteTolerance, kDownRelativeTolerance},
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
    std::puts("SwiGLU composition tests passed.");
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
