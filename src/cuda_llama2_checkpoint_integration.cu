#include "cuda_transformer/cublas_check.h"
#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/llama2_checkpoint.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

bool close_logits(const std::vector<float>& cpu, const std::vector<float>& gpu) {
  if (cpu.size() != gpu.size())
    return false;
  for (std::size_t index = 0; index < cpu.size(); ++index) {
    const float tolerance = 1e-3F + 2e-4F * fmaxf(1.0F, std::fabs(cpu[index]));
    if (!std::isfinite(cpu[index]) || !std::isfinite(gpu[index]) ||
        std::fabs(cpu[index] - gpu[index]) > tolerance) {
      std::fprintf(stderr,
                   "real-checkpoint logits mismatch at %zu: cpu=%.8g gpu=%.8g "
                   "tolerance=%.8g\n",
                   index, cpu[index], gpu[index], tolerance);
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "Usage: %s path/to/stories15M.bin\n", argv[0]);
    return EXIT_FAILURE;
  }
  cuda_transformer::Llama2CheckpointModel model;
  std::string error;
  const auto status = cuda_transformer::llama2_checkpoint_load(argv[1], &model, &error);
  if (status != cuda_transformer::Llama2CheckpointStatus::kSuccess) {
    std::fprintf(stderr, "checkpoint load failed (%s): %s\n",
                 cuda_transformer::llama2_checkpoint_status_string(status),
                 error.c_str());
    return EXIT_FAILURE;
  }
  cuda_transformer::TinyModelConfig config;
  const std::size_t sequence = model.checkpoint_config.max_sequence_length < 5
                                   ? model.checkpoint_config.max_sequence_length
                                   : 5;
  if (cuda_transformer::llama2_checkpoint_tiny_model_config(
          model.checkpoint_config, sequence, &config) !=
      cuda_transformer::Llama2CheckpointStatus::kSuccess) {
    std::fputs("could not create runtime config for checkpoint\n", stderr);
    return EXIT_FAILURE;
  }
  std::printf("loaded legacy llama2.c checkpoint: dim=%zu hidden_dim=%zu layers=%zu "
              "heads=%zu vocab=%zu max_seq=%zu classifier=%s\n",
              model.checkpoint_config.dim, model.checkpoint_config.hidden_dim,
              model.checkpoint_config.n_layers, model.checkpoint_config.n_heads,
              model.checkpoint_config.vocabulary_size,
              model.checkpoint_config.max_sequence_length,
              model.checkpoint_config.shared_classifier ? "tied" : "untied");

  std::vector<int> tokens(sequence);
  for (std::size_t index = 0; index < sequence; ++index)
    tokens[index] = static_cast<int>((index + 1) % config.vocabulary_size);
  std::vector<float> cpu_logits(sequence * config.vocabulary_size);
  if (!cuda_transformer::tiny_model_forward_cpu(tokens.data(), model.host_weights,
                                                config, cpu_logits.data())) {
    std::fputs("CPU real-checkpoint forward failed\n", stderr);
    return EXIT_FAILURE;
  }

  cublasHandle_t handle = nullptr;
  cuda_transformer::TinyModelWorkspace workspace;
  float* device_logits = nullptr;
  auto cleanup = [&] {
    bool ok = true;
    if (device_logits != nullptr && !CTR_CUDA_CHECK(cudaFree(device_logits)))
      ok = false;
    if (!CTR_CUDA_CHECK(cuda_transformer::tiny_model_workspace_destroy(&workspace)))
      ok = false;
    if (handle != nullptr && !CTR_CUBLAS_CHECK(cublasDestroy(handle)))
      ok = false;
    return ok;
  };
  if (!CTR_CUBLAS_CHECK(cublasCreate(&handle)) ||
      !CTR_CUDA_CHECK(cuda_transformer::tiny_model_workspace_create(&workspace,
                                                                      config)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&device_logits,
                                 cpu_logits.size() * sizeof(float)))) {
    cleanup();
    return EXIT_FAILURE;
  }
  std::vector<float> gpu_logits(cpu_logits.size());
  const bool ok = CTR_CUBLAS_CHECK(
                      cuda_transformer::tiny_model_forward_host_tokens_cuda(
                          handle, tokens.data(), model.device_weights, &workspace,
                          device_logits)) &&
                  CTR_CUDA_CHECK(cudaDeviceSynchronize()) &&
                  CTR_CUDA_CHECK(cudaMemcpy(gpu_logits.data(), device_logits,
                                            gpu_logits.size() * sizeof(float),
                                            cudaMemcpyDeviceToHost)) &&
                  close_logits(cpu_logits, gpu_logits);
  if (ok)
    std::puts("CPU and CUDA logits agree for the supplied integer-token sequence.");
  return cleanup() && ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
