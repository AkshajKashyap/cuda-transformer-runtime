#include "cuda_transformer/cublas_check.h"
#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/decoder_block.h"
#include "cuda_transformer/incremental_decoder_block.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr std::size_t kHidden = 256;
constexpr std::size_t kHeads = 4;
constexpr std::size_t kHeadDim = 64;
constexpr std::size_t kIntermediate = 512;
constexpr float kEpsilon = 1.0e-5F;
constexpr float kOutputAbsoluteTolerance = 1.0e-3F;
constexpr float kOutputRelativeTolerance = 2.0e-4F;
constexpr int kWarmups = 10;
constexpr int kBatches = 9;

struct DeviceWeights {
  float* attention_norm = nullptr;
  float* mlp_norm = nullptr;
  float* wq = nullptr;
  float* wk = nullptr;
  float* wv = nullptr;
  float* wo = nullptr;
  float* w_gate = nullptr;
  float* w_up = nullptr;
  float* w_down = nullptr;
};

int incremental_launches(std::size_t history) {
  if (history <= 128)
    return 500;
  if (history <= 256)
    return 300;
  if (history <= 512)
    return 200;
  return 100;
}

int full_launches(std::size_t history) {
  if (history <= 32)
    return 200;
  if (history <= 64)
    return 100;
  if (history <= 128)
    return 50;
  if (history <= 256)
    return 20;
  if (history <= 512)
    return 10;
  return 5;
}

template <typename Launch>
bool median_latency_ms(Launch launch, int launches_per_batch,
                       float* median_ms) {
  for (int i = 0; i < kWarmups; ++i) {
    if (!launch())
      return false;
  }
  if (!CTR_CUDA_CHECK(cudaDeviceSynchronize()))
    return false;

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (!CTR_CUDA_CHECK(cudaEventCreate(&start)) ||
      !CTR_CUDA_CHECK(cudaEventCreate(&stop))) {
    if (start != nullptr)
      CTR_CUDA_CHECK(cudaEventDestroy(start));
    if (stop != nullptr)
      CTR_CUDA_CHECK(cudaEventDestroy(stop));
    return false;
  }

  bool ok = true;
  std::vector<float> samples;
  samples.reserve(kBatches);
  for (int batch = 0; batch < kBatches && ok; ++batch) {
    ok = CTR_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < launches_per_batch && ok; ++i)
      ok = launch();
    if (ok)
      ok = CTR_CUDA_CHECK(cudaEventRecord(stop)) &&
           CTR_CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0F;
    if (ok)
      ok = CTR_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    if (ok)
      samples.push_back(elapsed_ms / static_cast<float>(launches_per_batch));
  }
  if (!CTR_CUDA_CHECK(cudaEventDestroy(start)))
    ok = false;
  if (!CTR_CUDA_CHECK(cudaEventDestroy(stop)))
    ok = false;
  if (!ok)
    return false;

  std::sort(samples.begin(), samples.end());
  *median_ms = samples[samples.size() / 2];
  return true;
}

bool check_output(const std::vector<float>& incremental,
                  const std::vector<float>& full, std::size_t history) {
  if (incremental.size() != full.size()) {
    std::fprintf(stderr, "sanity output size mismatch at history %zu\n", history);
    return false;
  }
  for (std::size_t i = 0; i < full.size(); ++i) {
    const float tolerance = kOutputAbsoluteTolerance +
                            kOutputRelativeTolerance *
                                fmaxf(1.0F, std::fabs(full[i]));
    if (!std::isfinite(incremental[i]) || !std::isfinite(full[i]) ||
        std::fabs(incremental[i] - full[i]) > tolerance) {
      std::fprintf(stderr,
                   "sanity mismatch: history=%zu index=%zu incremental=%.8g "
                   "full=%.8g error=%.8g tolerance=%.8g\n",
                   history, i, incremental[i], full[i],
                   std::fabs(incremental[i] - full[i]), tolerance);
      return false;
    }
  }
  return true;
}

bool run_history(cublasHandle_t handle, const DeviceWeights& device_weights,
                 const float* device_input, std::size_t history) {
  using namespace cuda_transformer;

  const std::size_t sequence = history + 1;
  const std::size_t activation_count = sequence * kHidden;
  const std::size_t activation_bytes = activation_count * sizeof(float);
  const std::size_t token_bytes = kHidden * sizeof(float);
  const DecoderBlockConfig full_config{
      {sequence, kHidden, kHeads, kHeadDim, kEpsilon},
      {sequence, kHidden, kIntermediate, kEpsilon},
  };
  const IncrementalDecoderBlockConfig incremental_config{
      kHidden, kHeads, kHeadDim, kIntermediate, kEpsilon, kEpsilon, sequence,
  };
  const DecoderBlockWeights full_weights{
      {device_weights.attention_norm, device_weights.wq, device_weights.wk,
       device_weights.wv, device_weights.wo},
      {device_weights.mlp_norm, device_weights.w_gate, device_weights.w_up,
       device_weights.w_down},
  };
  const IncrementalDecoderBlockWeights incremental_weights{
      {device_weights.attention_norm, device_weights.wq, device_weights.wk,
       device_weights.wv, device_weights.wo},
      {device_weights.mlp_norm, device_weights.w_gate, device_weights.w_up,
       device_weights.w_down},
  };

  DecoderBlockWorkspace full_workspace;
  IncrementalDecoderBlockWorkspace incremental_workspace;
  KvCache cache;
  float* full_output = nullptr;
  float* incremental_output = nullptr;
  auto cleanup = [&](bool ok) {
    if (full_output != nullptr && !CTR_CUDA_CHECK(cudaFree(full_output)))
      ok = false;
    if (incremental_output != nullptr &&
        !CTR_CUDA_CHECK(cudaFree(incremental_output)))
      ok = false;
    if (!CTR_CUDA_CHECK(decoder_block_workspace_destroy(&full_workspace)))
      ok = false;
    if (!CTR_CUDA_CHECK(
            incremental_decoder_block_workspace_destroy(&incremental_workspace)))
      ok = false;
    if (!CTR_CUDA_CHECK(kv_cache_destroy(&cache)))
      ok = false;
    return ok;
  };

  if (!CTR_CUDA_CHECK(decoder_block_workspace_create(&full_workspace,
                                                      full_config)) ||
      !CTR_CUDA_CHECK(incremental_decoder_block_workspace_create(
          &incremental_workspace, incremental_config)) ||
      !CTR_CUDA_CHECK(
          kv_cache_create(&cache, 1, kHeads, sequence, kHeadDim)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&full_output, activation_bytes)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&incremental_output, token_bytes)))
    return cleanup(false);

  // Build the exact fixed history once. This work is intentionally excluded
  // from every timed incremental decode launch.
  for (std::size_t token = 0; token < history; ++token) {
    if (!CTR_CUBLAS_CHECK(incremental_decoder_block_cuda(
            handle, device_input + token * kHidden, incremental_weights, &cache,
            &incremental_workspace, incremental_output)))
      return cleanup(false);
  }
  if (!CTR_CUDA_CHECK(cudaDeviceSynchronize()) ||
      cache.current_length != history) {
    std::fprintf(stderr, "cache prefill failed for history %zu\n", history);
    return cleanup(false);
  }

  // One untimed reference comparison confirms that both timed paths start from
  // the same prefix and produce the same final token.
  cache.current_length = history;
  if (!CTR_CUBLAS_CHECK(incremental_decoder_block_cuda(
          handle, device_input + history * kHidden, incremental_weights, &cache,
          &incremental_workspace, incremental_output)) ||
      !CTR_CUBLAS_CHECK(decoder_block_cuda(handle, device_input, full_weights,
                                            &full_workspace, full_output)) ||
      !CTR_CUDA_CHECK(cudaDeviceSynchronize()))
    return cleanup(false);
  std::vector<float> incremental_host(kHidden);
  std::vector<float> full_host(kHidden);
  if (!CTR_CUDA_CHECK(cudaMemcpy(incremental_host.data(), incremental_output,
                                 token_bytes, cudaMemcpyDeviceToHost)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(full_host.data(),
                                 full_output + history * kHidden, token_bytes,
                                 cudaMemcpyDeviceToHost)) ||
      !check_output(incremental_host, full_host, history))
    return cleanup(false);

  const auto full_launch = [&] {
    return CTR_CUBLAS_CHECK(decoder_block_cuda(handle, device_input,
                                                full_weights, &full_workspace,
                                                full_output));
  };
  const auto incremental_launch = [&] {
    // Only the logical length changes between launches. Position `history` is
    // outside the valid prefix and is deterministically overwritten.
    cache.current_length = history;
    return CTR_CUBLAS_CHECK(incremental_decoder_block_cuda(
        handle, device_input + history * kHidden, incremental_weights, &cache,
        &incremental_workspace, incremental_output));
  };

  float full_median_ms = 0.0F;
  float incremental_median_ms = 0.0F;
  if (!median_latency_ms(full_launch, full_launches(history), &full_median_ms) ||
      !median_latency_ms(incremental_launch, incremental_launches(history),
                         &incremental_median_ms))
    return cleanup(false);

  std::printf("%7zu %12zu %15.5f %15.5f %9.2fx %10d %10d\n", history,
              sequence, full_median_ms, incremental_median_ms,
              full_median_ms / incremental_median_ms, full_launches(history),
              incremental_launches(history));
  return cleanup(true);
}

bool allocate_and_copy(float** device, const std::vector<float>& host) {
  return CTR_CUDA_CHECK(cudaMalloc(device, host.size() * sizeof(float))) &&
         CTR_CUDA_CHECK(cudaMemcpy(*device, host.data(),
                                   host.size() * sizeof(float),
                                   cudaMemcpyHostToDevice));
}

bool setup_weights(DeviceWeights* device_weights) {
  const std::size_t square_weight_count = kHidden * kHidden;
  const std::size_t gate_weight_count = kHidden * kIntermediate;
  const std::size_t down_weight_count = kIntermediate * kHidden;
  std::vector<float> attention_norm(kHidden), mlp_norm(kHidden),
      wq(square_weight_count), wk(square_weight_count), wv(square_weight_count),
      wo(square_weight_count), w_gate(gate_weight_count), w_up(gate_weight_count),
      w_down(down_weight_count);
  for (std::size_t i = 0; i < kHidden; ++i) {
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

  return allocate_and_copy(&device_weights->attention_norm, attention_norm) &&
         allocate_and_copy(&device_weights->mlp_norm, mlp_norm) &&
         allocate_and_copy(&device_weights->wq, wq) &&
         allocate_and_copy(&device_weights->wk, wk) &&
         allocate_and_copy(&device_weights->wv, wv) &&
         allocate_and_copy(&device_weights->wo, wo) &&
         allocate_and_copy(&device_weights->w_gate, w_gate) &&
         allocate_and_copy(&device_weights->w_up, w_up) &&
         allocate_and_copy(&device_weights->w_down, w_down);
}

bool free_weights(DeviceWeights* weights) {
  bool ok = true;
  for (float* pointer : {weights->attention_norm, weights->mlp_norm, weights->wq,
                         weights->wk, weights->wv, weights->wo, weights->w_gate,
                         weights->w_up, weights->w_down}) {
    if (pointer != nullptr && !CTR_CUDA_CHECK(cudaFree(pointer)))
      ok = false;
  }
  *weights = {};
  return ok;
}

}  // namespace

int main() {
  constexpr std::array<std::size_t, 7> kHistories{16,  32,  64, 128,
                                                   256, 512, 1024};
  constexpr std::size_t kMaxSequence = kHistories.back() + 1;

  std::vector<float> input(kMaxSequence * kHidden);
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<float>(static_cast<int>((i * 5 + 3) % 37) - 18) /
               32.0F;

  cublasHandle_t handle = nullptr;
  float* device_input = nullptr;
  DeviceWeights device_weights;
  bool ok = CTR_CUBLAS_CHECK(cublasCreate(&handle)) &&
            allocate_and_copy(&device_input, input) &&
            setup_weights(&device_weights);
  if (ok) {
    std::puts("Decoder-block full-prefix vs cached incremental decode (FP32)");
    std::puts(" history full_tokens full_median_ms inc_median_ms   speedup "
              "full_launch inc_launch");
    for (const std::size_t history : kHistories) {
      if (!run_history(handle, device_weights, device_input, history)) {
        ok = false;
        break;
      }
    }
  }
  if (!free_weights(&device_weights))
    ok = false;
  if (device_input != nullptr && !CTR_CUDA_CHECK(cudaFree(device_input)))
    ok = false;
  if (handle != nullptr && !CTR_CUBLAS_CHECK(cublasDestroy(handle)))
    ok = false;
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
