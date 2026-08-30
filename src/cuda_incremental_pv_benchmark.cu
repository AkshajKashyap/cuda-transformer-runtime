#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/incremental_attention.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
constexpr std::size_t kHeads = 4;
constexpr std::size_t kHeadDim = 64;
constexpr int kWarmups = 10;
constexpr int kBatches = 9;

int launches(std::size_t history) {
  if (history <= 128)
    return 1000;
  if (history <= 256)
    return 500;
  if (history <= 512)
    return 200;
  return 100;
}

template <typename Launch>
bool median_us(Launch launch, int count, float* result) {
  for (int i = 0; i < kWarmups; ++i)
    if (!launch())
      return false;
  if (!CTR_CUDA_CHECK(cudaDeviceSynchronize()))
    return false;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (!CTR_CUDA_CHECK(cudaEventCreate(&start)) ||
      !CTR_CUDA_CHECK(cudaEventCreate(&stop)))
    return false;
  bool ok = true;
  std::vector<float> samples;
  for (int batch = 0; batch < kBatches && ok; ++batch) {
    ok = CTR_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < count && ok; ++i)
      ok = launch();
    if (ok)
      ok = CTR_CUDA_CHECK(cudaEventRecord(stop)) &&
           CTR_CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.0F;
    if (ok)
      ok = CTR_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    if (ok)
      samples.push_back(1000.0F * milliseconds / count);
  }
  if (!CTR_CUDA_CHECK(cudaEventDestroy(start)))
    ok = false;
  if (!CTR_CUDA_CHECK(cudaEventDestroy(stop)))
    ok = false;
  if (!ok)
    return false;
  std::sort(samples.begin(), samples.end());
  *result = samples[samples.size() / 2];
  return true;
}

bool run(std::size_t history) {
  const std::size_t probability_count = kHeads * history;
  const std::size_t value_count = kHeads * history * kHeadDim;
  const std::size_t output_count = kHeads * kHeadDim;
  std::vector<float> probabilities(probability_count);
  std::vector<float> values(value_count);
  for (std::size_t h = 0; h < kHeads; ++h)
    for (std::size_t k = 0; k < history; ++k) {
      probabilities[h * history + k] = 1.0F / history;
      for (std::size_t d = 0; d < kHeadDim; ++d)
        values[(h * history + k) * kHeadDim + d] =
            static_cast<float>((h * 17 + k * 3 + d) % 31) / 31.0F;
    }
  float *dp = nullptr, *dv = nullptr, *do_ = nullptr;
  auto cleanup = [&](bool ok) {
    for (float* pointer : {dp, dv, do_})
      if (pointer != nullptr && !CTR_CUDA_CHECK(cudaFree(pointer)))
        ok = false;
    return ok;
  };
  if (!CTR_CUDA_CHECK(cudaMalloc(&dp, probability_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&dv, value_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&do_, output_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMemcpy(dp, probabilities.data(),
                                 probability_count * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(dv, values.data(), value_count * sizeof(float),
                                 cudaMemcpyHostToDevice)))
    return cleanup(false);

  const auto serial = [&] {
    return CTR_CUDA_CHECK(cuda_transformer::incremental_probability_value_serial_cuda(
        dp, dv, do_, kHeads, history, history, kHeadDim));
  };
  const auto parallel = [&] {
    return CTR_CUDA_CHECK(cuda_transformer::incremental_probability_value_parallel_cuda(
        dp, dv, do_, kHeads, history, history, kHeadDim));
  };
  float serial_us = 0.0F;
  float parallel_us = 0.0F;
  if (!median_us(serial, launches(history), &serial_us) ||
      !median_us(parallel, launches(history), &parallel_us))
    return cleanup(false);
  std::printf("%7zu %9.3f %13.3f %8.2fx\n", history, serial_us, parallel_us,
              serial_us / parallel_us);
  return cleanup(true);
}
}  // namespace

int main() {
  std::puts("history    old_us  optimized_us  speedup");
  for (const std::size_t history :
       std::array<std::size_t, 8>{16, 32, 64, 128, 256, 512, 1024, 2048})
    if (!run(history))
      return EXIT_FAILURE;
}
