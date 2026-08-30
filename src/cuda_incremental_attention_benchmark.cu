#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/incremental_attention.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
struct Config {
  std::size_t heads, history, dim;
};
int launches(std::size_t history) {
  if (history <= 256)
    return 500;
  if (history <= 512)
    return 300;
  if (history <= 1024)
    return 200;
  return 100;
}
bool run(Config c) {
  cuda_transformer::KvCache cache;
  cuda_transformer::IncrementalAttentionWorkspace workspace;
  if (!CTR_CUDA_CHECK(cuda_transformer::kv_cache_create(&cache, 1, c.heads,
                                                        c.history + 1, c.dim)))
    return false;
  const std::size_t history_elements = c.heads * c.history * c.dim,
                    token_elements = c.heads * c.dim;
  std::vector<float> history(history_elements, 0.125F),
      token(token_elements, 0.25F);
  float *dk = nullptr, *dv = nullptr, *dq = nullptr, *do_ = nullptr;
  auto clean = [&](bool ok) {
    for (auto p : {dk, dv, dq, do_})
      if (p && !CTR_CUDA_CHECK(cudaFree(p)))
        ok = false;
    if (!CTR_CUDA_CHECK(
            cuda_transformer::incremental_attention_workspace_destroy(&workspace)))
      ok = false;
    return CTR_CUDA_CHECK(cuda_transformer::kv_cache_destroy(&cache)) && ok;
  };
  if (!CTR_CUDA_CHECK(cudaMalloc(&dk, history_elements * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&dv, history_elements * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&dq, token_elements * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&do_, token_elements * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMemcpy(dk, history.data(),
                                 history_elements * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(dv, history.data(),
                                 history_elements * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(dq, token.data(),
                                 token_elements * sizeof(float),
                                 cudaMemcpyHostToDevice)))
    return clean(false);
  if (!CTR_CUDA_CHECK(cuda_transformer::incremental_attention_workspace_create(
          &workspace, c.heads, c.dim, c.history + 1)))
    return clean(false);
  if (!CTR_CUDA_CHECK(
          cuda_transformer::kv_cache_prefill(&cache, dk, dv, c.history)) ||
      !CTR_CUDA_CHECK(cudaDeviceSynchronize()))
    return clean(false);
  constexpr int warmups = 10, batches = 9;
  const int count = launches(c.history);
  for (int i = 0; i < warmups; i++) {
    cache.current_length = c.history;
    if (!CTR_CUDA_CHECK(
            cuda_transformer::incremental_decode_with_workspace(
                &cache, dq, dq, dq, do_, &workspace)))
      return clean(false);
  }
  if (!CTR_CUDA_CHECK(cudaDeviceSynchronize()))
    return clean(false);
  cudaEvent_t a = nullptr, b = nullptr;
  if (!CTR_CUDA_CHECK(cudaEventCreate(&a)) ||
      !CTR_CUDA_CHECK(cudaEventCreate(&b)))
    return clean(false);
  std::vector<float> samples;
  for (int batch = 0; batch < batches; batch++) {
    if (!CTR_CUDA_CHECK(cudaEventRecord(a)))
      return clean(false);
    for (int i = 0; i < count; i++) {
      cache.current_length = c.history;
      if (!CTR_CUDA_CHECK(
              cuda_transformer::incremental_decode_with_workspace(
                  &cache, dq, dq, dq, do_, &workspace)))
        return clean(false);
    }
    if (!CTR_CUDA_CHECK(cudaEventRecord(b)) ||
        !CTR_CUDA_CHECK(cudaEventSynchronize(b)))
      return clean(false);
    float ms = 0;
    if (!CTR_CUDA_CHECK(cudaEventElapsedTime(&ms, a, b)))
      return clean(false);
    samples.push_back(ms / count);
  }
  std::sort(samples.begin(), samples.end());
  std::printf(
      "heads=%zu dim=%zu history=%zu: %.5f ms median (%d batches x %d)\n",
      c.heads, c.dim, c.history, samples[batches / 2], batches, count);
  CTR_CUDA_CHECK(cudaEventDestroy(a));
  CTR_CUDA_CHECK(cudaEventDestroy(b));
  return clean(true);
}
} // namespace
int main() {
  for (auto c : std::array<Config, 12>{{{4, 64, 64},
                                        {4, 128, 64},
                                        {4, 256, 64},
                                        {4, 512, 64},
                                        {4, 1024, 64},
                                        {4, 2048, 64},
                                        {8, 64, 64},
                                        {8, 128, 64},
                                        {8, 256, 64},
                                        {8, 512, 64},
                                        {8, 1024, 64},
                                        {8, 2048, 64}}})
    if (!run(c))
      return EXIT_FAILURE;
}
