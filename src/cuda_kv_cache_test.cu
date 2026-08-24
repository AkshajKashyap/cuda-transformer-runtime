#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/kv_cache.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
bool run_case(std::size_t heads, std::size_t max_seq, std::size_t dim) {
  cuda_transformer::KvCache cache;
  if (!CTR_CUDA_CHECK(
          cuda_transformer::kv_cache_create(&cache, 1, heads, max_seq, dim)))
    return false;
  bool ok = cache.current_length == 0;
  const std::size_t prefill = max_seq > 2 ? max_seq - 2 : 1;
  std::vector<float> k(heads * prefill * dim), v(k.size());
  for (std::size_t i = 0; i < k.size(); ++i) {
    k[i] = static_cast<float>(i);
    v[i] = -k[i] - 1.0F;
  }
  float *dk = nullptr, *dv = nullptr;
  if (ok && (!CTR_CUDA_CHECK(cudaMalloc(&dk, k.size() * sizeof(float))) ||
             !CTR_CUDA_CHECK(cudaMalloc(&dv, v.size() * sizeof(float))) ||
             !CTR_CUDA_CHECK(cudaMemcpy(dk, k.data(), k.size() * sizeof(float),
                                        cudaMemcpyHostToDevice)) ||
             !CTR_CUDA_CHECK(cudaMemcpy(dv, v.data(), v.size() * sizeof(float),
                                        cudaMemcpyHostToDevice)) ||
             !CTR_CUDA_CHECK(
                 cuda_transformer::kv_cache_prefill(&cache, dk, dv, prefill)) ||
             !CTR_CUDA_CHECK(cudaDeviceSynchronize())))
    ok = false;
  std::vector<float> got(heads * max_seq * dim), got_values(got.size());
  if (ok && !CTR_CUDA_CHECK(cudaMemcpy(got.data(), cache.keys,
                                       got.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost)))
    ok = false;
  for (std::size_t h = 0; h < heads && ok; ++h)
    for (std::size_t s = 0; s < prefill && ok; ++s)
      for (std::size_t d = 0; d < dim; ++d)
        if (got[(h * max_seq + s) * dim + d] != k[(h * prefill + s) * dim + d])
          ok = false;
  if (ok && !CTR_CUDA_CHECK(cudaMemcpy(got_values.data(), cache.values,
                                       got_values.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost)))
    ok = false;
  for (std::size_t h = 0; h < heads && ok; ++h)
    for (std::size_t s = 0; s < prefill && ok; ++s)
      for (std::size_t d = 0; d < dim; ++d)
        if (got_values[(h * max_seq + s) * dim + d] !=
            v[(h * prefill + s) * dim + d])
          ok = false;
  const std::vector<float> before_append = got;
  const std::vector<float> before_append_values = got_values;
  std::vector<float> one(heads * dim), one_v(heads * dim);
  for (std::size_t i = 0; i < one.size(); ++i) {
    one[i] = 1000 + i;
    one_v[i] = -1000.0F - static_cast<float>(i);
  }
  float *d1 = nullptr, *d1v = nullptr;
  if (ok &&
      (!CTR_CUDA_CHECK(cudaMalloc(&d1, one.size() * sizeof(float))) ||
       !CTR_CUDA_CHECK(cudaMalloc(&d1v, one.size() * sizeof(float))) ||
       !CTR_CUDA_CHECK(cudaMemcpy(d1, one.data(), one.size() * sizeof(float),
                                  cudaMemcpyHostToDevice)) ||
       !CTR_CUDA_CHECK(cudaMemcpy(d1v, one_v.data(),
                                  one_v.size() * sizeof(float),
                                  cudaMemcpyHostToDevice)) ||
       !CTR_CUDA_CHECK(cuda_transformer::kv_cache_append(&cache, d1, d1v)) ||
       !CTR_CUDA_CHECK(cudaDeviceSynchronize())))
    ok = false;
  if (ok && !CTR_CUDA_CHECK(cudaMemcpy(got.data(), cache.keys,
                                       got.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost)))
    ok = false;
  for (std::size_t h = 0; h < heads && ok; ++h)
    for (std::size_t d = 0; d < dim; ++d)
      if (got[(h * max_seq + prefill) * dim + d] != one[h * dim + d])
        ok = false;
  if (ok && !CTR_CUDA_CHECK(cudaMemcpy(got_values.data(), cache.values,
                                       got_values.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost)))
    ok = false;
  for (std::size_t h = 0; h < heads && ok; ++h)
    for (std::size_t s = 0; s < prefill && ok; ++s)
      for (std::size_t d = 0; d < dim; ++d)
        if (got[(h * max_seq + s) * dim + d] !=
                before_append[(h * max_seq + s) * dim + d] ||
            got_values[(h * max_seq + s) * dim + d] !=
                before_append_values[(h * max_seq + s) * dim + d])
          ok = false;
  for (std::size_t h = 0; h < heads && ok; ++h)
    for (std::size_t d = 0; d < dim; ++d)
      if (got_values[(h * max_seq + prefill) * dim + d] != one_v[h * dim + d])
        ok = false;
  if (ok &&
          !CTR_CUDA_CHECK(cuda_transformer::kv_cache_append(&cache, d1, d1v)) ||
      !CTR_CUDA_CHECK(cudaDeviceSynchronize()))
    ok = false;
  if (cache.current_length != max_seq ||
      cuda_transformer::kv_cache_append(&cache, d1, d1v) !=
          cudaErrorInvalidValue)
    ok = false;
  cuda_transformer::kv_cache_reset(&cache);
  if (cache.current_length != 0)
    ok = false;
  std::vector<float> full(heads * max_seq * dim, 42.0F);
  float *dfull = nullptr;
  if (ok && (!CTR_CUDA_CHECK(cudaMalloc(&dfull, full.size() * sizeof(float))) ||
             !CTR_CUDA_CHECK(cudaMemcpy(dfull, full.data(),
                                        full.size() * sizeof(float),
                                        cudaMemcpyHostToDevice)) ||
             !CTR_CUDA_CHECK(cuda_transformer::kv_cache_prefill(
                 &cache, dfull, dfull, max_seq)) ||
             !CTR_CUDA_CHECK(cudaDeviceSynchronize())))
    ok = false;
  if (cache.current_length != max_seq ||
      cuda_transformer::kv_cache_prefill(&cache, d1, d1v, max_seq + 1) !=
          cudaErrorInvalidValue)
    ok = false;
  cuda_transformer::kv_cache_reset(&cache);
  if (ok && !CTR_CUDA_CHECK(cuda_transformer::kv_cache_append(&cache, d1, d1v)))
    ok = false;
  if (dk)
    CTR_CUDA_CHECK(cudaFree(dk));
  if (dv)
    CTR_CUDA_CHECK(cudaFree(dv));
  if (d1)
    CTR_CUDA_CHECK(cudaFree(d1));
  if (d1v)
    CTR_CUDA_CHECK(cudaFree(d1v));
  if (dfull)
    CTR_CUDA_CHECK(cudaFree(dfull));
  return CTR_CUDA_CHECK(cuda_transformer::kv_cache_destroy(&cache)) && ok;
}
} // namespace
int main() {
  cuda_transformer::KvCache invalid;
  if (cuda_transformer::kv_cache_create(&invalid, 1, 1, 0, 4) !=
      cudaErrorInvalidValue)
    return EXIT_FAILURE;
  if (!run_case(1, 4, 4) || !run_case(2, 7, 8) || !run_case(4, 32, 64))
    return EXIT_FAILURE;
  std::puts("KV cache tests passed.");
}
