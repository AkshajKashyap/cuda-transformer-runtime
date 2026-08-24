#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cuda_transformer {

// Owns fixed-capacity FP32 buffers in [batch, heads, max_sequence, head_dim].
// Copying is disabled to prevent two cache objects freeing the same allocation.
struct KvCache {
  float *keys = nullptr;
  float *values = nullptr;
  std::size_t batch = 0;
  std::size_t heads = 0;
  std::size_t max_sequence = 0;
  std::size_t head_dim = 0;
  std::size_t current_length = 0;

  KvCache() = default;
  KvCache(const KvCache &) = delete;
  KvCache &operator=(const KvCache &) = delete;
};

cudaError_t kv_cache_create(KvCache *cache, std::size_t batch,
                            std::size_t heads, std::size_t max_sequence,
                            std::size_t head_dim);
cudaError_t kv_cache_destroy(KvCache *cache);
cudaError_t kv_cache_prefill(KvCache *cache, const float *keys,
                             const float *values, std::size_t sequence_length,
                             cudaStream_t stream = nullptr);
// current_length advances after the kernel launch is accepted. CUDA execution
// remains asynchronous; callers that need execution completion must synchronize
// their stream before consuming the newly written cache entries.
cudaError_t kv_cache_append(KvCache *cache, const float *key,
                            const float *value, cudaStream_t stream = nullptr);
void kv_cache_reset(KvCache *cache);

} // namespace cuda_transformer
