#pragma once
#include "cuda_transformer/kv_cache.h"
namespace cuda_transformer {
// Stores K after RoPE at cache.current_length; V is stored unchanged.
cudaError_t incremental_decode(KvCache *cache, const float *q, const float *k,
                               const float *v, float *output,
                               cudaStream_t stream = nullptr);
} // namespace cuda_transformer
