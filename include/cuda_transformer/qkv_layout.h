#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cuda_transformer {

void token_major_to_head_major_cpu(const float *input, float *output,
                                   std::size_t sequence, std::size_t heads,
                                   std::size_t head_dim);
void head_major_to_token_major_cpu(const float *input, float *output,
                                   std::size_t sequence, std::size_t heads,
                                   std::size_t head_dim);

cudaError_t token_major_to_head_major_cuda(const float *input, float *output,
                                           std::size_t sequence,
                                           std::size_t heads,
                                           std::size_t head_dim,
                                           cudaStream_t stream = nullptr);
cudaError_t head_major_to_token_major_cuda(const float *input, float *output,
                                           std::size_t sequence,
                                           std::size_t heads,
                                           std::size_t head_dim,
                                           cudaStream_t stream = nullptr);
} // namespace cuda_transformer
