#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cuda_transformer {

inline constexpr int kThreadsPerBlock = 256;

void vector_add_cpu(const float* a, const float* b, float* out, std::size_t count);
void scalar_multiply_cpu(const float* input, float alpha, float* output, std::size_t count);
float sum_cpu(const float* input, std::size_t count);

cudaError_t vector_add_cuda(const float* a,
                            const float* b,
                            float* out,
                            std::size_t count,
                            cudaStream_t stream = nullptr);
cudaError_t scalar_multiply_cuda(const float* input,
                                 float alpha,
                                 float* output,
                                 std::size_t count,
                                 cudaStream_t stream = nullptr);

// A simple single-output reduction using atomic addition. It is useful as a
// learning reference, but its summation order is not deterministic.
cudaError_t sum_atomic_cuda(const float* input,
                            std::size_t count,
                            float* output,
                            cudaStream_t stream = nullptr);

// Reduces each block in shared memory, writes one partial sum per block, then
// repeatedly reduces those partial sums until one result remains in output.
cudaError_t sum_shared_cuda(const float* input,
                            std::size_t count,
                            float* output,
                            cudaStream_t stream = nullptr);

}  // namespace cuda_transformer
