#pragma once
#include <cstddef>
#include <cuda_runtime.h>
namespace cuda_transformer {
inline constexpr int kPrimitiveThreads = 256;
void rmsnorm_cpu(const float *, const float *, float *, std::size_t,
                 std::size_t, float);
void softmax_cpu(const float *, float *, std::size_t, std::size_t);
void silu_cpu(const float *, float *, std::size_t);
void multiply_cpu(const float *, const float *, float *, std::size_t);
void residual_add_cpu(const float *, const float *, float *, std::size_t);
cudaError_t rmsnorm_cuda(const float *, const float *, float *, std::size_t,
                         std::size_t, float, cudaStream_t = nullptr);
cudaError_t softmax_cuda(const float *, float *, std::size_t, std::size_t,
                         cudaStream_t = nullptr);
cudaError_t silu_cuda(const float *, float *, std::size_t,
                      cudaStream_t = nullptr);
cudaError_t multiply_cuda(const float *, const float *, float *, std::size_t,
                          cudaStream_t = nullptr);
cudaError_t residual_add_cuda(const float *, const float *, float *,
                              std::size_t, cudaStream_t = nullptr);
} // namespace cuda_transformer
