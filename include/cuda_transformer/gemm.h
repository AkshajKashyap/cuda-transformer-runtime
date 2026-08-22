#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace cuda_transformer {

inline constexpr int kGemmTileSize = 16;

// All matrices are row-major: element (row, column) of a matrix with `columns`
// columns is stored at data[row * columns + column].
void gemm_cpu(const float* a,
              const float* b,
              float* c,
              std::size_t m,
              std::size_t k,
              std::size_t n);

cudaError_t gemm_naive_cuda(const float* a,
                            const float* b,
                            float* c,
                            std::size_t m,
                            std::size_t k,
                            std::size_t n,
                            cudaStream_t stream = nullptr);

cudaError_t gemm_tiled_cuda(const float* a,
                            const float* b,
                            float* c,
                            std::size_t m,
                            std::size_t k,
                            std::size_t n,
                            cudaStream_t stream = nullptr);

// Computes row-major C = A * B by viewing the buffers as column-major
// transposes and asking cuBLAS to compute C^T = B^T * A^T.
cublasStatus_t gemm_cublas_row_major(cublasHandle_t handle,
                                     const float* a,
                                     const float* b,
                                     float* c,
                                     std::size_t m,
                                     std::size_t k,
                                     std::size_t n,
                                     cudaStream_t stream = nullptr);

}  // namespace cuda_transformer
