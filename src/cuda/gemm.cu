#include "cuda_transformer/gemm.h"

#include <cstddef>
#include <limits>

namespace cuda_transformer {
namespace {

bool dimensions_fit_cublas_int(std::size_t m, std::size_t k, std::size_t n) {
    constexpr auto kMaxInt = static_cast<std::size_t>(std::numeric_limits<int>::max());
    return m <= kMaxInt && k <= kMaxInt && n <= kMaxInt;
}

__global__ void gemm_naive_kernel(const float* a,
                                  const float* b,
                                  float* c,
                                  std::size_t m,
                                  std::size_t k,
                                  std::size_t n) {
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const std::size_t column = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= m || column >= n) {
        return;
    }

    float sum = 0.0F;
    for (std::size_t inner = 0; inner < k; ++inner) {
        sum += a[row * k + inner] * b[inner * n + column];
    }
    c[row * n + column] = sum;
}

__global__ void gemm_tiled_kernel(const float* a,
                                  const float* b,
                                  float* c,
                                  std::size_t m,
                                  std::size_t k,
                                  std::size_t n) {
    __shared__ float a_tile[kGemmTileSize][kGemmTileSize];
    __shared__ float b_tile[kGemmTileSize][kGemmTileSize];

    const unsigned int local_row = threadIdx.y;
    const unsigned int local_column = threadIdx.x;
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * kGemmTileSize + local_row;
    const std::size_t column =
        static_cast<std::size_t>(blockIdx.x) * kGemmTileSize + local_column;
    float sum = 0.0F;

    const std::size_t tile_count = (k + kGemmTileSize - 1) / kGemmTileSize;
    for (std::size_t tile = 0; tile < tile_count; ++tile) {
        const std::size_t a_column = tile * kGemmTileSize + local_column;
        const std::size_t b_row = tile * kGemmTileSize + local_row;
        a_tile[local_row][local_column] = (row < m && a_column < k)
                                               ? a[row * k + a_column]
                                               : 0.0F;
        b_tile[local_row][local_column] = (b_row < k && column < n)
                                               ? b[b_row * n + column]
                                               : 0.0F;
        __syncthreads();

        for (int inner = 0; inner < kGemmTileSize; ++inner) {
            sum += a_tile[local_row][inner] * b_tile[inner][local_column];
        }
        // Do not overwrite either tile until every thread has consumed it.
        __syncthreads();
    }

    if (row < m && column < n) {
        c[row * n + column] = sum;
    }
}

dim3 gemm_block() {
    return dim3(kGemmTileSize, kGemmTileSize);
}

dim3 gemm_grid(std::size_t m, std::size_t n) {
    return dim3(static_cast<unsigned int>((n + kGemmTileSize - 1) / kGemmTileSize),
                static_cast<unsigned int>((m + kGemmTileSize - 1) / kGemmTileSize));
}

}  // namespace

void gemm_cpu(const float* a,
              const float* b,
              float* c,
              std::size_t m,
              std::size_t k,
              std::size_t n) {
    for (std::size_t row = 0; row < m; ++row) {
        for (std::size_t column = 0; column < n; ++column) {
            float sum = 0.0F;
            for (std::size_t inner = 0; inner < k; ++inner) {
                sum += a[row * k + inner] * b[inner * n + column];
            }
            c[row * n + column] = sum;
        }
    }
}

cudaError_t gemm_naive_cuda(const float* a,
                            const float* b,
                            float* c,
                            std::size_t m,
                            std::size_t k,
                            std::size_t n,
                            cudaStream_t stream) {
    if (m == 0 || k == 0 || n == 0) {
        return cudaErrorInvalidValue;
    }

    gemm_naive_kernel<<<gemm_grid(m, n), gemm_block(), 0, stream>>>(a, b, c, m, k, n);
    return cudaGetLastError();
}

cudaError_t gemm_tiled_cuda(const float* a,
                            const float* b,
                            float* c,
                            std::size_t m,
                            std::size_t k,
                            std::size_t n,
                            cudaStream_t stream) {
    if (m == 0 || k == 0 || n == 0) {
        return cudaErrorInvalidValue;
    }

    gemm_tiled_kernel<<<gemm_grid(m, n), gemm_block(), 0, stream>>>(a, b, c, m, k, n);
    return cudaGetLastError();
}

cublasStatus_t gemm_cublas_row_major(cublasHandle_t handle,
                                     const float* a,
                                     const float* b,
                                     float* c,
                                     std::size_t m,
                                     std::size_t k,
                                     std::size_t n,
                                     cudaStream_t stream) {
    if (m == 0 || k == 0 || n == 0 || !dimensions_fit_cublas_int(m, k, n)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    // Keep this milestone's cuBLAS reference in FP32 rather than allowing
    // Ampere's reduced-precision TF32 tensor-core path.
    cublasStatus_t status = cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = cublasSetStream(handle, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }

    constexpr float kAlpha = 1.0F;
    constexpr float kBeta = 0.0F;
    // A row-major MxK buffer A is the same bytes as a column-major KxM A^T.
    // Thus C^T (column-major NxM) = B^T (NxK) * A^T (KxM).
    return cublasSgemm(handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        static_cast<int>(n),
                        static_cast<int>(m),
                        static_cast<int>(k),
                        &kAlpha,
                        b,
                        static_cast<int>(n),
                        a,
                        static_cast<int>(k),
                        &kBeta,
                        c,
                        static_cast<int>(n));
}

}  // namespace cuda_transformer
