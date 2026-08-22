#include "cuda_transformer/cublas_check.h"
#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/gemm.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

struct GemmShape {
    std::size_t m;
    std::size_t k;
    std::size_t n;
};

float tolerance(float expected, std::size_t k) {
    // Different valid FP32 accumulation orders have error that grows with the
    // number of products. This remains tight for the deliberately bounded data.
    return 1.0e-4F + 2.0e-6F * static_cast<float>(k) * fmaxf(1.0F, std::fabs(expected));
}

bool release_device_memory(bool success, float* a, float* b, float* c) {
    if (a != nullptr && !CTR_CUDA_CHECK(cudaFree(a))) {
        success = false;
    }
    if (b != nullptr && !CTR_CUDA_CHECK(cudaFree(b))) {
        success = false;
    }
    if (c != nullptr && !CTR_CUDA_CHECK(cudaFree(c))) {
        success = false;
    }
    return success;
}

bool check_result(const char* implementation,
                  const GemmShape& shape,
                  const std::vector<float>& actual,
                  const std::vector<float>& expected) {
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float allowed_error = tolerance(expected[index], shape.k);
        if (std::fabs(actual[index] - expected[index]) > allowed_error) {
            const std::size_t row = index / shape.n;
            const std::size_t column = index % shape.n;
            std::fprintf(stderr,
                         "%s GEMM mismatch for %zux%zux%zu at (%zu, %zu): "
                         "expected %.9g, got %.9g, tolerance %.9g\n",
                         implementation,
                         shape.m,
                         shape.k,
                         shape.n,
                         row,
                         column,
                         expected[index],
                         actual[index],
                         allowed_error);
            return false;
        }
    }
    return true;
}

bool run_case(cublasHandle_t cublas_handle, const GemmShape& shape) {
    std::vector<float> a(shape.m * shape.k);
    std::vector<float> b(shape.k * shape.n);
    std::vector<float> expected(shape.m * shape.n);
    std::vector<float> actual(shape.m * shape.n);

    for (std::size_t index = 0; index < a.size(); ++index) {
        a[index] = static_cast<float>(static_cast<int>((index * 13) % 23) - 11) * 0.0625F;
    }
    for (std::size_t index = 0; index < b.size(); ++index) {
        b[index] = static_cast<float>(static_cast<int>((index * 7) % 19) - 9) * 0.0625F;
    }
    cuda_transformer::gemm_cpu(
        a.data(), b.data(), expected.data(), shape.m, shape.k, shape.n);

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;
    const std::size_t a_bytes = a.size() * sizeof(float);
    const std::size_t b_bytes = b.size() * sizeof(float);
    const std::size_t c_bytes = actual.size() * sizeof(float);

    if (!CTR_CUDA_CHECK(cudaMalloc(&device_a, a_bytes))) {
        return false;
    }
    if (!CTR_CUDA_CHECK(cudaMalloc(&device_b, b_bytes))) {
        return release_device_memory(false, device_a, device_b, device_c);
    }
    if (!CTR_CUDA_CHECK(cudaMalloc(&device_c, c_bytes)) ||
        !CTR_CUDA_CHECK(cudaMemcpy(device_a, a.data(), a_bytes, cudaMemcpyHostToDevice)) ||
        !CTR_CUDA_CHECK(cudaMemcpy(device_b, b.data(), b_bytes, cudaMemcpyHostToDevice))) {
        return release_device_memory(false, device_a, device_b, device_c);
    }

    if (!CTR_CUDA_CHECK(cuda_transformer::gemm_naive_cuda(
            device_a, device_b, device_c, shape.m, shape.k, shape.n)) ||
        !CTR_CUDA_CHECK(cudaDeviceSynchronize()) ||
        !CTR_CUDA_CHECK(
            cudaMemcpy(actual.data(), device_c, c_bytes, cudaMemcpyDeviceToHost))) {
        return release_device_memory(false, device_a, device_b, device_c);
    }
    bool success = check_result("naive", shape, actual, expected);

    if (success &&
        (!CTR_CUDA_CHECK(cuda_transformer::gemm_tiled_cuda(
             device_a, device_b, device_c, shape.m, shape.k, shape.n)) ||
         !CTR_CUDA_CHECK(cudaDeviceSynchronize()) ||
         !CTR_CUDA_CHECK(
             cudaMemcpy(actual.data(), device_c, c_bytes, cudaMemcpyDeviceToHost)))) {
        success = false;
    }
    if (success) {
        success = check_result("tiled", shape, actual, expected);
    }

    if (success &&
        (!CTR_CUBLAS_CHECK(cuda_transformer::gemm_cublas_row_major(
             cublas_handle, device_a, device_b, device_c, shape.m, shape.k, shape.n)) ||
         !CTR_CUDA_CHECK(cudaDeviceSynchronize()) ||
         !CTR_CUDA_CHECK(
             cudaMemcpy(actual.data(), device_c, c_bytes, cudaMemcpyDeviceToHost)))) {
        success = false;
    }
    if (success) {
        success = check_result("cuBLAS", shape, actual, expected);
    }

    return release_device_memory(success, device_a, device_b, device_c);
}

}  // namespace

int main() {
    constexpr std::array<GemmShape, 7> kShapes{{
        {1, 1, 1},
        {3, 5, 7},
        {8, 4, 13},
        {16, 16, 16},
        {17, 19, 23},
        {31, 33, 29},
        {128, 96, 112},
    }};

    cublasHandle_t cublas_handle = nullptr;
    if (!CTR_CUBLAS_CHECK(cublasCreate(&cublas_handle))) {
        return EXIT_FAILURE;
    }

    bool success = true;
    for (const GemmShape& shape : kShapes) {
        if (!run_case(cublas_handle, shape)) {
            success = false;
            break;
        }
    }
    if (!CTR_CUBLAS_CHECK(cublasDestroy(cublas_handle))) {
        success = false;
    }

    if (!success) {
        return EXIT_FAILURE;
    }
    std::puts("CUDA GEMM tests passed.");
    return EXIT_SUCCESS;
}
