#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/gpu_fundamentals.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

bool cleanup(bool success,
             float* device_a,
             float* device_b,
             float* device_out,
             cudaEvent_t start,
             cudaEvent_t stop) {
    if (start != nullptr && !CTR_CUDA_CHECK(cudaEventDestroy(start))) {
        success = false;
    }
    if (stop != nullptr && !CTR_CUDA_CHECK(cudaEventDestroy(stop))) {
        success = false;
    }
    if (device_a != nullptr && !CTR_CUDA_CHECK(cudaFree(device_a))) {
        success = false;
    }
    if (device_b != nullptr && !CTR_CUDA_CHECK(cudaFree(device_b))) {
        success = false;
    }
    if (device_out != nullptr && !CTR_CUDA_CHECK(cudaFree(device_out))) {
        success = false;
    }
    return success;
}

}  // namespace

int main() {
    constexpr std::size_t kCount = 1U << 20;
    constexpr int kWarmupIterations = 10;
    constexpr int kTimedIterations = 100;

    std::vector<float> a(kCount, 1.25F);
    std::vector<float> b(kCount, -0.75F);
    const std::size_t bytes = kCount * sizeof(float);
    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_out = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    if (!CTR_CUDA_CHECK(cudaMalloc(&device_a, bytes)) ||
        !CTR_CUDA_CHECK(cudaMalloc(&device_b, bytes)) ||
        !CTR_CUDA_CHECK(cudaMalloc(&device_out, bytes)) ||
        !CTR_CUDA_CHECK(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice)) ||
        !CTR_CUDA_CHECK(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice))) {
        cleanup(false, device_a, device_b, device_out, start, stop);
        return EXIT_FAILURE;
    }

    for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
        if (!CTR_CUDA_CHECK(
                cuda_transformer::vector_add_cuda(device_a, device_b, device_out, kCount))) {
            cleanup(false, device_a, device_b, device_out, start, stop);
            return EXIT_FAILURE;
        }
    }
    if (!CTR_CUDA_CHECK(cudaDeviceSynchronize()) || !CTR_CUDA_CHECK(cudaEventCreate(&start)) ||
        !CTR_CUDA_CHECK(cudaEventCreate(&stop)) || !CTR_CUDA_CHECK(cudaEventRecord(start))) {
        cleanup(false, device_a, device_b, device_out, start, stop);
        return EXIT_FAILURE;
    }

    for (int iteration = 0; iteration < kTimedIterations; ++iteration) {
        if (!CTR_CUDA_CHECK(
                cuda_transformer::vector_add_cuda(device_a, device_b, device_out, kCount))) {
            cleanup(false, device_a, device_b, device_out, start, stop);
            return EXIT_FAILURE;
        }
    }
    if (!CTR_CUDA_CHECK(cudaEventRecord(stop)) || !CTR_CUDA_CHECK(cudaEventSynchronize(stop))) {
        cleanup(false, device_a, device_b, device_out, start, stop);
        return EXIT_FAILURE;
    }

    float elapsed_milliseconds = 0.0F;
    if (!CTR_CUDA_CHECK(cudaEventElapsedTime(&elapsed_milliseconds, start, stop))) {
        cleanup(false, device_a, device_b, device_out, start, stop);
        return EXIT_FAILURE;
    }
    std::printf("Vector addition kernel-only average over %d iterations: %.4f ms\n",
                kTimedIterations,
                elapsed_milliseconds / kTimedIterations);

    return cleanup(true, device_a, device_b, device_out, start, stop) ? EXIT_SUCCESS
                                                                       : EXIT_FAILURE;
}
