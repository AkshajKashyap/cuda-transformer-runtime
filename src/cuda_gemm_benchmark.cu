#include "cuda_transformer/cublas_check.h"
#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/gemm.h"

#include <algorithm>
#include <array>
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

struct MeasurementConfig {
    int batches;
    int launches_per_batch;
};

MeasurementConfig measurement_config(const GemmShape& shape) {
    const std::size_t largest_dimension = std::max({shape.m, shape.k, shape.n});
    if (largest_dimension <= 128) {
        return {9, 1000};
    }
    if (largest_dimension <= 256) {
        return {9, 200};
    }
    if (largest_dimension <= 512) {
        return {9, 50};
    }
    return {9, 10};
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

bool destroy_events(bool success, cudaEvent_t start, cudaEvent_t stop) {
    if (start != nullptr && !CTR_CUDA_CHECK(cudaEventDestroy(start))) {
        success = false;
    }
    if (stop != nullptr && !CTR_CUDA_CHECK(cudaEventDestroy(stop))) {
        success = false;
    }
    return success;
}

template <typename Launch>
bool time_kernel(const char* name, const GemmShape& shape, Launch launch) {
    constexpr int kWarmupIterations = 10;
    const MeasurementConfig config = measurement_config(shape);

    for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
        if (!launch()) {
            return false;
        }
    }
    if (!CTR_CUDA_CHECK(cudaDeviceSynchronize())) {
        return false;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (!CTR_CUDA_CHECK(cudaEventCreate(&start)) || !CTR_CUDA_CHECK(cudaEventCreate(&stop))) {
        destroy_events(false, start, stop);
        return false;
    }

    std::vector<float> batch_latencies;
    batch_latencies.reserve(config.batches);
    for (int batch = 0; batch < config.batches; ++batch) {
        if (!CTR_CUDA_CHECK(cudaEventRecord(start))) {
            destroy_events(false, start, stop);
            return false;
        }
        for (int iteration = 0; iteration < config.launches_per_batch; ++iteration) {
            if (!launch()) {
                destroy_events(false, start, stop);
                return false;
            }
        }
        if (!CTR_CUDA_CHECK(cudaEventRecord(stop)) ||
            !CTR_CUDA_CHECK(cudaEventSynchronize(stop))) {
            destroy_events(false, start, stop);
            return false;
        }

        float elapsed_milliseconds = 0.0F;
        if (!CTR_CUDA_CHECK(cudaEventElapsedTime(&elapsed_milliseconds, start, stop))) {
            destroy_events(false, start, stop);
            return false;
        }
        batch_latencies.push_back(elapsed_milliseconds / config.launches_per_batch);
    }

    std::sort(batch_latencies.begin(), batch_latencies.end());
    const double median_milliseconds = batch_latencies[batch_latencies.size() / 2];
    const double flops = 2.0 * static_cast<double>(shape.m) * static_cast<double>(shape.n) *
                         static_cast<double>(shape.k);
    const double gflops = flops / (median_milliseconds * 1.0e6);
    std::printf("%-7s %4zux%4zux%4zu  %2d x %4d  %8.4f ms  %9.2f GFLOP/s\n",
                name,
                shape.m,
                shape.k,
                shape.n,
                config.batches,
                config.launches_per_batch,
                median_milliseconds,
                gflops);

    return destroy_events(true, start, stop);
}

bool run_shape(cublasHandle_t cublas_handle, const GemmShape& shape) {
    std::vector<float> a(shape.m * shape.k);
    std::vector<float> b(shape.k * shape.n);
    for (std::size_t index = 0; index < a.size(); ++index) {
        a[index] = static_cast<float>(index % 11) * 0.0625F;
    }
    for (std::size_t index = 0; index < b.size(); ++index) {
        b[index] = static_cast<float>(index % 13) * -0.0625F;
    }

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;
    const std::size_t a_bytes = a.size() * sizeof(float);
    const std::size_t b_bytes = b.size() * sizeof(float);
    const std::size_t c_bytes = shape.m * shape.n * sizeof(float);
    if (!CTR_CUDA_CHECK(cudaMalloc(&device_a, a_bytes)) ||
        !CTR_CUDA_CHECK(cudaMalloc(&device_b, b_bytes)) ||
        !CTR_CUDA_CHECK(cudaMalloc(&device_c, c_bytes)) ||
        !CTR_CUDA_CHECK(cudaMemcpy(device_a, a.data(), a_bytes, cudaMemcpyHostToDevice)) ||
        !CTR_CUDA_CHECK(cudaMemcpy(device_b, b.data(), b_bytes, cudaMemcpyHostToDevice))) {
        return release_device_memory(false, device_a, device_b, device_c);
    }

    bool success = time_kernel("naive", shape, [&] {
        return CTR_CUDA_CHECK(cuda_transformer::gemm_naive_cuda(
            device_a, device_b, device_c, shape.m, shape.k, shape.n));
    });
    if (success) {
        success = time_kernel("tiled", shape, [&] {
            return CTR_CUDA_CHECK(cuda_transformer::gemm_tiled_cuda(
                device_a, device_b, device_c, shape.m, shape.k, shape.n));
        });
    }
    if (success) {
        success = time_kernel("cuBLAS", shape, [&] {
            return CTR_CUBLAS_CHECK(cuda_transformer::gemm_cublas_row_major(
                cublas_handle, device_a, device_b, device_c, shape.m, shape.k, shape.n));
        });
    }

    return release_device_memory(success, device_a, device_b, device_c);
}

}  // namespace

int main() {
    constexpr std::array<GemmShape, 5> kShapes{{
        {64, 64, 64},
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
    }};

    cublasHandle_t cublas_handle = nullptr;
    if (!CTR_CUBLAS_CHECK(cublasCreate(&cublas_handle))) {
        return EXIT_FAILURE;
    }

    std::puts("Kernel-only FP32 GEMM timing (allocation and transfers excluded)");
    std::puts("method   MxKxN          batches x launches     median       throughput");
    bool success = true;
    for (const GemmShape& shape : kShapes) {
        if (!run_shape(cublas_handle, shape)) {
            success = false;
            break;
        }
    }
    if (!CTR_CUBLAS_CHECK(cublasDestroy(cublas_handle))) {
        success = false;
    }
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
