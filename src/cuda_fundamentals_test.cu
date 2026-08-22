#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/gpu_fundamentals.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr float kAbsoluteTolerance = 1.0e-5F;
constexpr float kRelativeTolerance = 1.0e-5F;

bool approximately_equal(float actual, float expected) {
    const float difference = std::fabs(actual - expected);
    const float scale = fmaxf(std::fabs(actual), std::fabs(expected));
    return difference <= kAbsoluteTolerance + kRelativeTolerance * scale;
}

bool free_device_memory(bool success, float* first, float* second, float* third) {
    if (first != nullptr && !CTR_CUDA_CHECK(cudaFree(first))) {
        success = false;
    }
    if (second != nullptr && !CTR_CUDA_CHECK(cudaFree(second))) {
        success = false;
    }
    if (third != nullptr && !CTR_CUDA_CHECK(cudaFree(third))) {
        success = false;
    }
    return success;
}

bool check_vector(const char* operation,
                  std::size_t count,
                  const std::vector<float>& actual,
                  const std::vector<float>& expected) {
    for (std::size_t index = 0; index < count; ++index) {
        if (!approximately_equal(actual[index], expected[index])) {
            std::fprintf(stderr,
                         "%s mismatch for count %zu at index %zu: expected %.9g, got %.9g\n",
                         operation,
                         count,
                         index,
                         expected[index],
                         actual[index]);
            return false;
        }
    }
    return true;
}

bool run_vector_case(std::size_t count) {
    std::vector<float> a(count);
    std::vector<float> b(count);
    std::vector<float> addition_expected(count);
    std::vector<float> scalar_expected(count);
    std::vector<float> actual(count);

    for (std::size_t index = 0; index < count; ++index) {
        a[index] = static_cast<float>(static_cast<int>(index % 31) - 15) * 0.125F;
        b[index] = static_cast<float>(static_cast<int>(index % 19) - 9) * 0.25F;
    }
    constexpr float kAlpha = -1.75F;
    cuda_transformer::vector_add_cpu(
        a.data(), b.data(), addition_expected.data(), count);
    cuda_transformer::scalar_multiply_cpu(a.data(), kAlpha, scalar_expected.data(), count);

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_out = nullptr;
    const std::size_t bytes = count * sizeof(float);

    if (!CTR_CUDA_CHECK(cudaMalloc(&device_a, bytes))) {
        return false;
    }
    if (!CTR_CUDA_CHECK(cudaMalloc(&device_b, bytes))) {
        return free_device_memory(false, device_a, device_b, device_out);
    }
    if (!CTR_CUDA_CHECK(cudaMalloc(&device_out, bytes))) {
        return free_device_memory(false, device_a, device_b, device_out);
    }
    if (!CTR_CUDA_CHECK(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice)) ||
        !CTR_CUDA_CHECK(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice))) {
        return free_device_memory(false, device_a, device_b, device_out);
    }

    if (!CTR_CUDA_CHECK(
            cuda_transformer::vector_add_cuda(device_a, device_b, device_out, count)) ||
        !CTR_CUDA_CHECK(cudaDeviceSynchronize()) ||
        !CTR_CUDA_CHECK(cudaMemcpy(actual.data(),
                                   device_out,
                                   bytes,
                                   cudaMemcpyDeviceToHost))) {
        return free_device_memory(false, device_a, device_b, device_out);
    }
    bool success = check_vector("vector addition", count, actual, addition_expected);

    if (success &&
        (!CTR_CUDA_CHECK(cuda_transformer::scalar_multiply_cuda(
             device_a, kAlpha, device_out, count)) ||
         !CTR_CUDA_CHECK(cudaDeviceSynchronize()) ||
         !CTR_CUDA_CHECK(cudaMemcpy(actual.data(),
                                    device_out,
                                    bytes,
                                    cudaMemcpyDeviceToHost)))) {
        success = false;
    }
    if (success) {
        success = check_vector("scalar multiplication", count, actual, scalar_expected);
    }

    return free_device_memory(success, device_a, device_b, device_out);
}

bool run_reduction_case(std::size_t count) {
    std::vector<float> input(count);
    for (std::size_t index = 0; index < count; ++index) {
        // Binary-fraction values make this deterministic input easy to inspect.
        input[index] = static_cast<float>(1 + index % 7) * 0.125F;
    }
    const float expected = cuda_transformer::sum_cpu(input.data(), count);

    float* device_input = nullptr;
    float* device_sum = nullptr;
    const std::size_t bytes = count * sizeof(float);
    if (!CTR_CUDA_CHECK(cudaMalloc(&device_input, bytes))) {
        return false;
    }
    if (!CTR_CUDA_CHECK(cudaMalloc(&device_sum, sizeof(float)))) {
        return free_device_memory(false, device_input, device_sum, nullptr);
    }
    if (!CTR_CUDA_CHECK(
            cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice))) {
        return free_device_memory(false, device_input, device_sum, nullptr);
    }

    float actual = 0.0F;
    if (!CTR_CUDA_CHECK(cuda_transformer::sum_atomic_cuda(device_input, count, device_sum)) ||
        !CTR_CUDA_CHECK(cudaDeviceSynchronize()) ||
        !CTR_CUDA_CHECK(cudaMemcpy(
            &actual, device_sum, sizeof(float), cudaMemcpyDeviceToHost))) {
        return free_device_memory(false, device_input, device_sum, nullptr);
    }
    bool success = approximately_equal(actual, expected);
    if (!success) {
        std::fprintf(stderr,
                     "atomic reduction mismatch for count %zu: expected %.9g, got %.9g\n",
                     count,
                     expected,
                     actual);
    }

    if (success &&
        (!CTR_CUDA_CHECK(cuda_transformer::sum_shared_cuda(device_input, count, device_sum)) ||
         !CTR_CUDA_CHECK(cudaDeviceSynchronize()) ||
         !CTR_CUDA_CHECK(cudaMemcpy(
             &actual, device_sum, sizeof(float), cudaMemcpyDeviceToHost)))) {
        success = false;
    }
    if (success && !approximately_equal(actual, expected)) {
        std::fprintf(stderr,
                     "shared-memory reduction mismatch for count %zu: expected %.9g, got %.9g\n",
                     count,
                     expected,
                     actual);
        success = false;
    }

    return free_device_memory(success, device_input, device_sum, nullptr);
}

}  // namespace

int main() {
    constexpr std::array<std::size_t, 7> kCounts{
        1, 17, 255, 256, 257, 513, 65'537};

    for (const std::size_t count : kCounts) {
        if (!run_vector_case(count) || !run_reduction_case(count)) {
            return EXIT_FAILURE;
        }
    }

    std::puts("CUDA fundamentals tests passed.");
    return EXIT_SUCCESS;
}
