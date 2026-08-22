#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace {

// CUDA calls return an error code instead of throwing exceptions. Checking every
// call makes setup and transfer failures visible at the operation that caused them.
bool check_cuda(cudaError_t error, const char* expression, const char* file, int line) {
    if (error == cudaSuccess) {
        return true;
    }

    std::fprintf(stderr,
                 "CUDA error at %s:%d while evaluating %s: %s\n",
                 file,
                 line,
                 expression,
                 cudaGetErrorString(error));
    return false;
}

#define CUDA_CHECK(expression) \
    check_cuda((expression), #expression, __FILE__, __LINE__)

// Each CUDA thread updates one array element. The bounds check permits a launch
// configuration whose total thread count is larger than the input array.
__global__ void add_constant(int* values, std::size_t count, int constant) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        values[index] += constant;
    }
}

}  // namespace

int main() {
    constexpr int kConstant = 7;
    std::array<int, 8> host_values{0, 1, 2, 3, 4, 5, 6, 7};
    const std::array<int, 8> expected_values{7, 8, 9, 10, 11, 12, 13, 14};

    int* device_values = nullptr;
    const std::size_t bytes = host_values.size() * sizeof(host_values[0]);

    if (!CUDA_CHECK(cudaMalloc(&device_values, bytes))) {
        return EXIT_FAILURE;
    }

    if (!CUDA_CHECK(cudaMemcpy(device_values,
                               host_values.data(),
                               bytes,
                               cudaMemcpyHostToDevice))) {
        (void)CUDA_CHECK(cudaFree(device_values));
        return EXIT_FAILURE;
    }

    constexpr int kThreadsPerBlock = 32;
    const int blocks = static_cast<int>(
        (host_values.size() + kThreadsPerBlock - 1) / kThreadsPerBlock);
    add_constant<<<blocks, kThreadsPerBlock>>>(
        device_values, host_values.size(), kConstant);

    // Kernel launches are asynchronous. These two checks surface launch errors
    // and errors encountered while the GPU executes the kernel.
    if (!CUDA_CHECK(cudaGetLastError()) || !CUDA_CHECK(cudaDeviceSynchronize())) {
        (void)CUDA_CHECK(cudaFree(device_values));
        return EXIT_FAILURE;
    }

    if (!CUDA_CHECK(cudaMemcpy(host_values.data(),
                               device_values,
                               bytes,
                               cudaMemcpyDeviceToHost))) {
        (void)CUDA_CHECK(cudaFree(device_values));
        return EXIT_FAILURE;
    }

    if (!CUDA_CHECK(cudaFree(device_values))) {
        return EXIT_FAILURE;
    }

    for (std::size_t index = 0; index < host_values.size(); ++index) {
        if (host_values[index] != expected_values[index]) {
            std::fprintf(stderr,
                         "Incorrect result at index %zu: expected %d, got %d\n",
                         index,
                         expected_values[index],
                         host_values[index]);
            return EXIT_FAILURE;
        }
    }

    std::puts("CUDA smoke test passed.");
    return EXIT_SUCCESS;
}
