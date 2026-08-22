#include "cuda_transformer/gpu_fundamentals.h"

#include <algorithm>
#include <cstddef>

namespace cuda_transformer {
namespace {

constexpr std::size_t kElementsPerBlock = 2 * kThreadsPerBlock;

int blocks_for(std::size_t count, std::size_t elements_per_block) {
    return static_cast<int>((count + elements_per_block - 1) / elements_per_block);
}

__global__ void vector_add_kernel(const float* a,
                                  const float* b,
                                  float* out,
                                  std::size_t count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        // Consecutive threads access consecutive floats, which is coalesced.
        out[index] = a[index] + b[index];
    }
}

__global__ void scalar_multiply_kernel(const float* input,
                                       float alpha,
                                       float* output,
                                       std::size_t count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = alpha * input[index];
    }
}

__global__ void sum_atomic_kernel(const float* input, std::size_t count, float* output) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        atomicAdd(output, input[index]);
    }
}

__global__ void block_sum_kernel(const float* input, std::size_t count, float* partial_sums) {
    extern __shared__ float shared[];

    const unsigned int thread = threadIdx.x;
    const std::size_t first_index =
        static_cast<std::size_t>(blockIdx.x) * (2 * blockDim.x) + thread;
    float value = 0.0F;

    if (first_index < count) {
        value = input[first_index];
    }
    const std::size_t second_index = first_index + blockDim.x;
    if (second_index < count) {
        value += input[second_index];
    }

    shared[thread] = value;
    __syncthreads();

    // All threads must finish writing shared memory before any thread reads a
    // partner value from it. The barrier is needed after every reduction step.
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) {
            shared[thread] += shared[thread + stride];
        }
        __syncthreads();
    }

    if (thread == 0) {
        partial_sums[blockIdx.x] = shared[0];
    }
}

cudaError_t free_buffers(float* first, float* second, cudaError_t prior_error) {
    cudaError_t cleanup_error = cudaSuccess;
    if (first != nullptr) {
        cleanup_error = cudaFree(first);
    }
    if (second != nullptr) {
        const cudaError_t second_error = cudaFree(second);
        if (cleanup_error == cudaSuccess) {
            cleanup_error = second_error;
        }
    }
    return prior_error != cudaSuccess ? prior_error : cleanup_error;
}

}  // namespace

void vector_add_cpu(const float* a, const float* b, float* out, std::size_t count) {
    for (std::size_t index = 0; index < count; ++index) {
        out[index] = a[index] + b[index];
    }
}

void scalar_multiply_cpu(const float* input, float alpha, float* output, std::size_t count) {
    for (std::size_t index = 0; index < count; ++index) {
        output[index] = alpha * input[index];
    }
}

float sum_cpu(const float* input, std::size_t count) {
    float total = 0.0F;
    for (std::size_t index = 0; index < count; ++index) {
        total += input[index];
    }
    return total;
}

cudaError_t vector_add_cuda(const float* a,
                            const float* b,
                            float* out,
                            std::size_t count,
                            cudaStream_t stream) {
    if (count == 0) {
        return cudaSuccess;
    }

    vector_add_kernel<<<blocks_for(count, kThreadsPerBlock), kThreadsPerBlock, 0, stream>>>(
        a, b, out, count);
    return cudaGetLastError();
}

cudaError_t scalar_multiply_cuda(const float* input,
                                 float alpha,
                                 float* output,
                                 std::size_t count,
                                 cudaStream_t stream) {
    if (count == 0) {
        return cudaSuccess;
    }

    scalar_multiply_kernel<<<blocks_for(count, kThreadsPerBlock), kThreadsPerBlock, 0, stream>>>(
        input, alpha, output, count);
    return cudaGetLastError();
}

cudaError_t sum_atomic_cuda(const float* input,
                            std::size_t count,
                            float* output,
                            cudaStream_t stream) {
    if (count == 0) {
        return cudaErrorInvalidValue;
    }

    cudaError_t error = cudaMemsetAsync(output, 0, sizeof(float), stream);
    if (error != cudaSuccess) {
        return error;
    }

    sum_atomic_kernel<<<blocks_for(count, kThreadsPerBlock), kThreadsPerBlock, 0, stream>>>(
        input, count, output);
    return cudaGetLastError();
}

cudaError_t sum_shared_cuda(const float* input,
                            std::size_t count,
                            float* output,
                            cudaStream_t stream) {
    if (count == 0) {
        return cudaErrorInvalidValue;
    }

    const std::size_t max_partials = (count + kElementsPerBlock - 1) / kElementsPerBlock;
    float* first_buffer = nullptr;
    float* second_buffer = nullptr;
    cudaError_t error = cudaMalloc(&first_buffer, max_partials * sizeof(float));
    if (error != cudaSuccess) {
        return error;
    }
    error = cudaMalloc(&second_buffer, max_partials * sizeof(float));
    if (error != cudaSuccess) {
        return free_buffers(first_buffer, second_buffer, error);
    }

    const float* current_input = input;
    std::size_t current_count = count;
    float* current_output = first_buffer;

    while (true) {
        const int blocks = blocks_for(current_count, kElementsPerBlock);
        block_sum_kernel<<<blocks,
                           kThreadsPerBlock,
                           kThreadsPerBlock * sizeof(float),
                           stream>>>(current_input, current_count, current_output);
        error = cudaGetLastError();
        if (error != cudaSuccess) {
            return free_buffers(first_buffer, second_buffer, error);
        }

        if (blocks == 1) {
            error = cudaMemcpyAsync(
                output, current_output, sizeof(float), cudaMemcpyDeviceToDevice, stream);
            return free_buffers(first_buffer, second_buffer, error);
        }

        current_input = current_output;
        current_count = static_cast<std::size_t>(blocks);
        current_output = current_output == first_buffer ? second_buffer : first_buffer;
    }
}

}  // namespace cuda_transformer
