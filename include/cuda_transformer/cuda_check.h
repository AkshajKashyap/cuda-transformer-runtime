#pragma once

#include <cuda_runtime.h>

#include <cstdio>

namespace cuda_transformer {

// CUDA reports failures through return codes. Keep the diagnostic close to the
// failing call so asynchronous GPU work does not hide the source of an error.
inline bool check_cuda(cudaError_t error, const char* expression, const char* file, int line) {
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

}  // namespace cuda_transformer

#define CTR_CUDA_CHECK(expression) \
    ::cuda_transformer::check_cuda((expression), #expression, __FILE__, __LINE__)
