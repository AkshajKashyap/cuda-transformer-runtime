#pragma once

#include <cublas_v2.h>

#include <cstdio>

namespace cuda_transformer {

inline const char* cublas_status_name(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
        default:
            return "unknown cuBLAS status";
    }
}

inline bool check_cublas(cublasStatus_t status,
                         const char* expression,
                         const char* file,
                         int line) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return true;
    }

    std::fprintf(stderr,
                 "cuBLAS error at %s:%d while evaluating %s: %s\n",
                 file,
                 line,
                 expression,
                 cublas_status_name(status));
    return false;
}

}  // namespace cuda_transformer

#define CTR_CUBLAS_CHECK(expression) \
    ::cuda_transformer::check_cublas((expression), #expression, __FILE__, __LINE__)
