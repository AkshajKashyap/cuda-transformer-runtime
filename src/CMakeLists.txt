add_library(cuda_fundamentals STATIC cuda/gpu_fundamentals.cu)

target_include_directories(cuda_fundamentals
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(cuda_fundamentals PUBLIC cxx_std_20)

add_library(cuda_gemm STATIC cuda/gemm.cu)

target_include_directories(cuda_gemm
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(cuda_gemm PUBLIC cxx_std_20)
target_link_libraries(cuda_gemm PUBLIC CUDA::cublas)

add_executable(cuda_smoke_test cuda_smoke_test.cu)

target_compile_features(cuda_smoke_test PRIVATE cxx_std_20)
target_include_directories(cuda_smoke_test PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_test(NAME cuda_smoke_test COMMAND cuda_smoke_test)

add_executable(cuda_fundamentals_test cuda_fundamentals_test.cu)
target_link_libraries(cuda_fundamentals_test PRIVATE cuda_fundamentals)
add_test(NAME cuda_fundamentals_test COMMAND cuda_fundamentals_test)

add_executable(cuda_vector_timing cuda_vector_timing.cu)
target_link_libraries(cuda_vector_timing PRIVATE cuda_fundamentals)

add_executable(cuda_gemm_test cuda_gemm_test.cu)
target_link_libraries(cuda_gemm_test PRIVATE cuda_gemm)
add_test(NAME cuda_gemm_test COMMAND cuda_gemm_test)

add_executable(cuda_gemm_benchmark cuda_gemm_benchmark.cu)
target_link_libraries(cuda_gemm_benchmark PRIVATE cuda_gemm)
