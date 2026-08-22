cmake_minimum_required(VERSION 3.24)

# Milestone 0 is developed and benchmarked on an Ampere RTX 3050 Laptop GPU.
# Set this before enabling CUDA so CMake passes explicit sm_86 code generation
# to nvcc. A -DCMAKE_CUDA_ARCHITECTURES=... command-line value takes precedence.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "86-real" CACHE STRING
        "CUDA architectures to generate")
endif()

project(
    cuda_transformer_runtime
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()

add_subdirectory(src)
