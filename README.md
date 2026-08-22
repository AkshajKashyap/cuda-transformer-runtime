# CUDA Transformer Runtime

A small C++/CUDA transformer inference runtime built to study GPU
architecture, transformer inference, numerical correctness, profiling,
and performance engineering.

## Status

Milestone 2 — matrix multiplication / GEMM.

## CUDA smoke test

The Milestone 0 smoke test verifies the complete CUDA execution path: device
allocation, host-to-device transfer, a kernel launch, synchronization,
device-to-host transfer, and CPU result checking.

The current development and benchmarking machine has an NVIDIA GeForce RTX
3050 Laptop GPU (Ampere, compute capability 8.6). By default, CMake therefore
generates explicit `sm_86` SASS code (`CMAKE_CUDA_ARCHITECTURES=86-real`). This
avoids relying on the compiler/toolchain's architecture default and matches the
validated local environment.

```bash
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

To build for another GPU, configure from a clean build directory with an
explicit override. For example, a GPU with compute capability 7.5 can use:

```bash
cmake -S . -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=75-real
```

The command-line value overrides the project's `86-real` default. This
configuration produces SASS only for the selected architecture; it
does not yet include multi-architecture or PTX-forward-compatibility support.

## GPU fundamentals

Milestone 1 adds CPU references and CUDA implementations for float vector
addition, scalar multiplication, and summation. The vector kernels launch 256
threads per block. Thread `blockIdx.x * blockDim.x + threadIdx.x` handles one
element, so adjacent threads access adjacent floats in global memory. A bounds
check is necessary because the final block is often only partly occupied.

The reduction has two learning-oriented implementations:

- `sum_atomic_cuda` is a direct reference that atomically adds one element per
  thread into one result. Its floating-point summation order is nondeterministic.
- `sum_shared_cuda` has each 256-thread block load and reduce up to 512 values
  in shared memory, then writes one partial sum. It repeatedly launches the
  same block reduction on the partial sums until one result remains.

`__syncthreads()` is required after loading shared memory and after every
reduction step: without it, a thread could read a partner value before that
thread has written it.

`cuda_fundamentals_test` compares every GPU result with a deterministic CPU
reference for sizes 1, 17, 255, 256, 257, 513, and 65,537. It uses an absolute
tolerance of `1e-5` plus a relative tolerance of `1e-5`, which accommodates
the different valid floating-point addition orders in reductions.

Run the timing sample after building:

```bash
./build/src/cuda_vector_timing
```

It warms up vector addition ten times, records CUDA events around 100 further
kernel launches in the default stream, waits for the stop event, and reports
the average kernel-only latency. Host/device copies and allocation are outside
the timed interval. This is measurement infrastructure, not a performance
claim.

Current limitations: the reduction implementations are intentionally simple;
the atomic version is not deterministic, and the shared-memory version does
not yet use warp shuffles or other production optimizations. There are no
tensor, allocator, or transformer abstractions in this milestone.

## GEMM

Milestone 2 implements FP32 row-major matrix multiplication, `C = A * B`, for
`A[M, K]`, `B[K, N]`, and `C[M, N]`. Element `(row, column)` of a matrix with
`columns` columns is stored at `data[row * columns + column]`.

The naive CUDA kernel uses 16 x 16 thread blocks. A thread at
`(blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x)`
computes one `C[row, column]`, reading `K` values from each input matrix. It
has straightforward but poor data reuse: neighboring output threads repeatedly
load overlapping A and B values from global memory.

The tiled kernel uses the same 16 x 16 output mapping. A block cooperatively
loads one 16 x 16 tile from A and one from B into shared memory (2,048 bytes
total), accumulates products in a register, and repeats over K tiles. Partial
tiles load zero for out-of-range elements and the final store is bounds checked.
`__syncthreads()` is required after loading a tile and after consuming it, so
no thread reads incomplete shared data or overwrites data that another thread
still needs.

cuBLAS is also tested and benchmarked. It is column-major, so the row-major
operation is issued as `C^T = B^T * A^T`: the row-major B buffer is passed as
the first column-major operand with dimensions `N x K`, followed by row-major A
as `K x M`; the output has leading dimension `N`. No input transpose or copy is
needed. The wrapper selects `CUBLAS_PEDANTIC_MATH` so this FP32 milestone does
not silently use Ampere's reduced-precision TF32 tensor-core path.

`cuda_gemm_test` compares CPU, naive CUDA, tiled CUDA, and cuBLAS results for
`1x1x1`, `3x5x7`, `8x4x13`, `16x16x16`, `17x19x23`, `31x33x29`, and
`128x96x112`. Its tolerance is
`1e-4 + 2e-6 * K * max(1, abs(expected))`, scaling modestly with the number of
FP32 products accumulated into each output element.

Run the focused kernel-only benchmark after building:

```bash
./build/src/cuda_gemm_benchmark
```

It benchmarks naive CUDA, tiled CUDA, and cuBLAS for square sizes 64, 128,
256, 512, and 1024. Device memory allocation and host/device copies occur
before the measurement. Each method warms up ten times, then CUDA events time
nine independent batches in the default stream. Each batch contains 1,000
launches for 64–128, 200 for 256, 50 for 512, or 10 for 1024; the benchmark
reports the median per-launch batch latency and its corresponding
`2*M*N*K / time` GFLOP/s. This is kernel-only timing.

Laptop GPU clocks, thermals, and power limits can vary between runs. Small
GEMMs are especially sensitive to fixed launch and scheduling overhead. Treat
results as workload- and hardware-specific measurements, not general
performance claims.

Current GEMM limitations: the handwritten kernels are instructional FP32
implementations only. They do not use tensor cores, WMMA, asynchronous copies,
double buffering, register blocking, warp-specialized code, or tuned cuBLAS
algorithm selection. No transformer operations have been added.
