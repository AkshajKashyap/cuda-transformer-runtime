# CUDA Transformer Runtime

A small C++/CUDA transformer inference runtime built to study GPU
architecture, transformer inference, numerical correctness, profiling,
and performance engineering.

## Status

Milestone 1 — GPU fundamentals.

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
tensor, allocator, GEMM, or transformer abstractions in this milestone.
