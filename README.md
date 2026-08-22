# CUDA Transformer Runtime

A small C++/CUDA transformer inference runtime built to study GPU
architecture, transformer inference, numerical correctness, profiling,
and performance engineering.

## Status

Milestone 0 — environment and architecture setup.

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
Milestone 0 configuration produces SASS only for the selected architecture; it
does not yet include multi-architecture or PTX-forward-compatibility support.
