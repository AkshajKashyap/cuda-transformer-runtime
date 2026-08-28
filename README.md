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

## Transformer primitives

Milestone 3 adds FP32 RMSNorm, numerically stable row-wise softmax, SiLU,
elementwise multiply, and residual addition. RMSNorm uses one 256-thread block
per row: threads accumulate strided squared values into 1 KiB shared memory,
synchronize for the sum reduction, then scale their row elements. Softmax uses
the same per-row structure for explicit maximum and exponent-sum reductions;
subtracting the maximum before `exp` prevents overflow. Both require barriers
after shared-memory writes and reduction steps.

SiLU, multiply, and residual addition are 1D 256-thread, bounds-checked
kernels and are expected to be memory-bandwidth-bound. RMSNorm and softmax are
reduction/synchronization-bound hypotheses, not profiler conclusions. Tests use
`1e-5 + 2e-5 * max(1, abs(expected))`, include rows/widths 1, 17, 256, 257,
and 32x768, and verify softmax row sums. Zero sizes and nonpositive RMSNorm
epsilon are rejected. `cuda_transformer_primitives_benchmark` uses ten warmups,
nine CUDA-event batches of 200 launches, and median kernel-only latency; copies
and allocations are outside measurement.

## Causal attention baseline

Milestone 4 uses contiguous FP32 Q, K, and V buffers in `[batch, heads,
sequence, head_dim]` row-major order. RoPE rotates even/odd Q and K pairs with
the standard position-dependent frequency; V is unchanged. The visible stages
are RoPE → scaled QK^T → causal mask → stable row softmax → P×V. Future keys
receive a large negative logit before max-subtracted softmax.

Scores and probabilities are materialized as `[batch, heads, query, key]`.
Each FP32 buffer consumes `batch*heads*sequence*sequence*sizeof(float)` bytes,
so this baseline has intentional O(sequence²) intermediate storage. It is a
correctness-first, non-fused baseline for future comparisons.

`cuda_attention_benchmark` times RoPE, QK^T, softmax, P×V, and full staged
attention for `(heads, seq, dim)` `(4,32,64)`, `(4,64,64)`, `(8,128,64)`, and
`(8,256,64)`. Allocation/copies are excluded; each stage has 10 warm-ups and 9
CUDA-event batches, using 200, 50, or 10 launches per batch as sequence grows.
It reports median kernel-only latency and per-shape score/probability size.

## KV-cache incremental decode

The contiguous FP32 K and V caches each use `[batch, heads, max_sequence,
head_dim]`. K is stored after RoPE at its original absolute position; V is
stored unchanged, and historical K is never rotated again. Prefill supplies
already-rotated prompt K and raw V. At decode position `t = current_length`,
new Q/K receive RoPE at `t`, rotated K and V are appended, and the new Q attends
only to positions `0..t` before producing one output token.

K+V cache storage is `2*batch*heads*max_sequence*head_dim*4` bytes. For batch
1 and head_dim 64: heads=4 uses 0.25, 0.50, and 1.00 MiB at max_seq 128, 256,
and 512; heads=8 uses 0.50, 1.00, and 2.00 MiB. Temporary decode workspaces
are separate from this persistent storage.

Full attention over S tokens performs O(S^2*head_dim) score work and materializes
O(S^2) values per head. Cached single-token decode reads S K/V entries and does
O(S*head_dim) score and weighted-V work; it avoids recomputing older K/V
projections, but per-token work still grows with context length.

`cuda_incremental_attention_benchmark` measures fixed-history full staged decode
for heads 4/8, head_dim 64, and histories 64/128/256/512/1024/2048. Before every
launch it restores only `current_length=S`; position S is outside the valid
prefix and is deterministically overwritten. Allocation/transfers are outside
timing. Ten warm-ups precede nine CUDA-event batches, with 500 launches at
64–256, 300 at 512, 200 at 1024, and 100 at 2048; median per-launch kernel-only
latency is reported. Short-context measurements may be dominated by fixed launch
overhead, so they need not be monotonic. Q·K and P×V work grows roughly linearly
with valid history; measurements determine where that scaling becomes visible.

## Decoder-block decode benchmark

`cuda_decoder_block_decode_benchmark` compares complete FP32 decoder-block
full-prefix recomputation with the production cached incremental decoder block.
It uses batch 1, hidden size 256, 4 heads of dimension 64, intermediate size
512, and RMSNorm epsilons of `1e-5`; both paths share one deterministic weight
set and input prefix. It covers fixed histories 16, 32, 64, 128, 256, 512, and
1024. For history `S`, the full path processes `S+1` tokens, while the cached
path pre-fills tokens `0..S-1` before timing and measures token `S` only.

Run it after building:

```bash
./build/src/cuda_decoder_block_decode_benchmark
```

Allocation, host/device transfers, cache prefill, and one per-history output
sanity check are outside timing. Each path has 10 warm-ups and 9 independent
CUDA-event timing batches; the reported value is median per-execution latency.
Before every cached launch, only the logical cache length is restored to `S`, so
the deterministic token at position `S` overwrites the same invalid cache slot
without rebuilding history. Full-prefix batches use 200/100/50/20/10/5
launches at histories 16/32, 64, 128, 256, 512, and 1024 respectively; cached
batches use 500/500/500/500/300/200/100 launches. This correctness-first
comparison is a baseline for later decode optimizations, not a production
performance claim.

## Incremental decoder-block stage profiler

`cuda_incremental_decoder_block_profile` profiles the production-equivalent
FP32 one-token decoder-block composition at fixed histories 16, 64, 256, and
1024. It uses the same batch-1 shape as the decode benchmark: hidden 256, four
heads of dimension 64, intermediate size 512, and both RMSNorm epsilons
`1e-5`.

```bash
./build/src/cuda_incremental_decoder_block_profile
```

The profiler pre-fills cache history outside timing, verifies its explicit
primitive composition against `incremental_decoder_block_cuda`, then records
CUDA-event boundaries around stages in the same default-stream order. It uses
10 complete warm-ups, 9 timed batches, and 500/500/300/100 executions per batch
at the four histories. It reports median per-execution stage latency, each
stage's percentage of the measured composition, a direct end-to-end timing, and
the corresponding 6C1 baseline for comparison. The incremental-attention core
is reported as one group because its RoPE, cache append, QK, softmax, and P×V
kernels are intentionally encapsulated by the existing API. Event boundaries
can add profiling overhead; compare the stage sum with the direct timing rather
than treating either as a new performance claim.

If Nsight Systems is available, an optional complementary trace is:

```bash
nsys profile --trace=cuda,cublas -o incremental_decoder_block_profile \
  ./build/src/cuda_incremental_decoder_block_profile
```
