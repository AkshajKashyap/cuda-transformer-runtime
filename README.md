# CUDA Transformer Runtime

A from-scratch CUDA/C++ implementation of a LLaMA-style decoder block with
KV-cached incremental decoding, CPU-vs-GPU correctness tests,
profiling-driven optimization, and reproducible performance benchmarks.

This is a portfolio and educational systems project, not a complete LLM serving
engine. It focuses on the parts of transformer inference that expose GPU memory
layout, CUDA kernels, cuBLAS integration, numerical correctness, and practical
performance engineering.

## At a glance

- FP32 pre-norm LLaMA-style decoder block in C++20/CUDA.
- Full-sequence causal attention and cached one-token incremental decoding.
- CPU reference implementations and 18/18 CTests passing on the tested RTX
  3050 Laptop GPU.
- CUDA-event benchmarks and Nsight Systems trace support.
- Two Nsight-driven optimizations: no steady-state GPU allocation and a
  long-context cooperative probability × value (P×V) reduction.

For the representative single-block FP32 workload below, cached decoding at
1024 tokens of history reduced next-token latency from 7.86 ms to 0.42 ms
(18.81×) on the tested RTX 3050 Laptop GPU. This is not full-model tokens/sec.

## Quickstart

The project defaults to explicit Ampere `sm_86` SASS for its development GPU.

```bash
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

Run selected benchmarks after a successful build:

```bash
./build/src/cuda_gemm_benchmark
./build/src/cuda_incremental_attention_benchmark
./build/src/cuda_incremental_pv_benchmark
./build/src/cuda_decoder_block_decode_benchmark
```

For a different GPU, supply an explicit override—for example:

```bash
cmake -S . -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=75-real
```

The override replaces the `86-real` default. The project has been validated on
one CUDA environment only; compatible GPU, driver, and toolkit versions are
required.

## What runs on CUDA

The runtime includes CUDA implementations of vector/reduction fundamentals,
naive/tiled/cuBLAS GEMM, RMSNorm, LayerNorm, stable softmax, RoPE, causal attention,
SwiGLU, KV-cache operations, and a complete decoder block. The full-sequence
path materializes causal-attention scores/probabilities for clarity; the
incremental path uses a persistent K/V cache and workspace-owned scratch.

### Full-sequence decoder block

```text
X
→ RMSNorm
→ Q/K/V projections
→ token-major → head-major repack
→ RoPE(Q, K)
→ causal attention
→ head-major → token-major repack
→ output projection
→ residual
→ RMSNorm
→ gate/up projections
→ SiLU(gate) * up
→ down projection
→ residual
```

### Incremental cached decode

```text
X_t
→ attention RMSNorm
→ Q/K/V projections
→ token-major → head-major repack
→ RoPE(Q_t, K_t) at absolute cache position t
→ append rotated K_t + raw V_t
→ attention against cached K/V
→ Wo
→ residual
→ SwiGLU MLP
→ residual
```

QKV projections naturally produce token-major `[sequence, hidden]` buffers,
while attention accesses each head’s sequence contiguously. Explicit repacks
make that change visible rather than hiding it behind a tensor abstraction.
The cache uses `[batch, heads, max_sequence, head_dim]`: K is stored after
RoPE, V remains raw, and `current_length` is the absolute position for the next
decoded token. Historical K is never rotated again.

More detail: [architecture](docs/architecture.md).

## Performance

All figures below are real-GPU measurements for batch 1, one FP32 decoder
block, hidden size 256, 4 heads × 64 dimensions, MLP intermediate size 512.
CUDA events measure kernel-only work; allocations and host/device transfers are
outside the timed region. Laptop clocks, power limits, and DVFS can vary, so
these results are workload- and hardware-specific.

### Full prefix vs cached next-token decode

| History | Full-prefix ms | Cached ms | Speedup |
| ---: | ---: | ---: | ---: |
| 16 | 0.56269 | 0.44469 | 1.27× |
| 32 | 0.47918 | 0.38257 | 1.25× |
| 64 | 0.48190 | 0.41284 | 1.17× |
| 128 | 0.47434 | 0.39564 | 1.20× |
| 256 | 0.85171 | 0.42340 | 2.01× |
| 512 | 2.42635 | 0.34323 | 7.07× |
| 1024 | 7.86487 | 0.41812 | 18.81× |

Full-prefix recomputation processes `S+1` tokens. Cached decode pre-fills
history outside timing, resets only logical cache length before each repeated
launch, and overwrites the deterministic next-token slot.

### P×V microbenchmark

| History | Serial µs | Cooperative µs | Speedup |
| ---: | ---: | ---: | ---: |
| 16 | 21.465 | 16.667 | 1.29× |
| 32 | 18.334 | 22.112 | 0.83× |
| 64 | 22.493 | 27.177 | 0.83× |
| 128 | 17.183 | 22.036 | 0.78× |
| 256 | 23.681 | 22.128 | 1.07× |
| 512 | 30.474 | 22.697 | 1.34× |
| 1024 | 64.338 | 25.118 | 2.56× |
| 2048 | 211.456 | 47.870 | 4.42× |

The production path remains serial below history 512 and uses the cooperative
kernel at 512 and above. That cutoff is empirical for this RTX 3050 benchmark
environment and model shape, not a portability claim.

See [performance notes](docs/performance.md) for methodology and the complete
profiling narrative.

## Profiling and optimization

Correctness came first. Nsight Systems then guided two isolated hot-path
changes:

1. The initial incremental path made three `cudaMalloc` and three `cudaFree`
   calls per decoded token for rotated Q, rotated K, and score scratch.
   `IncrementalAttentionWorkspace` moved those buffers to workspace creation,
   so successful steady-state production decode makes zero dynamic GPU
   allocation calls.
2. Reprofiling showed serial P×V dominating long-context GPU kernel time.
   P×V now has one cooperative CUDA block per `(head, output_dimension)` for
   long histories, selected using the measured cutoff above.

No precise whole-runtime gain is attributed to the allocation change because
cross-run laptop-GPU conditions were noisy. The P×V table is the controlled
same-session old/new evidence.

Collect a low-volume production trace:

```bash
nsys profile --trace=cuda,cublas --sample=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o incremental_decode_h1024 \
  ./build/src/cuda_incremental_decoder_block_profile \
  --trace --history 1024 --iterations 20

nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum \
  incremental_decode_h1024.nsys-rep
```

## Correctness methodology

Trusted CPU references are compared against CUDA results using operation-specific
FP32 absolute-plus-relative tolerances. Tests include awkward and
non-power-of-two shapes, causal masking and row-sum checks, cache-prefix
preservation, and token-by-token full-sequence versus incremental equivalence.
They also validate the complete decoder block. The authoritative real-GPU
result is **18/18 CTests passing**. Exact test cases and tolerances are kept in
the focused test sources under `src/`.

## Repository map

- `src/cuda/gemm.cu` and `include/.../gemm.h`: row-major GEMM and cuBLAS
  wrappers.
- `src/cuda/transformer_primitives.cu`: RMSNorm, LayerNorm, softmax, SiLU,
  multiply, and residual operations.
- `src/cuda/attention.cu`, `incremental_attention.cu`, `kv_cache.cu`: causal
  attention, cached attention, and cache ownership.
- `src/cuda/qkv_projection.cu`, `qkv_layout.cu`: QKV projection workspace and
  explicit layout conversions.
- `src/cuda/attention_sublayer.cu`, `mlp_sublayer.cu`, `decoder_block.cu`:
  full decoder-block composition.
- `src/cuda/incremental_decoder_block.cu`: production cached one-token block.
- `src/*benchmark.cu`: standalone CUDA-event benchmarks; `src/*test.cu`:
  CTest executables and CPU/GPU checks.

## Environment and reproducibility

Validated environment: WSL2 Ubuntu, NVIDIA GeForce RTX 3050 Laptop GPU
(compute capability 8.6), CUDA Toolkit 13.3 / nvcc 13.3.73, GCC 13.3, CMake
3.28, and Ninja. The CMake default is `CMAKE_CUDA_ARCHITECTURES=86-real`, which
produces explicit `sm_86` code for this development machine while permitting a
command-line override.

## Limitations

- FP32 only; no FP16/BF16, Tensor Core, or quantized path.
- One decoder block, not complete pretrained-model inference.
- No checkpoint loader, tokenizer, embeddings, LM head, sampling, or generation
  loop.
- No FlashAttention, paged KV cache, CUDA Graphs, or multi-layer/multi-GPU
  execution.
- Incremental decode is batch-1 focused.
- Handwritten kernels favor educational clarity and measured bottlenecks, not
  competition with mature inference libraries.
- Results were measured on one laptop GPU and may not generalize.

## Further reading

- [Architecture](docs/architecture.md)
- [Performance](docs/performance.md)
- [Interview notes](docs/interview_notes.md)
- [Release checklist](docs/release_checklist.md)
- [Portfolio measurement report](reports/portfolio/v1.0.0.md)

## License

MIT; see [LICENSE](LICENSE).
