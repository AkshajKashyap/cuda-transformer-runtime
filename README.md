# CUDA Transformer Runtime

A from-scratch CUDA/C++ implementation of a LLaMA-style decoder block with
KV-cached incremental decoding, CPU-vs-GPU correctness tests,
profiling-driven optimization, and reproducible performance benchmarks.

This is a portfolio and educational systems project, not a complete LLM serving
engine. It focuses on the parts of transformer inference that expose GPU memory
layout, CUDA kernels, cuBLAS integration, numerical correctness, and practical
performance engineering.

## At a glance

- FP32 pre-norm LLaMA-style decoder blocks and tiny model-level forward path
  in C++20/CUDA.
- Full-sequence, model-level KV-cached one-token decoding, and deterministic
  greedy generation.
- Complete model-level LLaMA-style forward execution from integer token IDs to
  vocabulary logits using deterministic or caller-supplied weights.
- CPU reference implementations and focused CTest coverage for each composed
  stage.
- Legacy `llama2.c` FP32 checkpoint loading with one-time layout adaptation for
  the `stories15M.bin` TinyStories model.
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

### Optional real checkpoint

The repository does not track model files. Download the legacy FP32
`llama2.c` checkpoint locally, then run the opt-in CPU-vs-CUDA integration
check with integer token IDs:

```bash
mkdir -p models
curl -L https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin \
  -o models/stories15M.bin
./build/src/cuda_llama2_checkpoint_integration models/stories15M.bin
```

`models/*.bin` is gitignored. The loader supports the original 28-byte legacy
FP32 header used by this file, not the newer versioned or quantized llama2.c
formats.

### Optional standard tokenizer and text generation

The matching legacy 32k tokenizer is also local-only. The tokenizer file has
no vocabulary-size field, so the loader receives and verifies the vocabulary
size from the loaded checkpoint.

```bash
curl -L https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin \
  -o models/tokenizer.bin
./build/src/cuda_llama2_tokenizer_integration models/tokenizer.bin
./build/src/cuda_llama2_generate models/stories15M.bin models/tokenizer.bin \
  "Once upon a time" 64
```

The generation CLI prints the original prompt plus continuation and the
generated token IDs. It follows the authoritative llama2.c **non-chat** loop:
a generated BOS token (`1`) is the sequence delimiter. This is intentionally
scoped to the target reference behavior; EOS remains token `2` and is not a
universal stopping convention. BOS/EOS marker strings are omitted from the
user-facing rendered continuation.

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
naive/tiled/cuBLAS GEMM, RMSNorm, LayerNorm, stable softmax, RoPE, causal
attention, SwiGLU, KV-cache operations, decoder blocks, and a tiny complete
model forward path. The full-sequence path materializes causal-attention
scores/probabilities for clarity; the incremental path uses a persistent K/V
cache and workspace-owned scratch.

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

### Tiny model-level forward

```text
device token IDs [sequence]
→ embedding lookup [sequence, hidden]
→ configurable count of full-sequence decoder blocks
→ final RMSNorm
→ LM-head projection
→ logits [sequence, vocabulary]
```

The core model API is device-native and enqueues this path without a host-to-
device token copy or synchronization. A separate convenience wrapper accepts
host token IDs, validates them, and copies them into workspace-owned device
storage. The model uses two reusable activation buffers: layers alternate their
input/output roles, so odd and even layer counts both select the correct final
activation without dynamic GPU allocation during forward execution.

### Legacy llama2.c checkpoint boundary

`Llama2CheckpointModel` owns adapted host tensors for the independent CPU
oracle and one-time CUDA allocations for the runtime. It provides non-owning
`TinyModelWeights` views for each path, so ordinary full forward, cached decode,
and greedy generation do not allocate or upload model weights.

The target legacy file stores PyTorch linear matrices as `[output, input]`,
whereas this runtime evaluates row-major `X[rows, input] * W[input, output]`.
The loader therefore explicitly transposes Wq/Wk/Wv/Wo, W1/W2/W3, and the
classifier during setup. Embeddings and RMSNorm scales are copied directly.
For tied classifiers, the embedding table is still transposed into a separate
`[hidden, vocabulary]` LM-head allocation; this preserves mathematical tying
while satisfying the two runtime layouts. Legacy RoPE frequency arrays are
validated in the file size then skipped because RoPE is computed algorithmically.

Only equal query/KV head counts are currently supported: checkpoints with
`n_kv_heads != n_heads` are rejected rather than treated as grouped-query
attention. The runtime and legacy reference both use even/odd RoPE pairs with
base 10000 and RMSNorm epsilon `1e-5`.

### Legacy llama2.c tokenizer

`Llama2Tokenizer` is a CPU-only RAII loader for the standard legacy binary:
one native little-endian `uint32` maximum token length, then a caller-supplied
number of entries, each `float32` score, `uint32` byte length, and exact raw
bytes. It rejects truncation, invalid sizes/scores, missing dummy-space token,
and trailing bytes.

Encoding matches `llama2.c`: optional BOS `1`, optional EOS `2`, a dummy token
whose bytes are exactly one ASCII space for non-empty input, codepoint lookup,
byte fallback `byte + 3`, then repeated highest-score adjacent BPE merging.
Exact score ties keep the leftmost candidate. Decode removes one leading ASCII
space after BOS and turns exact `<0xXX>` pieces back into raw bytes. The
standard tokenizer integration checks the authoritative reference prompt
`I believe the meaning of life is` against IDs
`[1, 306, 4658, 278, 6593, 310, 2834, 338]`.

### Tiny model-level incremental decode

```text
device token ID [1]
→ embedding lookup [1, hidden]
→ incremental decoder block 0 with cache 0
→ ...
→ incremental decoder block N-1 with cache N-1
→ final RMSNorm → LM-head projection → logits [1, vocabulary]
```

Each layer owns independent K/V cache storage and one-token scratch. The
incremental workspace receives an explicit `max_sequence_length` cache capacity;
it is distinct from the full-sequence length used by `tiny_model_forward_cuda`.
Before decoding, the runtime checks every layer cache/workspace for matching
shape, capacity, and logical length, preventing predictable partial cache
advancement. The correctness-oriented prefill helper repeats this same
incremental path for each prefix token; it is not an optimized parallel prefill.

### Deterministic greedy generation

`tiny_model_generate_greedy_cuda` is a small host-orchestrated wrapper around
the existing cached decode path. For a non-empty host prompt, it logically
resets the incremental workspace, processes every prompt token once, copies the
last prompt token's `[vocabulary]` logits to the host, and selects the largest
logit. Exact ties select the lowest token ID. Each selected token is decoded
only when its successor logits are needed; no custom GPU argmax is used.

For prompt `[7, 19, 4]`, if the logits after `4` choose `82`, logits after
processing `82` choose `11`, and logits after processing `11` choose `39`, then
`max_new_tokens=3` returns `[82, 11, 39]`. The cache contains
`[7, 19, 4, 82, 11]` and has logical length 5: `39` is returned but not decoded,
because its successor is not requested. In general a nonzero request processes
`prompt_length + max_new_tokens - 1` tokens. This requirement is preflighted
against cache capacity before the helper resets or mutates predictable state.

Greedy generation synchronizes after each device-to-host logits copy so the CPU
can select the next ID. The underlying device-native one-token decode remains
asynchronous and allocation-free after setup. This is correctness-first model
composition, not a tokens/sec benchmark or a production generation loop.

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
They also validate complete decoder blocks and the tiny model's CPU/GPU logits.
Exact test cases and tolerances are kept in the focused test sources under
`src/`.

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
- `src/cuda/tiny_model.cu`: device-native embedding-to-logits model
  orchestration, CPU correctness oracle, and model-level cached decoding.
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
- Legacy FP32 stories15M checkpoint loading, 32k llama2.c tokenizer-compatible
  text encoding/decoding, and CUDA greedy text generation are supported.
- No tokenizer variants beyond the legacy llama2.c binary, no probabilistic
  sampling, and no general pretrained-model compatibility layer.
- Model-level prefill is sequential and correctness-first, not optimized.
- Only legacy FP32 llama2.c checkpoints with equal query/KV head counts are
  supported; no newer versioned/quantized formats or grouped-query attention.
- No FlashAttention, paged KV cache, CUDA Graphs, or multi-GPU execution.
- Batch size is 1 for both the tiny full-sequence model and incremental decode.
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
