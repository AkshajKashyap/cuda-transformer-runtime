# Performance record

This is the authoritative measurement record for this repository. Results are
specific to the stated workload and RTX 3050 Laptop GPU; they are not claims
about other GPUs or production-serving throughput.

## Environment and timing boundaries

- WSL2 Ubuntu, NVIDIA GeForce RTX 3050 Laptop GPU (4 GiB), CUDA 13.3,
  GCC 13.3, CMake/Ninja.
- CMake generates explicit `sm_86` SASS for this development machine.
- Real-model results use FP32 `stories15M`: dim 288, intermediate 768,
  6 layers, 6 heads, vocabulary 32,000, context 256, batch 1.

| Measurement | Clock | Excluded work |
| --- | --- | --- |
| GPU cached decode and sequential prefill | CUDA events | checkpoint/tokenizer loading, workspace allocation, uploads |
| Logit selection | wall clock | setup; separates D2H+stream completion from CPU argmax when reported |
| End-to-end generation | wall clock | checkpoint/tokenizer file loading and prompt tokenization |
| Nsight trace | profiler timestamps | setup and prefill before the trace range |

The benchmark warms up before sampling and reports repeated measurements. For
fixed-history decode, history is prepared before timing and logical cache length
is restored between launches, so every interval represents the same position.

## Current normal-runtime baseline

| Workload | Result |
| --- | ---: |
| 64-token greedy continuation | about 266.8 tokens/s |
| 64-token greedy continuation | about 3.748 ms/generated token |
| Cached GPU decode at context 128 | about 2.30 ms in the original baseline sweep |
| 32k logits D2H + synchronization + CPU greedy argmax | about 0.248 ms/token |

The supplied release validation retained the headline values above. The benchmark
still emits the per-row values below; rerun it before publishing a table that
requires every raw row. This is deliberately explicit rather than inventing
numbers that were not retained in the release notes.

### Sequential correctness-first prefill scaling

| Prefix length | GPU/event result |
| ---: | --- |
| 8 | Rerun benchmark for recorded raw row |
| 16 | Rerun benchmark for recorded raw row |
| 32 | Rerun benchmark for recorded raw row |
| 64 | Rerun benchmark for recorded raw row |
| 128 | Rerun benchmark for recorded raw row |

Prefill is intentionally a repeated one-token cached decode path, not an
optimized parallel prompt pass. It may compute final RMSNorm and LM-head logits
for prefix tokens whose logits are not needed.

### Fixed-context cached decode scaling

| History | GPU/event result |
| ---: | --- |
| 8 | Rerun benchmark for recorded raw row |
| 16 | Rerun benchmark for recorded raw row |
| 32 | Rerun benchmark for recorded raw row |
| 64 | Rerun benchmark for recorded raw row |
| 128 | about 2.30 ms (original baseline sweep) |
| 192 | Rerun benchmark for recorded raw row |
| 255 | Rerun benchmark for recorded raw row |

### Host logits selection

| Component for device-produced `[1, 32000]` FP32 logits | Median/average record |
| --- | --- |
| D2H copy + required stream synchronization + CPU argmax | about 0.248 ms combined |
| D2H + synchronization alone | Rerun benchmark for separate row |
| CPU argmax scan alone | Rerun benchmark for separate row |

### End-to-end greedy generation

| New tokens | Wall-clock result |
| ---: | --- |
| 16 | Rerun benchmark for recorded raw row |
| 32 | Rerun benchmark for recorded raw row |
| 64 | about 266.8 tokens/s; about 3.748 ms/token |

The end-to-end number includes current GPU cached decode, logit readback,
stream synchronization, CPU argmax, and small host orchestration. It should
not be compared directly to the CUDA-event-only decode number.

## Nsight Systems diagnostic

A context-128 fixed-context decode trace found the following per token:

| Finding | Value |
| --- | ---: |
| GPU kernel launches | 141 |
| cuBLAS GEMV launches | 43 |
| Other, mostly tiny custom launches | 98 |
| Summed GPU kernel execution | about 0.96 ms |

This is profiling evidence that launch/orchestration overhead materially affects
the current runtime. It is not a claim that summed kernel time equals API or
end-to-end latency.

## Isolated fixed-context CUDA Graph experiment

This is a feasibility diagnostic, not the normal runtime path. At history 128,
the graph captures one already-prefilled decode with fixed position/length and
compares its replay with ordinary decode in the same run.

| Measurement | Result |
| --- | ---: |
| Ordinary decode median | 2.54505 ms |
| Graph replay median | 1.02687 ms |
| Reduction | 1.51818 ms (59.65%) |
| Replay speedup | 2.478× |
| Graph logits error vs ordinary | 0 |
| K/V cache state | expected byte-identical result |
| Capture cost | 3.38975 ms |
| Instantiation cost | 0.98026 ms |

The graph cannot serve arbitrary generation: RoPE position, cache append slot,
attention length, and some launch parameters are captured at one context. No
2.478× general-generation or production-runtime claim is made.

## Reproduction and profiling commands

```bash
./build/src/cuda_llama2_benchmark models/stories15M.bin models/tokenizer.bin

nsys --version
ncu --version
nsys profile --trace=cuda,cublas --sample=none --capture-range=cudaProfilerApi \
  --capture-range-end=stop --force-overwrite true -o decode_context128 \
  ./build/src/cuda_llama2_benchmark models/stories15M.bin models/tokenizer.bin \
  --trace-decode-context 128 --trace-iterations 20

./build/src/cuda_llama2_benchmark models/stories15M.bin models/tokenizer.bin \
  --graph-decode-context 128
```

Use Nsight Systems to identify CPU/GPU timeline gaps, API overhead, launch
counts, and cuBLAS activity. Use Nsight Compute only after a specific kernel is
identified; it answers per-kernel occupancy, memory, and instruction questions
but is not a whole-runtime timeline tool.

## Interpretation limits

Laptop clocks, power limits, thermals, and background work can vary. Small
workloads can be dominated by fixed launch overhead, making rows non-monotonic.
CUDA events measure queued GPU work, not host API time; wall-clock generation
includes host work. Measure again before drawing an optimization conclusion.
