# CUDA Transformer Runtime

A from-scratch C++20/CUDA educational transformer inference runtime. It loads
a real legacy `llama2.c`-compatible `stories15M.bin` checkpoint and runs the
complete path from UTF-8 prompt through BPE tokenization, LLaMA-style decoder
layers, KV-cached greedy decoding, and generated text.

The project is correctness-first: CUDA results are checked against independent
CPU/reference paths, and its performance work uses real GPU benchmarks and
Nsight Systems. It is a learning and portfolio project, not production LLM
serving infrastructure.

## Key results

| Result | Validated result on the RTX 3050 Laptop GPU |
| --- | --- |
| Correctness suite | 21/21 CTest tests passed |
| Real model | `stories15M`: dim 288, intermediate 768, 6 layers, 6 heads, vocab 32,000, context 256 |
| Tokenizer reference | `I believe the meaning of life is` → `[1, 306, 4658, 278, 6593, 310, 2834, 338]` |
| CPU vs CUDA logits | maximum absolute error `2.5629997e-05` |
| Greedy generation | matches the tested authoritative `llama2.c` output |
| Normal 64-token continuation | about 266.8 tokens/s, 3.748 ms/generated token end-to-end |
| Normal cached decode, context 128 | about 2.30 ms GPU latency in the baseline sweep |
| Logits transfer + sync + CPU argmax | about 0.248 ms/token |
| Fixed-context CUDA Graph diagnostic | 2.54505 ms ordinary → 1.02687 ms replay at history 128 (59.65% reduction, 2.478×) |

The CUDA Graph number is deliberately narrow: it is an isolated replay of one
captured, fixed-context decode. It is not integrated into arbitrary
autoregressive generation and is not an end-to-end generation speedup claim.

## Execution path

```text
UTF-8 prompt → legacy BPE tokenizer → token IDs
    → embeddings → N pre-norm LLaMA-style decoder layers
    → final RMSNorm → LM-head projection → vocabulary logits
    → host greedy argmax → next token → rendered text
```

The core model APIs are device-native and asynchronous. The generation driver
uses small host wrappers only where it must supply token IDs or read logits for
the current host-side greedy selection. More detail is in
[architecture notes](docs/architecture.md).

## Build and test

The development machine is an Ampere RTX 3050 Laptop GPU. CMake defaults to
explicit `sm_86` SASS generation; users on another GPU can override it.

```bash
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure

# Example override for another CUDA architecture
cmake -S . -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=89-real
```

## Reproduce real-model integration

Model and tokenizer binaries are intentionally local-only and gitignored.

```bash
mkdir -p models
curl -L https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin \
  -o models/stories15M.bin
curl -L https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin \
  -o models/tokenizer.bin

./build/src/cuda_llama2_tokenizer_integration models/tokenizer.bin
./build/src/cuda_llama2_checkpoint_integration models/stories15M.bin
./build/src/cuda_llama2_generate models/stories15M.bin models/tokenizer.bin \
  "I believe the meaning of life is" 64
./build/src/cuda_llama2_benchmark models/stories15M.bin models/tokenizer.bin
```

Generation follows the authoritative `llama2.c` non-chat loop: a generated
token ID `1` terminates that loop. This is a compatibility choice for this
specific reference path, not a universal Llama stopping convention.

## Profiling and graph diagnostic

The normal benchmark reports production-path measurements. The following
commands are optional diagnostics; they do not change the runtime path.

```bash
# Capture only repeated, already-prefilled fixed-context decode work.
nsys profile --trace=cuda,cublas --sample=none --capture-range=cudaProfilerApi \
  --capture-range-end=stop --force-overwrite true \
  -o decode_context128 \
  ./build/src/cuda_llama2_benchmark models/stories15M.bin models/tokenizer.bin \
  --trace-decode-context 128 --trace-iterations 20

# Isolated fixed-context CUDA Graph feasibility experiment.
./build/src/cuda_llama2_benchmark models/stories15M.bin models/tokenizer.bin \
  --graph-decode-context 128
```

See [performance measurements](docs/performance.md) for timing boundaries,
known results, profiling evidence, and interpretation limits.

## Current limitations

- FP32 only; batch size 1 only.
- Legacy FP32 `llama2.c` checkpoint format and legacy 32k tokenizer target.
- Equal query/KV head counts only: no GQA or MQA.
- Correctness-first sequential incremental prefill, not optimized parallel
  prefill.
- Host-orchestrated greedy selection; no sampling or production generation
  scheduler.
- No FlashAttention, paged KV cache, production CUDA Graph path, continuous
  batching, or multi-GPU execution.
- Not intended to compete with vLLM, TensorRT-LLM, llama.cpp, or a production
  model-serving stack.

## Repository guide

- [Architecture](docs/architecture.md): data flow, layouts, ownership, and
  full versus cached execution.
- [Performance](docs/performance.md): authoritative real-model and diagnostic
  measurement record.
- [Interview notes](docs/interview_notes.md): concise technical study guide.
- [Release checklist](docs/release_checklist.md): final validation steps.

## License

This repository is released under the MIT License. See [LICENSE](LICENSE).
