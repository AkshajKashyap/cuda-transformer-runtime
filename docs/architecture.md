# Architecture

`cuda-transformer-runtime` is a small FP32 C++20/CUDA runtime with LLaMA-style
pre-norm decoder blocks and a tiny model-level embedding-to-logits path. It
deliberately exposes buffers and stages instead of hiding them behind a general
tensor framework.

## Layouts

Full-sequence activations use contiguous row-major `[sequence, hidden]` storage.
QKV projection initially produces the same token-major form. Attention instead
uses contiguous per-head sequences, so Q, K, and V are explicitly repacked to
`[batch=1, heads, sequence, head_dim]`. The reverse repack returns attention
output to token-major form for the output projection.

The persistent KV cache owns separate FP32 K and V buffers with layout
`[batch, heads, max_sequence, head_dim]`. This is intentionally fixed capacity
and caller-owned: cache lifetime is independent of decoder-block workspace
lifetime.

## Full-sequence flow

`decoder_block_cuda` composes existing attention and MLP sublayers:

```text
X → RMSNorm → QKV → repack → RoPE(Q,K) → causal attention
  → reverse repack → Wo → X + attention
  → RMSNorm → W_gate/W_up → SiLU(gate) * up → W_down
  → attention residual + MLP output
```

Full attention materializes `[batch, heads, query, key]` score and probability
buffers. This O(sequence²) baseline is intentional: it is easy to audit and
acts as a correctness reference for later work.

## Tiny model flow

`tiny_model_forward_cuda` is device-native: callers supply device token IDs
`[sequence]`, a caller-owned logits buffer `[sequence, vocabulary_size]`, a
cuBLAS handle, and caller-owned device weights. It enqueues:

```text
token IDs → embedding lookup → decoder block 0 → ... → decoder block N-1
          → final RMSNorm → LM-head GEMM → vocabulary logits
```

The model workspace owns two `[sequence, hidden]` activation buffers. Embedding
writes A, each decoder block reads one activation buffer and writes the other,
and final RMSNorm writes to the unused buffer before LM-head projection. Thus
both odd and even layer counts use the same ping-pong loop. It also owns a
device token-ID buffer only for the optional host-token convenience wrapper and
an array of reusable `DecoderBlockWorkspace` objects. Setup allocates these
buffers once; successful core forwards allocate and free nothing.

## Incremental flow

`incremental_decoder_block_cuda` accepts one `[1, hidden]` token plus an
external `KvCache` and produces one output token:

```text
X_t → attention RMSNorm → QKV → token/head repack
    → RoPE at position cache.current_length
    → append rotated K_t and raw V_t
    → cached QK / softmax / P×V → reverse repack → Wo → residual
    → MLP RMSNorm → gate/up → SwiGLU → down → residual
```

`current_length` is the absolute position of the new token before append. Q
and new K are rotated once at that position. K is stored after RoPE because
future queries should dot against position-aware keys; V is not position-rotated
and is stored raw. Historical keys are never rotated again.

## Ownership and workspaces

Callers own model-weight buffers, input/output buffers, the cuBLAS handle, and
the KV cache. Workspaces are non-copyable owners of temporary device buffers:

- `QkvProjectionWorkspace` owns token/head-major QKV buffers.
- `IncrementalAttentionWorkspace` owns rotated-Q, rotated-K, and
  `[heads, max_sequence]` score/probability scratch.
- `IncrementalDecoderBlockWorkspace` embeds both the QKV and incremental
  attention workspaces plus decoder-block intermediates.

The decoder-block workspace stores `max_sequence`; its attention workspace and
the external cache must have matching capacities. This makes the allocation
boundary explicit: after cache and workspace creation, successful production
incremental decode does not call `cudaMalloc` or `cudaFree`.

## Dependency direction

The project keeps low-level ownership separate from higher-level composition:

```text
transformer primitives, GEMM, QKV layout, KV cache
        ↓
attention / incremental attention / QKV projection
        ↓
attention sublayer and MLP sublayer
        ↓
decoder block and incremental decoder block
```

`cuda_kv_cache` does not depend on attention. This allows cache semantics to be
tested independently.

## cuBLAS and row-major matrices

Repository matrices are row-major. cuBLAS is column-major, so the linear helper
views the same bytes as transposes and computes `C^T = B^T * A^T`; this produces
row-major `C = A * B` without physical transpose buffers. The wrapper requests
pedantic FP32 math so Ampere TF32 is not silently selected in this project.
