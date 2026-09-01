# Technical interview notes

These notes are a study guide for explaining the implementation precisely.
They describe the repository, not a claim that it is a production serving stack.

## Concepts to explain

- **Kernel versus runtime:** a kernel is one GPU function launch; the runtime
  owns composition, memory/workspace lifetime, streams, error boundaries,
  weights, token flow, and generation control.
- **Training versus inference:** training needs backward passes, gradients, and
  optimizer state. This project performs FP32 forward inference only.
- **Transformer path:** token IDs select embedding rows; each pre-norm decoder
  layer applies attention plus residual, then SwiGLU MLP plus residual; final
  RMSNorm and the LM head produce vocabulary logits.
- **GEMM and cuBLAS:** learned projections dominate dense arithmetic and cuBLAS
  is the serious baseline. Handwritten kernels remain useful for specialized
  layout and attention stages. Row-major matrices are mapped through the
  column-major cuBLAS interface without temporary transpose buffers.
- **RMSNorm:** each token row is scaled by the reciprocal square root of its
  mean squared value plus epsilon, then by learned per-channel weights. Unlike
  LayerNorm, it does not subtract the mean.
- **RoPE:** even/odd Q and K pairs are rotated by angles determined by absolute
  position. K is cached after rotation; V is cached raw.
- **Attention:** scores are `QK^T / sqrt(head_dim)`, future keys are causally
  masked, stable softmax normalizes each row, and probabilities weight V.
- **KV cache:** each layer owns independent fixed-capacity K/V buffers. It
  avoids recomputing historical K/V, but new-token attention still grows
  roughly linearly with valid history.
- **Full prefix versus incremental decode:** full mode processes all positions
  together; incremental mode processes one token at the cache length. CPU/GPU
  tests compare their outputs.
- **Prefill:** this project sequentially invokes the same one-token decode for
  each prompt token. That maximizes reuse and correctness coverage, not speed.
- **Logits and greedy decoding:** the LM head emits one score per vocabulary
  item. The current driver copies logits to host, synchronizes, and uses a
  deterministic lowest-ID-on-tie argmax.
- **Tokenizer/BPE:** the legacy tokenizer starts from UTF-8 pieces/byte fallback
  then repeatedly chooses the highest-score adjacent merge. Its vocabulary size
  comes from the matching checkpoint, not the tokenizer file itself.
- **Checkpoint transpose boundary:** external linear weights are
  `[output, input]`; runtime row-major linear input is `[input, output]`.
  Loading transposes once rather than paying a transpose in each forward.
- **CPU versus CUDA validation:** CPU references independently assemble the
  same math with deterministic inputs. Tests check finite values, shapes,
  cache preservation, and expected outputs rather than CUDA calling CPU code.
- **Floating-point tolerance:** reductions and different execution orders make
  FP32 bitwise equality inappropriate. Tolerances are justified against output
  magnitude and are not loosened merely to hide failures.
- **Launch overhead:** many small kernels and cuBLAS calls can make host/API
  orchestration visible even when total GPU arithmetic is modest.
- **CUDA events:** events measure elapsed GPU work on a stream; they do not
  include CPU orchestration or arbitrary host synchronization.
- **Nsight Systems versus Nsight Compute:** Systems gives a whole timeline,
  launch/API counts, and gaps; Compute diagnoses a selected kernel's memory,
  occupancy, and instruction behavior.
- **Why the graph experiment exists:** a context-128 profile showed 141
  launches/token and about 0.96 ms summed kernels, motivating a fixed-context
  replay experiment to test launch-overhead sensitivity.
- **Why it cannot accelerate arbitrary generation yet:** capture fixes RoPE
  position, cache slot, valid attention length, and launch parameters. Dynamic
  contexts would need a designed graph-update/cache-state scheme, correctness
  validation, and a cost/benefit measurement.

## Project-specific interview questions

1. **What is the complete path from prompt text to the next token?**
   - Tokenize text to IDs, run cached model decode, copy logits to host, choose
     greedy argmax, then render/decode the selected piece.

2. **Why retain a device-native model forward API?**
   - It avoids making an H2D token copy fundamental to execution and keeps
     composition usable by a future device-side scheduler.

3. **What makes an incremental cache append safe at model level?**
   - Before launching any layer, validate every cache length, capacity, shape,
     layer count, and workspace so predictable failures cannot advance only
     early layers. It cannot roll back arbitrary asynchronous device failures.

4. **Why does every layer need its own KV cache?**
   - Each layer produces a different representation and therefore distinct K/V
     tensors. Sharing a cache would corrupt attention history.

5. **Why cache rotated K rather than raw K?**
   - Future queries must attend to each key at the key's original absolute
     position; rotating once at append preserves that relationship.

6. **Why use stable softmax?**
   - Subtracting the row maximum before exponentiation prevents overflow while
     preserving normalized probabilities.

7. **Why use cuBLAS if the project is about CUDA?**
   - cuBLAS is the practical baseline for dense projections. Custom kernels
     teach and cover stages where the specialized layout is explicit.

8. **How is row-major GEMM expressed through column-major cuBLAS?**
   - View row-major byte storage as transposed column-major matrices and swap
     operand order so cuBLAS computes the transpose of the desired product.

9. **Why is sequential prefill intentionally retained?**
   - It exercises exactly the validated decode logic and avoids duplicated
     algorithms. It is a known performance limitation, not a hidden feature.

10. **What does the CPU oracle prove, and what does it not prove?**
    - It catches indexing, layout, masking, and numerical composition errors;
      it does not prove high performance or exact equality across all FP modes.

11. **Why use magnitude-aware FP32 tolerances?**
    - Dot products and reductions accumulate rounding error proportional to
      value scale; a fixed bitwise test would reject valid alternate orders.

12. **What did Nsight Systems reveal in the real-model decode?**
    - At context 128, 141 launches/token—43 GEMVs and 98 mostly tiny kernels—
      with about 0.96 ms summed kernel execution, motivating overhead study.

13. **What exactly did the CUDA Graph experiment demonstrate?**
    - For one fixed history 128 capture, replay reduced ordinary same-run
      latency from 2.54505 ms to 1.02687 ms with zero logit error; it validates
      launch overhead as a bottleneck, not general graph generation.

14. **Why does fixed-context graph replay not work for arbitrary decoding?**
    - Each new token changes position, cache length, append address, and
      attention work. Those dependencies are baked into the current capture.

15. **What would dynamic-context graph support require?**
    - A carefully designed update or family-of-graphs policy, stable buffers
      and launch contracts, cache-length handling, and measurements showing
      replay/update costs beat ordinary execution.

16. **Why keep logits selection on the host for now?**
    - It is simple and deterministic for correctness. The measured combined
      D2H/sync/argmax cost is a concrete baseline before changing that design.
