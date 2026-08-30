# Interview notes

## Why use cuBLAS as well as handwritten GEMM?

Handwritten GEMM teaches indexing, tiling, shared memory, and correctness.
cuBLAS is the serious baseline for linear projections: it is mature, tuned, and
lets the project focus optimization effort where profiling shows a bottleneck.

## Why is row-major cuBLAS non-obvious?

cuBLAS expects column-major storage. A row-major `A[M,K]` has the same bytes as
a column-major `A^T[K,M]`, so the wrapper computes `C^T = B^T * A^T` with
swapped operands. No transpose allocation is required.

## Why repack token-major QKV?

Projection output is naturally `[sequence, hidden]`, but attention wants each
head’s time sequence contiguous. Repacking into `[heads, sequence, head_dim]`
makes the attention indexing direct and visible.

## What is RoPE, and why cache rotated K?

RoPE rotates even/odd Q/K pairs by an angle determined by absolute position.
Each new Q and K is rotated once. Cached K stays rotated because later queries
must use its original position; V is raw because RoPE does not apply to V.

## Why does KV caching help?

Without a cache, each next token recomputes projections and attention for the
whole prefix. Cached decode stores old K/V and computes only the new token,
although its QK and P×V work still grows linearly with valid history.

## Why can cached decode lose at short contexts?

At small histories, several small kernel launches and cuBLAS calls dominate.
Avoided arithmetic is not yet large enough to overcome fixed GPU launch/API
costs.

## Why did per-token allocation matter?

Nsight exposed three `cudaMalloc` and three `cudaFree` calls per token inside
incremental attention. Moving that scratch into a reusable workspace removed
allocation from the steady-state hot path.

## Why did P×V dominate at long context?

The original kernel had only one thread per output `(head, dimension)`, each
serially reducing every cached value. At long histories that leaves too little
parallel work. The replacement uses one block per output and cooperatively
reduces history in shared memory.

## Why an empirical dispatch threshold?

The cooperative kernel adds block and reduction overhead, so it is not always
faster. The 512 cutoff came from the RTX 3050 P×V benchmark; another GPU or
shape may need a different measurement.

## Why do GPU timings vary on a laptop?

DVFS, thermals, power limits, launch overhead, and background activity affect
short timings. CUDA-event kernel time, CPU API overhead, and end-to-end latency
measure different things.

## What would be optimized next?

Profile again first. Likely candidates include launch count/fusion, small GEMM
behavior, and remaining long-context attention work—but none should be changed
without new evidence.

## What is needed for full LLM inference?

A model-weight loader, tokenizer, embeddings, LM head, sampling/generation
loop, multiple layers, lower-precision paths, and production memory-management
strategies such as a paged KV cache.
