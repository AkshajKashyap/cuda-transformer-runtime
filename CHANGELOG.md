# Changelog

## v1.0.0 — 2026-08-29

Portfolio release of the educational CUDA transformer runtime.

- FP32 full-sequence and KV-cached incremental LLaMA-style decoder block.
- CPU/GPU correctness coverage with 18/18 real-GPU CTests passing.
- CUDA-event benchmarks, stage profiling, and Nsight Systems trace support.
- Reusable incremental-attention workspace removes steady-state dynamic GPU
  allocation.
- Cooperative long-context P×V reduction with RTX 3050 measured dispatch.
- Architecture, performance, interview, release, and portfolio-result docs.
