# Changelog

## Unreleased — recommended v1.1.0

Backward-compatible portfolio-release hardening after `v1.0.0`.

- Added real `stories15M` checkpoint loading, legacy tokenizer integration, and
  deterministic end-to-end greedy text generation.
- Added model-level full and KV-cached execution, per-layer cache validation,
  real-model correctness coverage, and CPU/CUDA logit comparison.
- Added real-model benchmark, Nsight Systems trace mode, and an isolated
  fixed-context CUDA Graph feasibility experiment.
- Updated architecture, performance, README, release, and interview materials
  to distinguish normal runtime results from diagnostic graph measurements.

## v1.0.0 — 2026-08-29

Portfolio release of the educational CUDA transformer runtime.

- FP32 full-sequence and KV-cached incremental LLaMA-style decoder block.
- CPU/GPU correctness coverage with 18/18 real-GPU CTests passing.
- CUDA-event benchmarks, stage profiling, and Nsight Systems trace support.
- Reusable incremental-attention workspace removes steady-state dynamic GPU
  allocation.
- Cooperative long-context P×V reduction with RTX 3050 measured dispatch.
- Architecture, performance, interview, release, and portfolio-result docs.
