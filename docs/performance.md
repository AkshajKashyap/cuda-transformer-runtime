# Performance and profiling

## Measured configuration

Results below were collected on the RTX 3050 Laptop GPU in WSL2 for one FP32
decoder block: batch 1, hidden 256, 4 heads × 64 head dimensions, intermediate
size 512. CUDA events measure kernel-only latency. Allocations and host/device
transfers are outside timed regions.

Each benchmark uses warmups, nine independent timing batches, and median
per-execution latency. For fixed-history incremental measurements, K/V history
is prefilled outside timing and `current_length` is restored before every
launch, so the same deterministic position is overwritten each time.

## End-to-end decoder-block decode

| History | Full-prefix ms | Cached ms | Speedup |
| ---: | ---: | ---: | ---: |
| 16 | 0.56269 | 0.44469 | 1.27× |
| 32 | 0.47918 | 0.38257 | 1.25× |
| 64 | 0.48190 | 0.41284 | 1.17× |
| 128 | 0.47434 | 0.39564 | 1.20× |
| 256 | 0.85171 | 0.42340 | 2.01× |
| 512 | 2.42635 | 0.34323 | 7.07× |
| 1024 | 7.86487 | 0.41812 | 18.81× |

The full path recomputes a prefix of `S+1` tokens. Cached decode computes one
new token against history `S`; it does not represent full-model throughput.

## P×V microbenchmark

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

The serial kernel assigns one thread per `(head, output_dimension)` and loops
over history. The cooperative version assigns one block per output element and
reduces strided history partials in shared memory. Production uses serial below
512 and cooperative at 512+, an empirical cutoff for this GPU and shape.

## Nsight-driven changes

Initial traces of 20 production decode calls showed three `cudaMalloc` and
three `cudaFree` calls per token from rotated Q, rotated K, and score scratch.
Those buffers moved into `IncrementalAttentionWorkspace`; steady-state decode
now has zero dynamic GPU allocation. No precise cross-run speedup is claimed
because laptop conditions vary.

After that change, Nsight showed long-context P×V as the major custom-kernel
bottleneck: at history 1024, the serial kernel was about 94 µs in the earlier
profile. The controlled P×V benchmark above motivated the cooperative kernel.

## Interpretation limits

Laptop DVFS, thermal state, power limits, other GPU activity, and fixed launch
overhead can make small workloads non-monotonic. CUDA event latency is neither
CPU API time nor end-to-end application latency. Treat these as reproducible
measurements for the stated workload, not universal performance claims.
