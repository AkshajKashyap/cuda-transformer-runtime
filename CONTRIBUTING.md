# Contributing

This project values small, measurable, correctness-first changes.

1. Keep new work scoped to one clear purpose.
2. Preserve CPU references and existing numerical tolerances unless a justified
   correctness issue requires changing them.
3. Build with CMake/Ninja and run CTest on a CUDA-capable system.
4. For performance changes, keep allocations and transfers outside CUDA-event
   timing, use warmups and repeated batches, and report the hardware/context.
5. Do not commit generated build directories, Nsight reports, SQLite exports,
   or other large profiler artifacts.

Open an issue or discussion before broad architectural changes. This repository
is intentionally a focused educational runtime, not a general framework.
