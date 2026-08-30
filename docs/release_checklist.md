# v1.0.0 release checklist

- [x] CMake/Ninja build succeeds in the validated WSL2 CUDA environment.
- [x] Real RTX 3050 result: 18/18 CTests pass.
- [x] Standalone GEMM, incremental-attention, P×V, and decoder-block benchmarks exist.
- [x] Authoritative benchmark tables are recorded in the README and performance report.
- [x] README, architecture, performance, and interview documentation are present.
- [x] Limitations and hardware-specific measurement caveats are explicit.
- [x] Generated Nsight artifacts are ignored; Markdown reports are tracked instead.
- [x] `git diff --check` is clean for the release change set.
- [ ] Run the final real-GPU build, CTest, and benchmark commands listed below.
- [ ] Review `git status`, commit the release, and create tag `v1.0.0`.
- [ ] Create the GitHub release and attach/link only textual benchmark results.

There is no GPU GitHub Actions workflow: hosted runners do not provide the
validated CUDA hardware path. Real-GPU validation is documented rather than
simulated.
