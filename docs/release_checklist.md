# Next release checklist

This checklist is for the post-`v1.0.0` portfolio release. The recommended
version is `v1.1.0`: the new model/generation/profiling surface is substantial
but does not break the existing public work.

- [x] CMake/Ninja build is documented for the validated WSL2 CUDA environment.
- [x] Real RTX 3050 validation recorded: 21/21 CTests pass.
- [x] `stories15M` checkpoint, standard tokenizer, model CPU/CUDA logits, and
  greedy text generation are validated against the target reference path.
- [x] README accurately says this is an educational correctness-first runtime,
  not a serving engine.
- [x] Architecture documents host/device responsibilities, persistent
  workspaces, asynchronous execution, cache ownership, and graph boundaries.
- [x] Performance document separates normal measurements, profiling evidence,
  and the isolated fixed-context CUDA Graph experiment.
- [x] External checkpoint/tokenizer binaries and generated profiling artifacts
  are ignored rather than tracked.
- [x] No unfinished required-work TODO/FIXME markers were found in source/docs.
- [ ] If publishing all performance tables, rerun the real-model benchmark and
  preserve every raw sequential-prefill/decode/generation row in the release.
- [ ] Run final real-GPU commands below after the documentation change.
- [ ] Review `git status`, commit, tag `v1.1.0`, and create a release. This
  repository intentionally has no GPU GitHub Actions validation path.

```bash
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
git diff --check

./build/src/cuda_llama2_tokenizer_integration models/tokenizer.bin
./build/src/cuda_llama2_checkpoint_integration models/stories15M.bin
./build/src/cuda_llama2_generate models/stories15M.bin models/tokenizer.bin \
  "I believe the meaning of life is" 64
./build/src/cuda_llama2_benchmark models/stories15M.bin models/tokenizer.bin
```
