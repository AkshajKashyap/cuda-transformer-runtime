#include "cuda_transformer/attention.h"
#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/incremental_attention.h"
#include "cuda_transformer/kv_cache.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
bool close(float a, float b) {
  return std::fabs(a - b) <= 4.0e-4F + 1.0e-4F * fmaxf(1.0F, std::fabs(b));
}

bool run_case(std::size_t heads, std::size_t sequence, std::size_t dim,
              std::size_t prefill) {
  cuda_transformer::AttentionShape shape{1, heads, sequence, dim};
  const std::size_t elements = heads * sequence * dim;
  std::vector<float> q(elements), k(elements), v(elements), rotated_q(elements),
      probabilities(heads * sequence * sequence), reference(elements);
  for (std::size_t i = 0; i < elements; ++i) {
    q[i] = static_cast<float>(static_cast<int>(i % 29) - 14) * 0.0625F;
    k[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.0625F;
    v[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125F;
  }
  cuda_transformer::attention_cpu(q.data(), k.data(), v.data(),
                                  rotated_q.data(), probabilities.data(),
                                  reference.data(), shape);
  cuda_transformer::KvCache cache;
  cuda_transformer::IncrementalAttentionWorkspace workspace;
  cuda_transformer::IncrementalAttentionWorkspace wrong_heads_workspace;
  cuda_transformer::IncrementalAttentionWorkspace wrong_dim_workspace;
  cuda_transformer::IncrementalAttentionWorkspace wrong_capacity_workspace;
  if (!CTR_CUDA_CHECK(
          cuda_transformer::kv_cache_create(&cache, 1, heads, sequence, dim)))
    return false;
  float *dq = nullptr, *dk = nullptr, *dv = nullptr, *do_ = nullptr,
        *dfull = nullptr, *drot = nullptr;
  auto clean = [&](bool ok) {
    for (auto p : {dq, dk, dv, do_, dfull, drot})
      if (p && !CTR_CUDA_CHECK(cudaFree(p)))
        ok = false;
    for (auto* candidate : {&workspace, &wrong_heads_workspace,
                            &wrong_dim_workspace,
                            &wrong_capacity_workspace})
      if (!CTR_CUDA_CHECK(
              cuda_transformer::incremental_attention_workspace_destroy(candidate)))
        ok = false;
    return CTR_CUDA_CHECK(cuda_transformer::kv_cache_destroy(&cache)) && ok;
  };
  const std::size_t token_bytes = heads * dim * sizeof(float);
  if (!CTR_CUDA_CHECK(cudaMalloc(&dq, token_bytes)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&dk, token_bytes)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&dv, token_bytes)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&do_, token_bytes)))
    return clean(false);
  if (!CTR_CUDA_CHECK(cuda_transformer::incremental_attention_workspace_create(
          &workspace, heads, dim, sequence)) ||
      !CTR_CUDA_CHECK(cuda_transformer::incremental_attention_workspace_create(
          &wrong_heads_workspace, heads + 1, dim, sequence)) ||
      !CTR_CUDA_CHECK(cuda_transformer::incremental_attention_workspace_create(
          &wrong_dim_workspace, heads, dim + 2, sequence)) ||
      !CTR_CUDA_CHECK(cuda_transformer::incremental_attention_workspace_create(
          &wrong_capacity_workspace, heads, dim, sequence + 1)))
    return clean(false);
  if (cuda_transformer::incremental_decode_with_workspace(
          &cache, dq, dk, dv, do_, nullptr) != cudaErrorInvalidValue ||
      cuda_transformer::incremental_decode_with_workspace(
          &cache, dq, dk, dv, do_, &wrong_heads_workspace) !=
          cudaErrorInvalidValue ||
      cuda_transformer::incremental_decode_with_workspace(
          &cache, dq, dk, dv, do_, &wrong_dim_workspace) !=
          cudaErrorInvalidValue ||
      cuda_transformer::incremental_decode_with_workspace(
          &cache, dq, dk, dv, do_, &wrong_capacity_workspace) !=
          cudaErrorInvalidValue) {
    std::fprintf(stderr, "incompatible incremental-attention workspace accepted\n");
    return clean(false);
  }
  const std::size_t saved_max_sequence = workspace.max_sequence;
  ++workspace.max_sequence;
  const bool mutated_workspace_rejected =
      cuda_transformer::incremental_decode_with_workspace(
          &cache, dq, dk, dv, do_, &workspace) == cudaErrorInvalidValue;
  workspace.max_sequence = saved_max_sequence;
  if (!mutated_workspace_rejected) {
    std::fprintf(stderr, "mutated incremental-attention workspace accepted\n");
    return clean(false);
  }
  bool ok = true;
  if (prefill) {
    if (!CTR_CUDA_CHECK(cudaMalloc(&dfull, elements * sizeof(float))) ||
        !CTR_CUDA_CHECK(cudaMalloc(&drot, elements * sizeof(float))) ||
        !CTR_CUDA_CHECK(cudaMemcpy(dfull, k.data(), elements * sizeof(float),
                                   cudaMemcpyHostToDevice)) ||
        !CTR_CUDA_CHECK(cuda_transformer::rope_cuda(dfull, drot, shape)) ||
        !CTR_CUDA_CHECK(cudaDeviceSynchronize()))
      return clean(false);
    std::vector<float> rotated_k(elements), compact(heads * prefill * dim),
        compact_v(compact.size());
    if (!CTR_CUDA_CHECK(cudaMemcpy(rotated_k.data(), drot,
                                   elements * sizeof(float),
                                   cudaMemcpyDeviceToHost)))
      return clean(false);
    for (size_t h = 0; h < heads; h++)
      for (size_t t = 0; t < prefill; t++)
        for (size_t d = 0; d < dim; d++) {
          compact[(h * prefill + t) * dim + d] =
              rotated_k[(h * sequence + t) * dim + d];
          compact_v[(h * prefill + t) * dim + d] =
              v[(h * sequence + t) * dim + d];
        }
    float *cp = nullptr, *cv = nullptr;
    if (!CTR_CUDA_CHECK(cudaMalloc(&cp, compact.size() * sizeof(float))) ||
        !CTR_CUDA_CHECK(cudaMalloc(&cv, compact.size() * sizeof(float))) ||
        !CTR_CUDA_CHECK(cudaMemcpy(cp, compact.data(),
                                   compact.size() * sizeof(float),
                                   cudaMemcpyHostToDevice)) ||
        !CTR_CUDA_CHECK(cudaMemcpy(cv, compact_v.data(),
                                   compact_v.size() * sizeof(float),
                                   cudaMemcpyHostToDevice)) ||
        !CTR_CUDA_CHECK(
            cuda_transformer::kv_cache_prefill(&cache, cp, cv, prefill)) ||
        !CTR_CUDA_CHECK(cudaDeviceSynchronize()))
      ok = false;
    CTR_CUDA_CHECK(cudaFree(cp));
    CTR_CUDA_CHECK(cudaFree(cv));
  }
  std::vector<float> output(heads * dim);
  for (size_t t = prefill; t < sequence && ok; t++) {
    std::vector<float> tq(heads * dim), tk(heads * dim), tv(heads * dim);
    for (size_t h = 0; h < heads; h++)
      for (size_t d = 0; d < dim; d++) {
        tq[h * dim + d] = q[(h * sequence + t) * dim + d];
        tk[h * dim + d] = k[(h * sequence + t) * dim + d];
        tv[h * dim + d] = v[(h * sequence + t) * dim + d];
      }
    const std::size_t history_length = cache.current_length;
    std::vector<float> prior_k(heads * history_length * dim);
    std::vector<float> prior_v(prior_k.size());
    for (size_t h = 0; h < heads; ++h)
      for (size_t position = 0; position < history_length; ++position)
        for (size_t d = 0; d < dim; ++d) {
          const size_t cache_index = (h * sequence + position) * dim + d;
          const size_t history_index =
              (h * history_length + position) * dim + d;
          if (!CTR_CUDA_CHECK(
                  cudaMemcpy(&prior_k[history_index], cache.keys + cache_index,
                             sizeof(float), cudaMemcpyDeviceToHost)) ||
              !CTR_CUDA_CHECK(cudaMemcpy(
                  &prior_v[history_index], cache.values + cache_index,
                  sizeof(float), cudaMemcpyDeviceToHost)))
            ok = false;
        }
    ok = CTR_CUDA_CHECK(
             cudaMemcpy(dq, tq.data(), token_bytes, cudaMemcpyHostToDevice)) &&
         CTR_CUDA_CHECK(
             cudaMemcpy(dk, tk.data(), token_bytes, cudaMemcpyHostToDevice)) &&
         CTR_CUDA_CHECK(
             cudaMemcpy(dv, tv.data(), token_bytes, cudaMemcpyHostToDevice)) &&
         CTR_CUDA_CHECK(
             cuda_transformer::incremental_decode_with_workspace(
                 &cache, dq, dk, dv, do_, &workspace)) &&
         CTR_CUDA_CHECK(cudaDeviceSynchronize()) &&
         CTR_CUDA_CHECK(cudaMemcpy(output.data(), do_, token_bytes,
                                   cudaMemcpyDeviceToHost));
    if (cache.current_length != t + 1) {
      std::fprintf(
          stderr,
          "shape 1x%zux%zux%zu token %zu: cache length got %zu expected %zu\n",
          heads, sequence, dim, t, cache.current_length, t + 1);
      ok = false;
    }
    for (size_t h = 0; h < heads && ok; h++)
      for (size_t d = 0; d < dim; d++)
        if (!close(output[h * dim + d],
                   reference[(h * sequence + t) * dim + d]) ||
            !std::isfinite(output[h * dim + d])) {
          std::fprintf(stderr,
                       "shape 1x%zux%zux%zu token %zu: output/finiteness "
                       "mismatch h=%zu d=%zu got %.8g expected %.8g\n",
                       heads, sequence, dim, t, h, d, output[h * dim + d],
                       reference[(h * sequence + t) * dim + d]);
          ok = false;
        }
    std::vector<float> current_k(heads * sequence * dim),
        current_v(current_k.size());
    ok = ok &&
         CTR_CUDA_CHECK(cudaMemcpy(current_k.data(), cache.keys,
                                   current_k.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost)) &&
         CTR_CUDA_CHECK(cudaMemcpy(current_v.data(), cache.values,
                                   current_v.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost));
    for (size_t h = 0; h < heads && ok; ++h)
      for (size_t position = 0; position < history_length && ok; ++position)
        for (size_t d = 0; d < dim; ++d) {
          const size_t ci = (h * sequence + position) * dim + d,
                       hi = (h * history_length + position) * dim + d;
          if (current_k[ci] != prior_k[hi] || current_v[ci] != prior_v[hi]) {
            std::fprintf(stderr,
                         "shape 1x%zux%zux%zu token %zu: K/V history changed "
                         "h=%zu position=%zu d=%zu\n",
                         heads, sequence, dim, t, h, position, d);
            ok = false;
          }
        }
    for (size_t h = 0; h < heads && ok; ++h)
      for (size_t d = 0; d < dim; ++d) {
        const size_t pair = d / 2, even = d & ~size_t(1), base = h * dim + even;
        const float theta = std::pow(10000.0F, -float(2 * pair) / dim) * t,
                    c = std::cos(theta), s = std::sin(theta);
        const float expected_k = (d & 1) ? tk[base] * s + tk[base + 1] * c
                                         : tk[base] * c - tk[base + 1] * s;
        const size_t ci = (h * sequence + t) * dim + d;
        if (!close(current_k[ci], expected_k) ||
            current_v[ci] != tv[h * dim + d]) {
          std::fprintf(
              stderr,
              "shape 1x%zux%zux%zu token %zu: appended K/V mismatch h=%zu "
              "d=%zu K got %.8g expected %.8g V got %.8g expected %.8g\n",
              heads, sequence, dim, t, h, d, current_k[ci], expected_k,
              current_v[ci], tv[h * dim + d]);
          ok = false;
        }
      }
  }
  if (cache.current_length == sequence &&
      cuda_transformer::incremental_decode_with_workspace(
          &cache, dq, dk, dv, do_, &workspace) !=
          cudaErrorInvalidValue)
    ok = false;
  return clean(ok);
}
} // namespace
int main() {
  cuda_transformer::IncrementalAttentionWorkspace invalid_workspace;
  if (cuda_transformer::incremental_attention_workspace_create(
          &invalid_workspace, 0, 8, 4) != cudaErrorInvalidValue ||
      cuda_transformer::incremental_attention_workspace_create(
          &invalid_workspace, 1, 7, 4) != cudaErrorInvalidValue ||
      cuda_transformer::incremental_attention_workspace_create(
          &invalid_workspace, 1, 8, 0) != cudaErrorInvalidValue) {
    std::fputs("invalid incremental-attention workspace accepted\n", stderr);
    return EXIT_FAILURE;
  }
  for (auto s : std::array<std::array<size_t, 3>, 5>{{{{1, 1, 2}},
                                                      {{1, 4, 4}},
                                                      {{2, 5, 8}},
                                                      {{2, 7, 16}},
                                                      {{4, 32, 64}}}})
    if (!run_case(s[0], s[1], s[2], 0))
      return EXIT_FAILURE;
  if (!run_case(1, 7, 8, 3) || !run_case(4, 32, 64, 16))
    return EXIT_FAILURE;
  std::puts("Incremental attention tests passed.");
}
