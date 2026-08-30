#include "cuda_transformer/cublas_check.h"
#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/decoder_block.h"
#include "cuda_transformer/gemm.h"
#include "cuda_transformer/incremental_attention.h"
#include "cuda_transformer/kv_cache.h"
#include "cuda_transformer/qkv_layout.h"
#include "cuda_transformer/qkv_projection.h"
#include "cuda_transformer/transformer_primitives.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr float kEpsilon = 1.0e-5F;
constexpr float kAttentionAbsoluteTolerance = 5.0e-4F;
constexpr float kAttentionRelativeTolerance = 1.0e-4F;
// Full-prefix and cached paths use different attention accumulation shapes.
constexpr float kOutputAbsoluteTolerance = 1.0e-3F;
constexpr float kOutputRelativeTolerance = 2.0e-4F;

struct Shape {
  std::size_t sequence;
  std::size_t hidden;
  std::size_t heads;
  std::size_t head_dim;
  std::size_t intermediate;
};

bool check_row(const char* stage, const std::vector<float>& incremental,
               const std::vector<float>& full, Shape shape, std::size_t token,
               float absolute, float relative) {
  if (incremental.size() != full.size()) {
    std::fprintf(stderr,
                 "%s size mismatch: seq=%zu hidden=%zu heads=%zu head_dim=%zu "
                 "intermediate=%zu token=%zu incremental=%zu full=%zu\n",
                 stage, shape.sequence, shape.hidden, shape.heads,
                 shape.head_dim, shape.intermediate, token, incremental.size(),
                 full.size());
    return false;
  }
  for (std::size_t i = 0; i < incremental.size(); ++i) {
    const float tolerance = absolute + relative * fmaxf(1.0F, std::fabs(full[i]));
    if (!std::isfinite(incremental[i]) || !std::isfinite(full[i]) ||
        std::fabs(incremental[i] - full[i]) > tolerance) {
      std::fprintf(stderr,
                   "%s mismatch: seq=%zu hidden=%zu heads=%zu head_dim=%zu "
                   "intermediate=%zu token=%zu index=%zu incremental=%.8g "
                   "full=%.8g error=%.8g tolerance=%.8g\n",
                   stage, shape.sequence, shape.hidden, shape.heads,
                   shape.head_dim, shape.intermediate, token, i, incremental[i],
                   full[i], std::fabs(incremental[i] - full[i]), tolerance);
      return false;
    }
  }
  return true;
}

bool check_cache_history(const std::vector<float>& before_keys,
                         const std::vector<float>& before_values,
                         const std::vector<float>& after_keys,
                         const std::vector<float>& after_values, Shape shape,
                         std::size_t token, std::size_t history_length) {
  for (std::size_t head = 0; head < shape.heads; ++head)
    for (std::size_t position = 0; position < history_length; ++position)
      for (std::size_t d = 0; d < shape.head_dim; ++d) {
        const std::size_t index =
            (head * shape.sequence + position) * shape.head_dim + d;
        if (before_keys[index] != after_keys[index] ||
            before_values[index] != after_values[index]) {
          std::fprintf(stderr,
                       "cache history changed: seq=%zu hidden=%zu heads=%zu "
                       "head_dim=%zu intermediate=%zu token=%zu head=%zu "
                       "position=%zu dimension=%zu\n",
                       shape.sequence, shape.hidden, shape.heads, shape.head_dim,
                       shape.intermediate, token, head, position, d);
          return false;
        }
      }
  return true;
}

bool run(cublasHandle_t handle, Shape shape) {
  using namespace cuda_transformer;

  const std::size_t activation_count = shape.sequence * shape.hidden;
  const std::size_t square_weight_count = shape.hidden * shape.hidden;
  const std::size_t gate_weight_count = shape.hidden * shape.intermediate;
  const std::size_t down_weight_count = shape.intermediate * shape.hidden;
  const std::size_t cache_count = shape.heads * shape.sequence * shape.head_dim;
  std::vector<float> input(activation_count), attention_norm(shape.hidden),
      mlp_norm(shape.hidden), wq(square_weight_count), wk(square_weight_count),
      wv(square_weight_count), wo(square_weight_count),
      w_gate(gate_weight_count), w_up(gate_weight_count),
      w_down(down_weight_count);
  for (std::size_t i = 0; i < activation_count; ++i)
    input[i] = static_cast<float>(static_cast<int>((i * 5 + 3) % 37) - 18) /
               32.0F;
  for (std::size_t i = 0; i < shape.hidden; ++i) {
    attention_norm[i] = 0.75F + static_cast<float>(i % 13) / 32.0F;
    mlp_norm[i] = 0.60F + static_cast<float>((i * 3 + 1) % 17) / 40.0F;
  }
  for (std::size_t i = 0; i < square_weight_count; ++i) {
    wq[i] = static_cast<float>(static_cast<int>((i * 3 + 1) % 41) - 20) / 64.0F;
    wk[i] = static_cast<float>(static_cast<int>((i * 5 + 7) % 43) - 21) / 64.0F;
    wv[i] = static_cast<float>(static_cast<int>((i * 7 + 11) % 47) - 23) / 64.0F;
    wo[i] = static_cast<float>(static_cast<int>((i * 11 + 13) % 53) - 26) / 80.0F;
  }
  for (std::size_t i = 0; i < gate_weight_count; ++i) {
    w_gate[i] =
        static_cast<float>(static_cast<int>((i * 13 + 5) % 59) - 29) / 96.0F;
    w_up[i] =
        static_cast<float>(static_cast<int>((i * 17 + 9) % 61) - 30) / 96.0F;
  }
  for (std::size_t i = 0; i < down_weight_count; ++i)
    w_down[i] =
        static_cast<float>(static_cast<int>((i * 19 + 7) % 67) - 33) / 112.0F;

  float *d_full_input = nullptr, *d_token_input = nullptr,
        *d_attention_norm = nullptr, *d_mlp_norm = nullptr, *d_wq = nullptr,
        *d_wk = nullptr, *d_wv = nullptr, *d_wo = nullptr,
        *d_w_gate = nullptr, *d_w_up = nullptr, *d_w_down = nullptr,
        *d_attention_head = nullptr, *d_attention_token = nullptr,
        *d_attention_projected = nullptr, *d_attention_residual = nullptr,
        *d_mlp_normalized = nullptr, *d_gate = nullptr, *d_up = nullptr,
        *d_activated_gate = nullptr, *d_gated = nullptr, *d_down = nullptr,
        *d_incremental_output = nullptr, *d_full_output = nullptr;
  KvCache cache;
  QkvProjectionWorkspace qkv;
  IncrementalAttentionWorkspace attention_workspace;
  auto cleanup = [&](bool ok) {
    for (float* pointer : {d_full_input, d_token_input, d_attention_norm,
                           d_mlp_norm, d_wq, d_wk, d_wv, d_wo, d_w_gate, d_w_up,
                           d_w_down, d_attention_head, d_attention_token,
                           d_attention_projected, d_attention_residual,
                           d_mlp_normalized, d_gate, d_up, d_activated_gate,
                           d_gated, d_down, d_incremental_output, d_full_output}) {
      if (pointer != nullptr && !CTR_CUDA_CHECK(cudaFree(pointer)))
        ok = false;
    }
    if (qkv.q_token != nullptr &&
        !CTR_CUDA_CHECK(qkv_projection_workspace_destroy(&qkv)))
      ok = false;
    if (!CTR_CUDA_CHECK(
            incremental_attention_workspace_destroy(&attention_workspace)))
      ok = false;
    if (!CTR_CUDA_CHECK(kv_cache_destroy(&cache)))
      ok = false;
    return ok;
  };

  const std::size_t hidden_bytes = shape.hidden * sizeof(float);
  const std::size_t activation_bytes = activation_count * sizeof(float);
  const std::size_t intermediate_bytes =
      shape.intermediate * sizeof(float);
  if (!CTR_CUDA_CHECK(kv_cache_create(&cache, 1, shape.heads, shape.sequence,
                                      shape.head_dim)) ||
      !CTR_CUDA_CHECK(qkv_projection_workspace_create(
          &qkv, {1, shape.hidden, shape.heads, shape.head_dim})) ||
      !CTR_CUDA_CHECK(incremental_attention_workspace_create(
          &attention_workspace, shape.heads, shape.head_dim, shape.sequence)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_full_input, activation_bytes)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_token_input, hidden_bytes)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_attention_norm, hidden_bytes)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_mlp_norm, hidden_bytes)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_wq, square_weight_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_wk, square_weight_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_wv, square_weight_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_wo, square_weight_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(
          cudaMalloc(&d_w_gate, gate_weight_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_w_up, gate_weight_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(
          cudaMalloc(&d_w_down, down_weight_count * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_attention_head, hidden_bytes)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_attention_token, hidden_bytes)) ||
      !CTR_CUDA_CHECK(
          cudaMalloc(&d_attention_projected, hidden_bytes)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_attention_residual, hidden_bytes)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_mlp_normalized, hidden_bytes)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_gate, intermediate_bytes)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_up, intermediate_bytes)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_activated_gate, intermediate_bytes)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_gated, intermediate_bytes)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_down, hidden_bytes)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_incremental_output, hidden_bytes)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&d_full_output, activation_bytes)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_full_input, input.data(), activation_bytes,
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_attention_norm, attention_norm.data(),
                                 hidden_bytes, cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_mlp_norm, mlp_norm.data(), hidden_bytes,
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_wq, wq.data(), square_weight_count * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_wk, wk.data(), square_weight_count * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_wv, wv.data(), square_weight_count * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_wo, wo.data(), square_weight_count * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_w_gate, w_gate.data(),
                                 gate_weight_count * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_w_up, w_up.data(),
                                 gate_weight_count * sizeof(float),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(d_w_down, w_down.data(),
                                 down_weight_count * sizeof(float),
                                 cudaMemcpyHostToDevice)))
    return cleanup(false);

  if (cache.current_length != 0) {
    std::fprintf(stderr, "initial cache length is %zu, expected zero\n",
                 cache.current_length);
    return cleanup(false);
  }

  const DecoderBlockWeights block_weights{
      {d_attention_norm, d_wq, d_wk, d_wv, d_wo},
      {d_mlp_norm, d_w_gate, d_w_up, d_w_down},
  };
  std::vector<float> incremental_attention(shape.hidden), full_attention(shape.hidden),
      incremental_output(shape.hidden), full_output(shape.hidden),
      cache_keys_before(cache_count), cache_values_before(cache_count),
      cache_keys_after(cache_count), cache_values_after(cache_count);
  bool ok = true;
  for (std::size_t token = 0; token < shape.sequence && ok; ++token) {
    const std::size_t history_length = cache.current_length;
    if (!CTR_CUDA_CHECK(cudaMemcpy(cache_keys_before.data(), cache.keys,
                                   cache_count * sizeof(float),
                                   cudaMemcpyDeviceToHost)) ||
        !CTR_CUDA_CHECK(cudaMemcpy(cache_values_before.data(), cache.values,
                                   cache_count * sizeof(float),
                                   cudaMemcpyDeviceToHost))) {
      ok = false;
      break;
    }

    const float* token_input = input.data() + token * shape.hidden;
    ok = CTR_CUDA_CHECK(cudaMemcpy(d_token_input, token_input, hidden_bytes,
                                   cudaMemcpyHostToDevice)) &&
         CTR_CUDA_CHECK(rmsnorm_cuda(d_token_input, d_attention_norm,
                                     d_attention_residual, 1, shape.hidden,
                                     kEpsilon));
    // Reuse the attention-residual buffer as the one-token normalized input
    // until the first residual overwrites it later in this same stream.
    if (ok)
      ok = CTR_CUBLAS_CHECK(project_qkv_cuda(handle, d_attention_residual, d_wq,
                                              d_wk, d_wv, &qkv));
    if (ok)
      ok = CTR_CUDA_CHECK(incremental_decode_with_workspace(
          &cache, qkv.q_head, qkv.k_head, qkv.v_head, d_attention_head,
          &attention_workspace));
    if (ok)
      ok = CTR_CUDA_CHECK(head_major_to_token_major_cuda(
          d_attention_head, d_attention_token, 1, shape.heads, shape.head_dim));
    if (ok)
      ok = CTR_CUBLAS_CHECK(linear_cublas_row_major(
          handle, d_attention_token, d_wo, d_attention_projected, 1,
          shape.hidden, shape.hidden));
    if (ok)
      ok = CTR_CUDA_CHECK(residual_add_cuda(d_token_input, d_attention_projected,
                                             d_attention_residual,
                                             shape.hidden));
    if (ok)
      ok = CTR_CUDA_CHECK(rmsnorm_cuda(d_attention_residual, d_mlp_norm,
                                       d_mlp_normalized, 1, shape.hidden,
                                       kEpsilon));
    if (ok)
      ok = CTR_CUBLAS_CHECK(linear_cublas_row_major(
          handle, d_mlp_normalized, d_w_gate, d_gate, 1, shape.hidden,
          shape.intermediate));
    if (ok)
      ok = CTR_CUBLAS_CHECK(linear_cublas_row_major(
          handle, d_mlp_normalized, d_w_up, d_up, 1, shape.hidden,
          shape.intermediate));
    if (ok)
      ok = CTR_CUDA_CHECK(silu_cuda(d_gate, d_activated_gate, shape.intermediate));
    if (ok)
      ok = CTR_CUDA_CHECK(multiply_cuda(d_activated_gate, d_up, d_gated,
                                         shape.intermediate));
    if (ok)
      ok = CTR_CUBLAS_CHECK(linear_cublas_row_major(
          handle, d_gated, d_w_down, d_down, 1, shape.intermediate,
          shape.hidden));
    if (ok)
      ok = CTR_CUDA_CHECK(residual_add_cuda(d_attention_residual, d_down,
                                             d_incremental_output,
                                             shape.hidden));

    DecoderBlockWorkspace prefix_workspace;
    const DecoderBlockConfig prefix_config{{token + 1, shape.hidden, shape.heads,
                                             shape.head_dim, kEpsilon},
                                            {token + 1, shape.hidden,
                                             shape.intermediate, kEpsilon}};
    if (ok)
      ok = CTR_CUDA_CHECK(
          decoder_block_workspace_create(&prefix_workspace, prefix_config));
    if (ok)
      ok = CTR_CUBLAS_CHECK(decoder_block_cuda(handle, d_full_input, block_weights,
                                                &prefix_workspace, d_full_output));
    if (ok)
      ok = CTR_CUDA_CHECK(cudaDeviceSynchronize()) &&
           CTR_CUDA_CHECK(cudaMemcpy(incremental_attention.data(),
                                     d_attention_residual, hidden_bytes,
                                     cudaMemcpyDeviceToHost)) &&
           CTR_CUDA_CHECK(cudaMemcpy(incremental_output.data(),
                                     d_incremental_output, hidden_bytes,
                                     cudaMemcpyDeviceToHost)) &&
           CTR_CUDA_CHECK(cudaMemcpy(full_attention.data(),
                                     prefix_workspace.attention_output +
                                         token * shape.hidden,
                                     hidden_bytes, cudaMemcpyDeviceToHost)) &&
           CTR_CUDA_CHECK(cudaMemcpy(full_output.data(),
                                     d_full_output + token * shape.hidden,
                                     hidden_bytes, cudaMemcpyDeviceToHost)) &&
           CTR_CUDA_CHECK(cudaMemcpy(cache_keys_after.data(), cache.keys,
                                     cache_count * sizeof(float),
                                     cudaMemcpyDeviceToHost)) &&
           CTR_CUDA_CHECK(cudaMemcpy(cache_values_after.data(), cache.values,
                                     cache_count * sizeof(float),
                                     cudaMemcpyDeviceToHost));
    if (cache.current_length != token + 1) {
      std::fprintf(stderr,
                   "cache length mismatch: seq=%zu hidden=%zu heads=%zu "
                   "head_dim=%zu intermediate=%zu token=%zu got=%zu expected=%zu\n",
                   shape.sequence, shape.hidden, shape.heads, shape.head_dim,
                   shape.intermediate, token, cache.current_length, token + 1);
      ok = false;
    }
    ok = ok && check_cache_history(cache_keys_before, cache_values_before,
                                   cache_keys_after, cache_values_after, shape,
                                   token, history_length) &&
         check_row("attention residual", incremental_attention, full_attention,
                   shape, token, kAttentionAbsoluteTolerance,
                   kAttentionRelativeTolerance) &&
         check_row("decoder block output", incremental_output, full_output,
                   shape, token, kOutputAbsoluteTolerance,
                   kOutputRelativeTolerance);
    if (!CTR_CUDA_CHECK(decoder_block_workspace_destroy(&prefix_workspace)))
      ok = false;
  }
  if (ok && cache.current_length > cache.max_sequence) {
    std::fprintf(stderr, "cache capacity exceeded\n");
    ok = false;
  }
  return cleanup(ok);
}

}  // namespace

int main() {
  cublasHandle_t handle = nullptr;
  if (!CTR_CUBLAS_CHECK(cublasCreate(&handle)))
    return EXIT_FAILURE;
  bool ok = true;
  for (const Shape shape : std::array<Shape, 4>{{
           {1, 8, 1, 8, 16}, {4, 16, 2, 8, 32}, {7, 32, 4, 8, 48},
           {16, 256, 4, 64, 512},
       }}) {
    if (!run(handle, shape)) {
      ok = false;
      break;
    }
  }
  if (!CTR_CUBLAS_CHECK(cublasDestroy(handle)))
    ok = false;
  if (ok)
    std::puts("Incremental decoder block equivalence tests passed.");
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
