#include "cuda_transformer/cublas_check.h"
#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/decoder_block.h"
#include "cuda_transformer/incremental_decoder_block.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

constexpr float kEpsilon = 1.0e-5F;
constexpr float kOutputAbsoluteTolerance = 1.0e-3F;
constexpr float kOutputRelativeTolerance = 2.0e-4F;

struct Shape {
  std::size_t sequence;
  std::size_t hidden;
  std::size_t heads;
  std::size_t head_dim;
  std::size_t intermediate;
};

bool check_row(const std::vector<float>& incremental,
               const std::vector<float>& full, Shape shape,
               std::size_t token) {
  if (incremental.size() != full.size()) {
    std::fprintf(stderr, "output size mismatch at token %zu\n", token);
    return false;
  }
  for (std::size_t i = 0; i < incremental.size(); ++i) {
    const float tolerance = kOutputAbsoluteTolerance +
                            kOutputRelativeTolerance *
                                fmaxf(1.0F, std::fabs(full[i]));
    if (!std::isfinite(incremental[i]) || !std::isfinite(full[i]) ||
        std::fabs(incremental[i] - full[i]) > tolerance) {
      std::fprintf(stderr,
                   "decoder output mismatch: seq=%zu hidden=%zu heads=%zu "
                   "head_dim=%zu intermediate=%zu token=%zu index=%zu "
                   "incremental=%.8g full=%.8g error=%.8g tolerance=%.8g\n",
                   shape.sequence, shape.hidden, shape.heads, shape.head_dim,
                   shape.intermediate, token, i, incremental[i], full[i],
                   std::fabs(incremental[i] - full[i]), tolerance);
      return false;
    }
  }
  return true;
}

bool invalid_config_tests() {
  using cuda_transformer::IncrementalDecoderBlockConfig;
  using cuda_transformer::IncrementalDecoderBlockWorkspace;
  using cuda_transformer::incremental_decoder_block_workspace_create;
  using cuda_transformer::valid_incremental_decoder_block_config;

  const IncrementalDecoderBlockConfig valid{8, 1, 8, 16, kEpsilon, kEpsilon,
                                            1};
  auto invalid = std::array<IncrementalDecoderBlockConfig, 10>{
      valid, valid, valid, valid, valid, valid, valid, valid, valid, valid};
  invalid[0].hidden = 0;
  invalid[1].heads = 0;
  invalid[2].head_dim = 7;
  invalid[3].hidden = 15;
  invalid[4].intermediate = 0;
  invalid[5].attention_rmsnorm_epsilon = 0.0F;
  invalid[6].mlp_rmsnorm_epsilon = -kEpsilon;
  invalid[7].mlp_rmsnorm_epsilon = std::numeric_limits<float>::infinity();
  invalid[8].attention_rmsnorm_epsilon = std::numeric_limits<float>::infinity();
  invalid[9].max_sequence = 0;
  for (const IncrementalDecoderBlockConfig config : invalid) {
    IncrementalDecoderBlockWorkspace workspace;
    if (valid_incremental_decoder_block_config(config) ||
        incremental_decoder_block_workspace_create(&workspace, config) !=
            cudaErrorInvalidValue) {
      std::fprintf(stderr, "invalid incremental decoder-block config accepted\n");
      return false;
    }
  }
  return true;
}

bool run(cublasHandle_t handle, Shape shape) {
  using namespace cuda_transformer;

  const IncrementalDecoderBlockConfig incremental_config{
      shape.hidden, shape.heads, shape.head_dim, shape.intermediate, kEpsilon,
      kEpsilon, shape.sequence};
  const std::size_t activation_count = shape.sequence * shape.hidden;
  const std::size_t square_weight_count = shape.hidden * shape.hidden;
  const std::size_t gate_weight_count = shape.hidden * shape.intermediate;
  const std::size_t down_weight_count = shape.intermediate * shape.hidden;

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
        *d_incremental_output = nullptr, *d_full_output = nullptr;
  KvCache cache;
  IncrementalDecoderBlockWorkspace workspace;
  auto cleanup = [&](bool ok) {
    for (float* pointer : {d_full_input, d_token_input, d_attention_norm,
                           d_mlp_norm, d_wq, d_wk, d_wv, d_wo, d_w_gate, d_w_up,
                           d_w_down, d_incremental_output, d_full_output}) {
      if (pointer != nullptr && !CTR_CUDA_CHECK(cudaFree(pointer)))
        ok = false;
    }
    if (!CTR_CUDA_CHECK(incremental_decoder_block_workspace_destroy(&workspace)))
      ok = false;
    if (!CTR_CUDA_CHECK(kv_cache_destroy(&cache)))
      ok = false;
    return ok;
  };

  const std::size_t hidden_bytes = shape.hidden * sizeof(float);
  const std::size_t activation_bytes = activation_count * sizeof(float);
  if (!CTR_CUDA_CHECK(kv_cache_create(&cache, 1, shape.heads, shape.sequence,
                                      shape.head_dim)) ||
      !CTR_CUDA_CHECK(incremental_decoder_block_workspace_create(
          &workspace, incremental_config)) ||
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

  const IncrementalDecoderBlockWeights incremental_weights{
      {d_attention_norm, d_wq, d_wk, d_wv, d_wo},
      {d_mlp_norm, d_w_gate, d_w_up, d_w_down},
  };
  const DecoderBlockWeights full_weights{
      {d_attention_norm, d_wq, d_wk, d_wv, d_wo},
      {d_mlp_norm, d_w_gate, d_w_up, d_w_down},
  };
  bool ok = cache.current_length == 0;
  if (!ok)
    std::fprintf(stderr, "new cache does not start empty\n");
  ok = ok &&
      incremental_decoder_block_cuda(handle, nullptr, incremental_weights, &cache,
                                     &workspace, d_incremental_output) ==
          CUBLAS_STATUS_INVALID_VALUE &&
      incremental_decoder_block_cuda(handle, d_token_input,
                                     {{nullptr, d_wq, d_wk, d_wv, d_wo},
                                      {d_mlp_norm, d_w_gate, d_w_up, d_w_down}},
                                     &cache, &workspace, d_incremental_output) ==
          CUBLAS_STATUS_INVALID_VALUE &&
      incremental_decoder_block_cuda(handle, d_token_input,
                                     {{d_attention_norm, d_wq, d_wk, d_wv, d_wo},
                                      {d_mlp_norm, d_w_gate, d_w_up, nullptr}},
                                     &cache, &workspace,
                                     d_incremental_output) ==
          CUBLAS_STATUS_INVALID_VALUE &&
      incremental_decoder_block_cuda(handle, d_token_input, incremental_weights,
                                     nullptr, &workspace,
                                     d_incremental_output) ==
          CUBLAS_STATUS_INVALID_VALUE &&
      incremental_decoder_block_cuda(handle, d_token_input, incremental_weights,
                                     &cache, &workspace, nullptr) ==
          CUBLAS_STATUS_INVALID_VALUE;
  const IncrementalDecoderBlockConfig saved_workspace_config = workspace.config;
  ++workspace.config.intermediate;
  ok = ok && incremental_decoder_block_cuda(handle, d_token_input,
                                             incremental_weights, &cache,
                                             &workspace,
                                             d_incremental_output) ==
                 CUBLAS_STATUS_INVALID_VALUE;
  workspace.config = saved_workspace_config;
  ++workspace.attention.max_sequence;
  ok = ok && incremental_decoder_block_cuda(handle, d_token_input,
                                             incremental_weights, &cache,
                                             &workspace,
                                             d_incremental_output) ==
                 CUBLAS_STATUS_INVALID_VALUE;
  --workspace.attention.max_sequence;
  const std::size_t saved_heads = cache.heads;
  ++cache.heads;
  ok = ok && incremental_decoder_block_cuda(handle, d_token_input,
                                             incremental_weights, &cache,
                                             &workspace,
                                             d_incremental_output) ==
                 CUBLAS_STATUS_INVALID_VALUE;
  cache.heads = saved_heads;
  const std::size_t saved_max_sequence = cache.max_sequence;
  ++cache.max_sequence;
  ok = ok && incremental_decoder_block_cuda(handle, d_token_input,
                                             incremental_weights, &cache,
                                             &workspace,
                                             d_incremental_output) ==
                 CUBLAS_STATUS_INVALID_VALUE;
  cache.max_sequence = saved_max_sequence;

  std::vector<float> incremental_output(shape.hidden), full_output(shape.hidden);
  for (std::size_t token = 0; token < shape.sequence && ok; ++token) {
    ok = CTR_CUDA_CHECK(cudaMemcpy(d_token_input,
                                   input.data() + token * shape.hidden,
                                   hidden_bytes, cudaMemcpyHostToDevice)) &&
         CTR_CUBLAS_CHECK(incremental_decoder_block_cuda(
             handle, d_token_input, incremental_weights, &cache, &workspace,
             d_incremental_output));

    DecoderBlockWorkspace prefix_workspace;
    const DecoderBlockConfig prefix_config{{token + 1, shape.hidden, shape.heads,
                                             shape.head_dim, kEpsilon},
                                            {token + 1, shape.hidden,
                                             shape.intermediate, kEpsilon}};
    if (ok)
      ok = CTR_CUDA_CHECK(
          decoder_block_workspace_create(&prefix_workspace, prefix_config));
    if (ok)
      ok = CTR_CUBLAS_CHECK(decoder_block_cuda(handle, d_full_input, full_weights,
                                                &prefix_workspace, d_full_output));
    if (ok)
      ok = CTR_CUDA_CHECK(cudaDeviceSynchronize()) &&
           CTR_CUDA_CHECK(cudaMemcpy(incremental_output.data(),
                                     d_incremental_output, hidden_bytes,
                                     cudaMemcpyDeviceToHost)) &&
           CTR_CUDA_CHECK(cudaMemcpy(full_output.data(),
                                     d_full_output + token * shape.hidden,
                                     hidden_bytes, cudaMemcpyDeviceToHost));
    if (cache.current_length != token + 1) {
      std::fprintf(stderr,
                   "cache length mismatch: token=%zu got=%zu expected=%zu\n",
                   token, cache.current_length, token + 1);
      ok = false;
    }
    ok = ok && check_row(incremental_output, full_output, shape, token);
    if (!CTR_CUDA_CHECK(decoder_block_workspace_destroy(&prefix_workspace)))
      ok = false;
  }
  const std::size_t full_length = cache.current_length;
  if (ok &&
      incremental_decoder_block_cuda(handle, d_token_input, incremental_weights,
                                     &cache, &workspace,
                                     d_incremental_output) !=
          CUBLAS_STATUS_INVALID_VALUE) {
    std::fprintf(stderr, "full cache was accepted by incremental decoder block\n");
    ok = false;
  }
  if (cache.current_length != full_length) {
    std::fprintf(stderr, "full-cache execution changed cache length\n");
    ok = false;
  }
  return cleanup(ok);
}

}  // namespace

int main() {
  if (!invalid_config_tests())
    return EXIT_FAILURE;

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
    std::puts("Incremental decoder block tests passed.");
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
