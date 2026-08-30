#include "cuda_transformer/cublas_check.h"
#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/gemm.h"
#include "cuda_transformer/incremental_attention.h"
#include "cuda_transformer/incremental_decoder_block.h"
#include "cuda_transformer/qkv_layout.h"
#include "cuda_transformer/transformer_primitives.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include <cuda_profiler_api.h>

namespace {

constexpr std::size_t kHidden = 256;
constexpr std::size_t kHeads = 4;
constexpr std::size_t kHeadDim = 64;
constexpr std::size_t kIntermediate = 512;
constexpr float kEpsilon = 1.0e-5F;
constexpr float kOutputAbsoluteTolerance = 1.0e-3F;
constexpr float kOutputRelativeTolerance = 2.0e-4F;
constexpr int kWarmups = 10;
constexpr int kBatches = 9;
constexpr int kTraceWarmups = 5;

enum Stage : std::size_t {
  kAttentionRmsNorm,
  kQProjection,
  kKProjection,
  kVProjection,
  kQkvLayout,
  kIncrementalAttention,
  kAttentionOutputLayout,
  kWoProjection,
  kAttentionResidual,
  kMlpRmsNorm,
  kGateUpProjections,
  kSiluMultiply,
  kDownProjection,
  kFinalResidual,
  kStageCount,
};

constexpr std::array<const char*, kStageCount> kStageNames{
    "attention RMSNorm",       "Q projection",
    "K projection",            "V projection",
    "QKV layout",              "incremental attention core",
    "attention output layout", "Wo projection",
    "attention residual",      "MLP RMSNorm",
    "gate + up projections",   "SiLU + multiply",
    "down projection",         "final residual",
};

struct ProfileConfig {
  std::size_t history;
  int launches_per_batch;
  float baseline_6c1_ms;
};

struct DeviceWeights {
  float* attention_norm = nullptr;
  float* mlp_norm = nullptr;
  float* wq = nullptr;
  float* wk = nullptr;
  float* wv = nullptr;
  float* wo = nullptr;
  float* w_gate = nullptr;
  float* w_up = nullptr;
  float* w_down = nullptr;
};

struct ProfileResult {
  ProfileConfig config{};
  std::array<float, kStageCount> stage_ms{};
  float stage_total_ms = 0.0F;
  float direct_total_ms = 0.0F;
};

struct TraceOptions {
  bool enabled = false;
  std::size_t history = 0;
  int iterations = 0;
};

enum class CommandLineResult { kOk, kHelp, kError };

void print_usage(const char* program) {
  std::printf(
      "Usage:\n"
      "  %s\n"
      "  %s --trace --history <N> --iterations <N>\n",
      program, program);
}

bool parse_positive_size(const char* text, std::size_t* value) {
  char* end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(text, &end, 10);
  if (errno == ERANGE || end == text || *end != '\0' || parsed == 0 ||
      parsed > std::numeric_limits<std::size_t>::max())
    return false;
  *value = static_cast<std::size_t>(parsed);
  return true;
}

bool parse_positive_int(const char* text, int* value) {
  std::size_t parsed = 0;
  if (!parse_positive_size(text, &parsed) ||
      parsed > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    return false;
  *value = static_cast<int>(parsed);
  return true;
}

CommandLineResult parse_command_line(int argc, char** argv,
                                     TraceOptions* options) {
  if (argc == 1)
    return CommandLineResult::kOk;

  bool saw_trace = false;
  bool saw_history = false;
  bool saw_iterations = false;
  for (int index = 1; index < argc; ++index) {
    if (std::strcmp(argv[index], "--help") == 0) {
      print_usage(argv[0]);
      return CommandLineResult::kHelp;
    }
    if (std::strcmp(argv[index], "--trace") == 0 && !saw_trace) {
      saw_trace = true;
      continue;
    }
    if (std::strcmp(argv[index], "--history") == 0 && !saw_history &&
        index + 1 < argc && parse_positive_size(argv[++index], &options->history)) {
      saw_history = true;
      continue;
    }
    if (std::strcmp(argv[index], "--iterations") == 0 && !saw_iterations &&
        index + 1 < argc &&
        parse_positive_int(argv[++index], &options->iterations)) {
      saw_iterations = true;
      continue;
    }
    std::fprintf(stderr, "invalid command-line argument: %s\n", argv[index]);
    print_usage(argv[0]);
    return CommandLineResult::kError;
  }

  if (!saw_trace || !saw_history || !saw_iterations) {
    std::fputs("--trace requires both --history and --iterations\n", stderr);
    print_usage(argv[0]);
    return CommandLineResult::kError;
  }
  options->enabled = true;
  return CommandLineResult::kOk;
}

bool profiler_range_call(cudaError_t error, const char* expression) {
  // Running trace mode without Nsight may leave CUDA profiling disabled. The
  // production workload is still useful in that case, just not captured.
  if (error == cudaSuccess || error == cudaErrorProfilerDisabled)
    return true;
  return cuda_transformer::check_cuda(error, expression, __FILE__, __LINE__);
}

// One event boundary before the first stage and after each following stage.
// Events are kept for every execution in a batch, then all intervals are read
// after synchronizing only the batch's final event.
struct BatchEvents {
  std::vector<cudaEvent_t> events;
  std::size_t stride = kStageCount + 1;

  bool create(int executions) {
    events.resize(static_cast<std::size_t>(executions) * stride, nullptr);
    for (cudaEvent_t& event : events) {
      if (!CTR_CUDA_CHECK(cudaEventCreate(&event)))
        return false;
    }
    return true;
  }

  bool destroy() {
    bool ok = true;
    for (cudaEvent_t event : events) {
      if (event != nullptr && !CTR_CUDA_CHECK(cudaEventDestroy(event)))
        ok = false;
    }
    events.clear();
    return ok;
  }

  cudaEvent_t* boundaries(int execution) {
    return events.data() + static_cast<std::size_t>(execution) * stride;
  }
};

bool record(cudaEvent_t* boundaries, Stage stage) {
  return boundaries == nullptr ||
         CTR_CUDA_CHECK(cudaEventRecord(boundaries[static_cast<std::size_t>(stage)]));
}

bool compose_incremental(
    cublasHandle_t handle, const float* input,
    cuda_transformer::IncrementalDecoderBlockWeights weights,
    cuda_transformer::KvCache* cache,
    cuda_transformer::IncrementalDecoderBlockWorkspace* workspace, float* output,
    cudaEvent_t* boundaries = nullptr) {
  using namespace cuda_transformer;
  const IncrementalDecoderBlockConfig config = workspace->config;

  if (!record(boundaries, kAttentionRmsNorm) ||
      !CTR_CUDA_CHECK(rmsnorm_cuda(
          input, weights.attention.norm_weight, workspace->attention_normalized, 1,
          config.hidden, config.attention_rmsnorm_epsilon)) ||
      !record(boundaries, kQProjection) ||
      !CTR_CUBLAS_CHECK(linear_cublas_row_major(
          handle, workspace->attention_normalized, weights.attention.wq,
          workspace->qkv.q_token, 1, config.hidden, config.hidden)) ||
      !record(boundaries, kKProjection) ||
      !CTR_CUBLAS_CHECK(linear_cublas_row_major(
          handle, workspace->attention_normalized, weights.attention.wk,
          workspace->qkv.k_token, 1, config.hidden, config.hidden)) ||
      !record(boundaries, kVProjection) ||
      !CTR_CUBLAS_CHECK(linear_cublas_row_major(
          handle, workspace->attention_normalized, weights.attention.wv,
          workspace->qkv.v_token, 1, config.hidden, config.hidden)) ||
      !record(boundaries, kQkvLayout) ||
      !CTR_CUDA_CHECK(token_major_to_head_major_cuda(
          workspace->qkv.q_token, workspace->qkv.q_head, 1, config.heads,
          config.head_dim)) ||
      !CTR_CUDA_CHECK(token_major_to_head_major_cuda(
          workspace->qkv.k_token, workspace->qkv.k_head, 1, config.heads,
          config.head_dim)) ||
      !CTR_CUDA_CHECK(token_major_to_head_major_cuda(
          workspace->qkv.v_token, workspace->qkv.v_head, 1, config.heads,
          config.head_dim)) ||
      !record(boundaries, kIncrementalAttention) ||
      !CTR_CUDA_CHECK(incremental_decode_with_workspace(
          cache, workspace->qkv.q_head, workspace->qkv.k_head,
          workspace->qkv.v_head, workspace->attention_head,
          &workspace->attention)) ||
      !record(boundaries, kAttentionOutputLayout) ||
      !CTR_CUDA_CHECK(head_major_to_token_major_cuda(
          workspace->attention_head, workspace->attention_token, 1, config.heads,
          config.head_dim)) ||
      !record(boundaries, kWoProjection) ||
      !CTR_CUBLAS_CHECK(linear_cublas_row_major(
          handle, workspace->attention_token, weights.attention.wo,
          workspace->attention_projected, 1, config.hidden, config.hidden)) ||
      !record(boundaries, kAttentionResidual) ||
      !CTR_CUDA_CHECK(residual_add_cuda(input, workspace->attention_projected,
                                        workspace->attention_residual,
                                        config.hidden)) ||
      !record(boundaries, kMlpRmsNorm) ||
      !CTR_CUDA_CHECK(rmsnorm_cuda(
          workspace->attention_residual, weights.mlp.norm_weight,
          workspace->mlp_normalized, 1, config.hidden,
          config.mlp_rmsnorm_epsilon)) ||
      !record(boundaries, kGateUpProjections) ||
      !CTR_CUBLAS_CHECK(linear_cublas_row_major(
          handle, workspace->mlp_normalized, weights.mlp.w_gate, workspace->gate,
          1, config.hidden, config.intermediate)) ||
      !CTR_CUBLAS_CHECK(linear_cublas_row_major(
          handle, workspace->mlp_normalized, weights.mlp.w_up, workspace->up, 1,
          config.hidden, config.intermediate)) ||
      !record(boundaries, kSiluMultiply) ||
      !CTR_CUDA_CHECK(silu_cuda(workspace->gate, workspace->activated_gate,
                                config.intermediate)) ||
      !CTR_CUDA_CHECK(multiply_cuda(workspace->activated_gate, workspace->up,
                                    workspace->gated, config.intermediate)) ||
      !record(boundaries, kDownProjection) ||
      !CTR_CUBLAS_CHECK(linear_cublas_row_major(
          handle, workspace->gated, weights.mlp.w_down, workspace->down, 1,
          config.intermediate, config.hidden)) ||
      !record(boundaries, kFinalResidual) ||
      !CTR_CUDA_CHECK(residual_add_cuda(workspace->attention_residual,
                                        workspace->down, output, config.hidden)))
    return false;

  return boundaries == nullptr ||
         CTR_CUDA_CHECK(cudaEventRecord(boundaries[kStageCount]));
}

bool check_output(const std::vector<float>& profiled,
                  const std::vector<float>& production, std::size_t history) {
  if (profiled.size() != production.size()) {
    std::fprintf(stderr, "sanity output size mismatch at history %zu\n", history);
    return false;
  }
  for (std::size_t i = 0; i < profiled.size(); ++i) {
    const float tolerance = kOutputAbsoluteTolerance +
                            kOutputRelativeTolerance *
                                fmaxf(1.0F, std::fabs(production[i]));
    if (!std::isfinite(profiled[i]) || !std::isfinite(production[i]) ||
        std::fabs(profiled[i] - production[i]) > tolerance) {
      std::fprintf(stderr,
                   "sanity mismatch: history=%zu index=%zu profiled=%.8g "
                   "production=%.8g error=%.8g tolerance=%.8g\n",
                   history, i, profiled[i], production[i],
                   std::fabs(profiled[i] - production[i]), tolerance);
      return false;
    }
  }
  return true;
}

template <typename Launch>
bool direct_median_latency_ms(Launch launch, int launches_per_batch,
                              float* median_ms) {
  for (int i = 0; i < kWarmups; ++i) {
    if (!launch())
      return false;
  }
  if (!CTR_CUDA_CHECK(cudaDeviceSynchronize()))
    return false;

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (!CTR_CUDA_CHECK(cudaEventCreate(&start)) ||
      !CTR_CUDA_CHECK(cudaEventCreate(&stop))) {
    if (start != nullptr)
      CTR_CUDA_CHECK(cudaEventDestroy(start));
    if (stop != nullptr)
      CTR_CUDA_CHECK(cudaEventDestroy(stop));
    return false;
  }

  bool ok = true;
  std::vector<float> samples;
  samples.reserve(kBatches);
  for (int batch = 0; batch < kBatches && ok; ++batch) {
    ok = CTR_CUDA_CHECK(cudaEventRecord(start));
    for (int execution = 0; execution < launches_per_batch && ok; ++execution)
      ok = launch();
    if (ok)
      ok = CTR_CUDA_CHECK(cudaEventRecord(stop)) &&
           CTR_CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0F;
    if (ok)
      ok = CTR_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    if (ok)
      samples.push_back(elapsed_ms / static_cast<float>(launches_per_batch));
  }
  if (!CTR_CUDA_CHECK(cudaEventDestroy(start)))
    ok = false;
  if (!CTR_CUDA_CHECK(cudaEventDestroy(stop)))
    ok = false;
  if (!ok)
    return false;
  std::sort(samples.begin(), samples.end());
  *median_ms = samples[samples.size() / 2];
  return true;
}

bool profile_composition(
    cublasHandle_t handle, const float* input,
    cuda_transformer::IncrementalDecoderBlockWeights weights,
    cuda_transformer::KvCache* cache,
    cuda_transformer::IncrementalDecoderBlockWorkspace* workspace, float* output,
    ProfileConfig config, std::array<float, kStageCount>* median_stage_ms) {
  for (int i = 0; i < kWarmups; ++i) {
    cache->current_length = config.history;
    if (!compose_incremental(handle, input, weights, cache, workspace, output))
      return false;
  }
  if (!CTR_CUDA_CHECK(cudaDeviceSynchronize()))
    return false;

  BatchEvents events;
  if (!events.create(config.launches_per_batch)) {
    events.destroy();
    return false;
  }
  bool ok = true;
  std::array<std::vector<float>, kStageCount> samples;
  for (std::vector<float>& stage_samples : samples)
    stage_samples.reserve(kBatches);

  for (int batch = 0; batch < kBatches && ok; ++batch) {
    for (int execution = 0; execution < config.launches_per_batch && ok;
         ++execution) {
      cache->current_length = config.history;
      ok = compose_incremental(handle, input, weights, cache, workspace, output,
                               events.boundaries(execution));
    }
    if (!ok)
      break;
    ok = CTR_CUDA_CHECK(cudaEventSynchronize(
        events.boundaries(config.launches_per_batch - 1)[kStageCount]));
    std::array<float, kStageCount> batch_ms{};
    for (int execution = 0; execution < config.launches_per_batch && ok;
         ++execution) {
      cudaEvent_t* boundaries = events.boundaries(execution);
      for (std::size_t stage = 0; stage < kStageCount; ++stage) {
        float elapsed_ms = 0.0F;
        ok = CTR_CUDA_CHECK(cudaEventElapsedTime(
            &elapsed_ms, boundaries[stage], boundaries[stage + 1]));
        batch_ms[stage] += elapsed_ms;
      }
    }
    if (ok) {
      for (std::size_t stage = 0; stage < kStageCount; ++stage)
        samples[stage].push_back(batch_ms[stage] /
                                  static_cast<float>(config.launches_per_batch));
    }
  }
  if (!events.destroy())
    ok = false;
  if (!ok)
    return false;

  for (std::size_t stage = 0; stage < kStageCount; ++stage) {
    std::sort(samples[stage].begin(), samples[stage].end());
    (*median_stage_ms)[stage] = samples[stage][samples[stage].size() / 2];
  }
  return true;
}

bool run_history(cublasHandle_t handle, const DeviceWeights& device_weights,
                 const float* device_input, ProfileConfig profile_config,
                 ProfileResult* result) {
  using namespace cuda_transformer;

  const std::size_t sequence = profile_config.history + 1;
  const IncrementalDecoderBlockConfig config{
      kHidden, kHeads, kHeadDim, kIntermediate, kEpsilon, kEpsilon, sequence,
  };
  const IncrementalDecoderBlockWeights weights{
      {device_weights.attention_norm, device_weights.wq, device_weights.wk,
       device_weights.wv, device_weights.wo},
      {device_weights.mlp_norm, device_weights.w_gate, device_weights.w_up,
       device_weights.w_down},
  };
  KvCache cache;
  IncrementalDecoderBlockWorkspace workspace;
  float* profiled_output = nullptr;
  float* production_output = nullptr;
  auto cleanup = [&](bool ok) {
    if (profiled_output != nullptr && !CTR_CUDA_CHECK(cudaFree(profiled_output)))
      ok = false;
    if (production_output != nullptr &&
        !CTR_CUDA_CHECK(cudaFree(production_output)))
      ok = false;
    if (!CTR_CUDA_CHECK(incremental_decoder_block_workspace_destroy(&workspace)))
      ok = false;
    if (!CTR_CUDA_CHECK(kv_cache_destroy(&cache)))
      ok = false;
    return ok;
  };

  if (!CTR_CUDA_CHECK(kv_cache_create(&cache, 1, kHeads, sequence, kHeadDim)) ||
      !CTR_CUDA_CHECK(
          incremental_decoder_block_workspace_create(&workspace, config)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&profiled_output, kHidden * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&production_output, kHidden * sizeof(float))))
    return cleanup(false);

  // Build exactly the same rotated-K/raw-V history as production decode.
  for (std::size_t token = 0; token < profile_config.history; ++token) {
    if (!CTR_CUBLAS_CHECK(incremental_decoder_block_cuda(
            handle, device_input + token * kHidden, weights, &cache, &workspace,
            production_output)))
      return cleanup(false);
  }
  if (!CTR_CUDA_CHECK(cudaDeviceSynchronize()) ||
      cache.current_length != profile_config.history) {
    std::fprintf(stderr, "cache prefill failed for history %zu\n",
                 profile_config.history);
    return cleanup(false);
  }

  // Compare this explicitly composed path with the production API before timing.
  cache.current_length = profile_config.history;
  if (!compose_incremental(handle,
                           device_input + profile_config.history * kHidden,
                           weights, &cache, &workspace, profiled_output) ||
      !CTR_CUDA_CHECK(cudaDeviceSynchronize()))
    return cleanup(false);
  cache.current_length = profile_config.history;
  if (!CTR_CUBLAS_CHECK(incremental_decoder_block_cuda(
          handle, device_input + profile_config.history * kHidden, weights,
          &cache, &workspace, production_output)) ||
      !CTR_CUDA_CHECK(cudaDeviceSynchronize()))
    return cleanup(false);
  std::vector<float> profiled_host(kHidden);
  std::vector<float> production_host(kHidden);
  if (!CTR_CUDA_CHECK(cudaMemcpy(profiled_host.data(), profiled_output,
                                 kHidden * sizeof(float),
                                 cudaMemcpyDeviceToHost)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(production_host.data(), production_output,
                                 kHidden * sizeof(float),
                                 cudaMemcpyDeviceToHost)) ||
      !check_output(profiled_host, production_host, profile_config.history))
    return cleanup(false);

  result->config = profile_config;
  const auto direct_launch = [&] {
    cache.current_length = profile_config.history;
    return CTR_CUBLAS_CHECK(incremental_decoder_block_cuda(
        handle, device_input + profile_config.history * kHidden, weights, &cache,
        &workspace, production_output));
  };
  if (!profile_composition(handle,
                           device_input + profile_config.history * kHidden,
                           weights, &cache, &workspace, profiled_output,
                           profile_config, &result->stage_ms) ||
      !direct_median_latency_ms(direct_launch, profile_config.launches_per_batch,
                                &result->direct_total_ms))
    return cleanup(false);

  for (const float stage_ms : result->stage_ms)
    result->stage_total_ms += stage_ms;
  return cleanup(true);
}

bool run_trace(cublasHandle_t handle, const DeviceWeights& device_weights,
               const float* device_input, TraceOptions options) {
  using namespace cuda_transformer;

  if (options.history == std::numeric_limits<std::size_t>::max()) {
    std::fputs("history is too large\n", stderr);
    return false;
  }
  const std::size_t sequence = options.history + 1;
  const IncrementalDecoderBlockConfig config{
      kHidden, kHeads, kHeadDim, kIntermediate, kEpsilon, kEpsilon, sequence,
  };
  const IncrementalDecoderBlockWeights weights{
      {device_weights.attention_norm, device_weights.wq, device_weights.wk,
       device_weights.wv, device_weights.wo},
      {device_weights.mlp_norm, device_weights.w_gate, device_weights.w_up,
       device_weights.w_down},
  };
  KvCache cache;
  IncrementalDecoderBlockWorkspace workspace;
  float* composed_output = nullptr;
  float* production_output = nullptr;
  auto cleanup = [&](bool ok) {
    if (composed_output != nullptr && !CTR_CUDA_CHECK(cudaFree(composed_output)))
      ok = false;
    if (production_output != nullptr &&
        !CTR_CUDA_CHECK(cudaFree(production_output)))
      ok = false;
    if (!CTR_CUDA_CHECK(incremental_decoder_block_workspace_destroy(&workspace)))
      ok = false;
    if (!CTR_CUDA_CHECK(kv_cache_destroy(&cache)))
      ok = false;
    return ok;
  };

  if (!CTR_CUDA_CHECK(kv_cache_create(&cache, 1, kHeads, sequence, kHeadDim)) ||
      !CTR_CUDA_CHECK(
          incremental_decoder_block_workspace_create(&workspace, config)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&composed_output, kHidden * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&production_output, kHidden * sizeof(float))))
    return cleanup(false);

  // This prefill is deliberately before cudaProfilerStart(), so a capture
  // range contains only the requested fixed-history production decodes.
  for (std::size_t token = 0; token < options.history; ++token) {
    if (!CTR_CUBLAS_CHECK(incremental_decoder_block_cuda(
            handle, device_input + token * kHidden, weights, &cache, &workspace,
            production_output)))
      return cleanup(false);
  }
  if (!CTR_CUDA_CHECK(cudaDeviceSynchronize()) ||
      cache.current_length != options.history) {
    std::fprintf(stderr, "cache prefill failed for history %zu\n",
                 options.history);
    return cleanup(false);
  }

  // Keep the existing composition-vs-production sanity check out of the
  // capture range. It verifies trace mode still exercises the same API path.
  cache.current_length = options.history;
  if (!compose_incremental(handle, device_input + options.history * kHidden,
                           weights, &cache, &workspace, composed_output) ||
      !CTR_CUDA_CHECK(cudaDeviceSynchronize()))
    return cleanup(false);
  cache.current_length = options.history;
  if (!CTR_CUBLAS_CHECK(incremental_decoder_block_cuda(
          handle, device_input + options.history * kHidden, weights, &cache,
          &workspace, production_output)) ||
      !CTR_CUDA_CHECK(cudaDeviceSynchronize()))
    return cleanup(false);
  std::vector<float> composed_host(kHidden);
  std::vector<float> production_host(kHidden);
  if (!CTR_CUDA_CHECK(cudaMemcpy(composed_host.data(), composed_output,
                                 kHidden * sizeof(float),
                                 cudaMemcpyDeviceToHost)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(production_host.data(), production_output,
                                 kHidden * sizeof(float),
                                 cudaMemcpyDeviceToHost)) ||
      !check_output(composed_host, production_host, options.history))
    return cleanup(false);

  for (int warmup = 0; warmup < kTraceWarmups; ++warmup) {
    cache.current_length = options.history;
    if (!CTR_CUBLAS_CHECK(incremental_decoder_block_cuda(
            handle, device_input + options.history * kHidden, weights, &cache,
            &workspace, production_output)))
      return cleanup(false);
  }
  if (!CTR_CUDA_CHECK(cudaDeviceSynchronize()))
    return cleanup(false);

  std::printf("Trace mode: history=%zu iterations=%d warmups=%d\n",
              options.history, options.iterations, kTraceWarmups);
  if (!profiler_range_call(cudaProfilerStart(), "cudaProfilerStart()"))
    return cleanup(false);

  bool ok = true;
  for (int iteration = 0; iteration < options.iterations; ++iteration) {
    cache.current_length = options.history;
    if (!CTR_CUBLAS_CHECK(incremental_decoder_block_cuda(
            handle, device_input + options.history * kHidden, weights, &cache,
            &workspace, production_output))) {
      ok = false;
      break;
    }
  }
  if (ok && !CTR_CUDA_CHECK(cudaDeviceSynchronize()))
    ok = false;
  if (!profiler_range_call(cudaProfilerStop(), "cudaProfilerStop()"))
    ok = false;
  if (ok)
    std::puts("Trace mode complete.");
  return cleanup(ok);
}

bool allocate_and_copy(float** device, const std::vector<float>& host) {
  return CTR_CUDA_CHECK(cudaMalloc(device, host.size() * sizeof(float))) &&
         CTR_CUDA_CHECK(cudaMemcpy(*device, host.data(),
                                   host.size() * sizeof(float),
                                   cudaMemcpyHostToDevice));
}

bool setup_weights(DeviceWeights* device_weights) {
  const std::size_t square_weight_count = kHidden * kHidden;
  const std::size_t gate_weight_count = kHidden * kIntermediate;
  const std::size_t down_weight_count = kIntermediate * kHidden;
  std::vector<float> attention_norm(kHidden), mlp_norm(kHidden),
      wq(square_weight_count), wk(square_weight_count), wv(square_weight_count),
      wo(square_weight_count), w_gate(gate_weight_count), w_up(gate_weight_count),
      w_down(down_weight_count);
  for (std::size_t i = 0; i < kHidden; ++i) {
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

  return allocate_and_copy(&device_weights->attention_norm, attention_norm) &&
         allocate_and_copy(&device_weights->mlp_norm, mlp_norm) &&
         allocate_and_copy(&device_weights->wq, wq) &&
         allocate_and_copy(&device_weights->wk, wk) &&
         allocate_and_copy(&device_weights->wv, wv) &&
         allocate_and_copy(&device_weights->wo, wo) &&
         allocate_and_copy(&device_weights->w_gate, w_gate) &&
         allocate_and_copy(&device_weights->w_up, w_up) &&
         allocate_and_copy(&device_weights->w_down, w_down);
}

bool free_weights(DeviceWeights* weights) {
  bool ok = true;
  for (float* pointer : {weights->attention_norm, weights->mlp_norm, weights->wq,
                         weights->wk, weights->wv, weights->wo, weights->w_gate,
                         weights->w_up, weights->w_down}) {
    if (pointer != nullptr && !CTR_CUDA_CHECK(cudaFree(pointer)))
      ok = false;
  }
  *weights = {};
  return ok;
}

void print_profile(const ProfileResult& result) {
  std::printf("\nhistory=%zu (%d batches x %d executions)\n",
              result.config.history, kBatches, result.config.launches_per_batch);
  std::puts("stage                           us     percent");
  for (std::size_t stage = 0; stage < kStageCount; ++stage) {
    const float percent = 100.0F * result.stage_ms[stage] / result.stage_total_ms;
    std::printf("%-28s %9.2f %9.2f%%\n", kStageNames[stage],
                result.stage_ms[stage] * 1000.0F, percent);
  }
  std::printf("%-28s %9.2f\n", "TOTAL measured composition",
              result.stage_total_ms * 1000.0F);
  std::printf("%-28s %9.2f\n", "direct end-to-end",
              result.direct_total_ms * 1000.0F);
  std::printf("%-28s %9.2f\n", "6C1 RTX 3050 baseline",
              result.config.baseline_6c1_ms * 1000.0F);
}

float grouped_stage_ms(const ProfileResult& result, std::size_t first,
                       std::size_t last) {
  float total = 0.0F;
  for (std::size_t stage = first; stage <= last; ++stage)
    total += result.stage_ms[stage];
  return total;
}

void print_cross_history(const std::vector<ProfileResult>& results) {
  std::puts("\nCross-history major-stage latency (microseconds)");
  std::printf("%-28s", "stage");
  for (const ProfileResult& result : results)
    std::printf(" h%-9zu", result.config.history);
  std::puts("");
  const auto row = [&](const char* name, std::size_t first, std::size_t last) {
    std::printf("%-28s", name);
    for (const ProfileResult& result : results)
      std::printf(" %10.2f", grouped_stage_ms(result, first, last) * 1000.0F);
    std::puts("");
  };
  row("attention RMSNorm", kAttentionRmsNorm, kAttentionRmsNorm);
  row("QKV projections", kQProjection, kVProjection);
  row("QKV layout", kQkvLayout, kQkvLayout);
  row("incremental attention core", kIncrementalAttention,
      kIncrementalAttention);
  row("attention output layout", kAttentionOutputLayout,
      kAttentionOutputLayout);
  row("Wo projection", kWoProjection, kWoProjection);
  row("attention residual", kAttentionResidual, kAttentionResidual);
  row("MLP RMSNorm", kMlpRmsNorm, kMlpRmsNorm);
  row("gate + up projections", kGateUpProjections, kGateUpProjections);
  row("SiLU + multiply", kSiluMultiply, kSiluMultiply);
  row("down projection", kDownProjection, kDownProjection);
  row("final residual", kFinalResidual, kFinalResidual);
  std::printf("%-28s", "TOTAL measured composition");
  for (const ProfileResult& result : results)
    std::printf(" %10.2f", result.stage_total_ms * 1000.0F);
  std::puts("");
  std::printf("%-28s", "direct end-to-end");
  for (const ProfileResult& result : results)
    std::printf(" %10.2f", result.direct_total_ms * 1000.0F);
  std::puts("");
}

}  // namespace

int main(int argc, char** argv) {
  TraceOptions trace_options;
  const CommandLineResult command_line =
      parse_command_line(argc, argv, &trace_options);
  if (command_line == CommandLineResult::kHelp)
    return EXIT_SUCCESS;
  if (command_line == CommandLineResult::kError)
    return EXIT_FAILURE;
  if (trace_options.enabled &&
      (trace_options.history == std::numeric_limits<std::size_t>::max() ||
       trace_options.history + 1 >
           std::numeric_limits<std::size_t>::max() / kHidden)) {
    std::fputs("trace history is too large\n", stderr);
    return EXIT_FAILURE;
  }

  constexpr std::array<ProfileConfig, 4> kProfiles{{
      {16, 500, 0.47382F},
      {64, 500, 0.51515F},
      {256, 300, 0.69575F},
      {1024, 100, 0.52362F},
  }};
  const std::size_t max_sequence =
      trace_options.enabled ? trace_options.history + 1
                            : kProfiles.back().history + 1;

  std::vector<float> input(max_sequence * kHidden);
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<float>(static_cast<int>((i * 5 + 3) % 37) - 18) /
               32.0F;

  cublasHandle_t handle = nullptr;
  float* device_input = nullptr;
  DeviceWeights device_weights;
  bool ok = CTR_CUBLAS_CHECK(cublasCreate(&handle)) &&
            allocate_and_copy(&device_input, input) &&
            setup_weights(&device_weights);
  std::vector<ProfileResult> results;
  if (ok) {
    if (trace_options.enabled) {
      ok = run_trace(handle, device_weights, device_input, trace_options);
    } else {
      std::puts("Incremental decoder-block stage profile (FP32, batch=1, "
                "hidden=256, heads=4, head_dim=64, intermediate=512)");
      for (const ProfileConfig profile_config : kProfiles) {
        ProfileResult result;
        if (!run_history(handle, device_weights, device_input, profile_config,
                         &result)) {
          ok = false;
          break;
        }
        print_profile(result);
        results.push_back(result);
      }
      if (ok)
        print_cross_history(results);
    }
  }
  if (!free_weights(&device_weights))
    ok = false;
  if (device_input != nullptr && !CTR_CUDA_CHECK(cudaFree(device_input)))
    ok = false;
  if (handle != nullptr && !CTR_CUBLAS_CHECK(cublasDestroy(handle)))
    ok = false;
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
