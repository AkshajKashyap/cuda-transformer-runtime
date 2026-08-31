#include "cuda_transformer/cublas_check.h"
#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/llama2_checkpoint.h"
#include "cuda_transformer/llama2_tokenizer.h"
#include "cuda_transformer/tiny_model.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include <cuda_profiler_api.h>

namespace {

constexpr char kBenchmarkPrompt[] = "I believe the meaning of life is";
constexpr std::array<std::size_t, 5> kPrefillLengths{8, 16, 32, 64, 128};
constexpr std::array<std::size_t, 7> kDecodeContexts{8, 16, 32, 64,
                                                       128, 192, 255};
constexpr std::array<std::size_t, 3> kGenerationLengths{16, 32, 64};
constexpr int kWarmups = 3;
constexpr int kBatches = 9;
constexpr int kDecodeLaunchesPerBatch = 20;
constexpr int kHostSelectionIterationsPerBatch = 50;
constexpr int kTraceWarmups = 5;

struct Options {
  const char* checkpoint_path = nullptr;
  const char* tokenizer_path = nullptr;
  bool trace = false;
  std::size_t trace_context = 0;
  int trace_iterations = 0;
  bool graph_decode = false;
  std::size_t graph_context = 0;
};

struct Stats {
  float median_ms = 0.0F;
  float average_ms = 0.0F;
};

struct SelectionStats {
  Stats transfer_sync{};
  Stats argmax{};
  Stats total{};
};

bool parse_positive_size(const char* text, std::size_t* value) {
  if (text == nullptr || *text == '\0')
    return false;
  char* end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || parsed == 0 ||
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

void print_usage(const char* program) {
  std::printf(
      "Usage:\n"
      "  %s checkpoint.bin tokenizer.bin\n"
      "  %s checkpoint.bin tokenizer.bin --trace-decode-context <N> "
      "--trace-iterations <N>\n"
      "  %s checkpoint.bin tokenizer.bin --graph-decode-context <N>\n",
      program, program, program);
}

bool parse_options(int argc, char** argv, Options* options) {
  if (options == nullptr || argc < 3)
    return false;
  options->checkpoint_path = argv[1];
  options->tokenizer_path = argv[2];
  if (argc == 3)
    return true;

  bool saw_context = false;
  bool saw_iterations = false;
  bool saw_graph_context = false;
  for (int index = 3; index < argc; ++index) {
    if (std::strcmp(argv[index], "--trace-decode-context") == 0 &&
        !saw_context && index + 1 < argc &&
        parse_positive_size(argv[++index], &options->trace_context)) {
      saw_context = true;
      continue;
    }
    if (std::strcmp(argv[index], "--trace-iterations") == 0 &&
        !saw_iterations && index + 1 < argc &&
        parse_positive_int(argv[++index], &options->trace_iterations)) {
      saw_iterations = true;
      continue;
    }
    if (std::strcmp(argv[index], "--graph-decode-context") == 0 &&
        !saw_graph_context && index + 1 < argc &&
        parse_positive_size(argv[++index], &options->graph_context)) {
      saw_graph_context = true;
      continue;
    }
    return false;
  }
  if (saw_graph_context && !saw_context && !saw_iterations) {
    options->graph_decode = true;
    return true;
  }
  options->trace = saw_context && saw_iterations;
  return options->trace;
}

Stats summarize(std::vector<float> samples) {
  Stats result{};
  if (samples.empty())
    return result;
  std::sort(samples.begin(), samples.end());
  for (const float sample : samples)
    result.average_ms += sample;
  result.average_ms /= static_cast<float>(samples.size());
  result.median_ms = samples[samples.size() / 2];
  return result;
}

// Resetting only `current_length` is safe for this benchmark: each queued
// decode receives its length as a launch parameter and overwrites the same
// previously-unused K/V slot. The already-prefilled [0, history) entries are
// never modified.
bool restore_cache_lengths(cuda_transformer::TinyModelIncrementalWorkspace* workspace,
                           std::size_t history) {
  if (workspace == nullptr || workspace->layer_caches == nullptr ||
      history >= workspace->max_sequence_length)
    return false;
  for (std::size_t layer = 0; layer < workspace->model_config.layers; ++layer)
    workspace->layer_caches[layer].current_length = history;
  return true;
}

bool prefill_history(cublasHandle_t handle, const int* device_tokens,
                     std::size_t length,
                     cuda_transformer::TinyModelWeights device_weights,
                     cuda_transformer::TinyModelIncrementalWorkspace* workspace,
                     float* device_logits, cudaStream_t stream = nullptr) {
  using namespace cuda_transformer;
  return CTR_CUDA_CHECK(tiny_model_incremental_workspace_reset(workspace)) &&
         CTR_CUBLAS_CHECK(tiny_model_incremental_prefill_cuda(
             handle, device_tokens, length, device_weights, workspace,
             device_logits, stream));
}

// Each launch has its own event pair. Host-only logical-cache restoration is
// therefore outside every interval, avoiding idle CPU submission gaps in a
// single large event interval.
bool measure_gpu_batches(const std::function<bool()>& prepare,
                         const std::function<bool()>& launch,
                         int launches_per_batch, Stats* stats,
                         cudaStream_t stream = nullptr) {
  if (launches_per_batch <= 0 || stats == nullptr)
    return false;
  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    if (!prepare() || !launch())
      return false;
  }
  if (!CTR_CUDA_CHECK(cudaStreamSynchronize(stream)))
    return false;

  std::vector<cudaEvent_t> starts(static_cast<std::size_t>(launches_per_batch),
                                  nullptr);
  std::vector<cudaEvent_t> stops(static_cast<std::size_t>(launches_per_batch),
                                 nullptr);
  bool ok = true;
  for (int launch_index = 0; launch_index < launches_per_batch; ++launch_index) {
    ok = CTR_CUDA_CHECK(cudaEventCreate(&starts[launch_index])) &&
         CTR_CUDA_CHECK(cudaEventCreate(&stops[launch_index]));
    if (!ok)
      break;
  }

  std::vector<float> samples;
  samples.reserve(kBatches);
  for (int batch = 0; batch < kBatches && ok; ++batch) {
    float batch_ms = 0.0F;
    for (int launch_index = 0; launch_index < launches_per_batch && ok;
         ++launch_index) {
      ok = prepare() &&
           CTR_CUDA_CHECK(cudaEventRecord(starts[launch_index], stream)) &&
           launch() && CTR_CUDA_CHECK(cudaEventRecord(stops[launch_index], stream));
    }
    if (ok)
      ok = CTR_CUDA_CHECK(
          cudaEventSynchronize(stops[static_cast<std::size_t>(launches_per_batch - 1)]));
    for (int launch_index = 0; launch_index < launches_per_batch && ok;
         ++launch_index) {
      float elapsed_ms = 0.0F;
      ok = CTR_CUDA_CHECK(cudaEventElapsedTime(
          &elapsed_ms, starts[launch_index], stops[launch_index]));
      batch_ms += elapsed_ms;
    }
    if (ok)
      samples.push_back(batch_ms / static_cast<float>(launches_per_batch));
  }
  for (cudaEvent_t event : starts) {
    if (event != nullptr && !CTR_CUDA_CHECK(cudaEventDestroy(event)))
      ok = false;
  }
  for (cudaEvent_t event : stops) {
    if (event != nullptr && !CTR_CUDA_CHECK(cudaEventDestroy(event)))
      ok = false;
  }
  if (!ok)
    return false;
  *stats = summarize(std::move(samples));
  return true;
}

// This includes host API submission plus completion of the queued GPU work.
// `prepare` remains outside the clock so logical cache reset is not presented
// as model prefill latency.
bool measure_wall_batches(const std::function<bool()>& prepare,
                          const std::function<bool()>& launch, Stats* stats,
                          cudaStream_t stream = nullptr) {
  using Clock = std::chrono::steady_clock;
  if (stats == nullptr)
    return false;
  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    if (!prepare() || !launch() || !CTR_CUDA_CHECK(cudaStreamSynchronize(stream)))
      return false;
  }
  std::vector<float> samples;
  samples.reserve(kBatches);
  for (int batch = 0; batch < kBatches; ++batch) {
    if (!prepare())
      return false;
    const auto start = Clock::now();
    if (!launch() || !CTR_CUDA_CHECK(cudaStreamSynchronize(stream)))
      return false;
    const auto stop = Clock::now();
    samples.push_back(
        std::chrono::duration<float, std::milli>(stop - start).count());
  }
  *stats = summarize(std::move(samples));
  return true;
}

bool measure_selection(const float* device_logits, std::size_t vocabulary_size,
                       SelectionStats* stats) {
  using Clock = std::chrono::steady_clock;
  if (device_logits == nullptr || vocabulary_size == 0 || stats == nullptr)
    return false;
  std::vector<float> host_logits(vocabulary_size);
  auto one_selection = [&](float* transfer_ms, float* argmax_ms,
                           float* total_ms) {
    const auto total_start = Clock::now();
    if (!CTR_CUDA_CHECK(cudaMemcpyAsync(host_logits.data(), device_logits,
                                        vocabulary_size * sizeof(float),
                                        cudaMemcpyDeviceToHost)) ||
        !CTR_CUDA_CHECK(cudaStreamSynchronize(nullptr)))
      return false;
    const auto after_transfer = Clock::now();
    int selected_token = -1;
    if (!cuda_transformer::tiny_model_greedy_argmax_host(
            host_logits.data(), host_logits.size(), &selected_token))
      return false;
    const auto total_stop = Clock::now();
    *transfer_ms = std::chrono::duration<float, std::milli>(
                       after_transfer - total_start)
                       .count();
    *argmax_ms =
        std::chrono::duration<float, std::milli>(total_stop - after_transfer)
            .count();
    *total_ms =
        std::chrono::duration<float, std::milli>(total_stop - total_start).count();
    return true;
  };

  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    float transfer_ms = 0.0F;
    float argmax_ms = 0.0F;
    float total_ms = 0.0F;
    if (!one_selection(&transfer_ms, &argmax_ms, &total_ms))
      return false;
  }

  std::vector<float> transfer_samples;
  std::vector<float> argmax_samples;
  std::vector<float> total_samples;
  transfer_samples.reserve(kBatches);
  argmax_samples.reserve(kBatches);
  total_samples.reserve(kBatches);
  for (int batch = 0; batch < kBatches; ++batch) {
    float transfer_sum = 0.0F;
    float argmax_sum = 0.0F;
    float total_sum = 0.0F;
    for (int iteration = 0; iteration < kHostSelectionIterationsPerBatch;
         ++iteration) {
      float transfer_ms = 0.0F;
      float argmax_ms = 0.0F;
      float total_ms = 0.0F;
      if (!one_selection(&transfer_ms, &argmax_ms, &total_ms))
        return false;
      transfer_sum += transfer_ms;
      argmax_sum += argmax_ms;
      total_sum += total_ms;
    }
    const float divisor = static_cast<float>(kHostSelectionIterationsPerBatch);
    transfer_samples.push_back(transfer_sum / divisor);
    argmax_samples.push_back(argmax_sum / divisor);
    total_samples.push_back(total_sum / divisor);
  }
  stats->transfer_sync = summarize(std::move(transfer_samples));
  stats->argmax = summarize(std::move(argmax_samples));
  stats->total = summarize(std::move(total_samples));
  return true;
}

bool measure_generation(cublasHandle_t handle, const std::vector<int>& prompt,
                        cuda_transformer::TinyModelWeights device_weights,
                        cuda_transformer::TinyModelIncrementalWorkspace* workspace,
                        float* device_logits, std::size_t requested_tokens,
                        Stats* stats, std::size_t* generated_tokens) {
  using Clock = std::chrono::steady_clock;
  if (stats == nullptr || generated_tokens == nullptr)
    return false;
  std::vector<int> output(requested_tokens);
  auto run_once = [&](float* elapsed_ms, std::size_t* count) {
    const auto start = Clock::now();
    if (!CTR_CUBLAS_CHECK(cuda_transformer::tiny_model_generate_greedy_cuda(
            handle, prompt.data(), prompt.size(), device_weights, workspace,
            requested_tokens, output.data(), device_logits, nullptr,
            cuda_transformer::kLlama2BosTokenId, count)) ||
        // The generation helper already synchronizes after every logits copy.
        // This explicit benchmark-level fence guards the wall-clock boundary.
        !CTR_CUDA_CHECK(cudaStreamSynchronize(nullptr)))
      return false;
    const auto stop = Clock::now();
    *elapsed_ms = std::chrono::duration<float, std::milli>(stop - start).count();
    return true;
  };

  std::size_t expected_count = 0;
  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    float elapsed_ms = 0.0F;
    std::size_t count = 0;
    if (!run_once(&elapsed_ms, &count))
      return false;
    if (warmup == 0)
      expected_count = count;
    else if (count != expected_count)
      return false;
  }

  std::vector<float> samples;
  samples.reserve(kBatches);
  for (int batch = 0; batch < kBatches; ++batch) {
    float elapsed_ms = 0.0F;
    std::size_t count = 0;
    if (!run_once(&elapsed_ms, &count) || count != expected_count)
      return false;
    samples.push_back(elapsed_ms);
  }
  *stats = summarize(std::move(samples));
  *generated_tokens = expected_count;
  return true;
}

bool profiler_range_call(cudaError_t error, const char* expression) {
  // The trace workload remains runnable without an active Nsight capture.
  if (error == cudaSuccess || error == cudaErrorProfilerDisabled)
    return true;
  return cuda_transformer::check_cuda(error, expression, __FILE__, __LINE__);
}

bool trace_decode(cublasHandle_t handle, const int* device_tokens,
                  cuda_transformer::TinyModelWeights device_weights,
                  cuda_transformer::TinyModelIncrementalWorkspace* workspace,
                  float* device_logits, std::size_t history, int iterations) {
  if (!prefill_history(handle, device_tokens, history, device_weights, workspace,
                       device_logits) ||
      !CTR_CUDA_CHECK(cudaDeviceSynchronize()))
    return false;
  for (int warmup = 0; warmup < kTraceWarmups; ++warmup) {
    if (!restore_cache_lengths(workspace, history) ||
        !CTR_CUBLAS_CHECK(cuda_transformer::tiny_model_incremental_decode_cuda(
            handle, device_tokens + history, device_weights, workspace,
            device_logits)))
      return false;
  }
  if (!CTR_CUDA_CHECK(cudaDeviceSynchronize()))
    return false;

  std::printf("trace mode decode_context=%zu iterations=%d warmups=%d\n", history,
              iterations, kTraceWarmups);
  if (!profiler_range_call(cudaProfilerStart(), "cudaProfilerStart()"))
    return false;
  bool ok = true;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (!restore_cache_lengths(workspace, history) ||
        !CTR_CUBLAS_CHECK(cuda_transformer::tiny_model_incremental_decode_cuda(
            handle, device_tokens + history, device_weights, workspace,
            device_logits))) {
      ok = false;
      break;
    }
  }
  if (ok && !CTR_CUDA_CHECK(cudaDeviceSynchronize()))
    ok = false;
  if (!profiler_range_call(cudaProfilerStop(), "cudaProfilerStop()"))
    ok = false;
  if (ok)
    std::puts("trace mode complete.");
  return ok;
}

struct CacheSnapshot {
  std::vector<std::vector<float>> keys;
  std::vector<std::vector<float>> values;
};

bool cache_lengths_are(const cuda_transformer::TinyModelIncrementalWorkspace& workspace,
                       std::size_t expected) {
  if (workspace.layer_caches == nullptr)
    return false;
  for (std::size_t layer = 0; layer < workspace.model_config.layers; ++layer) {
    if (workspace.layer_caches[layer].current_length != expected)
      return false;
  }
  return true;
}

bool snapshot_caches(const cuda_transformer::TinyModelIncrementalWorkspace& workspace,
                     CacheSnapshot* snapshot, cudaStream_t stream) {
  if (snapshot == nullptr || workspace.layer_caches == nullptr)
    return false;
  const std::size_t layer_count = workspace.model_config.layers;
  const std::size_t elements = workspace.model_config.heads *
                               workspace.max_sequence_length *
                               workspace.model_config.head_dim;
  snapshot->keys.assign(layer_count, std::vector<float>(elements));
  snapshot->values.assign(layer_count, std::vector<float>(elements));
  for (std::size_t layer = 0; layer < layer_count; ++layer) {
    const auto& cache = workspace.layer_caches[layer];
    if (!CTR_CUDA_CHECK(cudaMemcpyAsync(
            snapshot->keys[layer].data(), cache.keys, elements * sizeof(float),
            cudaMemcpyDeviceToHost, stream)) ||
        !CTR_CUDA_CHECK(cudaMemcpyAsync(
            snapshot->values[layer].data(), cache.values,
            elements * sizeof(float), cudaMemcpyDeviceToHost, stream)))
      return false;
  }
  return CTR_CUDA_CHECK(cudaStreamSynchronize(stream));
}

bool cache_bytes_equal(const CacheSnapshot& left, const CacheSnapshot& right) {
  if (left.keys.size() != right.keys.size() ||
      left.values.size() != right.values.size())
    return false;
  for (std::size_t layer = 0; layer < left.keys.size(); ++layer) {
    if (left.keys[layer].size() != right.keys[layer].size() ||
        left.values[layer].size() != right.values[layer].size() ||
        std::memcmp(left.keys[layer].data(), right.keys[layer].data(),
                    left.keys[layer].size() * sizeof(float)) != 0 ||
        std::memcmp(left.values[layer].data(), right.values[layer].data(),
                    left.values[layer].size() * sizeof(float)) != 0)
      return false;
  }
  return true;
}

bool cache_only_appended(const CacheSnapshot& before, const CacheSnapshot& after,
                         std::size_t history, std::size_t heads,
                         std::size_t max_sequence, std::size_t head_dim) {
  if (before.keys.size() != after.keys.size() ||
      before.values.size() != after.values.size() || history >= max_sequence)
    return false;
  const std::size_t bytes = head_dim * sizeof(float);
  for (std::size_t layer = 0; layer < before.keys.size(); ++layer) {
    for (std::size_t head = 0; head < heads; ++head) {
      for (std::size_t position = 0; position < max_sequence; ++position) {
        if (position == history)
          continue;
        const std::size_t offset =
            (head * max_sequence + position) * head_dim;
        if (std::memcmp(before.keys[layer].data() + offset,
                        after.keys[layer].data() + offset, bytes) != 0 ||
            std::memcmp(before.values[layer].data() + offset,
                        after.values[layer].data() + offset, bytes) != 0)
          return false;
      }
    }
  }
  return true;
}

bool close_logits(const std::vector<float>& ordinary,
                  const std::vector<float>& graph, float* max_error) {
  if (ordinary.size() != graph.size() || max_error == nullptr)
    return false;
  *max_error = 0.0F;
  for (std::size_t index = 0; index < ordinary.size(); ++index) {
    const float tolerance = 1.0e-3F +
                            2.0e-4F * fmaxf(1.0F, std::fabs(ordinary[index]));
    const float error = std::fabs(ordinary[index] - graph[index]);
    *max_error = fmaxf(*max_error, error);
    if (!std::isfinite(ordinary[index]) || !std::isfinite(graph[index]) ||
        error > tolerance)
      return false;
  }
  return true;
}

bool graph_decode_experiment(
    cublasHandle_t handle, const int* device_tokens,
    cuda_transformer::TinyModelWeights device_weights,
    cuda_transformer::TinyModelIncrementalWorkspace* workspace,
    float* device_logits, std::size_t history, cudaStream_t stream) {
  using Clock = std::chrono::steady_clock;
  if (workspace == nullptr || stream == nullptr ||
      !prefill_history(handle, device_tokens, history, device_weights, workspace,
                       device_logits, stream) ||
      !CTR_CUDA_CHECK(cudaStreamSynchronize(stream)) ||
      !cache_lengths_are(*workspace, history))
    return false;

  CacheSnapshot prefix_cache;
  if (!snapshot_caches(*workspace, &prefix_cache, stream))
    return false;

  if (!CTR_CUBLAS_CHECK(cuda_transformer::tiny_model_incremental_decode_cuda(
          handle, device_tokens + history, device_weights, workspace,
          device_logits, stream)) ||
      !CTR_CUDA_CHECK(cudaStreamSynchronize(stream)) ||
      !cache_lengths_are(*workspace, history + 1))
    return false;
  std::vector<float> ordinary_logits(workspace->model_config.vocabulary_size);
  if (!CTR_CUDA_CHECK(cudaMemcpyAsync(
          ordinary_logits.data(), device_logits,
          ordinary_logits.size() * sizeof(float), cudaMemcpyDeviceToHost,
          stream)) ||
      !CTR_CUDA_CHECK(cudaStreamSynchronize(stream)))
    return false;
  CacheSnapshot ordinary_cache;
  if (!snapshot_caches(*workspace, &ordinary_cache, stream) ||
      !cache_only_appended(prefix_cache, ordinary_cache, history,
                           workspace->model_config.heads,
                           workspace->max_sequence_length,
                           workspace->model_config.head_dim)) {
    std::fputs("ordinary graph-reference decode modified cache outside append slot\n",
               stderr);
    return false;
  }

  if (!restore_cache_lengths(workspace, history))
    return false;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  bool ok = true;
  const auto capture_start = Clock::now();
  if (!CTR_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal))) {
    ok = false;
  } else {
    const cublasStatus_t status = cuda_transformer::tiny_model_incremental_decode_cuda(
        handle, device_tokens + history, device_weights, workspace,
        device_logits, stream);
    const cudaError_t end_error = cudaStreamEndCapture(stream, &graph);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::fprintf(stderr, "graph capture decode failed: %s\n",
                   cuda_transformer::cublas_status_name(status));
      ok = false;
    }
    if (!CTR_CUDA_CHECK(end_error) || graph == nullptr)
      ok = false;
  }
  const auto capture_stop = Clock::now();
  const float capture_ms = std::chrono::duration<float, std::milli>(
                               capture_stop - capture_start)
                               .count();
  if (ok && !cache_lengths_are(*workspace, history + 1)) {
    std::fputs("capture did not advance host cache lengths exactly once\n", stderr);
    ok = false;
  }
  const auto instantiate_start = Clock::now();
  if (ok && !CTR_CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr,
                                                   nullptr, 0)))
    ok = false;
  const auto instantiate_stop = Clock::now();
  const float instantiate_ms = std::chrono::duration<float, std::milli>(
                                   instantiate_stop - instantiate_start)
                                   .count();
  if (ok && !restore_cache_lengths(workspace, history))
    ok = false;

  std::vector<float> graph_logits(workspace->model_config.vocabulary_size);
  CacheSnapshot graph_cache;
  float max_error = 0.0F;
  if (ok && (!CTR_CUDA_CHECK(cudaGraphLaunch(graph_exec, stream)) ||
             !CTR_CUDA_CHECK(cudaStreamSynchronize(stream)) ||
             !cache_lengths_are(*workspace, history) ||
             !CTR_CUDA_CHECK(cudaMemcpyAsync(
                 graph_logits.data(), device_logits,
                 graph_logits.size() * sizeof(float), cudaMemcpyDeviceToHost,
                 stream)) ||
             !CTR_CUDA_CHECK(cudaStreamSynchronize(stream)) ||
             !close_logits(ordinary_logits, graph_logits, &max_error) ||
             !snapshot_caches(*workspace, &graph_cache, stream) ||
             !cache_bytes_equal(ordinary_cache, graph_cache))) {
    std::fputs("graph replay correctness or cache-integrity check failed\n", stderr);
    ok = false;
  }
  if (ok) {
    std::printf("graph_capture context=%zu capture_ms=%.5f instantiate_ms=%.5f\n",
                history, capture_ms, instantiate_ms);
    std::printf("graph_correctness context=%zu max_logit_error=%.8g "
                "cache_lengths=%zu prefix_and_append=pass\n",
                history, max_error, history);
  }

  Stats ordinary_stats{};
  Stats graph_stats{};
  Stats graph_wall_stats{};
  const auto prepare = [&] { return restore_cache_lengths(workspace, history); };
  const auto ordinary_launch = [&] {
    return CTR_CUBLAS_CHECK(cuda_transformer::tiny_model_incremental_decode_cuda(
        handle, device_tokens + history, device_weights, workspace,
        device_logits, stream));
  };
  const auto graph_launch = [&] {
    return CTR_CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
  };
  if (ok && (!measure_gpu_batches(prepare, ordinary_launch,
                                  kDecodeLaunchesPerBatch, &ordinary_stats,
                                  stream) ||
             !measure_gpu_batches(prepare, graph_launch, kDecodeLaunchesPerBatch,
                                  &graph_stats, stream) ||
             !measure_wall_batches(prepare, graph_launch, &graph_wall_stats,
                                   stream)))
    ok = false;
  if (ok) {
    const float reduction_ms = ordinary_stats.median_ms - graph_stats.median_ms;
    const float reduction_percent =
        100.0F * reduction_ms / ordinary_stats.median_ms;
    const float speedup = ordinary_stats.median_ms / graph_stats.median_ms;
    std::printf("ordinary_decode context=%zu median_ms=%.5f average_ms=%.5f "
                "batches=%d launches_per_batch=%d\n",
                history, ordinary_stats.median_ms, ordinary_stats.average_ms,
                kBatches, kDecodeLaunchesPerBatch);
    std::printf("graph_decode context=%zu median_ms=%.5f average_ms=%.5f "
                "wall_median_ms=%.5f wall_average_ms=%.5f batches=%d "
                "launches_per_batch=%d\n",
                history, graph_stats.median_ms, graph_stats.average_ms,
                graph_wall_stats.median_ms, graph_wall_stats.average_ms,
                kBatches, kDecodeLaunchesPerBatch);
    std::printf("graph_comparison context=%zu median_reduction_ms=%.5f "
                "median_reduction_percent=%.2f speedup=%.3fx\n",
                history, reduction_ms, reduction_percent, speedup);
  }
  if (graph_exec != nullptr && !CTR_CUDA_CHECK(cudaGraphExecDestroy(graph_exec)))
    ok = false;
  if (graph != nullptr && !CTR_CUDA_CHECK(cudaGraphDestroy(graph)))
    ok = false;
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse_options(argc, argv, &options)) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  cuda_transformer::Llama2CheckpointModel model;
  std::string error;
  const auto checkpoint_status = cuda_transformer::llama2_checkpoint_load(
      options.checkpoint_path, &model, &error);
  if (checkpoint_status != cuda_transformer::Llama2CheckpointStatus::kSuccess) {
    std::fprintf(stderr, "checkpoint load failed (%s): %s\n",
                 cuda_transformer::llama2_checkpoint_status_string(checkpoint_status),
                 error.c_str());
    return EXIT_FAILURE;
  }
  cuda_transformer::Llama2Tokenizer tokenizer;
  const auto tokenizer_status = tokenizer.load(
      options.tokenizer_path, model.checkpoint_config.vocabulary_size, &error);
  if (tokenizer_status != cuda_transformer::Llama2TokenizerStatus::kSuccess) {
    std::fprintf(stderr, "tokenizer load failed (%s): %s\n",
                 cuda_transformer::llama2_tokenizer_status_string(tokenizer_status),
                 error.c_str());
    return EXIT_FAILURE;
  }
  const std::vector<int> prompt = tokenizer.encode(kBenchmarkPrompt, true, false);
  if (prompt.empty()) {
    std::fputs("benchmark prompt tokenization failed\n", stderr);
    return EXIT_FAILURE;
  }

  cuda_transformer::TinyModelConfig config;
  if (cuda_transformer::llama2_checkpoint_tiny_model_config(
          model.checkpoint_config, 1, &config) !=
      cuda_transformer::Llama2CheckpointStatus::kSuccess) {
    std::fputs("could not construct one-token runtime configuration\n", stderr);
    return EXIT_FAILURE;
  }
  const std::size_t max_context = model.checkpoint_config.max_sequence_length;
  if (max_context <= *std::max_element(kDecodeContexts.begin(),
                                       kDecodeContexts.end()) ||
      prompt.size() > max_context) {
    std::fputs("checkpoint context capacity is too small for this benchmark\n",
               stderr);
    return EXIT_FAILURE;
  }
  if ((options.trace && options.trace_context >= max_context) ||
      (options.graph_decode && options.graph_context >= max_context)) {
    std::fprintf(stderr,
                 "decode context must be in [1, %zu] for this checkpoint\n",
                 max_context - 1);
    return EXIT_FAILURE;
  }

  cublasHandle_t handle = nullptr;
  cuda_transformer::TinyModelIncrementalWorkspace workspace;
  int* device_tokens = nullptr;
  float* device_logits = nullptr;
  cudaStream_t graph_stream = nullptr;
  auto cleanup = [&] {
    bool ok = true;
    if (graph_stream != nullptr && !CTR_CUDA_CHECK(cudaStreamDestroy(graph_stream)))
      ok = false;
    if (device_logits != nullptr && !CTR_CUDA_CHECK(cudaFree(device_logits)))
      ok = false;
    if (device_tokens != nullptr && !CTR_CUDA_CHECK(cudaFree(device_tokens)))
      ok = false;
    if (!CTR_CUDA_CHECK(
            cuda_transformer::tiny_model_incremental_workspace_destroy(&workspace)))
      ok = false;
    if (handle != nullptr && !CTR_CUBLAS_CHECK(cublasDestroy(handle)))
      ok = false;
    return ok;
  };
  if (!CTR_CUBLAS_CHECK(cublasCreate(&handle)) ||
      !CTR_CUDA_CHECK(cuda_transformer::tiny_model_incremental_workspace_create(
          &workspace, config, max_context)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&device_tokens, max_context * sizeof(int))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&device_logits,
                                 config.vocabulary_size * sizeof(float)))) {
    cleanup();
    return EXIT_FAILURE;
  }

  // These deterministic, valid IDs are copied once during setup. Repeating the
  // real prompt supplies all requested context lengths without treating host
  // token transfer as part of GPU-only prefill/decode timing.
  std::vector<int> context_tokens(max_context);
  for (std::size_t position = 0; position < context_tokens.size(); ++position)
    context_tokens[position] = prompt[position % prompt.size()];
  if (!CTR_CUDA_CHECK(cudaMemcpy(device_tokens, context_tokens.data(),
                                 context_tokens.size() * sizeof(int),
                                 cudaMemcpyHostToDevice))) {
    cleanup();
    return EXIT_FAILURE;
  }

  std::printf("stories15M benchmark: hidden=%zu layers=%zu heads=%zu vocab=%zu "
              "max_context=%zu\n",
              config.hidden, config.layers, config.heads, config.vocabulary_size,
              max_context);
  std::printf("method warmups=%d batches=%d decode_launches_per_batch=%d\n",
              kWarmups, kBatches, kDecodeLaunchesPerBatch);

  bool ok = true;
  if (options.graph_decode) {
    if (!CTR_CUDA_CHECK(
            cudaStreamCreateWithFlags(&graph_stream, cudaStreamNonBlocking))) {
      cleanup();
      return EXIT_FAILURE;
    }
    ok = graph_decode_experiment(handle, device_tokens, model.device_weights,
                                 &workspace, device_logits,
                                 options.graph_context, graph_stream);
    return cleanup() && ok ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  if (options.trace) {
    ok = trace_decode(handle, device_tokens, model.device_weights, &workspace,
                      device_logits, options.trace_context,
                      options.trace_iterations);
    return cleanup() && ok ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  for (const std::size_t length : kPrefillLengths) {
    Stats gpu_stats{};
    Stats wall_stats{};
    const auto prepare = [&] {
      return CTR_CUDA_CHECK(
          cuda_transformer::tiny_model_incremental_workspace_reset(&workspace));
    };
    const auto launch = [&] {
      return CTR_CUBLAS_CHECK(cuda_transformer::tiny_model_incremental_prefill_cuda(
          handle, device_tokens, length, model.device_weights, &workspace,
          device_logits));
    };
    ok = measure_gpu_batches(prepare, launch, 1, &gpu_stats) &&
         measure_wall_batches(prepare, launch, &wall_stats);
    if (!ok)
      break;
    std::printf("prefill length=%zu method=sequential_incremental_gpu "
                "gpu_median_ms=%.5f gpu_average_ms=%.5f "
                "wall_median_ms=%.5f wall_average_ms=%.5f "
                "wall_median_ms_per_token=%.5f "
                "wall_average_ms_per_token=%.5f batches=%d\n",
                length, gpu_stats.median_ms, gpu_stats.average_ms,
                wall_stats.median_ms, wall_stats.average_ms,
                wall_stats.median_ms / static_cast<float>(length),
                wall_stats.average_ms / static_cast<float>(length), kBatches);
  }

  for (const std::size_t history : kDecodeContexts) {
    if (!ok || !prefill_history(handle, device_tokens, history,
                               model.device_weights, &workspace, device_logits) ||
        !CTR_CUDA_CHECK(cudaDeviceSynchronize())) {
      ok = false;
      break;
    }
    Stats stats{};
    const auto prepare = [&] { return restore_cache_lengths(&workspace, history); };
    const auto launch = [&] {
      return CTR_CUBLAS_CHECK(cuda_transformer::tiny_model_incremental_decode_cuda(
          handle, device_tokens + history, model.device_weights, &workspace,
          device_logits));
    };
    ok = measure_gpu_batches(prepare, launch, kDecodeLaunchesPerBatch, &stats);
    if (!ok)
      break;
    std::printf("decode context=%zu method=gpu_cached_model "
                "median_ms=%.5f average_ms=%.5f batches=%d "
                "launches_per_batch=%d\n",
                history, stats.median_ms, stats.average_ms, kBatches,
                kDecodeLaunchesPerBatch);
  }

  // Produce representative real-model logits once. The following metric then
  // measures the existing D2H + synchronization + CPU selection path alone.
  if (ok && (!prefill_history(handle, device_tokens, 128, model.device_weights,
                              &workspace, device_logits) ||
             !CTR_CUDA_CHECK(cudaDeviceSynchronize()) ||
             !restore_cache_lengths(&workspace, 128) ||
             !CTR_CUBLAS_CHECK(
                 cuda_transformer::tiny_model_incremental_decode_cuda(
                     handle, device_tokens + 128, model.device_weights,
                     &workspace, device_logits)) ||
             !CTR_CUDA_CHECK(cudaDeviceSynchronize())))
    ok = false;
  SelectionStats selection{};
  if (ok)
    ok = measure_selection(device_logits, config.vocabulary_size, &selection);
  if (ok) {
    std::printf("logits_selection vocab=%zu logits_bytes=%zu "
                "transfer_sync_median_ms=%.5f transfer_sync_average_ms=%.5f "
                "argmax_median_ms=%.5f argmax_average_ms=%.5f "
                "total_median_ms=%.5f total_average_ms=%.5f batches=%d "
                "iterations_per_batch=%d\n",
                config.vocabulary_size,
                config.vocabulary_size * sizeof(float),
                selection.transfer_sync.median_ms,
                selection.transfer_sync.average_ms, selection.argmax.median_ms,
                selection.argmax.average_ms, selection.total.median_ms,
                selection.total.average_ms, kBatches,
                kHostSelectionIterationsPerBatch);
  }

  for (const std::size_t requested : kGenerationLengths) {
    if (!ok)
      break;
    Stats stats{};
    std::size_t generated = 0;
    ok = measure_generation(handle, prompt, model.device_weights, &workspace,
                            device_logits, requested, &stats, &generated);
    if (!ok || generated == 0)
      break;
    std::printf("generation requested_new_tokens=%zu generated_tokens=%zu "
                "method=greedy_end_to_end median_ms=%.5f average_ms=%.5f "
                "median_tokens_per_s=%.5f average_tokens_per_s=%.5f "
                "median_ms_per_generated_token=%.5f "
                "average_ms_per_generated_token=%.5f batches=%d\n",
                requested, generated, stats.median_ms, stats.average_ms,
                1000.0F * static_cast<float>(generated) / stats.median_ms,
                1000.0F * static_cast<float>(generated) / stats.average_ms,
                stats.median_ms / static_cast<float>(generated),
                stats.average_ms / static_cast<float>(generated), kBatches);
  }
  return cleanup() && ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
