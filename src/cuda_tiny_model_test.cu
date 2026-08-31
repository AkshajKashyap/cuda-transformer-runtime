#include "cuda_transformer/cublas_check.h"
#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/tiny_model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr float kAbsoluteTolerance = 5e-4F;
constexpr float kRelativeTolerance = 1e-4F;

struct HostLayerWeights {
  std::vector<float> attention_norm;
  std::vector<float> wq;
  std::vector<float> wk;
  std::vector<float> wv;
  std::vector<float> wo;
  std::vector<float> mlp_norm;
  std::vector<float> w_gate;
  std::vector<float> w_up;
  std::vector<float> w_down;
};

struct HostWeights {
  std::vector<float> embeddings;
  std::vector<float> final_norm;
  std::vector<float> lm_head;
  std::vector<HostLayerWeights> layers;
  std::vector<cuda_transformer::TinyModelLayerWeights> layer_views;
  cuda_transformer::TinyModelWeights view{};
};

struct DeviceLayerWeights {
  float* attention_norm = nullptr;
  float* wq = nullptr;
  float* wk = nullptr;
  float* wv = nullptr;
  float* wo = nullptr;
  float* mlp_norm = nullptr;
  float* w_gate = nullptr;
  float* w_up = nullptr;
  float* w_down = nullptr;
};

struct DeviceWeights {
  float* embeddings = nullptr;
  float* final_norm = nullptr;
  float* lm_head = nullptr;
  std::vector<DeviceLayerWeights> layers;
  std::vector<cuda_transformer::TinyModelLayerWeights> layer_views;
  cuda_transformer::TinyModelWeights view{};
};

float next_value(unsigned int* state, float scale) {
  *state = *state * 1664525U + 1013904223U;
  const float unit = static_cast<float>((*state >> 8) & 0xFFFFU) / 65535.0F;
  return scale * (2.0F * unit - 1.0F);
}

void fill(std::vector<float>* values, unsigned int* state, float scale) {
  for (float& value : *values)
    value = next_value(state, scale);
}

void make_host_weights(HostWeights* host,
                       cuda_transformer::TinyModelConfig config) {
  unsigned int state = 0x13579BDFU;
  const std::size_t hidden_square = config.hidden * config.hidden;
  const std::size_t hidden_intermediate = config.hidden * config.intermediate;
  const std::size_t intermediate_hidden = config.intermediate * config.hidden;
  host->embeddings.resize(config.vocabulary_size * config.hidden);
  host->final_norm.resize(config.hidden);
  host->lm_head.resize(config.hidden * config.vocabulary_size);
  fill(&host->embeddings, &state, 0.25F);
  fill(&host->lm_head, &state, 0.05F);
  for (float& value : host->final_norm)
    value = 1.0F + next_value(&state, 0.15F);

  host->layers.resize(config.layers);
  host->layer_views.resize(config.layers);
  for (std::size_t layer = 0; layer < config.layers; ++layer) {
    HostLayerWeights& weights = host->layers[layer];
    weights.attention_norm.resize(config.hidden);
    weights.wq.resize(hidden_square);
    weights.wk.resize(hidden_square);
    weights.wv.resize(hidden_square);
    weights.wo.resize(hidden_square);
    weights.mlp_norm.resize(config.hidden);
    weights.w_gate.resize(hidden_intermediate);
    weights.w_up.resize(hidden_intermediate);
    weights.w_down.resize(intermediate_hidden);
    for (float& value : weights.attention_norm)
      value = 1.0F + next_value(&state, 0.15F);
    for (float& value : weights.mlp_norm)
      value = 1.0F + next_value(&state, 0.15F);
    fill(&weights.wq, &state, 0.05F);
    fill(&weights.wk, &state, 0.05F);
    fill(&weights.wv, &state, 0.05F);
    fill(&weights.wo, &state, 0.05F);
    fill(&weights.w_gate, &state, 0.05F);
    fill(&weights.w_up, &state, 0.05F);
    fill(&weights.w_down, &state, 0.05F);
    host->layer_views[layer].decoder.attention = {
        weights.attention_norm.data(), weights.wq.data(), weights.wk.data(),
        weights.wv.data(), weights.wo.data()};
    host->layer_views[layer].decoder.mlp = {
        weights.mlp_norm.data(), weights.w_gate.data(), weights.w_up.data(),
        weights.w_down.data()};
  }
  host->view = {host->embeddings.data(), host->layer_views.data(), config.layers,
                host->final_norm.data(), host->lm_head.data()};
}

bool allocate_and_copy(float** device, const std::vector<float>& host) {
  return CTR_CUDA_CHECK(cudaMalloc(device, host.size() * sizeof(float))) &&
         CTR_CUDA_CHECK(cudaMemcpy(*device, host.data(),
                                   host.size() * sizeof(float),
                                   cudaMemcpyHostToDevice));
}

bool make_device_weights(DeviceWeights* device, const HostWeights& host,
                         cuda_transformer::TinyModelConfig config) {
  if (!allocate_and_copy(&device->embeddings, host.embeddings) ||
      !allocate_and_copy(&device->final_norm, host.final_norm) ||
      !allocate_and_copy(&device->lm_head, host.lm_head))
    return false;
  device->layers.resize(config.layers);
  device->layer_views.resize(config.layers);
  for (std::size_t layer = 0; layer < config.layers; ++layer) {
    const HostLayerWeights& source = host.layers[layer];
    DeviceLayerWeights& target = device->layers[layer];
    if (!allocate_and_copy(&target.attention_norm, source.attention_norm) ||
        !allocate_and_copy(&target.wq, source.wq) ||
        !allocate_and_copy(&target.wk, source.wk) ||
        !allocate_and_copy(&target.wv, source.wv) ||
        !allocate_and_copy(&target.wo, source.wo) ||
        !allocate_and_copy(&target.mlp_norm, source.mlp_norm) ||
        !allocate_and_copy(&target.w_gate, source.w_gate) ||
        !allocate_and_copy(&target.w_up, source.w_up) ||
        !allocate_and_copy(&target.w_down, source.w_down))
      return false;
    device->layer_views[layer].decoder.attention = {
        target.attention_norm, target.wq, target.wk, target.wv, target.wo};
    device->layer_views[layer].decoder.mlp = {
        target.mlp_norm, target.w_gate, target.w_up, target.w_down};
  }
  device->view = {device->embeddings, device->layer_views.data(), config.layers,
                  device->final_norm, device->lm_head};
  return true;
}

bool free_device_weights(DeviceWeights* device) {
  bool ok = true;
  for (DeviceLayerWeights& layer : device->layers) {
    for (float* pointer : {layer.attention_norm, layer.wq, layer.wk, layer.wv,
                           layer.wo, layer.mlp_norm, layer.w_gate, layer.w_up,
                           layer.w_down}) {
      if (pointer != nullptr && !CTR_CUDA_CHECK(cudaFree(pointer)))
        ok = false;
    }
  }
  for (float* pointer : {device->embeddings, device->final_norm, device->lm_head})
    if (pointer != nullptr && !CTR_CUDA_CHECK(cudaFree(pointer)))
      ok = false;
  *device = {};
  return ok;
}

bool check_values(const char* name, const std::vector<float>& gpu,
                  const std::vector<float>& cpu,
                  cuda_transformer::TinyModelConfig config) {
  if (gpu.size() != config.sequence * config.vocabulary_size ||
      cpu.size() != config.sequence * config.vocabulary_size ||
      gpu.size() != cpu.size()) {
    std::fprintf(stderr, "%s logits shape mismatch: gpu=%zu cpu=%zu expected=%zu\n",
                 name, gpu.size(), cpu.size(),
                 config.sequence * config.vocabulary_size);
    return false;
  }
  for (std::size_t index = 0; index < gpu.size(); ++index) {
    const float tolerance =
        kAbsoluteTolerance + kRelativeTolerance * fmaxf(1.0F, std::fabs(cpu[index]));
    if (!std::isfinite(gpu[index]) || !std::isfinite(cpu[index]) ||
        std::fabs(gpu[index] - cpu[index]) > tolerance) {
      std::fprintf(stderr,
                   "%s mismatch: layers=%zu sequence=%zu hidden=%zu vocab=%zu "
                   "index=%zu gpu=%.8g cpu=%.8g error=%.8g tolerance=%.8g\n",
                   name, config.layers, config.sequence, config.hidden,
                   config.vocabulary_size, index, gpu[index], cpu[index],
                   std::fabs(gpu[index] - cpu[index]), tolerance);
      return false;
    }
  }
  return true;
}

bool check_identical(const std::vector<float>& left,
                     const std::vector<float>& right, const char* name) {
  if (left.size() != right.size())
    return false;
  for (std::size_t index = 0; index < left.size(); ++index)
    if (left[index] != right[index]) {
      std::fprintf(stderr, "%s is not deterministic at index %zu: %.8g %.8g\n",
                   name, index, left[index], right[index]);
      return false;
    }
  return true;
}

bool embedding_test() {
  constexpr std::size_t kVocabulary = 4;
  constexpr std::size_t kHidden = 3;
  constexpr std::size_t kSequence = 3;
  const std::vector<float> embeddings{0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F,
                                      6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F};
  const std::vector<int> token_ids{2, 0, 2};
  const std::vector<float> expected{6.0F, 7.0F, 8.0F, 0.0F, 1.0F,
                                    2.0F, 6.0F, 7.0F, 8.0F};
  std::vector<float> cpu(expected.size()), gpu(expected.size());
  cuda_transformer::embedding_lookup_cpu(token_ids.data(), embeddings.data(),
                                         cpu.data(), kSequence, kVocabulary,
                                         kHidden);
  if (cpu != expected) {
    std::fputs("embedding CPU hand-check failed\n", stderr);
    return false;
  }

  int* device_token_ids = nullptr;
  float *device_embeddings = nullptr, *device_output = nullptr;
  auto cleanup = [&] {
    bool ok = true;
    if (device_token_ids != nullptr && !CTR_CUDA_CHECK(cudaFree(device_token_ids)))
      ok = false;
    if (device_embeddings != nullptr && !CTR_CUDA_CHECK(cudaFree(device_embeddings)))
      ok = false;
    if (device_output != nullptr && !CTR_CUDA_CHECK(cudaFree(device_output)))
      ok = false;
    return ok;
  };
  if (!CTR_CUDA_CHECK(cudaMalloc(&device_token_ids,
                                 token_ids.size() * sizeof(int))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&device_embeddings,
                                 embeddings.size() * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&device_output, expected.size() * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMemcpy(device_token_ids, token_ids.data(),
                                 token_ids.size() * sizeof(int),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(device_embeddings, embeddings.data(),
                                 embeddings.size() * sizeof(float),
                                 cudaMemcpyHostToDevice))) {
    cleanup();
    return false;
  }
  const bool ok =
      CTR_CUDA_CHECK(cuda_transformer::embedding_lookup_cuda(
          device_token_ids, device_embeddings, device_output, kSequence,
          kVocabulary, kHidden)) &&
      CTR_CUDA_CHECK(cudaDeviceSynchronize()) &&
      CTR_CUDA_CHECK(cudaMemcpy(gpu.data(), device_output,
                                gpu.size() * sizeof(float),
                                cudaMemcpyDeviceToHost)) &&
      gpu == expected;
  if (!ok)
    std::fputs("embedding CUDA hand-check failed\n", stderr);
  return cleanup() && ok;
}

bool model_test(cublasHandle_t handle, std::size_t layers,
                const std::vector<int>& token_ids, const char* name) {
  const cuda_transformer::TinyModelConfig config{128, 5, 64, layers, 4, 16,
                                                  128, 1e-5F};
  HostWeights host;
  make_host_weights(&host, config);
  std::vector<float> cpu_logits(config.sequence * config.vocabulary_size);
  if (!cuda_transformer::tiny_model_forward_cpu(token_ids.data(), host.view,
                                                config, cpu_logits.data())) {
    std::fprintf(stderr, "%s CPU reference failed\n", name);
    return false;
  }

  DeviceWeights device_weights;
  cuda_transformer::TinyModelWorkspace workspace;
  int* device_token_ids = nullptr;
  float* device_logits = nullptr;
  auto cleanup = [&] {
    bool ok = true;
    if (device_token_ids != nullptr && !CTR_CUDA_CHECK(cudaFree(device_token_ids)))
      ok = false;
    if (device_logits != nullptr && !CTR_CUDA_CHECK(cudaFree(device_logits)))
      ok = false;
    if (!CTR_CUDA_CHECK(cuda_transformer::tiny_model_workspace_destroy(&workspace)))
      ok = false;
    if (!free_device_weights(&device_weights))
      ok = false;
    return ok;
  };
  if (!make_device_weights(&device_weights, host, config) ||
      !CTR_CUDA_CHECK(cuda_transformer::tiny_model_workspace_create(&workspace,
                                                                      config)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&device_token_ids,
                                 config.sequence * sizeof(int))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&device_logits,
                                 cpu_logits.size() * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMemcpy(device_token_ids, token_ids.data(),
                                 config.sequence * sizeof(int),
                                 cudaMemcpyHostToDevice))) {
    cleanup();
    return false;
  }

  const std::vector<int> invalid_negative{-1, 3, 7, 3, 11};
  const std::vector<int> invalid_large{128, 3, 7, 3, 11};
  bool ok =
      cuda_transformer::tiny_model_forward_host_tokens_cuda(
          handle, invalid_negative.data(), device_weights.view, &workspace,
          device_logits) == CUBLAS_STATUS_INVALID_VALUE &&
      cuda_transformer::tiny_model_forward_host_tokens_cuda(
          handle, invalid_large.data(), device_weights.view, &workspace,
          device_logits) == CUBLAS_STATUS_INVALID_VALUE &&
      cuda_transformer::tiny_model_forward_cuda(
          handle, nullptr, device_weights.view, &workspace, device_logits) ==
          CUBLAS_STATUS_INVALID_VALUE;

  std::vector<float> gpu_logits(cpu_logits.size()), repeated_logits(cpu_logits.size()),
      wrapper_logits(cpu_logits.size());
  ok = ok &&
       CTR_CUBLAS_CHECK(cuda_transformer::tiny_model_forward_cuda(
           handle, device_token_ids, device_weights.view, &workspace,
           device_logits)) &&
       CTR_CUDA_CHECK(cudaDeviceSynchronize()) &&
       CTR_CUDA_CHECK(cudaMemcpy(gpu_logits.data(), device_logits,
                                 gpu_logits.size() * sizeof(float),
                                 cudaMemcpyDeviceToHost)) &&
       check_values(name, gpu_logits, cpu_logits, config);
  ok = ok &&
       CTR_CUBLAS_CHECK(cuda_transformer::tiny_model_forward_cuda(
           handle, device_token_ids, device_weights.view, &workspace,
           device_logits)) &&
       CTR_CUDA_CHECK(cudaDeviceSynchronize()) &&
       CTR_CUDA_CHECK(cudaMemcpy(repeated_logits.data(), device_logits,
                                 repeated_logits.size() * sizeof(float),
                                 cudaMemcpyDeviceToHost)) &&
       check_identical(gpu_logits, repeated_logits, name);
  ok = ok &&
       CTR_CUBLAS_CHECK(cuda_transformer::tiny_model_forward_host_tokens_cuda(
           handle, token_ids.data(), device_weights.view, &workspace,
           device_logits)) &&
       CTR_CUDA_CHECK(cudaDeviceSynchronize()) &&
       CTR_CUDA_CHECK(cudaMemcpy(wrapper_logits.data(), device_logits,
                                 wrapper_logits.size() * sizeof(float),
                                 cudaMemcpyDeviceToHost)) &&
       check_values("host token wrapper", wrapper_logits, cpu_logits, config);
  return cleanup() && ok;
}

bool check_row_values(const char* name, const std::vector<float>& gpu,
                      const std::vector<float>& reference,
                      cuda_transformer::TinyModelConfig config) {
  if (gpu.size() != config.vocabulary_size ||
      reference.size() != config.vocabulary_size) {
    std::fprintf(stderr, "%s logits row shape mismatch: gpu=%zu reference=%zu\n",
                 name, gpu.size(), reference.size());
    return false;
  }
  for (std::size_t index = 0; index < gpu.size(); ++index) {
    const float tolerance =
        1e-3F + 2e-4F * fmaxf(1.0F, std::fabs(reference[index]));
    if (!std::isfinite(gpu[index]) || !std::isfinite(reference[index]) ||
        std::fabs(gpu[index] - reference[index]) > tolerance) {
      std::fprintf(stderr,
                   "%s mismatch: layers=%zu index=%zu gpu=%.8g reference=%.8g "
                   "error=%.8g tolerance=%.8g\n",
                   name, config.layers, index, gpu[index], reference[index],
                   std::fabs(gpu[index] - reference[index]), tolerance);
      return false;
    }
  }
  return true;
}

bool check_cache_history(const std::vector<float>& before_keys,
                         const std::vector<float>& before_values,
                         const std::vector<float>& after_keys,
                         const std::vector<float>& after_values,
                         cuda_transformer::TinyModelConfig config,
                         std::size_t max_sequence_length,
                         std::size_t history_length, std::size_t layer) {
  for (std::size_t head = 0; head < config.heads; ++head)
    for (std::size_t position = 0; position < history_length; ++position)
      for (std::size_t dimension = 0; dimension < config.head_dim; ++dimension) {
        const std::size_t index =
            (head * max_sequence_length + position) * config.head_dim + dimension;
        if (before_keys[index] != after_keys[index] ||
            before_values[index] != after_values[index]) {
          std::fprintf(stderr,
                       "cache history changed: layer=%zu head=%zu position=%zu "
                       "dimension=%zu\n",
                       layer, head, position, dimension);
          return false;
        }
      }
  return true;
}

bool cache_lengths_are(const cuda_transformer::TinyModelIncrementalWorkspace& state,
                       std::size_t expected) {
  for (std::size_t layer = 0; layer < state.model_config.layers; ++layer)
    if (state.layer_caches[layer].current_length != expected) {
      std::fprintf(stderr, "cache length mismatch: layer=%zu got=%zu expected=%zu\n",
                   layer, state.layer_caches[layer].current_length, expected);
      return false;
    }
  return true;
}

bool incremental_model_test(cublasHandle_t handle, std::size_t layers,
                            const std::vector<int>& token_ids,
                            const char* name) {
  using namespace cuda_transformer;
  constexpr std::size_t kPrefixLength = 4;
  constexpr std::size_t kMaxSequenceLength = 8;
  const TinyModelConfig config{128, 5, 64, layers, 4, 16, 128, 1e-5F};
  HostWeights host;
  make_host_weights(&host, config);
  std::vector<float> cpu_logits(config.sequence * config.vocabulary_size);
  if (!tiny_model_forward_cpu(token_ids.data(), host.view, config,
                              cpu_logits.data())) {
    std::fprintf(stderr, "%s CPU model reference failed\n", name);
    return false;
  }

  DeviceWeights device_weights;
  TinyModelWorkspace full_workspace;
  TinyModelIncrementalWorkspace incremental_workspace;
  int *device_token_ids = nullptr, *device_capacity_tokens = nullptr;
  float *device_full_logits = nullptr, *device_incremental_logits = nullptr;
  auto cleanup = [&] {
    bool ok = true;
    for (int* pointer : {device_token_ids, device_capacity_tokens})
      if (pointer != nullptr && !CTR_CUDA_CHECK(cudaFree(pointer)))
        ok = false;
    for (float* pointer : {device_full_logits, device_incremental_logits})
      if (pointer != nullptr && !CTR_CUDA_CHECK(cudaFree(pointer)))
        ok = false;
    if (!CTR_CUDA_CHECK(tiny_model_workspace_destroy(&full_workspace)))
      ok = false;
    if (!CTR_CUDA_CHECK(
            tiny_model_incremental_workspace_destroy(&incremental_workspace)))
      ok = false;
    if (!free_device_weights(&device_weights))
      ok = false;
    return ok;
  };
  const std::vector<int> capacity_tokens{7, 19, 4, 82, 11, 6, 6, 3};
  if (!make_device_weights(&device_weights, host, config) ||
      !CTR_CUDA_CHECK(tiny_model_workspace_create(&full_workspace, config)) ||
      !CTR_CUDA_CHECK(tiny_model_incremental_workspace_create(
          &incremental_workspace, config, kMaxSequenceLength)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&device_token_ids,
                                 token_ids.size() * sizeof(int))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&device_capacity_tokens,
                                 capacity_tokens.size() * sizeof(int))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&device_full_logits,
                                 cpu_logits.size() * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&device_incremental_logits,
                                 config.vocabulary_size * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMemcpy(device_token_ids, token_ids.data(),
                                 token_ids.size() * sizeof(int),
                                 cudaMemcpyHostToDevice)) ||
      !CTR_CUDA_CHECK(cudaMemcpy(device_capacity_tokens, capacity_tokens.data(),
                                 capacity_tokens.size() * sizeof(int),
                                 cudaMemcpyHostToDevice))) {
    cleanup();
    return false;
  }

  std::vector<float> full_gpu(cpu_logits.size());
  bool ok = CTR_CUBLAS_CHECK(tiny_model_forward_cuda(
      handle, device_token_ids, device_weights.view, &full_workspace,
      device_full_logits)) &&
            CTR_CUDA_CHECK(cudaDeviceSynchronize()) &&
            CTR_CUDA_CHECK(cudaMemcpy(full_gpu.data(), device_full_logits,
                                      full_gpu.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost)) &&
            check_values("full model before incremental equivalence", full_gpu,
                         cpu_logits, config);
  std::vector<float> full_final(config.vocabulary_size);
  for (std::size_t index = 0; index < config.vocabulary_size; ++index)
    full_final[index] = full_gpu[kPrefixLength * config.vocabulary_size + index];

  ok = ok && CTR_CUBLAS_CHECK(tiny_model_incremental_prefill_cuda(
                 handle, device_token_ids, kPrefixLength, device_weights.view,
                 &incremental_workspace, device_incremental_logits)) &&
       CTR_CUDA_CHECK(cudaDeviceSynchronize()) &&
       cache_lengths_are(incremental_workspace, kPrefixLength);

  TinyModelWeights wrong_layer_count = device_weights.view;
  --wrong_layer_count.layer_count;
  ok = ok && tiny_model_incremental_decode_cuda(
                 handle, device_token_ids + kPrefixLength, wrong_layer_count,
                 &incremental_workspace, device_incremental_logits) ==
                 CUBLAS_STATUS_INVALID_VALUE &&
       cache_lengths_are(incremental_workspace, kPrefixLength);

  const std::size_t cache_count =
      config.heads * kMaxSequenceLength * config.head_dim;
  std::vector<std::vector<float>> before_keys(
      layers, std::vector<float>(cache_count));
  std::vector<std::vector<float>> before_values(
      layers, std::vector<float>(cache_count));
  for (std::size_t layer = 0; layer < layers && ok; ++layer) {
    ok = CTR_CUDA_CHECK(cudaMemcpy(before_keys[layer].data(),
                                   incremental_workspace.layer_caches[layer].keys,
                                   cache_count * sizeof(float),
                                   cudaMemcpyDeviceToHost)) &&
         CTR_CUDA_CHECK(cudaMemcpy(before_values[layer].data(),
                                   incremental_workspace.layer_caches[layer].values,
                                   cache_count * sizeof(float),
                                   cudaMemcpyDeviceToHost));
  }

  const float* first_keys = incremental_workspace.layer_caches[0].keys;
  const float* first_values = incremental_workspace.layer_caches[0].values;
  for (std::size_t layer = 1; layer < layers && ok; ++layer) {
    if (incremental_workspace.layer_caches[layer].keys == first_keys ||
        incremental_workspace.layer_caches[layer].values == first_values) {
      std::fprintf(stderr, "%s shares cache storage between layers\n", name);
      ok = false;
    }
  }

  std::vector<float> incremental_logits(config.vocabulary_size);
  ok = ok && CTR_CUBLAS_CHECK(tiny_model_incremental_decode_cuda(
                 handle, device_token_ids + kPrefixLength, device_weights.view,
                 &incremental_workspace, device_incremental_logits)) &&
       CTR_CUDA_CHECK(cudaDeviceSynchronize()) &&
       CTR_CUDA_CHECK(cudaMemcpy(incremental_logits.data(),
                                 device_incremental_logits,
                                 incremental_logits.size() * sizeof(float),
                                 cudaMemcpyDeviceToHost)) &&
       check_row_values(name, incremental_logits, full_final, config) &&
       cache_lengths_are(incremental_workspace, kPrefixLength + 1) &&
       tiny_model_incremental_decode_host_token_cuda(
           handle, -1, device_weights.view, &incremental_workspace,
           device_incremental_logits) == CUBLAS_STATUS_INVALID_VALUE &&
       cache_lengths_are(incremental_workspace, kPrefixLength + 1);

  for (std::size_t layer = 0; layer < layers && ok; ++layer) {
    std::vector<float> after_keys(cache_count), after_values(cache_count);
    ok = CTR_CUDA_CHECK(cudaMemcpy(after_keys.data(),
                                   incremental_workspace.layer_caches[layer].keys,
                                   cache_count * sizeof(float),
                                   cudaMemcpyDeviceToHost)) &&
         CTR_CUDA_CHECK(cudaMemcpy(after_values.data(),
                                   incremental_workspace.layer_caches[layer].values,
                                   cache_count * sizeof(float),
                                   cudaMemcpyDeviceToHost)) &&
         check_cache_history(before_keys[layer], before_values[layer], after_keys,
                             after_values, config, kMaxSequenceLength,
                             kPrefixLength, layer);
  }

  std::vector<const float*> saved_keys(layers), saved_values(layers);
  for (std::size_t layer = 0; layer < layers; ++layer) {
    saved_keys[layer] = incremental_workspace.layer_caches[layer].keys;
    saved_values[layer] = incremental_workspace.layer_caches[layer].values;
  }
  const float* saved_activation_a = incremental_workspace.activation_a;
  ok = ok &&
       CTR_CUBLAS_CHECK(tiny_model_incremental_prefill_cuda(
           handle, device_capacity_tokens + kPrefixLength, 3, device_weights.view,
           &incremental_workspace, device_incremental_logits)) &&
       CTR_CUDA_CHECK(cudaDeviceSynchronize()) &&
       cache_lengths_are(incremental_workspace, kMaxSequenceLength) &&
       tiny_model_incremental_decode_cuda(
           handle, device_capacity_tokens, device_weights.view,
           &incremental_workspace, device_incremental_logits) ==
           CUBLAS_STATUS_INVALID_VALUE &&
       cache_lengths_are(incremental_workspace, kMaxSequenceLength);

  ok = ok && CTR_CUDA_CHECK(
                 tiny_model_incremental_workspace_reset(&incremental_workspace)) &&
       cache_lengths_are(incremental_workspace, 0) &&
       incremental_workspace.activation_a == saved_activation_a;
  for (std::size_t layer = 0; layer < layers && ok; ++layer)
    ok = incremental_workspace.layer_caches[layer].keys == saved_keys[layer] &&
         incremental_workspace.layer_caches[layer].values == saved_values[layer];

  if (layers > 1) {
    // Deliberately corrupt public logical metadata. Preflight must reject
    // before launching layer 0, so neither cache can be advanced.
    incremental_workspace.layer_caches[1].current_length = 1;
    ok = ok && tiny_model_incremental_decode_cuda(
                   handle, device_token_ids, device_weights.view,
                   &incremental_workspace, device_incremental_logits) ==
                   CUBLAS_STATUS_INVALID_VALUE &&
         incremental_workspace.layer_caches[0].current_length == 0 &&
         incremental_workspace.layer_caches[1].current_length == 1;
    incremental_workspace.layer_caches[1].current_length = 0;
  }

  std::vector<float> repeated_logits(config.vocabulary_size);
  ok = ok && CTR_CUBLAS_CHECK(tiny_model_incremental_prefill_cuda(
                 handle, device_token_ids, kPrefixLength, device_weights.view,
                 &incremental_workspace, device_incremental_logits)) &&
       CTR_CUBLAS_CHECK(tiny_model_incremental_decode_cuda(
           handle, device_token_ids + kPrefixLength, device_weights.view,
           &incremental_workspace, device_incremental_logits)) &&
       CTR_CUDA_CHECK(cudaDeviceSynchronize()) &&
       CTR_CUDA_CHECK(cudaMemcpy(repeated_logits.data(), device_incremental_logits,
                                 repeated_logits.size() * sizeof(float),
                                 cudaMemcpyDeviceToHost)) &&
       check_identical(incremental_logits, repeated_logits,
                       "reset-prefill-decode determinism");
  return cleanup() && ok;
}

bool greedy_argmax_test() {
  int selected = -1;
  const std::vector<float> tied{2.0F, -1.0F, 2.0F, 1.0F};
  const std::vector<float> unique{-3.0F, 4.0F, 1.0F};
  return cuda_transformer::tiny_model_greedy_argmax_host(
             tied.data(), tied.size(), &selected) &&
         selected == 0 &&
         cuda_transformer::tiny_model_greedy_argmax_host(
             unique.data(), unique.size(), &selected) &&
         selected == 1 &&
         !cuda_transformer::tiny_model_greedy_argmax_host(nullptr, tied.size(),
                                                           &selected) &&
         !cuda_transformer::tiny_model_greedy_argmax_host(
             tied.data(), 0, &selected);
}

bool generation_test(cublasHandle_t handle, std::size_t layers,
                     const std::vector<int>& prompt,
                     std::size_t max_new_tokens,
                     std::size_t max_sequence_length, const char* name) {
  using namespace cuda_transformer;
  const TinyModelConfig config{128, 5, 64, layers, 4, 16, 128, 1e-5F};
  HostWeights host;
  make_host_weights(&host, config);

  DeviceWeights device_weights;
  TinyModelIncrementalWorkspace generated_workspace;
  TinyModelIncrementalWorkspace manual_workspace;
  float *device_generated_logits = nullptr, *device_manual_logits = nullptr;
  auto cleanup = [&] {
    bool ok = true;
    for (float* pointer : {device_generated_logits, device_manual_logits})
      if (pointer != nullptr && !CTR_CUDA_CHECK(cudaFree(pointer)))
        ok = false;
    if (!CTR_CUDA_CHECK(
            tiny_model_incremental_workspace_destroy(&generated_workspace)))
      ok = false;
    if (!CTR_CUDA_CHECK(tiny_model_incremental_workspace_destroy(&manual_workspace)))
      ok = false;
    if (!free_device_weights(&device_weights))
      ok = false;
    return ok;
  };
  if (!make_device_weights(&device_weights, host, config) ||
      !CTR_CUDA_CHECK(tiny_model_incremental_workspace_create(
          &generated_workspace, config, max_sequence_length)) ||
      !CTR_CUDA_CHECK(tiny_model_incremental_workspace_create(
          &manual_workspace, config, max_sequence_length)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&device_generated_logits,
                                 config.vocabulary_size * sizeof(float))) ||
      !CTR_CUDA_CHECK(cudaMalloc(&device_manual_logits,
                                 config.vocabulary_size * sizeof(float)))) {
    cleanup();
    return false;
  }

  const std::size_t expected_cache_length =
      prompt.size() + max_new_tokens - 1;
  std::vector<int> manual_generated(max_new_tokens);
  std::vector<float> host_logits(config.vocabulary_size);
  bool ok = CTR_CUDA_CHECK(tiny_model_incremental_workspace_reset(
      &manual_workspace));
  for (std::size_t position = 0; position < prompt.size() && ok; ++position)
    ok = CTR_CUBLAS_CHECK(tiny_model_incremental_decode_host_token_cuda(
        handle, prompt[position], device_weights.view, &manual_workspace,
        device_manual_logits));

  if (ok) {
    ok = CTR_CUDA_CHECK(cudaMemcpy(host_logits.data(), device_manual_logits,
                                   host_logits.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost));
  }
  for (std::size_t generated = 0; generated < max_new_tokens && ok;
       ++generated) {
    ok = tiny_model_greedy_argmax_host(host_logits.data(), host_logits.size(),
                                       &manual_generated[generated]);
    if (generated + 1 == max_new_tokens)
      break;
    ok = ok && CTR_CUBLAS_CHECK(tiny_model_incremental_decode_host_token_cuda(
                   handle, manual_generated[generated], device_weights.view,
                   &manual_workspace, device_manual_logits)) &&
         CTR_CUDA_CHECK(cudaMemcpy(host_logits.data(), device_manual_logits,
                                   host_logits.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost));
  }
  ok = ok && cache_lengths_are(manual_workspace, expected_cache_length);

  std::vector<int> generated(max_new_tokens), repeated(max_new_tokens);
  ok = ok && CTR_CUBLAS_CHECK(tiny_model_generate_greedy_cuda(
                 handle, prompt.data(), prompt.size(), device_weights.view,
                 &generated_workspace, max_new_tokens, generated.data(),
                 device_generated_logits)) &&
       generated == manual_generated &&
       cache_lengths_are(generated_workspace, expected_cache_length);
  for (std::size_t index = 0; index < generated.size() && ok; ++index)
    if (generated[index] < 0 ||
        static_cast<std::size_t>(generated[index]) >= config.vocabulary_size) {
      std::fprintf(stderr, "%s generated invalid token at %zu: %d\n", name,
                   index, generated[index]);
      ok = false;
    }

  // A repeated request resets logically and must reproduce exactly the same
  // host-orchestrated greedy sequence.
  ok = ok && CTR_CUBLAS_CHECK(tiny_model_generate_greedy_cuda(
                 handle, prompt.data(), prompt.size(), device_weights.view,
                 &generated_workspace, max_new_tokens, repeated.data(),
                 device_generated_logits)) &&
       generated == repeated &&
       cache_lengths_are(generated_workspace, expected_cache_length);

  // Zero generation is deliberately a no-op, even if a prior request filled
  // the cache. Empty and invalid prompts are rejected before reset/mutation.
  ok = ok && tiny_model_generate_greedy_cuda(
                 handle, prompt.data(), prompt.size(), device_weights.view,
                 &generated_workspace, 0, nullptr, nullptr) ==
                 CUBLAS_STATUS_SUCCESS &&
       cache_lengths_are(generated_workspace, expected_cache_length);
  const std::vector<int> invalid_prompt{-1};
  ok = ok && tiny_model_generate_greedy_cuda(
                 handle, invalid_prompt.data(), invalid_prompt.size(),
                 device_weights.view, &generated_workspace, 1, generated.data(),
                 device_generated_logits) == CUBLAS_STATUS_INVALID_VALUE &&
       tiny_model_generate_greedy_cuda(
           handle, nullptr, 0, device_weights.view, &generated_workspace, 1,
           generated.data(), device_generated_logits) == CUBLAS_STATUS_INVALID_VALUE &&
       cache_lengths_are(generated_workspace, expected_cache_length);

  // This requires one more decoded token than the exact-boundary request.
  // Generation preflights before reset, so the existing logical cache history
  // must survive the rejection unchanged.
  ok = ok && tiny_model_generate_greedy_cuda(
                 handle, prompt.data(), prompt.size(), device_weights.view,
                 &generated_workspace, max_new_tokens + 1, generated.data(),
                 device_generated_logits) == CUBLAS_STATUS_INVALID_VALUE &&
       cache_lengths_are(generated_workspace, expected_cache_length);
  return cleanup() && ok;
}

bool invalid_config_test() {
  cuda_transformer::TinyModelWorkspace workspace;
  const cuda_transformer::TinyModelConfig invalid{128, 5, 64, 0, 4, 16,
                                                   128, 1e-5F};
  return cuda_transformer::tiny_model_workspace_create(&workspace, invalid) ==
         cudaErrorInvalidValue;
}

}  // namespace

int main() {
  if (!invalid_config_test() || !greedy_argmax_test() || !embedding_test())
    return EXIT_FAILURE;
  cublasHandle_t handle = nullptr;
  if (!CTR_CUBLAS_CHECK(cublasCreate(&handle)))
    return EXIT_FAILURE;
  const bool ok =
      model_test(handle, 1, {3, 3, 7, 3, 11}, "one-layer tiny model") &&
      model_test(handle, 2, {0, 5, 127, 1, 42}, "two-layer tiny model") &&
      incremental_model_test(handle, 1, {7, 7, 4, 7, 11},
                             "one-layer repeated cached model") &&
      incremental_model_test(handle, 2, {7, 19, 4, 82, 11},
                             "two-layer cached model") &&
      generation_test(handle, 1, {7}, 1, 1,
                      "one-layer single-token greedy generation") &&
      generation_test(handle, 2, {7, 19, 4}, 3, 5,
                      "two-layer multi-token greedy generation");
  if (!CTR_CUBLAS_CHECK(cublasDestroy(handle)))
    return EXIT_FAILURE;
  if (!ok)
    return EXIT_FAILURE;
  std::puts("Tiny model tests passed.");
}
