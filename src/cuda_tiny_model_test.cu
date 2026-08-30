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
  host->view = {host->embeddings.data(), host->layer_views.data(),
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
  device->view = {device->embeddings, device->layer_views.data(),
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

bool invalid_config_test() {
  cuda_transformer::TinyModelWorkspace workspace;
  const cuda_transformer::TinyModelConfig invalid{128, 5, 64, 0, 4, 16,
                                                   128, 1e-5F};
  return cuda_transformer::tiny_model_workspace_create(&workspace, invalid) ==
         cudaErrorInvalidValue;
}

}  // namespace

int main() {
  if (!invalid_config_test() || !embedding_test())
    return EXIT_FAILURE;
  cublasHandle_t handle = nullptr;
  if (!CTR_CUBLAS_CHECK(cublasCreate(&handle)))
    return EXIT_FAILURE;
  const bool ok =
      model_test(handle, 1, {3, 3, 7, 3, 11}, "one-layer tiny model") &&
      model_test(handle, 2, {0, 5, 127, 1, 42}, "two-layer tiny model");
  if (!CTR_CUBLAS_CHECK(cublasDestroy(handle)))
    return EXIT_FAILURE;
  if (!ok)
    return EXIT_FAILURE;
  std::puts("Tiny model tests passed.");
}
