#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/llama2_checkpoint.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct SyntheticConfig {
  int dim = 4;
  int hidden_dim = 8;
  int layers = 2;
  int heads = 2;
  int kv_heads = 2;
  int vocabulary = 7;
  int sequence = 3;
};

void write_values(std::ofstream* file, std::size_t count, float* value) {
  for (std::size_t index = 0; index < count; ++index) {
    file->write(reinterpret_cast<const char*>(value), sizeof(*value));
    *value += 1.0F;
  }
}

bool write_checkpoint(const std::filesystem::path& path, SyntheticConfig config,
                      bool shared_classifier, bool add_extra_byte = false) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file)
    return false;
  const std::array<int, 7> header{
      config.dim, config.hidden_dim, config.layers, config.heads, config.kv_heads,
      shared_classifier ? config.vocabulary : -config.vocabulary, config.sequence};
  file.write(reinterpret_cast<const char*>(header.data()), sizeof(header));
  if (!file)
    return false;

  const std::size_t dim = static_cast<std::size_t>(config.dim);
  const std::size_t hidden = static_cast<std::size_t>(config.hidden_dim);
  const std::size_t layers = static_cast<std::size_t>(config.layers);
  const std::size_t vocabulary = static_cast<std::size_t>(config.vocabulary);
  const std::size_t head_dim = dim / static_cast<std::size_t>(config.heads);
  float value = 1.0F;
  for (const std::size_t count : {
           vocabulary * dim, layers * dim, layers * dim * dim,
           layers * dim * dim, layers * dim * dim, layers * dim * dim,
           layers * dim, layers * hidden * dim, layers * dim * hidden,
           layers * hidden * dim, dim,
           static_cast<std::size_t>(config.sequence) * head_dim / 2,
           static_cast<std::size_t>(config.sequence) * head_dim / 2,
           shared_classifier ? std::size_t{0} : vocabulary * dim}) {
    write_values(&file, count, &value);
  }
  if (add_extra_byte)
    file.put('\0');
  return static_cast<bool>(file);
}

bool check_transposes_and_views(const cuda_transformer::Llama2CheckpointModel& model,
                                bool shared_classifier) {
  const cuda_transformer::Llama2CheckpointConfig& config = model.checkpoint_config;
  if (config.dim != 4 || config.hidden_dim != 8 || config.n_layers != 2 ||
      config.n_heads != 2 || config.n_kv_heads != 2 ||
      config.vocabulary_size != 7 || config.max_sequence_length != 3 ||
      config.shared_classifier != shared_classifier || model.host_weights.layers == nullptr ||
      model.device_weights.layers == nullptr)
    return false;

  // Synthetic source values start at one. Embeddings have 28 floats and
  // attention norms have eight, so Wq starts at 37 in checkpoint row-major
  // [output, input] order. Runtime Wq is [input, output].
  const float expected_wq_input1_output2 = 37.0F + 2.0F * 4.0F + 1.0F;
  if (model.host_weights.layers[0].decoder.attention.wq[1 * 4 + 2] !=
      expected_wq_input1_output2)
    return false;
  const auto& attention = model.host_weights.layers[0].decoder.attention;
  if (attention.wk[2 * 4 + 1] != 69.0F + 1.0F * 4.0F + 2.0F ||
      attention.wv[3 * 4 + 0] != 101.0F + 0.0F * 4.0F + 3.0F ||
      attention.wo[0 * 4 + 3] != 133.0F + 3.0F * 4.0F + 0.0F)
    return false;

  // W1 begins after all preceding legacy groups. It is [hidden_dim, dim] in
  // the checkpoint and becomes runtime gate [dim, hidden_dim].
  const float w1_base = 1.0F + 28.0F + 8.0F + 4.0F * 32.0F + 8.0F;
  const float expected_gate_input3_output5 = w1_base + 5.0F * 4.0F + 3.0F;
  if (model.host_weights.layers[0].decoder.mlp.w_gate[3 * 8 + 5] !=
      expected_gate_input3_output5)
    return false;
  const auto& mlp = model.host_weights.layers[0].decoder.mlp;
  if (mlp.w_down[6 * 4 + 2] != 237.0F + 2.0F * 8.0F + 6.0F ||
      mlp.w_up[3 * 8 + 5] != 301.0F + 5.0F * 4.0F + 3.0F)
    return false;

  const float expected_lm_hidden2_vocab3 = shared_classifier
                                               ? 1.0F + 3.0F * 4.0F + 2.0F
                                               : 1.0F + 28.0F + 8.0F + 128.0F +
                                                     8.0F + 64.0F + 64.0F + 64.0F +
                                                     4.0F + 6.0F + 3.0F * 4.0F + 2.0F;
  if (model.host_weights.lm_head_weight[2 * 7 + 3] != expected_lm_hidden2_vocab3)
    return false;

  std::vector<float> copied_lm_head(4 * 7);
  return CTR_CUDA_CHECK(cudaMemcpy(copied_lm_head.data(),
                                   model.device_weights.lm_head_weight,
                                   copied_lm_head.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost)) &&
         copied_lm_head == std::vector<float>(
                               model.host_weights.lm_head_weight,
                               model.host_weights.lm_head_weight +
                                   copied_lm_head.size());
}

bool valid_checkpoint_test(const std::filesystem::path& path,
                           bool shared_classifier) {
  if (!write_checkpoint(path, {}, shared_classifier))
    return false;
  cuda_transformer::Llama2CheckpointModel model;
  std::string error;
  bool ok = cuda_transformer::llama2_checkpoint_load(path.c_str(), &model, &error) ==
                cuda_transformer::Llama2CheckpointStatus::kSuccess &&
            check_transposes_and_views(model, shared_classifier);
  cuda_transformer::TinyModelConfig tiny_config;
  ok = ok && cuda_transformer::llama2_checkpoint_tiny_model_config(
                 model.checkpoint_config, 2, &tiny_config) ==
                 cuda_transformer::Llama2CheckpointStatus::kSuccess &&
       tiny_config.sequence == 2 && tiny_config.hidden == 4 &&
       tiny_config.rmsnorm_epsilon == 1e-5F;
  ok = ok && CTR_CUDA_CHECK(model.reset()) && model.device_weights.layers == nullptr &&
       model.host_weights.layers == nullptr;
  std::filesystem::remove(path);
  return ok;
}

bool rejection_test(const std::filesystem::path& path, SyntheticConfig config,
                    bool shared_classifier,
                    cuda_transformer::Llama2CheckpointStatus expected,
                    bool extra_byte = false) {
  if (!write_checkpoint(path, config, shared_classifier, extra_byte))
    return false;
  cuda_transformer::Llama2CheckpointModel model;
  std::string error;
  const bool ok = cuda_transformer::llama2_checkpoint_load(path.c_str(), &model,
                                                             &error) == expected &&
                  !error.empty() && model.device_weights.layers == nullptr;
  std::filesystem::remove(path);
  return ok;
}

bool truncated_file_test(const std::filesystem::path& path) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  const int only_one_header_field = 4;
  file.write(reinterpret_cast<const char*>(&only_one_header_field),
             sizeof(only_one_header_field));
  file.close();
  cuda_transformer::Llama2CheckpointModel model;
  std::string error;
  const bool ok = cuda_transformer::llama2_checkpoint_load(path.c_str(), &model,
                                                             &error) ==
                      cuda_transformer::Llama2CheckpointStatus::kInvalidFormat &&
                  !error.empty();
  std::filesystem::remove(path);
  return ok;
}

}  // namespace

int main() {
  const std::filesystem::path base =
      std::filesystem::temp_directory_path() / "cuda_transformer_legacy_checkpoint";
  SyntheticConfig invalid_dim;
  invalid_dim.dim = 3;
  invalid_dim.heads = 2;
  SyntheticConfig grouped_query;
  grouped_query.kv_heads = 1;
  const bool ok = valid_checkpoint_test(base.string() + "_shared.bin", true) &&
                  valid_checkpoint_test(base.string() + "_unshared.bin", false) &&
                  truncated_file_test(base.string() + "_truncated.bin") &&
                  rejection_test(base.string() + "_extra.bin", {}, true,
                                 cuda_transformer::Llama2CheckpointStatus::kInvalidFormat,
                                 true) &&
                  rejection_test(base.string() + "_bad_dim.bin", invalid_dim, true,
                                 cuda_transformer::Llama2CheckpointStatus::kInvalidFormat) &&
                  rejection_test(base.string() + "_gqa.bin", grouped_query, true,
                                 cuda_transformer::Llama2CheckpointStatus::kUnsupportedArchitecture);
  if (!ok) {
    std::fputs("llama2 checkpoint tests failed\n", stderr);
    return EXIT_FAILURE;
  }
  std::puts("llama2 checkpoint tests passed.");
}
