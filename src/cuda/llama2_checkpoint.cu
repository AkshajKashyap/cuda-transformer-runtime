#include "cuda_transformer/llama2_checkpoint.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace cuda_transformer {
namespace {

constexpr std::size_t kHeaderInts = 7;
constexpr std::size_t kHeaderBytes = kHeaderInts * sizeof(std::int32_t);
constexpr float kLlama2RmsNormEpsilon = 1e-5F;

struct LegacyLayout {
  std::size_t embeddings = 0;
  std::size_t attention_norms = 0;
  std::size_t q = 0;
  std::size_t k = 0;
  std::size_t v = 0;
  std::size_t o = 0;
  std::size_t mlp_norms = 0;
  std::size_t w1 = 0;
  std::size_t w2 = 0;
  std::size_t w3 = 0;
  std::size_t final_norm = 0;
  std::size_t rope_cos = 0;
  std::size_t rope_sin = 0;
  std::size_t classifier = 0;
  std::size_t total = 0;
};

bool checked_add(std::size_t left, std::size_t right, std::size_t* result) {
  if (left > std::numeric_limits<std::size_t>::max() - right)
    return false;
  *result = left + right;
  return true;
}

bool checked_multiply(std::size_t left, std::size_t right,
                      std::size_t* result) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    return false;
  *result = left * right;
  return true;
}

bool checked_product(std::initializer_list<std::size_t> values,
                     std::size_t* result) {
  *result = 1;
  for (const std::size_t value : values)
    if (!checked_multiply(*result, value, result))
      return false;
  return true;
}

void set_error(std::string* error_message, const char* message) {
  if (error_message != nullptr)
    *error_message = message;
}

bool parse_positive(std::int32_t value, std::size_t* result) {
  if (value <= 0)
    return false;
  *result = static_cast<std::size_t>(value);
  return true;
}

Llama2CheckpointStatus parse_header(const std::array<std::int32_t, kHeaderInts>& raw,
                                    Llama2CheckpointConfig* config,
                                    std::string* error_message) {
  if (!parse_positive(raw[0], &config->dim) ||
      !parse_positive(raw[1], &config->hidden_dim) ||
      !parse_positive(raw[2], &config->n_layers) ||
      !parse_positive(raw[3], &config->n_heads) ||
      !parse_positive(raw[4], &config->n_kv_heads) ||
      !parse_positive(raw[6], &config->max_sequence_length)) {
    set_error(error_message, "legacy llama2.c header has non-positive dimensions");
    return Llama2CheckpointStatus::kInvalidFormat;
  }
  if (raw[5] == 0 || raw[5] == std::numeric_limits<std::int32_t>::min()) {
    set_error(error_message, "legacy llama2.c header has invalid vocabulary size");
    return Llama2CheckpointStatus::kInvalidFormat;
  }
  config->shared_classifier = raw[5] > 0;
  config->vocabulary_size = static_cast<std::size_t>(
      config->shared_classifier ? raw[5] : -raw[5]);
  if (config->dim % config->n_heads != 0) {
    set_error(error_message, "checkpoint dim must be divisible by n_heads");
    return Llama2CheckpointStatus::kInvalidFormat;
  }
  if (config->n_kv_heads != config->n_heads) {
    set_error(error_message,
              "grouped-query/multi-query checkpoints are unsupported: n_kv_heads must equal n_heads");
    return Llama2CheckpointStatus::kUnsupportedArchitecture;
  }
  return Llama2CheckpointStatus::kSuccess;
}

bool append_count(std::size_t count, LegacyLayout* layout) {
  return checked_add(layout->total, count, &layout->total);
}

Llama2CheckpointStatus make_layout(const Llama2CheckpointConfig& config,
                                   LegacyLayout* layout,
                                   std::string* error_message) {
  const std::size_t head_dim = config.dim / config.n_heads;
  if (head_dim % 2 != 0) {
    set_error(error_message, "checkpoint head_dim must be even for RoPE");
    return Llama2CheckpointStatus::kUnsupportedArchitecture;
  }
  if (!checked_product({config.vocabulary_size, config.dim}, &layout->embeddings) ||
      !checked_product({config.n_layers, config.dim}, &layout->attention_norms) ||
      !checked_product({config.n_layers, config.dim, config.dim}, &layout->q) ||
      !checked_product({config.n_layers, config.dim, config.dim}, &layout->k) ||
      !checked_product({config.n_layers, config.dim, config.dim}, &layout->v) ||
      !checked_product({config.n_layers, config.dim, config.dim}, &layout->o) ||
      !checked_product({config.n_layers, config.dim}, &layout->mlp_norms) ||
      !checked_product({config.n_layers, config.hidden_dim, config.dim},
                       &layout->w1) ||
      !checked_product({config.n_layers, config.dim, config.hidden_dim},
                       &layout->w2) ||
      !checked_product({config.n_layers, config.hidden_dim, config.dim},
                       &layout->w3) ||
      !checked_product({config.max_sequence_length, head_dim / 2},
                       &layout->rope_cos) ||
      !checked_product({config.max_sequence_length, head_dim / 2},
                       &layout->rope_sin)) {
    set_error(error_message, "checkpoint tensor size overflow");
    return Llama2CheckpointStatus::kInvalidFormat;
  }
  layout->final_norm = config.dim;
  layout->classifier = config.shared_classifier ? 0 : layout->embeddings;
  for (const std::size_t count : {layout->embeddings, layout->attention_norms,
                                  layout->q, layout->k, layout->v, layout->o,
                                  layout->mlp_norms, layout->w1, layout->w2,
                                  layout->w3, layout->final_norm,
                                  layout->rope_cos, layout->rope_sin,
                                  layout->classifier}) {
    if (!append_count(count, layout)) {
      set_error(error_message, "checkpoint total tensor size overflow");
      return Llama2CheckpointStatus::kInvalidFormat;
    }
  }
  return Llama2CheckpointStatus::kSuccess;
}

bool expected_file_size(const LegacyLayout& layout, std::size_t* bytes) {
  std::size_t weight_bytes = 0;
  return checked_multiply(layout.total, sizeof(float), &weight_bytes) &&
         checked_add(kHeaderBytes, weight_bytes, bytes);
}

bool read_floats(std::ifstream* file, std::vector<float>* values) {
  if (values->empty())
    return true;
  file->read(reinterpret_cast<char*>(values->data()),
             static_cast<std::streamsize>(values->size() * sizeof(float)));
  return file->good();
}

bool skip_floats(std::ifstream* file, std::size_t count) {
  std::size_t bytes = 0;
  if (!checked_multiply(count, sizeof(float), &bytes) ||
      bytes > static_cast<std::size_t>(std::numeric_limits<std::streamoff>::max()))
    return false;
  file->seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
  return file->good();
}

void transpose_per_layer(const std::vector<float>& source,
                         std::vector<float>* destination,
                         std::size_t layers, std::size_t rows,
                         std::size_t columns) {
  const std::size_t matrix_count = rows * columns;
  destination->resize(source.size());
  for (std::size_t layer = 0; layer < layers; ++layer)
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t column = 0; column < columns; ++column)
        (*destination)[layer * matrix_count + column * rows + row] =
            source[layer * matrix_count + row * columns + column];
}

}  // namespace

void Llama2CheckpointModel::bind_host_views_(Llama2CheckpointModel* model) {
  const Llama2CheckpointConfig& config = model->checkpoint_config;
  const std::size_t matrix = config.dim * config.dim;
  const std::size_t hidden_matrix = config.dim * config.hidden_dim;
  model->host_layers_.resize(config.n_layers);
  for (std::size_t layer = 0; layer < config.n_layers; ++layer) {
    TinyModelLayerWeights& view = model->host_layers_[layer];
    view.decoder.attention = {
        model->host_attention_norms_.data() + layer * config.dim,
        model->host_wq_.data() + layer * matrix,
        model->host_wk_.data() + layer * matrix,
        model->host_wv_.data() + layer * matrix,
        model->host_wo_.data() + layer * matrix};
    view.decoder.mlp = {
        model->host_mlp_norms_.data() + layer * config.dim,
        model->host_w_gate_.data() + layer * hidden_matrix,
        model->host_w_up_.data() + layer * hidden_matrix,
        model->host_w_down_.data() + layer * hidden_matrix};
  }
  model->host_weights = {model->host_embeddings_.data(), model->host_layers_.data(),
                         config.n_layers, model->host_final_norm_.data(),
                         model->host_lm_head_.data()};
}

cudaError_t first_error(cudaError_t first, cudaError_t next) {
  return first == cudaSuccess ? next : first;
}

cudaError_t allocate_and_copy(const std::vector<float>& host, float** device) {
  if (host.empty())
    return cudaErrorInvalidValue;
  cudaError_t error = cudaMalloc(device, host.size() * sizeof(float));
  if (error == cudaSuccess)
    error = cudaMemcpy(*device, host.data(), host.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
  return error;
}

void Llama2CheckpointModel::bind_device_views_(Llama2CheckpointModel* model) {
  const Llama2CheckpointConfig& config = model->checkpoint_config;
  const std::size_t matrix = config.dim * config.dim;
  const std::size_t hidden_matrix = config.dim * config.hidden_dim;
  model->device_layers_.resize(config.n_layers);
  for (std::size_t layer = 0; layer < config.n_layers; ++layer) {
    TinyModelLayerWeights& view = model->device_layers_[layer];
    view.decoder.attention = {
        model->device_attention_norms_ + layer * config.dim,
        model->device_wq_ + layer * matrix, model->device_wk_ + layer * matrix,
        model->device_wv_ + layer * matrix, model->device_wo_ + layer * matrix};
    view.decoder.mlp = {
        model->device_mlp_norms_ + layer * config.dim,
        model->device_w_gate_ + layer * hidden_matrix,
        model->device_w_up_ + layer * hidden_matrix,
        model->device_w_down_ + layer * hidden_matrix};
  }
  model->device_weights = {model->device_embeddings_, model->device_layers_.data(),
                           config.n_layers, model->device_final_norm_,
                           model->device_lm_head_};
}

Llama2CheckpointStatus Llama2CheckpointModel::upload_weights_(
    Llama2CheckpointModel* model, std::string* error_message) {
  cudaError_t error = cudaSuccess;
  for (const auto [host, device] : {
           std::pair{&model->host_embeddings_, &model->device_embeddings_},
           std::pair{&model->host_attention_norms_, &model->device_attention_norms_},
           std::pair{&model->host_wq_, &model->device_wq_},
           std::pair{&model->host_wk_, &model->device_wk_},
           std::pair{&model->host_wv_, &model->device_wv_},
           std::pair{&model->host_wo_, &model->device_wo_},
           std::pair{&model->host_mlp_norms_, &model->device_mlp_norms_},
           std::pair{&model->host_w_gate_, &model->device_w_gate_},
           std::pair{&model->host_w_up_, &model->device_w_up_},
           std::pair{&model->host_w_down_, &model->device_w_down_},
           std::pair{&model->host_final_norm_, &model->device_final_norm_},
           std::pair{&model->host_lm_head_, &model->device_lm_head_}}) {
    error = allocate_and_copy(*host, device);
    if (error != cudaSuccess) {
      set_error(error_message, cudaGetErrorString(error));
      model->reset();
      return error == cudaErrorMemoryAllocation
                 ? Llama2CheckpointStatus::kMemoryAllocationFailure
                 : Llama2CheckpointStatus::kCudaError;
    }
  }
  bind_device_views_(model);
  return Llama2CheckpointStatus::kSuccess;
}

const char* llama2_checkpoint_status_string(Llama2CheckpointStatus status) {
  switch (status) {
    case Llama2CheckpointStatus::kSuccess:
      return "success";
    case Llama2CheckpointStatus::kInvalidArgument:
      return "invalid argument";
    case Llama2CheckpointStatus::kIoError:
      return "I/O error";
    case Llama2CheckpointStatus::kInvalidFormat:
      return "invalid legacy llama2.c checkpoint format";
    case Llama2CheckpointStatus::kUnsupportedArchitecture:
      return "unsupported checkpoint architecture";
    case Llama2CheckpointStatus::kMemoryAllocationFailure:
      return "memory allocation failure";
    case Llama2CheckpointStatus::kCudaError:
      return "CUDA error";
  }
  return "unknown checkpoint status";
}

Llama2CheckpointModel::~Llama2CheckpointModel() { reset(); }

cudaError_t Llama2CheckpointModel::reset() {
  cudaError_t error = cudaSuccess;
  for (float** pointer : {&device_embeddings_, &device_attention_norms_,
                          &device_wq_, &device_wk_, &device_wv_, &device_wo_,
                          &device_mlp_norms_, &device_w_gate_, &device_w_up_,
                          &device_w_down_, &device_final_norm_, &device_lm_head_}) {
    if (*pointer != nullptr)
      error = first_error(error, cudaFree(*pointer));
    *pointer = nullptr;
  }
  checkpoint_config = {};
  host_weights = {};
  device_weights = {};
  host_embeddings_.clear();
  host_attention_norms_.clear();
  host_wq_.clear();
  host_wk_.clear();
  host_wv_.clear();
  host_wo_.clear();
  host_mlp_norms_.clear();
  host_w_gate_.clear();
  host_w_up_.clear();
  host_w_down_.clear();
  host_final_norm_.clear();
  host_lm_head_.clear();
  host_layers_.clear();
  device_layers_.clear();
  return error;
}

Llama2CheckpointStatus llama2_checkpoint_tiny_model_config(
    const Llama2CheckpointConfig& checkpoint_config, std::size_t sequence,
    TinyModelConfig* model_config) {
  if (model_config == nullptr || sequence == 0 ||
      sequence > checkpoint_config.max_sequence_length ||
      checkpoint_config.n_kv_heads != checkpoint_config.n_heads ||
      checkpoint_config.dim == 0 || checkpoint_config.hidden_dim == 0 ||
      checkpoint_config.n_layers == 0 || checkpoint_config.n_heads == 0 ||
      checkpoint_config.vocabulary_size == 0 ||
      checkpoint_config.dim % checkpoint_config.n_heads != 0)
    return Llama2CheckpointStatus::kInvalidArgument;
  *model_config = {checkpoint_config.vocabulary_size, sequence,
                   checkpoint_config.dim, checkpoint_config.n_layers,
                   checkpoint_config.n_heads,
                   checkpoint_config.dim / checkpoint_config.n_heads,
                   checkpoint_config.hidden_dim, kLlama2RmsNormEpsilon};
  return valid_tiny_model_config(*model_config)
             ? Llama2CheckpointStatus::kSuccess
             : Llama2CheckpointStatus::kInvalidArgument;
}

Llama2CheckpointStatus llama2_checkpoint_load(
    const char* path, Llama2CheckpointModel* model, std::string* error_message) {
  if (error_message != nullptr)
    error_message->clear();
  if (path == nullptr || model == nullptr) {
    set_error(error_message, "checkpoint path and destination model are required");
    return Llama2CheckpointStatus::kInvalidArgument;
  }
  static_assert(sizeof(float) == 4);
  static_assert(sizeof(std::int32_t) == 4);

  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    set_error(error_message, "could not open checkpoint file");
    return Llama2CheckpointStatus::kIoError;
  }
  const std::streamoff end = file.tellg();
  if (end < static_cast<std::streamoff>(kHeaderBytes)) {
    set_error(error_message, "checkpoint is smaller than the legacy header");
    return Llama2CheckpointStatus::kInvalidFormat;
  }
  file.seekg(0);
  std::array<std::int32_t, kHeaderInts> raw_header{};
  file.read(reinterpret_cast<char*>(raw_header.data()), kHeaderBytes);
  if (!file) {
    set_error(error_message, "could not read legacy checkpoint header");
    return Llama2CheckpointStatus::kIoError;
  }

  Llama2CheckpointConfig config;
  Llama2CheckpointStatus status = parse_header(raw_header, &config, error_message);
  if (status != Llama2CheckpointStatus::kSuccess)
    return status;
  LegacyLayout layout;
  status = make_layout(config, &layout, error_message);
  if (status != Llama2CheckpointStatus::kSuccess)
    return status;
  std::size_t expected_bytes = 0;
  if (!expected_file_size(layout, &expected_bytes) ||
      static_cast<std::uintmax_t>(end) != expected_bytes) {
    set_error(error_message, "checkpoint file size does not match its legacy tensor layout");
    return Llama2CheckpointStatus::kInvalidFormat;
  }

  // The complete file is structurally valid. Only now release an older model.
  if (model->reset() != cudaSuccess) {
    set_error(error_message, "could not release previous checkpoint device weights");
    return Llama2CheckpointStatus::kCudaError;
  }
  model->checkpoint_config = config;
  try {
    model->host_embeddings_.resize(layout.embeddings);
    model->host_attention_norms_.resize(layout.attention_norms);
    model->host_mlp_norms_.resize(layout.mlp_norms);
    model->host_final_norm_.resize(layout.final_norm);
    std::vector<float> source;
    if (!read_floats(&file, &model->host_embeddings_) ||
        !read_floats(&file, &model->host_attention_norms_)) {
      set_error(error_message, "could not read checkpoint embeddings or attention norms");
      model->reset();
      return Llama2CheckpointStatus::kIoError;
    }
    source.resize(layout.q);
    if (!read_floats(&file, &source))
      throw std::runtime_error("could not read Wq");
    transpose_per_layer(source, &model->host_wq_, config.n_layers, config.dim,
                        config.dim);
    if (!read_floats(&file, &source))
      throw std::runtime_error("could not read Wk");
    transpose_per_layer(source, &model->host_wk_, config.n_layers, config.dim,
                        config.dim);
    if (!read_floats(&file, &source))
      throw std::runtime_error("could not read Wv");
    transpose_per_layer(source, &model->host_wv_, config.n_layers, config.dim,
                        config.dim);
    if (!read_floats(&file, &source))
      throw std::runtime_error("could not read Wo");
    transpose_per_layer(source, &model->host_wo_, config.n_layers, config.dim,
                        config.dim);
    if (!read_floats(&file, &model->host_mlp_norms_))
      throw std::runtime_error("could not read FFN norms");
    source.resize(layout.w1);
    if (!read_floats(&file, &source))
      throw std::runtime_error("could not read W1");
    transpose_per_layer(source, &model->host_w_gate_, config.n_layers,
                        config.hidden_dim, config.dim);
    if (!read_floats(&file, &source))
      throw std::runtime_error("could not read W2");
    transpose_per_layer(source, &model->host_w_down_, config.n_layers,
                        config.dim, config.hidden_dim);
    if (!read_floats(&file, &source))
      throw std::runtime_error("could not read W3");
    transpose_per_layer(source, &model->host_w_up_, config.n_layers,
                        config.hidden_dim, config.dim);
    if (!read_floats(&file, &model->host_final_norm_) ||
        !skip_floats(&file, layout.rope_cos) ||
        !skip_floats(&file, layout.rope_sin))
      throw std::runtime_error("could not read final norm or skip legacy RoPE arrays");
    if (config.shared_classifier) {
      transpose_per_layer(model->host_embeddings_, &model->host_lm_head_, 1,
                          config.vocabulary_size, config.dim);
    } else {
      source.resize(layout.classifier);
      if (!read_floats(&file, &source))
        throw std::runtime_error("could not read classifier");
      transpose_per_layer(source, &model->host_lm_head_, 1,
                          config.vocabulary_size, config.dim);
    }
    Llama2CheckpointModel::bind_host_views_(model);
  } catch (const std::bad_alloc&) {
    set_error(error_message, "host allocation failed while loading checkpoint");
    model->reset();
    return Llama2CheckpointStatus::kMemoryAllocationFailure;
  } catch (const std::runtime_error& error) {
    set_error(error_message, error.what());
    model->reset();
    return Llama2CheckpointStatus::kIoError;
  }
  return Llama2CheckpointModel::upload_weights_(model, error_message);
}

}  // namespace cuda_transformer
