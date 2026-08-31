#include "cuda_transformer/cublas_check.h"
#include "cuda_transformer/cuda_check.h"
#include "cuda_transformer/llama2_checkpoint.h"
#include "cuda_transformer/llama2_tokenizer.h"
#include "cuda_transformer/tiny_model.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace {

bool parse_count(const char* text, std::size_t* count) {
  if (text == nullptr || *text == '\0')
    return false;
  errno = 0;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value == 0 ||
      value > std::numeric_limits<std::size_t>::max())
    return false;
  *count = static_cast<std::size_t>(value);
  return true;
}

void print_token_ids(const std::vector<int>& tokens, std::size_t count) {
  std::fputs("generated token IDs:", stdout);
  for (std::size_t index = 0; index < count; ++index)
    std::printf(" %d", tokens[index]);
  std::fputc('\n', stdout);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr,
                 "Usage: %s checkpoint.bin tokenizer.bin \"prompt\" max_new_tokens\n",
                 argv[0]);
    return EXIT_FAILURE;
  }
  std::size_t max_new_tokens = 0;
  if (!parse_count(argv[4], &max_new_tokens)) {
    std::fputs("max_new_tokens must be a positive integer\n", stderr);
    return EXIT_FAILURE;
  }

  cuda_transformer::Llama2CheckpointModel model;
  std::string error;
  const auto checkpoint_status =
      cuda_transformer::llama2_checkpoint_load(argv[1], &model, &error);
  if (checkpoint_status != cuda_transformer::Llama2CheckpointStatus::kSuccess) {
    std::fprintf(stderr, "checkpoint load failed (%s): %s\n",
                 cuda_transformer::llama2_checkpoint_status_string(checkpoint_status),
                 error.c_str());
    return EXIT_FAILURE;
  }
  cuda_transformer::Llama2Tokenizer tokenizer;
  const auto tokenizer_status = tokenizer.load(
      argv[2], model.checkpoint_config.vocabulary_size, &error);
  if (tokenizer_status != cuda_transformer::Llama2TokenizerStatus::kSuccess) {
    std::fprintf(stderr, "tokenizer load failed (%s): %s\n",
                 cuda_transformer::llama2_tokenizer_status_string(tokenizer_status),
                 error.c_str());
    return EXIT_FAILURE;
  }
  const std::vector<int> prompt_tokens = tokenizer.encode(argv[3], true, false);
  if (prompt_tokens.empty()) {
    std::fputs("tokenizer produced no prompt tokens\n", stderr);
    return EXIT_FAILURE;
  }
  const std::size_t context_capacity = model.checkpoint_config.max_sequence_length;
  if (prompt_tokens.size() > context_capacity ||
      max_new_tokens > context_capacity - prompt_tokens.size() + 1) {
    std::fprintf(stderr,
                 "prompt plus requested generation exceeds checkpoint context: "
                 "prompt=%zu max_new=%zu capacity=%zu\n",
                 prompt_tokens.size(), max_new_tokens,
                 model.checkpoint_config.max_sequence_length);
    return EXIT_FAILURE;
  }
  std::fputs("prompt token IDs:", stdout);
  for (const int token : prompt_tokens)
    std::printf(" %d", token);
  std::fputc('\n', stdout);

  cuda_transformer::TinyModelConfig config;
  if (cuda_transformer::llama2_checkpoint_tiny_model_config(
          model.checkpoint_config, 1, &config) !=
      cuda_transformer::Llama2CheckpointStatus::kSuccess) {
    std::fputs("could not construct incremental runtime configuration\n", stderr);
    return EXIT_FAILURE;
  }
  cublasHandle_t handle = nullptr;
  cuda_transformer::TinyModelIncrementalWorkspace workspace;
  float* device_logits = nullptr;
  auto cleanup = [&] {
    bool ok = true;
    if (device_logits != nullptr && !CTR_CUDA_CHECK(cudaFree(device_logits)))
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
          &workspace, config, model.checkpoint_config.max_sequence_length)) ||
      !CTR_CUDA_CHECK(cudaMalloc(&device_logits,
                                 config.vocabulary_size * sizeof(float)))) {
    cleanup();
    return EXIT_FAILURE;
  }

  std::vector<int> generated(max_new_tokens);
  std::size_t generated_count = 0;
  const bool ok = CTR_CUBLAS_CHECK(cuda_transformer::tiny_model_generate_greedy_cuda(
      handle, prompt_tokens.data(), prompt_tokens.size(), model.device_weights,
      &workspace, max_new_tokens, generated.data(), device_logits, nullptr,
      cuda_transformer::kLlama2BosTokenId, &generated_count));
  if (!ok) {
    cleanup();
    return EXIT_FAILURE;
  }

  std::string continuation;
  int previous_token = prompt_tokens.back();
  for (std::size_t index = 0; index < generated_count; ++index) {
    const int token = generated[index];
    if (token == cuda_transformer::kLlama2BosTokenId)
      break;
    // Keep special markers out of user-facing text. EOS does not terminate
    // this non-chat llama2.c-compatible loop; only generated BOS does.
    if (token != cuda_transformer::kLlama2EosTokenId) {
      const auto piece = tokenizer.decode_piece(previous_token, token);
      if (!piece) {
        std::fprintf(stderr, "could not decode generated token %d\n", token);
        cleanup();
        return EXIT_FAILURE;
      }
      continuation += *piece;
    }
    previous_token = token;
  }
  std::printf("prompt + continuation:\n%s%s\n", argv[3], continuation.c_str());
  print_token_ids(generated, generated_count);
  return cleanup() ? EXIT_SUCCESS : EXIT_FAILURE;
}
