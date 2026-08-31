#include "cuda_transformer/llama2_tokenizer.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

bool check_reference_prompt(const cuda_transformer::Llama2Tokenizer& tokenizer) {
  constexpr char kPrompt[] = "I believe the meaning of life is";
  const std::vector<int> expected{1, 306, 4658, 278, 6593, 310, 2834, 338};
  const std::vector<int> actual = tokenizer.encode(kPrompt, true, false);
  if (actual != expected) {
    std::fputs("reference tokenizer IDs differ\nexpected:", stderr);
    for (const int token : expected)
      std::fprintf(stderr, " %d", token);
    std::fputs("\nactual:", stderr);
    for (const int token : actual)
      std::fprintf(stderr, " %d", token);
    std::fputc('\n', stderr);
    return false;
  }
  std::string decoded;
  for (std::size_t index = 1; index < actual.size(); ++index) {
    const auto piece = tokenizer.decode_piece(actual[index - 1], actual[index]);
    if (!piece) {
      std::fputs("reference tokenizer decode failed\n", stderr);
      return false;
    }
    decoded += *piece;
  }
  if (decoded != kPrompt) {
    std::fprintf(stderr, "reference tokenizer decode differs: %s\n", decoded.c_str());
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "Usage: %s path/to/tokenizer.bin\n", argv[0]);
    return EXIT_FAILURE;
  }
  cuda_transformer::Llama2Tokenizer tokenizer;
  std::string error;
  const auto status = tokenizer.load(argv[1], 32000, &error);
  if (status != cuda_transformer::Llama2TokenizerStatus::kSuccess) {
    std::fprintf(stderr, "tokenizer load failed (%s): %s\n",
                 cuda_transformer::llama2_tokenizer_status_string(status),
                 error.c_str());
    return EXIT_FAILURE;
  }
  if (!check_reference_prompt(tokenizer))
    return EXIT_FAILURE;
  std::puts("32k llama2.c tokenizer reference encode/decode check passed.");
}
