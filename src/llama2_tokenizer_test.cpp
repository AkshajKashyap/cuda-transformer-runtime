#include "cuda_transformer/llama2_tokenizer.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kVocabularySize = 266;

struct TokenEntry {
  std::string piece;
  float score = -100.0F;
};

std::string raw_byte_token(unsigned char byte) {
  constexpr char kHex[] = "0123456789ABCDEF";
  std::string piece{"<0x00>"};
  piece[3] = kHex[byte >> 4];
  piece[4] = kHex[byte & 0x0F];
  return piece;
}

std::vector<TokenEntry> synthetic_vocabulary() {
  std::vector<TokenEntry> vocabulary(kVocabularySize);
  vocabulary[0].piece = "<unk>";
  vocabulary[1].piece = "\n<s>\n";
  vocabulary[2].piece = "\n</s>\n";
  for (int byte = 0; byte < 256; ++byte)
    vocabulary[byte + 3].piece = raw_byte_token(static_cast<unsigned char>(byte));
  vocabulary[259] = {" ", 0.0F};
  vocabulary[260] = {"a", 0.0F};
  vocabulary[261] = {"b", 0.0F};
  vocabulary[262] = {"ab", 10.0F};
  vocabulary[263] = {"\xC3\xA9", 0.0F};
  vocabulary[264] = {"c", 0.0F};
  vocabulary[265] = {"bc", 10.0F};
  return vocabulary;
}

bool write_tokenizer(const std::filesystem::path& path, bool extra = false,
                     bool bad_max_length = false) {
  const std::vector<TokenEntry> vocabulary = synthetic_vocabulary();
  std::size_t max_length = 0;
  for (const TokenEntry& token : vocabulary)
    max_length = std::max(max_length, token.piece.size());
  if (bad_max_length)
    max_length = 0;
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  const std::uint32_t encoded_max = static_cast<std::uint32_t>(max_length);
  file.write(reinterpret_cast<const char*>(&encoded_max), sizeof(encoded_max));
  for (const TokenEntry& token : vocabulary) {
    const std::uint32_t length = static_cast<std::uint32_t>(token.piece.size());
    file.write(reinterpret_cast<const char*>(&token.score), sizeof(token.score));
    file.write(reinterpret_cast<const char*>(&length), sizeof(length));
    file.write(token.piece.data(), static_cast<std::streamsize>(token.piece.size()));
  }
  if (extra)
    file.put('\0');
  return static_cast<bool>(file);
}

bool check_tokens(const std::vector<int>& actual, std::vector<int> expected,
                  const char* name) {
  if (actual == expected)
    return true;
  std::fprintf(stderr, "%s token mismatch\nexpected:", name);
  for (const int token : expected)
    std::fprintf(stderr, " %d", token);
  std::fprintf(stderr, "\nactual:");
  for (const int token : actual)
    std::fprintf(stderr, " %d", token);
  std::fputc('\n', stderr);
  return false;
}

bool valid_tokenizer_test(const std::filesystem::path& path) {
  if (!write_tokenizer(path))
    return false;
  cuda_transformer::Llama2Tokenizer tokenizer;
  std::string error;
  bool ok = tokenizer.load(path.c_str(), kVocabularySize, &error) ==
                cuda_transformer::Llama2TokenizerStatus::kSuccess &&
            tokenizer.vocabulary_size() == kVocabularySize &&
            tokenizer.max_token_length() == 6 &&
            check_tokens(tokenizer.encode("", true, false), {1}, "empty BOS") &&
            check_tokens(tokenizer.encode("", false, true), {2}, "empty EOS") &&
            check_tokens(tokenizer.encode("ab", true, false), {1, 259, 262},
                         "ASCII merge") &&
            check_tokens(tokenizer.encode("abc", false, false), {259, 262, 264},
                         "leftmost BPE score tie") &&
            check_tokens(tokenizer.encode("abab", false, false), {259, 262, 262},
                         "repeated-word BPE") &&
            check_tokens(tokenizer.encode(" ", false, false), {259, 259},
                         "space handling") &&
            check_tokens(tokenizer.encode("!", false, false), {259, 36},
                         "punctuation byte fallback") &&
            check_tokens(tokenizer.encode("\xC3\xA9", false, false), {259, 263},
                         "direct UTF-8 codepoint") &&
            check_tokens(tokenizer.encode("\xC3\x9F", false, false),
                         {259, 198, 162}, "UTF-8 byte fallback") &&
            check_tokens(tokenizer.encode("ab", false, true), {259, 262, 2},
                         "EOS append");
  const auto bos_space = tokenizer.decode_piece(1, 259);
  const auto raw_byte = tokenizer.decode_piece(260, 68);
  const auto invalid = tokenizer.decode_piece(0, static_cast<int>(kVocabularySize));
  ok = ok && bos_space && *bos_space == "" && raw_byte && *raw_byte == "A" &&
       !invalid;
  tokenizer.reset();
  ok = ok && !tokenizer.is_loaded() && tokenizer.encode("ab", true, false).empty();
  std::filesystem::remove(path);
  return ok;
}

bool rejection_test(const std::filesystem::path& path, bool extra,
                    bool bad_max_length,
                    cuda_transformer::Llama2TokenizerStatus expected) {
  if (!write_tokenizer(path, extra, bad_max_length))
    return false;
  cuda_transformer::Llama2Tokenizer tokenizer;
  std::string error;
  const bool ok = tokenizer.load(path.c_str(), kVocabularySize, &error) == expected &&
                  !error.empty() && !tokenizer.is_loaded();
  std::filesystem::remove(path);
  return ok;
}

bool truncated_test(const std::filesystem::path& path) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  const std::uint32_t max_length = 6;
  file.write(reinterpret_cast<const char*>(&max_length), sizeof(max_length));
  file.close();
  cuda_transformer::Llama2Tokenizer tokenizer;
  std::string error;
  const bool ok = tokenizer.load(path.c_str(), kVocabularySize, &error) ==
                      cuda_transformer::Llama2TokenizerStatus::kInvalidFormat &&
                  !error.empty();
  std::filesystem::remove(path);
  return ok;
}

}  // namespace

int main() {
  const std::filesystem::path base =
      std::filesystem::temp_directory_path() / "cuda_transformer_tokenizer";
  const bool ok = valid_tokenizer_test(base.string() + "_valid.bin") &&
                  truncated_test(base.string() + "_truncated.bin") &&
                  rejection_test(base.string() + "_extra.bin", true, false,
                                 cuda_transformer::Llama2TokenizerStatus::kInvalidFormat) &&
                  rejection_test(base.string() + "_bad_max.bin", false, true,
                                 cuda_transformer::Llama2TokenizerStatus::kInvalidFormat);
  if (!ok) {
    std::fputs("llama2 tokenizer tests failed\n", stderr);
    return EXIT_FAILURE;
  }
  std::puts("llama2 tokenizer tests passed.");
}
