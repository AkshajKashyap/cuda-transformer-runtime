#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace cuda_transformer {

inline constexpr int kLlama2BosTokenId = 1;
inline constexpr int kLlama2EosTokenId = 2;
inline constexpr int kLlama2ByteTokenOffset = 3;

enum class Llama2TokenizerStatus {
  kSuccess,
  kInvalidArgument,
  kIoError,
  kInvalidFormat,
  kMemoryAllocationFailure,
};

const char* llama2_tokenizer_status_string(Llama2TokenizerStatus status);

// Owns the original legacy llama2.c tokenizer vocabulary and BPE scores. The
// binary format has no vocabulary size field, so load() requires the expected
// value supplied by the matching checkpoint.
class Llama2Tokenizer {
 public:
  Llama2Tokenizer() = default;

  Llama2TokenizerStatus load(const char* path, std::size_t expected_vocabulary_size,
                             std::string* error_message = nullptr);
  void reset();

  bool is_loaded() const { return !vocabulary_.empty(); }
  std::size_t vocabulary_size() const { return vocabulary_.size(); }
  std::size_t max_token_length() const { return max_token_length_; }

  // Matches legacy llama2.c encode(): optional BOS/EOS, a non-empty-input
  // dummy " " token, complete UTF-8 codepoint lookup, byte fallback, and
  // repeated highest-score BPE merges. Calling on an unloaded tokenizer
  // returns an empty vector.
  std::vector<int> encode(std::string_view text, bool bos, bool eos) const;

  // Matches legacy decode() without terminal-oriented safe_printf filtering.
  // A leading ASCII space is removed when previous_token is BOS. A raw-byte
  // token written as <0xXX> becomes a one-byte string. Invalid IDs return null.
  std::optional<std::string> decode_piece(int previous_token, int token) const;
  std::optional<std::string> decode(const std::vector<int>& tokens) const;

 private:
  std::vector<std::string> vocabulary_;
  std::vector<float> scores_;
  std::unordered_map<std::string, int> lookup_;
  std::size_t max_token_length_ = 0;
};

}  // namespace cuda_transformer
