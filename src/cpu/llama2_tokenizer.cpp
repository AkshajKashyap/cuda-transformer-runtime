#include "cuda_transformer/llama2_tokenizer.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <new>

namespace cuda_transformer {
namespace {

constexpr std::size_t kMaxSaneTokenLength = 1U << 20;
constexpr std::size_t kMinimumByteFallbackVocabulary =
    kLlama2ByteTokenOffset + 256;

void set_error(std::string* error_message, const char* message) {
  if (error_message != nullptr)
    *error_message = message;
}

bool valid_token_id(int token, std::size_t vocabulary_size) {
  return token >= 0 && static_cast<std::size_t>(token) < vocabulary_size;
}

int hex_value(char value) {
  if (value >= '0' && value <= '9')
    return value - '0';
  if (value >= 'A' && value <= 'F')
    return value - 'A' + 10;
  if (value >= 'a' && value <= 'f')
    return value - 'a' + 10;
  return -1;
}

std::optional<std::string> raw_byte_piece(std::string_view piece) {
  if (piece.size() != 6 || piece[0] != '<' || piece[1] != '0' ||
      piece[2] != 'x' || piece[5] != '>')
    return std::nullopt;
  const int high = hex_value(piece[3]);
  const int low = hex_value(piece[4]);
  if (high < 0 || low < 0)
    return std::nullopt;
  return std::string(1, static_cast<char>((high << 4) | low));
}

}  // namespace

const char* llama2_tokenizer_status_string(Llama2TokenizerStatus status) {
  switch (status) {
    case Llama2TokenizerStatus::kSuccess:
      return "success";
    case Llama2TokenizerStatus::kInvalidArgument:
      return "invalid argument";
    case Llama2TokenizerStatus::kIoError:
      return "I/O error";
    case Llama2TokenizerStatus::kInvalidFormat:
      return "invalid llama2.c tokenizer format";
    case Llama2TokenizerStatus::kMemoryAllocationFailure:
      return "memory allocation failure";
  }
  return "unknown tokenizer status";
}

void Llama2Tokenizer::reset() {
  vocabulary_.clear();
  scores_.clear();
  lookup_.clear();
  max_token_length_ = 0;
}

Llama2TokenizerStatus Llama2Tokenizer::load(const char* path,
                                             std::size_t expected_vocabulary_size,
                                             std::string* error_message) {
  if (error_message != nullptr)
    error_message->clear();
  if (path == nullptr || expected_vocabulary_size < kMinimumByteFallbackVocabulary) {
    set_error(error_message,
              "tokenizer path and a vocabulary with byte-fallback entries are required");
    return Llama2TokenizerStatus::kInvalidArgument;
  }
  if (expected_vocabulary_size >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    set_error(error_message, "tokenizer vocabulary size exceeds legacy format limits");
    return Llama2TokenizerStatus::kInvalidArgument;
  }

  std::ifstream file(path, std::ios::binary);
  if (!file) {
    set_error(error_message, "could not open tokenizer file");
    return Llama2TokenizerStatus::kIoError;
  }
  std::uint32_t max_token_length = 0;
  file.read(reinterpret_cast<char*>(&max_token_length), sizeof(max_token_length));
  if (!file) {
    set_error(error_message, "could not read tokenizer max token length");
    return Llama2TokenizerStatus::kInvalidFormat;
  }
  if (max_token_length == 0 || max_token_length > kMaxSaneTokenLength) {
    set_error(error_message, "tokenizer max token length is not sane");
    return Llama2TokenizerStatus::kInvalidFormat;
  }

  std::vector<std::string> vocabulary;
  std::vector<float> scores;
  std::unordered_map<std::string, int> lookup;
  try {
    vocabulary.reserve(expected_vocabulary_size);
    scores.reserve(expected_vocabulary_size);
    lookup.reserve(expected_vocabulary_size);
    for (std::size_t token = 0; token < expected_vocabulary_size; ++token) {
      float score = 0.0F;
      std::uint32_t byte_length = 0;
      file.read(reinterpret_cast<char*>(&score), sizeof(score));
      file.read(reinterpret_cast<char*>(&byte_length), sizeof(byte_length));
      if (!file) {
        set_error(error_message, "tokenizer truncates in a vocabulary entry header");
        return Llama2TokenizerStatus::kInvalidFormat;
      }
      if (!std::isfinite(score) || byte_length > max_token_length ||
          byte_length > kMaxSaneTokenLength) {
        set_error(error_message, "tokenizer entry has invalid score or byte length");
        return Llama2TokenizerStatus::kInvalidFormat;
      }
      std::string piece(byte_length, '\0');
      if (byte_length != 0)
        file.read(piece.data(), static_cast<std::streamsize>(byte_length));
      if (!file) {
        set_error(error_message, "tokenizer truncates in vocabulary bytes");
        return Llama2TokenizerStatus::kInvalidFormat;
      }
      vocabulary.push_back(std::move(piece));
      scores.push_back(score);
      // Legacy lookup uses qsort/bsearch over the vocabulary. Standard Llama 2
      // has unique pieces; retain the first ID if malformed input duplicates one.
      lookup.emplace(vocabulary.back(), static_cast<int>(token));
    }
  } catch (const std::bad_alloc&) {
    set_error(error_message, "host allocation failed while loading tokenizer");
    return Llama2TokenizerStatus::kMemoryAllocationFailure;
  }
  if (file.peek() != std::ifstream::traits_type::eof()) {
    set_error(error_message, "tokenizer has trailing bytes after expected vocabulary");
    return Llama2TokenizerStatus::kInvalidFormat;
  }
  if (lookup.find(" ") == lookup.end()) {
    set_error(error_message, "tokenizer lacks the required dummy-prefix space token");
    return Llama2TokenizerStatus::kInvalidFormat;
  }

  reset();
  vocabulary_ = std::move(vocabulary);
  scores_ = std::move(scores);
  lookup_ = std::move(lookup);
  max_token_length_ = max_token_length;
  return Llama2TokenizerStatus::kSuccess;
}

std::vector<int> Llama2Tokenizer::encode(std::string_view text, bool bos,
                                         bool eos) const {
  if (!is_loaded())
    return {};
  std::vector<int> tokens;
  tokens.reserve(text.size() + static_cast<std::size_t>(bos) +
                 static_cast<std::size_t>(eos) + 1);
  if (bos)
    tokens.push_back(kLlama2BosTokenId);
  if (!text.empty())
    tokens.push_back(lookup_.at(" "));

  std::string codepoint;
  for (std::size_t position = 0; position < text.size(); ++position) {
    const unsigned char current = static_cast<unsigned char>(text[position]);
    if ((current & 0xC0U) != 0x80U)
      codepoint.clear();
    codepoint.push_back(static_cast<char>(current));
    const bool next_is_continuation =
        position + 1 < text.size() &&
        (static_cast<unsigned char>(text[position + 1]) & 0xC0U) == 0x80U;
    if (next_is_continuation && codepoint.size() < 4)
      continue;

    const auto found = lookup_.find(codepoint);
    if (found != lookup_.end()) {
      tokens.push_back(found->second);
    } else {
      for (const unsigned char byte : codepoint)
        tokens.push_back(static_cast<int>(byte) + kLlama2ByteTokenOffset);
    }
    codepoint.clear();
  }

  while (tokens.size() > 1) {
    float best_score = -1.0e10F;
    int best_id = -1;
    std::size_t best_position = 0;
    for (std::size_t position = 0; position + 1 < tokens.size(); ++position) {
      const std::string merged =
          vocabulary_[tokens[position]] + vocabulary_[tokens[position + 1]];
      const auto found = lookup_.find(merged);
      if (found != lookup_.end() && scores_[found->second] > best_score) {
        best_score = scores_[found->second];
        best_id = found->second;
        best_position = position;
      }
    }
    if (best_id == -1)
      break;
    tokens[best_position] = best_id;
    tokens.erase(tokens.begin() + static_cast<std::ptrdiff_t>(best_position + 1));
  }
  if (eos)
    tokens.push_back(kLlama2EosTokenId);
  return tokens;
}

std::optional<std::string> Llama2Tokenizer::decode_piece(int previous_token,
                                                          int token) const {
  if (!valid_token_id(token, vocabulary_.size()))
    return std::nullopt;
  std::string piece = vocabulary_[token];
  if (previous_token == kLlama2BosTokenId && !piece.empty() && piece[0] == ' ')
    piece.erase(piece.begin());
  if (const auto raw = raw_byte_piece(piece))
    return raw;
  return piece;
}

std::optional<std::string> Llama2Tokenizer::decode(
    const std::vector<int>& tokens) const {
  std::string text;
  int previous_token = -1;
  for (const int token : tokens) {
    const auto piece = decode_piece(previous_token, token);
    if (!piece)
      return std::nullopt;
    text += *piece;
    previous_token = token;
  }
  return text;
}

}  // namespace cuda_transformer
