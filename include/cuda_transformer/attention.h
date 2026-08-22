#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cuda_transformer {

// Contiguous row-major storage: [batch, heads, sequence, head_dim].
struct AttentionShape {
    std::size_t batch;
    std::size_t heads;
    std::size_t sequence;
    std::size_t head_dim;
};

bool valid_attention_shape(AttentionShape shape);

// Outputs rotated Q, causal-softmax probabilities [batch, heads, query, key],
// and attention output. RoPE is applied internally to both Q and K.
void attention_cpu(const float* q, const float* k, const float* v, float* rotated_q,
                   float* probabilities, float* output, AttentionShape shape);

cudaError_t rope_cuda(const float* input, float* output, AttentionShape shape,
                      cudaStream_t stream = nullptr);
cudaError_t attention_scores_cuda(const float* q, const float* k, float* scores,
                                  AttentionShape shape, cudaStream_t stream = nullptr);
cudaError_t attention_values_cuda(const float* probabilities, const float* v, float* output,
                                  AttentionShape shape, cudaStream_t stream = nullptr);

}  // namespace cuda_transformer
