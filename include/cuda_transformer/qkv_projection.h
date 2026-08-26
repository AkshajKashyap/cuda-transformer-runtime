#pragma once
#include "cuda_transformer/gemm.h"
#include <cstddef>
namespace cuda_transformer {
struct QkvProjectionConfig {
  std::size_t sequence, hidden, heads, head_dim;
};
struct QkvProjectionWorkspace {
  float *q_token = nullptr, *k_token = nullptr, *v_token = nullptr,
        *q_head = nullptr, *k_head = nullptr, *v_head = nullptr;
  QkvProjectionConfig config{};
  QkvProjectionWorkspace() = default;
  QkvProjectionWorkspace(const QkvProjectionWorkspace &) = delete;
  QkvProjectionWorkspace &operator=(const QkvProjectionWorkspace &) = delete;
};
bool valid_qkv_projection_config(QkvProjectionConfig);
cudaError_t qkv_projection_workspace_create(QkvProjectionWorkspace *,
                                            QkvProjectionConfig);
cudaError_t qkv_projection_workspace_destroy(QkvProjectionWorkspace *);
cublasStatus_t project_qkv_cuda(cublasHandle_t, const float *, const float *,
                                const float *, const float *,
                                QkvProjectionWorkspace *,
                                cudaStream_t = nullptr);
} // namespace cuda_transformer
