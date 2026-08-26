#include "cuda_transformer/qkv_layout.h"
#include "cuda_transformer/qkv_projection.h"
namespace cuda_transformer {
namespace {
void clear(QkvProjectionWorkspace *w) {
  w->q_token = w->k_token = w->v_token = w->q_head = w->k_head = w->v_head =
      nullptr;
  w->config = {};
}
bool matches(const QkvProjectionWorkspace *w) {
  return w && valid_qkv_projection_config(w->config) && w->q_token &&
         w->k_token && w->v_token && w->q_head && w->k_head && w->v_head;
}
} // namespace
bool valid_qkv_projection_config(QkvProjectionConfig c) {
  return c.sequence && c.hidden && c.heads && c.head_dim &&
         c.hidden == c.heads * c.head_dim;
}
cudaError_t qkv_projection_workspace_create(QkvProjectionWorkspace *w,
                                            QkvProjectionConfig c) {
  if (!w || !valid_qkv_projection_config(c))
    return cudaErrorInvalidValue;
  clear(w);
  size_t bytes = c.sequence * c.hidden * sizeof(float);
  cudaError_t e = cudaMalloc(&w->q_token, bytes);
  if (e == cudaSuccess)
    e = cudaMalloc(&w->k_token, bytes);
  if (e == cudaSuccess)
    e = cudaMalloc(&w->v_token, bytes);
  if (e == cudaSuccess)
    e = cudaMalloc(&w->q_head, bytes);
  if (e == cudaSuccess)
    e = cudaMalloc(&w->k_head, bytes);
  if (e == cudaSuccess)
    e = cudaMalloc(&w->v_head, bytes);
  if (e != cudaSuccess)
    qkv_projection_workspace_destroy(w);
  else
    w->config = c;
  return e;
}
cudaError_t qkv_projection_workspace_destroy(QkvProjectionWorkspace *w) {
  if (!w)
    return cudaErrorInvalidValue;
  cudaError_t e = cudaSuccess;
  for (auto p :
       {w->q_token, w->k_token, w->v_token, w->q_head, w->k_head, w->v_head})
    if (p) {
      auto x = cudaFree(p);
      if (e == cudaSuccess)
        e = x;
    }
  clear(w);
  return e;
}
cublasStatus_t project_qkv_cuda(cublasHandle_t h, const float *x,
                                const float *q, const float *k, const float *v,
                                QkvProjectionWorkspace *w, cudaStream_t s) {
  if (!h || !x || !q || !k || !v || !matches(w))
    return CUBLAS_STATUS_INVALID_VALUE;
  auto c = w->config;
  cublasStatus_t e = linear_cublas_row_major(h, x, q, w->q_token, c.sequence,
                                             c.hidden, c.hidden, s);
  if (e == CUBLAS_STATUS_SUCCESS)
    e = linear_cublas_row_major(h, x, k, w->k_token, c.sequence, c.hidden,
                                c.hidden, s);
  if (e == CUBLAS_STATUS_SUCCESS)
    e = linear_cublas_row_major(h, x, v, w->v_token, c.sequence, c.hidden,
                                c.hidden, s);
  if (e != CUBLAS_STATUS_SUCCESS)
    return e;
  cudaError_t ce = token_major_to_head_major_cuda(
      w->q_token, w->q_head, c.sequence, c.heads, c.head_dim, s);
  if (ce == cudaSuccess)
    ce = token_major_to_head_major_cuda(w->k_token, w->k_head, c.sequence,
                                        c.heads, c.head_dim, s);
  if (ce == cudaSuccess)
    ce = token_major_to_head_major_cuda(w->v_token, w->v_head, c.sequence,
                                        c.heads, c.head_dim, s);
  return ce == cudaSuccess ? CUBLAS_STATUS_SUCCESS
                           : CUBLAS_STATUS_EXECUTION_FAILED;
}
} // namespace cuda_transformer
