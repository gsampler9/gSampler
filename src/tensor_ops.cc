#include "./tensor_ops.h"
#include "cuda/batch/batch_ops.h"
#include "cuda/tensor_ops.h"

namespace gs {
torch::Tensor ListSampling(int64_t num_items, int64_t num_picks, bool replace) {
  return impl::ListSamplingCUDA(num_items, num_picks, replace);
}

torch::Tensor ListSamplingProbs(torch::Tensor probs, int64_t num_picks,
                                bool replace) {
  return impl::ListSamplingProbsCUDA(probs, num_picks, replace);
}

std::tuple<torch::Tensor, torch::Tensor> BatchListSamplingProbs(
    torch::Tensor probs, int64_t num_picks, bool replace, torch::Tensor range) {
  return impl::batch::BatchListSamplingProbsCUDA(probs, num_picks, replace,
                                                 range);
}

std::tuple<torch::Tensor, torch::Tensor> BatchListSampling(
    int64_t num_picks, bool replace, torch::Tensor range) {
  torch::Tensor probs = torch::ones(
      range.numel() - 1,
      torch::TensorOptions().dtype(torch::kFloat32).device(range.device()));
  return impl::batch::BatchListSamplingProbsCUDA(probs, num_picks, replace,
                                                 range);
}

}  // namespace gs
