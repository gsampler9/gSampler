#ifndef GS_CUDA_FUSION_SLICING_SAMPLING_H_
#define GS_CUDA_FUSION_SLICING_SAMPLING_H_
#include <torch/torch.h>

namespace gs {
namespace impl {
namespace fusion {
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
FusedCSCColSlicingSamplingCUDA(torch::Tensor indptr, torch::Tensor indices,
                               int64_t fanout, torch::Tensor node_ids,
                               bool replace, bool with_coo);
}  // namespace fusion
}  // namespace impl
}  // namespace gs

#endif