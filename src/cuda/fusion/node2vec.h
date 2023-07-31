#ifndef GS_CUDA_FUSION_NODE2VEC_H_
#define GS_CUDA_FUSION_NODE2VEC_H_
#include <torch/torch.h>

namespace gs {
namespace impl {
namespace fusion {
torch::Tensor FusedNode2VecCUDA(torch::Tensor seeds, int64_t walk_length,
                                torch::Tensor indices, torch::Tensor indptr,
                                double p, double q);
}  // namespace fusion
}  // namespace impl
}  // namespace gs

#endif