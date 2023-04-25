#ifndef GS_CUDA_FUSION_METAPATH_RANDOM_WALK_H_
#define GS_CUDA_FUSION_METAPATH_RANDOM_WALK_H_
#include <torch/torch.h>

namespace gs {
namespace impl {
namespace fusion {
torch::Tensor FusedMetapathRandomWalkCUDA(torch::Tensor seeds,
                                          torch::Tensor metapath,
                                          int64_t **all_indices,
                                          int64_t **all_indptr);
}
}  // namespace impl
}  // namespace gs

#endif