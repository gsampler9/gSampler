#ifndef GS_CUDA_FUSION_NODE2VEC_H_
#define GS_CUDA_FUSION_NODE2VEC_H_
#include <torch/torch.h>

namespace gs {
namespace impl {
namespace fusion {
torch::Tensor FusedNode2VecCUDA(torch::Tensor seeds, int64_t walk_length,
                                int64_t* indices, int64_t* indptr, double p,
                                double q);
}
}  // namespace impl
}  // namespace gs

#endif