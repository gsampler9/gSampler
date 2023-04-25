#ifndef GS_CUDA_FUSION_RANDOM_WALK_H_
#define GS_CUDA_FUSION_RANDOM_WALK_H_
#include <torch/torch.h>

namespace gs {
namespace impl {
namespace fusion {
torch::Tensor FusedRandomWalkCUDA(torch::Tensor seeds, int64_t walk_length,
                                  int64_t* indices, int64_t* indptr);
}
}  // namespace impl
}  // namespace gs

#endif