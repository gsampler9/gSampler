#ifndef GS_CUDA_FUSION_EDGE_MAP_REDUCE_H_
#define GS_CUDA_FUSION_EDGE_MAP_REDUCE_H_
#include <torch/torch.h>
#include "bcast.h"
#include "graph.h"

namespace gs {
namespace impl {
namespace fusion {
void COOEDivUSum(torch::Tensor row, torch::Tensor col,
                          torch::Tensor in_data, torch::Tensor divisor,
                          torch::Tensor out_data, torch::Tensor out_sum);
}
}  // namespace impl
}  // namespace gs

#endif