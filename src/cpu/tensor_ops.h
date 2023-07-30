#ifndef GS_CPU_TENSOR_OPS_H_
#define GS_CPU_TENSOR_OPS_H_

#include <torch/torch.h>

namespace gs {
namespace impl {
torch::Tensor SortIndicesCPU(torch::Tensor indptr, torch::Tensor indices);
}
}  // namespace gs

#endif