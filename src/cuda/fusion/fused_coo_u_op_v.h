#ifndef GS_CUDA_FUSION_FUSED_COO_U_OP_V_H_
#define GS_CUDA_FUSION_FUSED_COO_U_OP_V_H_

#include <torch/torch.h>
#include "bcast.h"
#include "graph.h"

namespace gs {
namespace impl {
namespace fusion {

void FUSED_COO_U_OP_V(const std::string& op, const BcastOff& bcast,
              const std::shared_ptr<COO> coo, torch::Tensor lhs1,
              torch::Tensor rhs1, torch::Tensor out1, torch::Tensor lhs2,
              torch::Tensor rhs2, torch::Tensor out2);


}
}  // namespace impl
}  // namespace gs


#endif