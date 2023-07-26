#ifndef GS_CUDA_FUSION_FUSED_CSC_U_OP_V_H_
#define GS_CUDA_FUSION_FUSED_CSC_U_OP_V_H_

#include <torch/torch.h>
#include "bcast.h"
#include "graph.h"

namespace gs {
namespace impl {
namespace fusion {

void FUSED_CSC_U_OP_V(const std::string& op, const BcastOff& bcast,
              const std::shared_ptr<CSC> csc, torch::Tensor lhs1,
              torch::Tensor rhs1, torch::Tensor out1, torch::Tensor lhs2,
              torch::Tensor rhs2, torch::Tensor out2);


}
}  // namespace impl
}  // namespace gs


#endif