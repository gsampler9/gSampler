#ifndef GS_CUDA_FUSION_FUSED_CSC_E_DIV_U_SUM_H_
#define GS_CUDA_FUSION_FUSED_CSC_E_DIV_U_SUM_H_

#include <torch/torch.h>
#include "bcast.h"
#include "graph.h"

namespace gs {
namespace impl {
namespace fusion {



void EDivUSumCSC(const std::string& op, const std::string& reduce,
             const BcastOff& bcast, const std::shared_ptr<CSC> csc,
             torch::Tensor ufeat, torch::Tensor efeat, torch::Tensor out,
             std::vector<torch::Tensor> out_aux);

}
}  // namespace impl
}  // namespace gs

#endif