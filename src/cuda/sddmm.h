#ifndef GS_CUDA_SDDMM_H_
#define GS_CUDA_SDDMM_H_

#include "../bcast.h"
#include "../graph_storage.h"

namespace gs {
namespace impl {
void SDDMMCSC(const std::string& op, const BcastOff& bcast,
              std::shared_ptr<CSC> csc, torch::optional<torch::Tensor> n_ids,
              torch::Tensor lhs, torch::Tensor rhs, torch::Tensor out,
              int lhs_target, int rhs_target);

void SDDMMCOO(const std::string& op, const BcastOff& bcast,
              std::shared_ptr<COO> coo, torch::Tensor lhs, torch::Tensor rhs,
              torch::Tensor out, int lhs_target, int rhs_target);

}  // namespace impl
}  // namespace gs

#endif