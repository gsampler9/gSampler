#ifndef GS_CUDA_SPMM_H_
#define GS_CUDA_SPMM_H_

#include "../bcast.h"
#include "../graph_storage.h"

namespace gs {
namespace impl {
void SpMMCSC(const std::string& op, const std::string& reduce,
             const BcastOff& bcast, const std::shared_ptr<CSC> csc,
             torch::Tensor ufeat, torch::Tensor efeat, torch::Tensor out,
             std::vector<torch::Tensor> out_aux);

void SpMMCOO(const std::string& op, const std::string& reduce,
             const BcastOff& bcast, const std::shared_ptr<COO> coo,
             torch::Tensor ufeat, torch::Tensor efeat, torch::Tensor out,
             int64_t u_target, std::vector<torch::Tensor> out_aux);
}  // namespace impl
}  // namespace gs

#endif