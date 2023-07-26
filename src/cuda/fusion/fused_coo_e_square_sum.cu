#include <thrust/execution_policy.h>
#include "../atomic.h"
#include "../cuda_common.h"
#include "../utils.h"
#include "fused_coo_e_square_sum.h"
#include "../macro.h"

namespace gs {
namespace impl {

namespace fusion {

void ESquareSumCOO(const std::string& op, const std::string& reduce,
             const BcastOff& bcast, const std::shared_ptr<COO> coo,
             torch::Tensor ufeat, torch::Tensor efeat, torch::Tensor out_sum,
             std::vector<torch::Tensor> out_aux) {
  int64_t num_edge = coo->row.numel();
  using it = thrust::counting_iterator<int64_t>;
  thrust::for_each(
      it(0), it(num_edge),
      [d_row = coo->row.data_ptr<int64_t>(), d_col = coo->col.data_ptr<int64_t>(),
       d_in = efeat.data_ptr<float>(),
       d_sum = out_sum.data_ptr<float>()] __device__(int64_t i) {
        AtomicAdd(d_sum + d_row[i], d_in[i]*d_in[i]);
      });
}



    }  // namespace fusion
}  // namespace impl
}  // namespace gs