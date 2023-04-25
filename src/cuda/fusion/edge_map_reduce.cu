#include <thrust/execution_policy.h>
#include "../atomic.h"
#include "../cuda_common.h"
#include "../utils.h"
#include "edge_map_reduce.h"

namespace gs {
namespace impl {
namespace fusion {
void COOEDivUSum(torch::Tensor row, torch::Tensor col, torch::Tensor in_data,
                 torch::Tensor divisor, torch::Tensor out_data,
                 torch::Tensor out_sum) {
  int64_t num_edge = row.numel();
  using it = thrust::counting_iterator<int64_t>;
  thrust::for_each(
      it(0), it(num_edge),
      [d_row = row.data_ptr<int64_t>(), d_col = col.data_ptr<int64_t>(),
       d_in = in_data.data_ptr<float>(), d_divisor = divisor.data_ptr<float>(),
       d_out = out_data.data_ptr<float>(),
       d_sum = out_sum.data_ptr<float>()] __device__(int64_t i) {
        d_out[i] = d_in[i] / d_divisor[d_row[i]];
        AtomicAdd(d_sum + d_col[i], d_out[i]);
      });
}
}  // namespace fusion
}  // namespace impl
}  // namespace gs