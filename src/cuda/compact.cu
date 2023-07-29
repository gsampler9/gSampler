#include "atomic.h"
#include "cuda_common.h"
#include "tensor_ops.h"
#include "utils.h"

namespace gs {
namespace impl {

// Compact
template <typename IdType>
std::tuple<torch::Tensor, torch::Tensor> TensorCompactCUDA(torch::Tensor data) {
  torch::Tensor unique_tensor = std::get<0>(torch::_unique2(data));
  int num_items = data.numel();
  int64_t map_size = unique_tensor.numel();
  torch::Tensor compact_tensor = torch::empty_like(data);

  // remap
  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(
      it(0), it(num_items),
      [in = data.data_ptr<IdType>(), map = unique_tensor.data_ptr<IdType>(),
       out = compact_tensor.data_ptr<IdType>(),
       map_size] __device__(IdType i) mutable {
        IdType index =
            cub::UpperBound<IdType*, int64_t, IdType>(map, map_size, in[i]);
        out[i] = index - 1;
      });

  return {unique_tensor, compact_tensor};
}

std::tuple<torch::Tensor, torch::Tensor> TensorCompact(torch::Tensor data) {
  return TensorCompactCUDA<int64_t>(data);
}

}  // namespace impl
}  // namespace gs