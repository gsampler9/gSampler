#include "tensor_ops.h"

#include "cuda_common.h"
#include "utils.h"

namespace gs {
namespace impl {
template <typename DType, typename IdType>
__global__ void IndexSelectSingleKernel(const DType* array, const IdType* index,
                                        const int64_t length,
                                        const int64_t arr_len, DType* out) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    assert(index[tx] >= 0 && index[tx] < arr_len);
    out[tx] = array[index[tx]];
    tx += stride_x;
  }
}

template <typename DType, typename IdType>
torch::Tensor _IndexSelectCPUFromGPU(torch::Tensor array, torch::Tensor index) {
  const DType* array_data = array.data_ptr<DType>();
  const IdType* idx_data = index.data_ptr<IdType>();
  const int64_t arr_len = array.numel();
  const int64_t len = index.numel();

  CHECK(array.is_pinned());
  CHECK_EQ(index.device().type(), torch::kCUDA);

  auto ret =
      torch::empty(len, torch::dtype(array.dtype()).device(index.device()));
  if (len == 0) return ret;
  DType* ret_data = ret.data_ptr<DType>();

  const int nt = FindNumThreads(len);
  const int nb = (len + nt - 1) / nt;
  CUDA_KERNEL_CALL((IndexSelectSingleKernel<DType, IdType>), nb, nt, array_data,
                   idx_data, len, arr_len, ret_data);
  return ret;
}

torch::Tensor IndexSelectCPUFromGPU(torch::Tensor array, torch::Tensor index) {
  return _IndexSelectCPUFromGPU<float, int64_t>(array, index);
}
}  // namespace impl
}  // namespace gs