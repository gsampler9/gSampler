#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "../atomic.h"
#include "../cuda_common.h"
#include "../utils.h"

#include "batch_ops.h"

namespace gs {
namespace impl {
namespace batch {
template <typename IdType>
__global__ void _BatchTensorEncodingKernel(IdType* out_data, IdType* in_data,
                                           int64_t* in_offsets,
                                           int64_t num_batch,
                                           int64_t seg_size) {
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  const int stride_y = blockDim.y * gridDim.y;
  while (ty < num_batch) {
    int64_t in_startoff = in_offsets[ty];
    int64_t in_endoff = in_offsets[ty + 1];
    int size = in_endoff - in_startoff;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride_x = blockDim.x * gridDim.x;
    for (; tx < size; tx += stride_x) {
      out_data[in_startoff + tx] = ty * seg_size + in_data[in_startoff + tx];
    }
    ty += stride_y;
  }
}

/*!
 * \brief Search for the insertion positions for needle in the hay.
 *
 * The hay is a list of sorted elements and the result is the insertion position
 * of each needle so that the insertion still gives sorted order.
 *
 * It essentially perform binary search to find upper bound for each needle
 * elements.
 *
 * For example:
 * hay = [0, 0, 1, 2, 2]
 * (implicit) needle = [0, 1, 2, 3]
 * then,
 * out = [2, 3, 5, 5]
 *
 * hay = [0, 0, 1, 3, 3]
 * (implicit) needle = [0, 1, 2, 3]
 * then,
 * out = [2, 3, 3, 5]
 */
template <typename IdType>
__global__ void _SortedSearchKernelUpperBound(const IdType* hay,
                                              int64_t hay_size,
                                              int64_t num_needles,
                                              IdType* pos) {
  IdType tx = static_cast<IdType>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < num_needles) {
    pos[tx] = cub::UpperBound(hay, hay_size, tx);
    tx += stride_x;
  }
}

torch::Tensor BatchEncodeCUDA(torch::Tensor data_tensor, torch::Tensor data_ptr,
                              int64_t encoding_size) {
  torch::Tensor out_data = torch::empty_like(data_tensor);
  int64_t numel = data_ptr.numel(), num_split = data_ptr.numel() - 1;
  torch::Tensor data_sizes =
      data_ptr.slice(0, 1, numel) - data_ptr.slice(0, 0, numel - 1);
  int64_t max_len = data_sizes.max().item<int64_t>();

  const int ntx = FindNumThreads(max_len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (max_len + ntx - 1) / ntx;
  const int nby = FindNumBlocks<'y'>((num_split + nty - 1) / nty);
  dim3 grid(nbx, nby);
  dim3 block(ntx, nty);
  CUDA_KERNEL_CALL((_BatchTensorEncodingKernel<int64_t>), grid, block,
                   out_data.data_ptr<int64_t>(),
                   data_tensor.data_ptr<int64_t>(),
                   data_ptr.data_ptr<int64_t>(), num_split, encoding_size);
  return out_data;
}

torch::Tensor BatchDecodeCUDA(torch::Tensor in_data, int64_t encoding_size) {
  torch::Tensor out_data = torch::empty_like(in_data);

  using it = thrust::counting_iterator<int64_t>;
  thrust::for_each(
      it(0), it(in_data.numel()),
      [in = in_data.data_ptr<int64_t>(), out = out_data.data_ptr<int64_t>(),
       size = encoding_size] __device__(int64_t i) mutable {
        out[i] = in[i] - (in[i] / size) * size;
      });
  return out_data;
}

std::tuple<torch::Tensor, torch::Tensor> GetBatchOffsets(
    torch::Tensor data_tensor, int64_t num_batches, int64_t encoding_size) {
  auto data_mask = torch::empty_like(data_tensor);
  auto decoded_data = torch::empty_like(data_tensor);
  using it = thrust::counting_iterator<int64_t>;
  thrust::for_each(it(0), it(data_tensor.numel()),
                   [in = data_tensor.data_ptr<int64_t>(),
                    out = decoded_data.data_ptr<int64_t>(),
                    mask = data_mask.data_ptr<int64_t>(),
                    size = encoding_size] __device__(int64_t i) mutable {
                     mask[i] = in[i] / size;
                     out[i] = in[i] - mask[i] * size;
                   });

  auto dataptr = torch::zeros(num_batches + 1,
                              torch::dtype(torch::kInt64).device(torch::kCUDA));

  dim3 block(128);
  dim3 grid((num_batches + block.x - 1) / block.x);
  CUDA_KERNEL_CALL((_SortedSearchKernelUpperBound<int64_t>), grid, block,
                   data_mask.data_ptr<int64_t>(), data_mask.numel(),
                   num_batches, dataptr.data_ptr<int64_t>() + 1);
  return {dataptr, decoded_data};
}
}  // namespace batch
}  // namespace impl
}  // namespace gs