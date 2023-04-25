#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "atomic.h"
#include "cuda_common.h"
#include "tensor_ops.h"
#include "utils.h"

namespace gs {
namespace impl {

template <typename IdType>
__global__ void _BatchSplitKernel(
    IdType* __restrict__ data_tensor, IdType* __restrict__ data_ptr,
    IdType* __restrict__ data_key, IdType** __restrict__ out_data,
    IdType** __restrict__ out_data_ptrs,
    IdType* __restrict__ out_data_ptrs_order_by_segments, int64_t num_segments,
    int64_t num_items) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int64_t index = tid; index < num_items;
       index += gridDim.x * blockDim.x) {
    int64_t batch_id = data_key[index];
    int64_t local_index = index - data_ptr[batch_id];
    int64_t segment_id =
        cub::UpperBound<IdType*, int64_t, IdType>(
            out_data_ptrs_order_by_segments + batch_id * (num_segments + 1),
            num_segments + 1, local_index) -
        1;

    IdType output_index = out_data_ptrs[segment_id][batch_id] + local_index -
                          (out_data_ptrs_order_by_segments +
                           batch_id * (num_segments + 1))[segment_id];
    out_data[segment_id][output_index] = data_tensor[index];
  }
}

// The opposite of what BatchConcat does
template <typename IdType>
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>> _BatchSplit(
    torch::Tensor data_tensor, torch::Tensor data_ptr_tensor,
    torch::Tensor data_key_tensor,
    const std::vector<torch::Tensor>& out_tensors,
    const std::vector<torch::Tensor>& out_ptr_tensors) {
  int64_t num_items = data_tensor.numel();
  int64_t num_segments = out_tensors.size();
  int64_t num_batchs = out_ptr_tensors[0].numel() - 1;

  thrust::host_vector<IdType*> h_out_tensors(num_segments);
  thrust::host_vector<IdType*> h_out_ptr_tensors(num_segments);

  for (int i = 0; i < num_segments; i++) {
    h_out_tensors[i] = out_tensors[i].data_ptr<IdType>();
    h_out_ptr_tensors[i] = out_ptr_tensors[i].data_ptr<IdType>();
  }

  IdType** d_out_tensors =
      reinterpret_cast<IdType**>(c10::cuda::CUDACachingAllocator::raw_alloc(
          sizeof(IdType*) * num_segments));
  cudaMemcpy(d_out_tensors, h_out_tensors.data(),
             sizeof(IdType*) * num_segments, cudaMemcpyHostToDevice);

  IdType** d_out_ptr_tensors =
      reinterpret_cast<IdType**>(c10::cuda::CUDACachingAllocator::raw_alloc(
          sizeof(IdType*) * num_segments));
  cudaMemcpy(d_out_ptr_tensors, h_out_ptr_tensors.data(),
             sizeof(IdType*) * num_segments, cudaMemcpyHostToDevice);

  torch::Tensor out_ptr_managed_by_segments =
      torch::empty((num_segments + 1) * num_batchs, data_ptr_tensor.options());

  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(it(0), it(num_batchs),
                   [in = d_out_ptr_tensors,
                    out = out_ptr_managed_by_segments.data_ptr<IdType>(),
                    num_segments] __device__(IdType i) mutable {
                     IdType acc = 0;
                     out[i * (num_segments + 1)] = acc;
                     for (int j = 0; j < num_segments; j++) {
                       acc += (in[j][i + 1] - in[j][i]);
                       out[j + 1 + i * (num_segments + 1)] = acc;
                     }
                   });

  constexpr int BLOCK_SIZE = 256;
  int num_blocks = (num_items + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 blocks(BLOCK_SIZE);
  dim3 grid(num_blocks);

  _BatchSplitKernel<IdType><<<grid, blocks>>>(
      data_tensor.data_ptr<IdType>(), data_ptr_tensor.data_ptr<IdType>(),
      data_key_tensor.data_ptr<IdType>(), d_out_tensors, d_out_ptr_tensors,
      out_ptr_managed_by_segments.data_ptr<IdType>(), num_segments, num_items);

  c10::cuda::CUDACachingAllocator::raw_delete(d_out_tensors);
  c10::cuda::CUDACachingAllocator::raw_delete(d_out_ptr_tensors);

  return {out_tensors, out_ptr_tensors};
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>>
BatchSplit2CUDA(torch::Tensor data_tensor, torch::Tensor data_ptr_tensor,
                torch::Tensor data_key_tensor,
                const std::vector<torch::Tensor>& out_tensors,
                const std::vector<torch::Tensor>& out_ptr_tensors) {
  return _BatchSplit<int64_t>(data_tensor, data_ptr_tensor, data_key_tensor,
                              out_tensors, out_ptr_tensors);
}
}  // namespace impl
}  // namespace gs