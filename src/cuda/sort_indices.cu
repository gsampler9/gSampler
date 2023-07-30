#include <c10/cuda/CUDACachingAllocator.h>
#include <cub/cub.cuh>
#include "atomic.h"
#include "cuda_common.h"
#include "tensor_ops.h"
#include "utils.h"

namespace gs {
namespace impl {

torch::Tensor SortIndicesCUDA(torch::Tensor indptr, torch::Tensor indices) {
  int64_t num_segments = indptr.numel() - 1;
  int64_t num_items = indices.numel();

  torch::Tensor sorted_indices = torch::empty_like(indices);

  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  cub::DeviceSegmentedRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, indices.data_ptr<int64_t>(),
      sorted_indices.data_ptr<int64_t>(), num_items, num_segments,
      indptr.data_ptr<int64_t>(), indptr.data_ptr<int64_t>() + 1);

  d_temp_storage =
      c10::cuda::CUDACachingAllocator::raw_alloc(temp_storage_bytes);

  cub::DeviceSegmentedRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, indices.data_ptr<int64_t>(),
      sorted_indices.data_ptr<int64_t>(), num_items, num_segments,
      indptr.data_ptr<int64_t>(), indptr.data_ptr<int64_t>() + 1);
  c10::cuda::CUDACachingAllocator::raw_delete(d_temp_storage);
  return sorted_indices;
}

}  // namespace impl
}  // namespace gs