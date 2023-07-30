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

  if (sorted_indices.device().type() != torch::kCUDA &&
      !sorted_indices.is_pinned()) {
    sorted_indices = sorted_indices.pin_memory();
  }

  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  cub::DeviceSegmentedRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, indices.data_ptr<int64_t>(),
      sorted_indices.data_ptr<int64_t>(), num_items, num_segments,
      indptr.data_ptr<int64_t>(), indptr.data_ptr<int64_t>() + 1);

  CUDA_CALL(cudaMallocManaged(&d_temp_storage, temp_storage_bytes));

  cub::DeviceSegmentedRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, indices.data_ptr<int64_t>(),
      sorted_indices.data_ptr<int64_t>(), num_items, num_segments,
      indptr.data_ptr<int64_t>(), indptr.data_ptr<int64_t>() + 1);

  CUDA_CALL(cudaFree(d_temp_storage));
  return sorted_indices;
}

}  // namespace impl
}  // namespace gs