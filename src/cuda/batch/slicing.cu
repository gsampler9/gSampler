#include <thrust/execution_policy.h>
#include "../atomic.h"
#include "../cuda_common.h"
#include "../utils.h"
#include "batch_ops.h"

namespace gs {
namespace impl {
namespace batch {
///////////////////// indptr slicing with indices encoding /////////////////////
template <typename IdType, bool WITH_COO, bool Encoding>
__global__ void _BatchGetSubIndicesKernel(IdType* out_indices,
                                          IdType* select_index, IdType* out_row,
                                          IdType* indptr, IdType* indices,
                                          IdType* sub_indptr,
                                          IdType* column_ids, IdType* nid_ptr,
                                          int64_t encoding_size, int64_t size) {
  int64_t row_in_batch = blockIdx.x * blockDim.y + threadIdx.y;
  int64_t row = nid_ptr[blockIdx.y] + row_in_batch;
  while (row < nid_ptr[blockIdx.y + 1]) {
    IdType in_start = indptr[column_ids[row]];
    IdType out_start = sub_indptr[row];
    IdType n_edges = sub_indptr[row + 1] - out_start;
    for (int idx = threadIdx.x; idx < n_edges; idx += blockDim.x) {
      if (Encoding) {
        out_indices[out_start + idx] =
            indices[in_start + idx] + blockIdx.y * encoding_size;
      } else {
        out_indices[out_start + idx] = indices[in_start + idx];
      }
      select_index[out_start + idx] = in_start + idx;
      if (WITH_COO) {
        out_row[out_start + idx] = row;
      }
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType, bool WITH_COO, bool Encoding>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
_BatchOnIndptrSlicing(torch::Tensor indptr, torch::Tensor indices,
                      torch::Tensor column_ids, torch::Tensor nid_ptr,
                      int64_t encoding_size) {
  int64_t num_items = column_ids.numel();

  // compute indptr
  auto Id_option = (indptr.is_pinned())
                       ? torch::dtype(indptr.dtype()).device(torch::kCUDA)
                       : indptr.options();
  torch::Tensor sub_indptr = torch::empty(num_items + 1, Id_option);
  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(
      thrust::device, it(0), it(num_items),
      [in = column_ids.data_ptr<IdType>(),
       in_indptr = indptr.data_ptr<IdType>(),
       out = sub_indptr.data_ptr<IdType>()] __device__(int i) mutable {
        IdType begin = in_indptr[in[i]];
        IdType end = in_indptr[in[i] + 1];
        out[i] = end - begin;
      });
  cub_exclusiveSum<IdType>(sub_indptr.data_ptr<IdType>(), num_items + 1);

  // compute indices
  thrust::device_ptr<IdType> item_prefix(
      static_cast<IdType*>(sub_indptr.data_ptr<IdType>()));
  int nnz = item_prefix[num_items];  // cpu
  torch::Tensor sub_indices = torch::empty(nnz, Id_option);
  torch::Tensor select_index = torch::empty(nnz, Id_option);
  torch::Tensor coo_offsets = sub_indptr.index({nid_ptr});

  torch::Tensor coo_col;
  IdType* coo_col_ptr;
  if (WITH_COO) {
    coo_col = torch::empty(nnz, Id_option);
    coo_col_ptr = coo_col.data_ptr<IdType>();
  } else {
    coo_col = torch::Tensor();
    coo_col_ptr = nullptr;
  }

  int64_t batch_num = nid_ptr.numel() - 1;
  torch::Tensor batch_size =
      nid_ptr.slice(0, 1, batch_num + 1) - nid_ptr.slice(0, 0, batch_num);
  int64_t max_batch_size = batch_size.max().item<int64_t>();
  dim3 block(16, 32);
  dim3 grid((max_batch_size + block.y - 1) / block.y, batch_num);
  CUDA_KERNEL_CALL((_BatchGetSubIndicesKernel<IdType, WITH_COO, Encoding>),
                   grid, block, sub_indices.data_ptr<IdType>(),
                   select_index.data_ptr<IdType>(), coo_col_ptr,
                   indptr.data_ptr<IdType>(), indices.data_ptr<IdType>(),
                   sub_indptr.data_ptr<IdType>(), column_ids.data_ptr<IdType>(),
                   nid_ptr.data_ptr<IdType>(), encoding_size, num_items);
  return {sub_indptr, coo_col, sub_indices, select_index, coo_offsets};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
BatchOnIndptrSlicingCUDA(torch::Tensor indptr, torch::Tensor indices,
                         torch::Tensor column_ids, torch::Tensor nid_ptr,
                         bool with_coo, bool encoding, int64_t encoding_size) {
  if (with_coo) {
    if (encoding) {
      return _BatchOnIndptrSlicing<int64_t, true, true>(
          indptr, indices, column_ids, nid_ptr, encoding_size);
    } else {
      return _BatchOnIndptrSlicing<int64_t, true, false>(
          indptr, indices, column_ids, nid_ptr, encoding_size);
    }
  } else {
    if (encoding) {
      return _BatchOnIndptrSlicing<int64_t, false, true>(
          indptr, indices, column_ids, nid_ptr, encoding_size);
    } else {
      return _BatchOnIndptrSlicing<int64_t, false, false>(
          indptr, indices, column_ids, nid_ptr, encoding_size);
    }
  }
}
}  // namespace batch
}  // namespace impl
}  // namespace gs