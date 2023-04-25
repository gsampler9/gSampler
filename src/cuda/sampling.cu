#include <curand_kernel.h>
#include <thrust/execution_policy.h>
#include "atomic.h"
#include "cuda_common.h"
#include "graph_ops.h"
#include "utils.h"

namespace gs {
namespace impl {

/////////////////////////// CSCColSamplingCUDA /////////////////////////////////
template <typename IdType, bool WITH_COO>
__global__ void _SampleSubIndicesReplaceKernel(IdType* sub_indices,
                                               IdType* select_index,
                                               IdType* coo_row, IdType* indptr,
                                               IdType* indices,
                                               IdType* sub_indptr, int64_t size,
                                               const uint64_t random_seed) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
  curandStatePhilox4_32_10_t rng;
  curand_init(random_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (row < size) {
    IdType in_start = indptr[row];
    IdType out_start = sub_indptr[row];
    IdType degree = indptr[row + 1] - in_start;
    IdType fanout = sub_indptr[row + 1] - out_start;
    IdType out_pos, in_pos;
    for (int idx = threadIdx.x; idx < fanout; idx += blockDim.x) {
      const IdType edge = curand(&rng) % degree;
      out_pos = out_start + idx;
      in_pos = in_start + edge;
      sub_indices[out_pos] = indices[in_pos];
      select_index[out_pos] = in_pos;
      if (WITH_COO) {
        coo_row[out_pos] = row;
      }
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType, bool WITH_COO>
__global__ void _SampleSubIndicesKernel(IdType* sub_indices,
                                        IdType* select_index, IdType* coo_row,
                                        IdType* indptr, IdType* indices,
                                        IdType* sub_indptr, int64_t size,
                                        const uint64_t random_seed) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
  curandStatePhilox4_32_10_t rng;
  curand_init(random_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (row < size) {
    IdType in_start = indptr[row];
    IdType out_start = sub_indptr[row];
    IdType degree = indptr[row + 1] - in_start;
    IdType fanout = sub_indptr[row + 1] - out_start;
    IdType out_pos, in_pos;
    if (degree <= fanout) {
      for (int idx = threadIdx.x; idx < degree; idx += blockDim.x) {
        out_pos = out_start + idx;
        in_pos = in_start + idx;
        sub_indices[out_pos] = indices[in_pos];
        select_index[out_pos] = in_pos;
        if (WITH_COO) {
          coo_row[out_pos] = row;
        }
      }
    } else {
      // reservoir algorithm
      for (int idx = threadIdx.x; idx < fanout; idx += blockDim.x) {
        sub_indices[out_start + idx] = idx;
      }
      __syncthreads();

      for (int idx = fanout + threadIdx.x; idx < degree; idx += blockDim.x) {
        const int num = curand(&rng) % (idx + 1);
        if (num < fanout) {
          AtomicMax(sub_indices + out_start + num, IdType(idx));
        }
      }
      __syncthreads();

      for (int idx = threadIdx.x; idx < fanout; idx += blockDim.x) {
        out_pos = out_start + idx;
        const IdType perm_idx = in_start + sub_indices[out_pos];
        sub_indices[out_pos] = indices[perm_idx];
        select_index[out_pos] = perm_idx;
        if (WITH_COO) {
          coo_row[out_pos] = row;
        }
      }
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType, bool WITH_COO>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
_CSCColSampling(torch::Tensor indptr, torch::Tensor indices, int64_t fanout,
                bool replace) {
  int64_t num_items = indptr.numel() - 1;
  auto sub_indptr = torch::empty(num_items + 1, indptr.options());

  // compute indptr
  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(thrust::device, it(0), it(num_items),
                   [in_indptr = indptr.data_ptr<IdType>(),
                    out = sub_indptr.data_ptr<IdType>(), if_replace = replace,
                    num_fanout = fanout] __device__(int i) mutable {
                     IdType begin = in_indptr[i];
                     IdType end = in_indptr[i + 1];
                     if (if_replace) {
                       out[i] = (end - begin) == 0 ? 0 : num_fanout;
                     } else {
                       out[i] = min(end - begin, num_fanout);
                     }
                   });
  cub_exclusiveSum<IdType>(sub_indptr.data_ptr<IdType>(), num_items + 1);

  // compute indices
  thrust::device_ptr<IdType> item_prefix(
      static_cast<IdType*>(sub_indptr.data_ptr<IdType>()));
  int nnz = item_prefix[num_items];  // cpu
  auto sub_indices = torch::empty(nnz, indices.options());
  auto select_index = torch::empty(nnz, indices.options());

  torch::Tensor coo_col;
  IdType* coo_col_ptr;
  if (WITH_COO) {
    coo_col = torch::empty(nnz, indices.options());
    coo_col_ptr = coo_col.data_ptr<IdType>();
  } else {
    coo_col = torch::Tensor();
    coo_col_ptr = nullptr;
  }

  const uint64_t random_seed = 7777;
  dim3 block(16, 32);
  dim3 grid((num_items + block.y - 1) / block.y);
  if (replace) {
    _SampleSubIndicesReplaceKernel<IdType, WITH_COO><<<grid, block>>>(
        sub_indices.data_ptr<IdType>(), select_index.data_ptr<IdType>(),
        coo_col_ptr, indptr.data_ptr<IdType>(), indices.data_ptr<IdType>(),
        sub_indptr.data_ptr<IdType>(), num_items, random_seed);
  } else {
    _SampleSubIndicesKernel<IdType, WITH_COO><<<grid, block>>>(
        sub_indices.data_ptr<IdType>(), select_index.data_ptr<IdType>(),
        coo_col_ptr, indptr.data_ptr<IdType>(), indices.data_ptr<IdType>(),
        sub_indptr.data_ptr<IdType>(), num_items, random_seed);
  }

  return {sub_indptr, coo_col, sub_indices, select_index};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
CSCColSamplingCUDA(torch::Tensor indptr, torch::Tensor indices, int64_t fanout,
                   bool replace, bool with_coo) {
  torch::Tensor out_indptr, out_coo_col, out_indices, out_selected_index;
  if (with_coo)
    std::tie(out_indptr, out_coo_col, out_indices, out_selected_index) =
        _CSCColSampling<int64_t, true>(indptr, indices, fanout, replace);
  else
    std::tie(out_indptr, out_coo_col, out_indices, out_selected_index) =
        _CSCColSampling<int64_t, false>(indptr, indices, fanout, replace);
  return {out_indptr, out_coo_col, out_indices, out_selected_index};
}

}  // namespace impl
}  // namespace gs