#include <curand_kernel.h>
#include <thrust/execution_policy.h>
#include "../atomic.h"
#include "../cuda_common.h"
#include "../utils.h"
#include "slice_sampling.h"

namespace gs {
namespace impl {
namespace fusion {

/////////////////// FusedCSCColSlicingSamplingCUDA //////////////////////
template <typename IdType>
__global__ void _FusedSliceSampleSubIndicesReplaceKernel(
    IdType* sub_indices, IdType* select_index, IdType* indptr, IdType* indices,
    IdType* sub_indptr, IdType* column_ids, int64_t size,
    const uint64_t random_seed) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
  curandStatePhilox4_32_10_t rng;
  curand_init(random_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (row < size) {
    IdType col = column_ids[row];
    IdType in_start = indptr[col];
    IdType out_start = sub_indptr[row];
    IdType degree = indptr[col + 1] - indptr[col];
    IdType fanout = sub_indptr[row + 1] - sub_indptr[row];
    IdType out_pos, in_pos;
    for (int idx = threadIdx.x; idx < fanout; idx += blockDim.x) {
      const IdType edge = curand(&rng) % degree;
      out_pos = out_start + idx;
      in_pos = in_start + edge;
      sub_indices[out_pos] = indices[in_pos];
      select_index[out_pos] = in_pos;
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType>
__global__ void _FusedSliceSampleSubIndicesKernel(
    IdType* sub_indices, IdType* select_index, IdType* indptr, IdType* indices,
    IdType* sub_indptr, IdType* column_ids, int64_t size,
    const uint64_t random_seed) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
  curandStatePhilox4_32_10_t rng;
  curand_init(random_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (row < size) {
    IdType col = column_ids[row];
    IdType in_start = indptr[col];
    IdType out_start = sub_indptr[row];
    IdType degree = indptr[col + 1] - indptr[col];
    IdType fanout = sub_indptr[row + 1] - sub_indptr[row];
    IdType out_pos, in_pos;
    if (degree <= fanout) {
      for (int idx = threadIdx.x; idx < degree; idx += blockDim.x) {
        out_pos = out_start + idx;
        in_pos = in_start + idx;
        sub_indices[out_pos] = indices[in_pos];
        select_index[out_pos] = in_pos;
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
      }
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
_FusedCSCColSlicingSampling(torch::Tensor indptr, torch::Tensor indices,
                            torch::Tensor column_ids, int64_t fanout,
                            bool replace) {
  int64_t num_items = column_ids.numel();
  auto Id_option = (indptr.is_pinned())
                       ? torch::dtype(indptr.dtype()).device(torch::kCUDA)
                       : indptr.options();

  // compute indptr
  auto sub_indptr = torch::empty(num_items + 1, Id_option);
  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(thrust::device, it(0), it(num_items),
                   [in = column_ids.data_ptr<IdType>(),
                    in_indptr = indptr.data_ptr<IdType>(),
                    out = sub_indptr.data_ptr<IdType>(), if_replace = replace,
                    num_fanout = fanout] __device__(int i) mutable {
                     IdType begin = in_indptr[in[i]];
                     IdType end = in_indptr[in[i] + 1];
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
  int n_edges = item_prefix[num_items];  // cpu
  auto sub_indices = torch::empty(n_edges, Id_option);
  auto select_index = torch::empty(n_edges, Id_option);

  const uint64_t random_seed = 7777;
  dim3 block(16, 32);
  dim3 grid((num_items + block.y - 1) / block.y);
  if (replace) {
    _FusedSliceSampleSubIndicesReplaceKernel<IdType><<<grid, block>>>(
        sub_indices.data_ptr<IdType>(), select_index.data_ptr<IdType>(),
        indptr.data_ptr<IdType>(), indices.data_ptr<IdType>(),
        sub_indptr.data_ptr<IdType>(), column_ids.data_ptr<IdType>(), num_items,
        random_seed);
  } else {
    _FusedSliceSampleSubIndicesKernel<IdType><<<grid, block>>>(
        sub_indices.data_ptr<IdType>(), select_index.data_ptr<IdType>(),
        indptr.data_ptr<IdType>(), indices.data_ptr<IdType>(),
        sub_indptr.data_ptr<IdType>(), column_ids.data_ptr<IdType>(), num_items,
        random_seed);
  }

  return {sub_indptr, sub_indices, select_index};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
FusedCSCColSlicingSamplingCUDA(torch::Tensor indptr, torch::Tensor indices,
                               torch::Tensor column_ids, int64_t fanout,
                               bool replace) {
  return _FusedCSCColSlicingSampling<int64_t>(indptr, indices, column_ids,
                                              fanout, replace);
}

/////////////////// FusedCSCColSlicingSamplingOneKeepDimCUDA ////////////
// for fanout = 1

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
FusedCSCColSlicingSamplingOneKeepDimCUDA(torch::Tensor indptr,
                                         torch::Tensor indices,
                                         torch::Tensor column_ids) {
  // get subptr
  int64_t num_items = column_ids.numel();
  auto sub_indptr = torch::ones(num_items + 1, indptr.options());
  thrust::device_ptr<int64_t> item_prefix(
      static_cast<int64_t*>(sub_indptr.data_ptr<int64_t>()));
  cub_exclusiveSum<int64_t>(thrust::raw_pointer_cast(item_prefix),
                            num_items + 1);
  auto select_index = torch::empty(num_items, indices.options());
  // get subindices
  auto sub_indices = torch::empty(num_items, indices.options());
  using it = thrust::counting_iterator<int64_t>;
  thrust::for_each(
      thrust::device, it(0), it(num_items),
      [sub_indices_ptr = sub_indices.data_ptr<int64_t>(),
       indptr_ptr = indptr.data_ptr<int64_t>(),
       indices_ptr = indices.data_ptr<int64_t>(),
       sub_indptr_ptr = sub_indptr.data_ptr<int64_t>(),
       select_index_ptr = select_index.data_ptr<int64_t>(),
       column_ids_ptr =
           column_ids.data_ptr<int64_t>()] __device__(int i) mutable {
        const uint64_t random_seed = 7777777;
        curandState rng;
        curand_init(random_seed + i, 0, 0, &rng);
        int64_t col = column_ids_ptr[i];
        int64_t in_start = indptr_ptr[col];
        int64_t out_start = sub_indptr_ptr[i];
        int64_t degree = indptr_ptr[col + 1] - indptr_ptr[col];
        if (degree == 0) {
          sub_indices_ptr[out_start] = -1;
          select_index_ptr[out_start] = -1;
        } else {
          // Sequential Sampling
          // const int64_t edge = tid % degree;
          // Random Sampling
          const int64_t edge = curand(&rng) % degree;
          sub_indices_ptr[out_start] = indices_ptr[in_start + edge];
          select_index_ptr[out_start] = in_start + edge;
        }
      });
  return {sub_indptr, sub_indices, select_index};
}
}  // namespace fusion
}  // namespace impl
}  // namespace gs