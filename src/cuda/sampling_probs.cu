#include <curand_kernel.h>
#include <thrust/execution_policy.h>
#include "atomic.h"
#include "cuda_common.h"
#include "graph_ops.h"
#include "utils.h"
#include "warpselect/WarpSelect.cuh"

namespace gs {
namespace impl {

// from
// https://github.com/facebookresearch/faiss/tree/151e3d7be54aec844b6328dc3e7dd0b83fcfa5bc/faiss/gpu/utils/warpselect
// warp Q to thread Q:
// 1, 1
// 32, 2
// 64, 3
// 128, 3
// 256, 4
// 512, 8
// 1024, 8
// 2048, 8
template <typename IdType, typename FloatType, int TILE_SIZE, int BLOCK_WARPS,
          int NumWarpQ, int NumThreadQ, bool WITH_COO>
__global__ void _CSRRowWiseSampleKernel(
    const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
    const IdType *const in_indptr, const IdType *const in_indices,
    const FloatType *const prob, const IdType *const out_ptr,
    IdType *const out_indices, IdType *const selected_index,
    IdType *const coo_row) {
  // we assign one warp per row
  assert(num_picks <= NumWarpQ);
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == BLOCK_WARPS);
  int laneid = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.y;
  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x,
              threadIdx.y * WARP_SIZE + threadIdx.x, 0, &rng);

  __shared__ IdType warpselect_out_index[WARP_SIZE * BLOCK_WARPS];
  IdType *warpselect_out_index_per_warp =
      warpselect_out_index + warp_id * WARP_SIZE;

  // init warpselect
  gs::impl::warpselect::WarpSelect<FloatType, IdType,
                                   true,  // produce largest values
                                   gs::impl::warpselect::Comparator<FloatType>,
                                   NumWarpQ, NumThreadQ,
                                   WARP_SIZE * BLOCK_WARPS>
      heap(gs::impl::warpselect::_Limits<FloatType>::getMin(), -1, num_picks);

  int64_t out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  const int64_t last_row =
      MIN(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  while (out_row < last_row) {
    const IdType row = out_row;
    const IdType in_row_start = in_indptr[row];
    const IdType deg = in_indptr[row + 1] - in_row_start;
    const IdType out_row_start = out_ptr[out_row];
    // A-Res value needs to be calculated only if deg is greater than num_picks
    // in weighted rowwise sampling without replacement
    if (deg > num_picks) {
      heap.reset();
      int limit = gs::impl::warpselect::roundDown(deg, WARP_SIZE);
      IdType i = laneid;

      for (; i < limit; i += WARP_SIZE) {
        FloatType item_prob = prob[in_row_start + i];
        FloatType ares_prob = __powf(curand_uniform(&rng), 1.0f / item_prob);
        heap.add(ares_prob, i);
      }

      if (i < deg) {
        FloatType item_prob = prob[in_row_start + i];
        FloatType ares_prob = __powf(curand_uniform(&rng), 1.0f / item_prob);
        heap.addThreadQ(ares_prob, i);
        i += WARP_SIZE;
      }

      heap.reduce();
      heap.writeOutV(warpselect_out_index_per_warp, num_picks);

      for (int idx = laneid; idx < num_picks; idx += WARP_SIZE) {
        const IdType out_idx = out_row_start + idx;
        const IdType in_idx = in_row_start + warpselect_out_index_per_warp[idx];
        out_indices[out_idx] = in_indices[in_idx];
        selected_index[out_idx] = in_idx;
        if (WITH_COO) {
          coo_row[out_idx] = out_row;
        }
      }
    } else {
      for (int idx = laneid; idx < deg; idx += WARP_SIZE) {
        // get in and out index
        const IdType out_idx = out_row_start + idx;
        const IdType in_idx = in_row_start + idx;
        // copy permutation over
        out_indices[out_idx] = in_indices[in_idx];
        selected_index[out_idx] = in_idx;
        if (WITH_COO) {
          coo_row[out_idx] = out_row;
        }
      }
    }

    out_row += BLOCK_WARPS;
  }
}

template <typename IdType, typename FloatType, int TILE_SIZE, int BLOCK_WARPS,
          bool WITH_COO>
__global__ void _CSRRowWiseSampleReplaceKernel(
    const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
    const IdType *const in_indptr, const IdType *const in_indices,
    const FloatType *const prob, const IdType *const sub_indptr,
    const IdType *const cdf_indptr, FloatType *const cdf,
    IdType *const out_indices, IdType *const selected_index,
    IdType *const coo_row) {
  // we assign one warp per row
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == BLOCK_WARPS);
  int warp_id = threadIdx.y;
  int laneid = threadIdx.x;
  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x,
              threadIdx.y * BLOCK_WARPS + threadIdx.x, 0, &rng);

  typedef cub::WarpScan<FloatType> WarpScan;
  __shared__ typename WarpScan::TempStorage temp_storage[BLOCK_WARPS];

  int64_t out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  const int64_t last_row =
      MIN(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  while (out_row < last_row) {
    const IdType row = out_row;
    const IdType in_row_start = in_indptr[row];
    const IdType deg = in_indptr[row + 1] - in_row_start;
    const IdType out_row_start = sub_indptr[out_row];
    const IdType cdf_row_start = cdf_indptr[out_row];
    const FloatType MIN_THREAD_DATA = static_cast<FloatType>(0.0f);

    if (deg > 0) {
      IdType max_iter = (1 + (deg - 1) / WARP_SIZE) * WARP_SIZE;
      // Have the block iterate over segments of items

      FloatType warp_aggregate = static_cast<FloatType>(0.0f);
      for (int idx = laneid; idx < max_iter; idx += WARP_SIZE) {
        FloatType thread_data =
            idx < deg ? prob[in_row_start + idx] : MIN_THREAD_DATA;
        if (laneid == 0) thread_data += warp_aggregate;
        thread_data = max(thread_data, MIN_THREAD_DATA);

        WarpScan(temp_storage[warp_id])
            .InclusiveSum(thread_data, thread_data, warp_aggregate);
        __syncwarp();
        // Store scanned items to cdf array
        if (idx < deg) {
          cdf[cdf_row_start + idx] = thread_data;
        }
      }
      __syncwarp();

      for (int idx = laneid; idx < num_picks; idx += WARP_SIZE) {
        // get random value
        FloatType sum = cdf[cdf_row_start + deg - 1];
        FloatType rand = static_cast<FloatType>(curand_uniform(&rng) * sum);
        // get the offset of the first value within cdf array which is greater
        // than random value.
        IdType item = cub::UpperBound<FloatType *, IdType, FloatType>(
            &cdf[cdf_row_start], deg, rand);
        item = MIN(item, deg - 1);
        // get in and out index
        const IdType in_idx = in_row_start + item;
        const IdType out_idx = out_row_start + idx;
        // copy permutation over
        out_indices[out_idx] = in_indices[in_idx];
        selected_index[out_idx] = in_idx;
        if (WITH_COO) {
          coo_row[out_idx] = out_row;
        }
      }
    }
    out_row += BLOCK_WARPS;
  }
}

template <typename IdType, typename FloatType, bool WITH_COO>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
_CSCColSamplingProbs(torch::Tensor indptr, torch::Tensor indices,
                     torch::Tensor probs, int64_t num_picks, bool replace) {
  int64_t num_items = indptr.numel() - 1;
  torch::Tensor sub_indptr = torch::empty_like(indptr);

  // temp_indptr is used to measure how much extra space we need.
  torch::Tensor temp_indptr = torch::empty_like(indptr);

  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(thrust::device, it(0), it(num_items),
                   [in_indptr = indptr.data_ptr<IdType>(),
                    sub_ptr = sub_indptr.data_ptr<IdType>(),
                    tmp_ptr = temp_indptr.data_ptr<IdType>(), replace,
                    num_picks] __device__(IdType i) mutable {
                     IdType begin = in_indptr[i];
                     IdType end = in_indptr[i + 1];
                     IdType deg = end - begin;
                     if (replace) {
                       sub_ptr[i] = deg == 0 ? 0 : num_picks;
                       tmp_ptr[i] = deg;
                     } else {
                       sub_ptr[i] = MIN(deg, num_picks);
                       tmp_ptr[i] = deg > num_picks ? deg : 0;
                     }
                   });
  cub_exclusiveSum<IdType>(sub_indptr.data_ptr<IdType>(), num_items + 1);
  cub_exclusiveSum<IdType>(temp_indptr.data_ptr<IdType>(), num_items + 1);

  thrust::device_ptr<IdType> sub_prefix(
      static_cast<IdType *>(sub_indptr.data_ptr<IdType>()));
  thrust::device_ptr<IdType> temp_prefix(
      static_cast<IdType *>(temp_indptr.data_ptr<IdType>()));
  IdType nnz = sub_prefix[num_items];
  IdType temp_size = temp_prefix[num_items];

  torch::Tensor temp = torch::empty(temp_size, probs.options());
  torch::Tensor sub_indices = torch::empty(nnz, indices.options());
  torch::Tensor select_index = torch::empty(nnz, indices.options());

  torch::Tensor coo_col;
  IdType *coo_col_ptr;
  if (WITH_COO) {
    coo_col = torch::empty(nnz, indices.options());
    coo_col_ptr = coo_col.data_ptr<IdType>();
  } else {
    coo_col = torch::Tensor();
    coo_col_ptr = nullptr;
  }

  const uint64_t random_seed = 7777;
  constexpr int BLOCK_SIZE = 128;
  constexpr int BLOCK_WARPS = BLOCK_SIZE / WARP_SIZE;
  constexpr int TILE_SIZE = 16;
  const dim3 block(WARP_SIZE, BLOCK_WARPS);
  const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);
  if (replace) {
    _CSRRowWiseSampleReplaceKernel<IdType, FloatType, TILE_SIZE, BLOCK_WARPS,
                                   WITH_COO><<<grid, block>>>(
        random_seed, num_picks, num_items, indptr.data_ptr<IdType>(),
        indices.data_ptr<IdType>(), probs.data_ptr<FloatType>(),
        sub_indptr.data_ptr<IdType>(), temp_indptr.data_ptr<IdType>(),
        temp.data_ptr<FloatType>(), sub_indices.data_ptr<IdType>(),
        select_index.data_ptr<IdType>(), coo_col_ptr);
  } else {
    _CSRRowWiseSampleKernel<IdType, FloatType, TILE_SIZE, BLOCK_WARPS, 32, 2,
                            WITH_COO><<<grid, block>>>(
        random_seed, num_picks, num_items, indptr.data_ptr<IdType>(),
        indices.data_ptr<IdType>(), probs.data_ptr<FloatType>(),
        sub_indptr.data_ptr<IdType>(), sub_indices.data_ptr<IdType>(),
        select_index.data_ptr<IdType>(), coo_col_ptr);
  }

  return {sub_indptr, coo_col, sub_indices, select_index};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
CSCColSamplingProbsCUDA(torch::Tensor indptr, torch::Tensor indices,
                        torch::Tensor probs, int64_t fanout, bool replace,
                        bool with_coo) {
  torch::Tensor out_indptr, out_coo_col, out_indices, out_selected_index;
  if (with_coo)
    std::tie(out_indptr, out_coo_col, out_indices, out_selected_index) =
        _CSCColSamplingProbs<int64_t, float, true>(indptr, indices, probs,
                                                   fanout, replace);
  else
    std::tie(out_indptr, out_coo_col, out_indices, out_selected_index) =
        _CSCColSamplingProbs<int64_t, float, false>(indptr, indices, probs,
                                                    fanout, replace);
  return {out_indptr, out_coo_col, out_indices, out_selected_index};
}
}  // namespace impl
}  // namespace gs