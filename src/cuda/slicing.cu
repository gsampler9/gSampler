#include <thrust/execution_policy.h>
#include "atomic.h"
#include "cuda_common.h"
#include "graph_ops.h"
#include "utils.h"

namespace gs {
namespace impl {
inline __host__ __device__ int UpPower(int key) {
  int ret = 1 << static_cast<uint32_t>(std::log2(key) + 1);
  return ret;
}

__device__ inline uint32_t Hash32Shift(uint32_t key) {
  key = ~key + (key << 15);  // # key = (key << 15) - key - 1;
  key = key ^ (key >> 12);
  key = key + (key << 2);
  key = key ^ (key >> 4);
  key = key * 2057;  // key = (key + (key << 3)) + (key << 11);
  key = key ^ (key >> 16);
  return key;
}

__device__ inline uint64_t Hash64Shift(uint64_t key) {
  key = (~key) + (key << 21);  // key = (key << 21) - key - 1;
  key = key ^ (key >> 24);
  key = (key + (key << 3)) + (key << 8);  // key * 265
  key = key ^ (key >> 14);
  key = (key + (key << 2)) + (key << 4);  // key * 21
  key = key ^ (key >> 28);
  key = key + (key << 31);
  return key;
}

/**
 * @brief Used to judge whether a node is in a node set
 *
 * @tparam IdType
 */
template <typename IdType>
struct NodeQueryHashmap {
  __device__ inline NodeQueryHashmap(IdType* Kptr, IdType* Vptr, size_t numel)
      : kptr(Kptr), vptr(Vptr), capacity(numel){};

  __device__ inline void Insert(IdType key, IdType value) {
    uint32_t delta = 1;
    uint32_t pos = hash(key);
    IdType prev = AtomicCAS(&kptr[pos], kEmptyKey, key);

    while (prev != key and prev != kEmptyKey) {
      pos = hash(pos + delta);
      delta += 1;
      prev = AtomicCAS(&kptr[pos], kEmptyKey, key);
    }

    vptr[pos] = value;
  }

  __device__ inline IdType Query(IdType key) {
    uint32_t delta = 1;
    uint32_t pos = hash(key);

    while (true) {
      if (kptr[pos] == key) {
        return vptr[pos];
      }
      if (kptr[pos] == kEmptyKey) {
        return -1;
      }
      pos = hash(pos + delta);
      delta += 1;
    }

    return -1;
  }

  __device__ inline uint32_t hash(int32_t key) {
    return Hash32Shift(key) & (capacity - 1);
  }

  __device__ inline uint32_t hash(uint32_t key) {
    return Hash32Shift(key) & (capacity - 1);
  }

  __device__ inline uint32_t hash(int64_t key) {
    return static_cast<uint32_t>(Hash64Shift(key)) & (capacity - 1);
  }

  __device__ inline uint32_t hash(uint64_t key) {
    return static_cast<uint32_t>(Hash64Shift(key)) & (capacity - 1);
  }

  IdType kEmptyKey{-1};
  IdType* kptr;
  IdType* vptr;
  uint32_t capacity{0};
};

////////////////////////////// slicing on indptr (CSC) ///////////////////////
template <typename IdType, bool WITH_COO>
__global__ void _GetSubIndicesKernel(IdType* out_indices, IdType* select_index,
                                     IdType* out_row, IdType* indptr,
                                     IdType* indices, IdType* sub_indptr,
                                     IdType* seeds, int64_t num_items) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
  while (row < num_items) {
    IdType in_start = indptr[seeds[row]];
    IdType out_start = sub_indptr[row];
    IdType n_edges = sub_indptr[row + 1] - out_start;
    for (int idx = threadIdx.x; idx < n_edges; idx += blockDim.x) {
      out_indices[out_start + idx] = indices[in_start + idx];
      select_index[out_start + idx] = in_start + idx;
      if (WITH_COO) {
        out_row[out_start + idx] = row;
      }
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType, bool WITH_COO>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
_OnIndptrSlicing(torch::Tensor indptr, torch::Tensor indices,
                 torch::Tensor seeds) {
  int64_t num_items = seeds.numel();

  // compute indptr
  auto Id_option = (indptr.is_pinned())
                       ? torch::dtype(indptr.dtype()).device(torch::kCUDA)
                       : indptr.options();
  torch::Tensor sub_indptr = torch::empty(num_items + 1, Id_option);
  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(
      thrust::device, it(0), it(num_items),
      [in = seeds.data_ptr<IdType>(), in_indptr = indptr.data_ptr<IdType>(),
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

  torch::Tensor coo_col;
  IdType* coo_col_ptr;
  if (WITH_COO) {
    coo_col = torch::empty(nnz, Id_option);
    coo_col_ptr = coo_col.data_ptr<IdType>();
  } else {
    coo_col = torch::Tensor();
    coo_col_ptr = nullptr;
  }

  dim3 block(16, 32);
  dim3 grid((num_items + block.y - 1) / block.y);
  _GetSubIndicesKernel<IdType, WITH_COO><<<grid, block>>>(
      sub_indices.data_ptr<IdType>(), select_index.data_ptr<IdType>(),
      coo_col_ptr, indptr.data_ptr<IdType>(), indices.data_ptr<IdType>(),
      sub_indptr.data_ptr<IdType>(), seeds.data_ptr<IdType>(), num_items);
  return {sub_indptr, coo_col, sub_indices, select_index};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
OnIndptrSlicingCUDA(torch::Tensor indptr, torch::Tensor indices,
                    torch::Tensor seeds, bool with_coo) {
  torch::Tensor out_indptr, out_coo_col, out_indices, out_selected_index;
  if (with_coo)
    std::tie(out_indptr, out_coo_col, out_indices, out_selected_index) =
        _OnIndptrSlicing<int64_t, true>(indptr, indices, seeds);
  else
    std::tie(out_indptr, out_coo_col, out_indices, out_selected_index) =
        _OnIndptrSlicing<int64_t, false>(indptr, indices, seeds);

  return {out_indptr, out_coo_col, out_indices, out_selected_index};
}

////////////////////////////// slicing on indices (CSC)////////////////////
template <typename IdType, int BLOCK_WARPS, int TILE_SIZE, bool WITH_COO>
__global__ void _OnIndicesSlicinigQueryKernel(
    const IdType* const in_indptr, const IdType* const in_indices,
    IdType* const key_buffer, IdType* const value_buffer, IdType* const out_deg,
    IdType* out_coo_row, IdType* const out_indices, IdType* const out_mask,
    const int num_items, const int dir_size) {
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == BLOCK_WARPS);

  IdType out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  IdType last_row =
      MIN(static_cast<IdType>(blockIdx.x + 1) * TILE_SIZE, num_items);

  int warp_id = threadIdx.y;
  int laneid = threadIdx.x;

  NodeQueryHashmap<IdType> hashmap(key_buffer, value_buffer, dir_size);

  typedef cub::WarpReduce<IdType> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage[BLOCK_WARPS];

  while (out_row < last_row) {
    IdType count = 0;
    IdType in_row_start = in_indptr[out_row];
    IdType in_row_end = in_indptr[out_row + 1];

    for (int idx = in_row_start + laneid; idx < in_row_end; idx += WARP_SIZE) {
      IdType value = hashmap.Query(in_indices[idx]);
      if (value != -1) {
        count += 1;
        out_mask[idx] = 1;
        out_indices[idx] = value;
        if (WITH_COO) {
          out_coo_row[idx] = out_row;
        }
      }
    }

    int deg = WarpReduce(temp_storage[warp_id]).Sum(count);
    if (laneid == 0) {
      out_deg[out_row] = deg;
    }

    out_row += BLOCK_WARPS;
  }
}

template <typename IdType, bool WITH_COO>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
_OnIndicesSlicing(torch::Tensor indptr, torch::Tensor indices,
                  torch::Tensor seeds) {
  int num_items = indptr.numel() - 1;
  int num_seeds = seeds.numel();

  // construct NodeQueryHashMap
  int dir_size = UpPower(num_seeds) * 2;
  torch::Tensor key_buffer = torch::full(dir_size, -1, indptr.options());
  torch::Tensor value_buffer = torch::full(dir_size, -1, indices.options());

  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(thrust::device, it(0), it(num_seeds),
                   [key = seeds.data_ptr<IdType>(),
                    _key_buffer = key_buffer.data_ptr<IdType>(),
                    _value_buffer = value_buffer.data_ptr<IdType>(),
                    dir_size] __device__(IdType i) {
                     NodeQueryHashmap<IdType> hashmap(_key_buffer,
                                                      _value_buffer, dir_size);
                     hashmap.Insert(key[i], i);
                   });

  constexpr int BLOCK_WARP = 128 / WARP_SIZE;
  constexpr int TILE_SIZE = 16;
  const dim3 block(WARP_SIZE, BLOCK_WARP);
  const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);

  torch::Tensor out_indptr = torch::empty_like(indptr);
  torch::Tensor out_indices = torch::empty_like(indices);
  torch::Tensor out_mask = torch::zeros_like(indices);

  torch::Tensor coo_col;
  IdType* coo_col_ptr;
  if (WITH_COO) {
    coo_col = torch::empty_like(indices);
    coo_col_ptr = coo_col.data_ptr<IdType>();
  } else {
    coo_col = torch::Tensor();
    coo_col_ptr = nullptr;
  }

  // query hashmap to get mask
  _OnIndicesSlicinigQueryKernel<IdType, BLOCK_WARP, TILE_SIZE, WITH_COO>
      <<<grid, block>>>(indptr.data_ptr<IdType>(), indices.data_ptr<IdType>(),
                        key_buffer.data_ptr<IdType>(),
                        value_buffer.data_ptr<IdType>(),
                        out_indptr.data_ptr<IdType>(), coo_col_ptr,
                        out_indices.data_ptr<IdType>(),
                        out_mask.data_ptr<IdType>(), num_items, dir_size);

  // prefix sum to get out_indptr and out_indices_index
  cub_exclusiveSum<IdType>(out_indptr.data_ptr<IdType>(), num_items + 1);
  torch::Tensor select_index = torch::nonzero(out_mask).reshape({
      -1,
  });

  if (WITH_COO)
    return {out_indptr, coo_col.index({select_index}),
            out_indices.index({select_index}), select_index};
  else
    return {out_indptr, torch::Tensor(), out_indices.index({select_index}),
            select_index};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
OnIndicesSlicingCUDA(torch::Tensor indptr, torch::Tensor indices,
                     torch::Tensor row_ids, bool with_coo) {
  torch::Tensor out_indptr, out_coo_col, out_indices, out_selected_index;
  if (with_coo)
    std::tie(out_indptr, out_coo_col, out_indices, out_selected_index) =
        _OnIndicesSlicing<int64_t, true>(indptr, indices, row_ids);
  else
    std::tie(out_indptr, out_coo_col, out_indices, out_selected_index) =
        _OnIndicesSlicing<int64_t, false>(indptr, indices, row_ids);
  return {out_indptr, out_coo_col, out_indices, out_selected_index};
};

////////////////////////////// COORowSlicingCUDA //////////////////////////
// reuse hashmap in CSCRowSlicingCUDA
template <typename IdType>
__global__ void _COORowSlicingKernel(const IdType* const in_coo_row,
                                     IdType* const key_buffer,
                                     IdType* const value_buffer,
                                     IdType* const out_mask,
                                     IdType* const out_coo_row,
                                     const int num_items, const int dir_size) {
  IdType tid = threadIdx.x + blockIdx.x * blockDim.x;
  IdType stride = gridDim.x * blockDim.x;

  NodeQueryHashmap<IdType> hashmap(key_buffer, value_buffer, dir_size);

  while (tid < num_items) {
    IdType value = hashmap.Query(in_coo_row[tid]);
    if (value != -1) {
      out_mask[tid] = 1;
      out_coo_row[tid] = value;
    }
    tid += stride;
  }
}

template <typename IdType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> _COORowSlicing(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor row_ids) {
  int num_items = coo_row.numel();
  int num_row_ids = row_ids.numel();

  // construct NodeQueryHashMap
  int dir_size = UpPower(num_row_ids) * 2;
  torch::Tensor key_buffer = torch::full(dir_size, -1, row_ids.options());
  torch::Tensor value_buffer = torch::full(dir_size, -1, row_ids.options());

  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(thrust::device, it(0), it(num_row_ids),
                   [key = row_ids.data_ptr<IdType>(),
                    _key_buffer = key_buffer.data_ptr<IdType>(),
                    _value_buffer = value_buffer.data_ptr<IdType>(),
                    dir_size] __device__(IdType i) {
                     NodeQueryHashmap<IdType> hashmap(_key_buffer,
                                                      _value_buffer, dir_size);
                     hashmap.Insert(key[i], i);
                   });

  torch::Tensor out_mask = torch::zeros_like(coo_row);
  torch::Tensor out_relabel_row = torch::empty_like(coo_row);

  constexpr int TILE_SIZE = 16;
  const dim3 block(256);
  const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);

  _COORowSlicingKernel<IdType><<<grid, block>>>(
      coo_row.data_ptr<IdType>(), key_buffer.data_ptr<IdType>(),
      value_buffer.data_ptr<IdType>(), out_mask.data_ptr<IdType>(),
      out_relabel_row.data_ptr<IdType>(), num_items, dir_size);

  torch::Tensor select_index = torch::nonzero(out_mask).reshape({
      -1,
  });
  return {out_relabel_row.index({select_index}), coo_col.index({select_index}),
          select_index};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> COORowSlicingCUDA(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor row_ids) {
  return _COORowSlicing<int64_t>(coo_row, coo_col, row_ids);
};

}  // namespace impl
}  // namespace gs