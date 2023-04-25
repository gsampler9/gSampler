#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include "atomic.h"
#include "cuda_common.h"
#include "tensor_ops.h"
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

template <typename IdType>
__global__ void _RepeatKernel(const IdType* pos, IdType* out, int64_t n_col,
                              int64_t length) {
  IdType tx = static_cast<IdType>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    IdType i = cub::UpperBound(pos, n_col, tx) - 1;
    out[tx] = i;
    tx += stride_x;
  }
}

template <typename IdType>
__global__ void _InsertHashmapsForCOOSlicing(
    IdType* __restrict__ data_tensor, IdType* __restrict__ data_key_tensor,
    IdType* __restrict__ hashmap_key_tensor,
    IdType* __restrict__ hashmap_value_tensor, IdType* __restrict__ hashmap_ptr,
    int64_t num_items) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int64_t index = tid; index < num_items;
       index += gridDim.x * blockDim.x) {
    int64_t batch_index = data_key_tensor[index];
    int64_t hashmap_begin = hashmap_ptr[batch_index];
    int64_t dir_size = hashmap_ptr[batch_index + 1] - hashmap_begin;
    NodeQueryHashmap<IdType> table(hashmap_key_tensor + hashmap_begin,
                                   hashmap_value_tensor + hashmap_begin,
                                   dir_size);
    table.Insert(data_tensor[index], index);
  }
}

template <typename IdType>
__global__ void _SearchHashmapsForCOOSlicing(
    IdType* __restrict__ data_tensor, IdType* __restrict__ data_key_tensor,
    IdType* __restrict__ hashmap_key_tensor,
    IdType* __restrict__ hashmap_value_tensor, IdType* __restrict__ hashmap_ptr,
    IdType* __restrict__ mask_tensor, int64_t num_items) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int64_t index = tid; index < num_items;
       index += gridDim.x * blockDim.x) {
    int64_t batch_index = data_key_tensor[index];
    int64_t hashmap_begin = hashmap_ptr[batch_index];
    int64_t dir_size = hashmap_ptr[batch_index + 1] - hashmap_begin;
    NodeQueryHashmap<IdType> table(hashmap_key_tensor + hashmap_begin,
                                   hashmap_value_tensor + hashmap_begin,
                                   dir_size);
    IdType value = table.Query(data_tensor[index]);
    mask_tensor[index] = value != -1 ? 1 : 0;
  }
}

template <typename IdType>
__global__ void _SortedSearchKernelUpperBound(const IdType* __restrict__ hay,
                                              int64_t hay_size,
                                              int64_t num_needles,
                                              IdType* __restrict__ pos) {
  IdType tx = static_cast<IdType>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < num_needles) {
    pos[tx] = cub::UpperBound(hay, hay_size, tx);
    tx += stride_x;
  }
}

template <typename IdType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> _BatchCOOSlicing(
    int64_t axis, torch::Tensor coo_row, torch::Tensor coo_col,
    torch::Tensor batch_ptr, torch::Tensor neigbhors,
    torch::Tensor neighbors_ptr, torch::Tensor neighbors_key) {
  torch::Tensor target_tensor = axis == 0 ? coo_col : coo_row;
  int64_t num_items = target_tensor.numel();
  int64_t num_batchs = batch_ptr.numel() - 1;

  torch::Tensor target_key = torch::empty_like(target_tensor);
  dim3 block_repeat(128);
  dim3 grid_repeat((target_tensor.numel() + block_repeat.x - 1) /
                   block_repeat.x);
  _RepeatKernel<IdType><<<grid_repeat, block_repeat>>>(
      batch_ptr.data_ptr<IdType>(), target_key.data_ptr<IdType>(),
      batch_ptr.numel(), num_items);

  torch::Tensor hashmap_ptr = torch::empty_like(batch_ptr);

  // Create Hashmaps
  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(
      it(0), it(num_batchs),
      [in = neighbors_ptr.data_ptr<IdType>(),
       out = hashmap_ptr.data_ptr<IdType>()] __device__(IdType i) mutable {
        out[i] = 2 * (1 << static_cast<uint32_t>(log2(in[i + 1] - in[i]) + 1));
      });

  cub_exclusiveSum<IdType>(hashmap_ptr.data_ptr<IdType>(), num_batchs + 1);
  thrust::device_ptr<IdType> wrapper_hashmap_ptr(
      static_cast<IdType*>(hashmap_ptr.data_ptr<IdType>()));
  IdType total_dir_size = wrapper_hashmap_ptr[num_batchs];

  IdType MAX = std::numeric_limits<IdType>::max();
  torch::Tensor key_tensor =
      torch::full(total_dir_size, -1, target_tensor.options());
  torch::Tensor value_tensor =
      torch::full(total_dir_size, MAX, target_tensor.options());

  constexpr int BLOCK_SIZE = 256;
  int num_blocks = (neigbhors.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 block_insert(BLOCK_SIZE);
  dim3 grid_insert(num_blocks);

  _InsertHashmapsForCOOSlicing<IdType><<<grid_insert, block_insert>>>(
      neigbhors.data_ptr<IdType>(), neighbors_key.data_ptr<IdType>(),
      key_tensor.data_ptr<IdType>(), value_tensor.data_ptr<IdType>(),
      hashmap_ptr.data_ptr<IdType>(), neigbhors.numel());

  torch::Tensor mask_tensor = torch::empty(num_items, target_tensor.options());

  dim3 block_search(BLOCK_SIZE);
  dim3 grid_search((target_tensor.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE);
  _SearchHashmapsForCOOSlicing<IdType><<<grid_search, block_search>>>(
      target_tensor.data_ptr<IdType>(), target_key.data_ptr<IdType>(),
      key_tensor.data_ptr<IdType>(), value_tensor.data_ptr<IdType>(),
      hashmap_ptr.data_ptr<IdType>(), mask_tensor.data_ptr<IdType>(),
      num_items);

  torch::Tensor sub_index = torch::nonzero(mask_tensor).reshape({-1});
  torch::Tensor sub_coo_row = coo_row.index({sub_index});
  torch::Tensor sub_coo_col = coo_col.index({sub_index});
  torch::Tensor sub_key = target_key.index({sub_index});

  torch::Tensor sub_ptr = torch::zeros_like(batch_ptr);

  dim3 block(128);
  dim3 grid((num_batchs + block.x - 1) / block.x);
  _SortedSearchKernelUpperBound<IdType>
      <<<grid, block>>>(sub_key.data_ptr<IdType>(), sub_key.numel(), num_batchs,
                        sub_ptr.data_ptr<IdType>() + 1);

  return {sub_coo_row, sub_coo_col, sub_ptr};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> BatchCOOSlicingCUDA(
    int64_t axis, torch::Tensor coo_row, torch::Tensor coo_col,
    torch::Tensor batch_ptr, torch::Tensor neigbhors,
    torch::Tensor neighbors_ptr) {
  torch::Tensor neighbors_key = torch::empty_like(neigbhors);

  dim3 block(128);
  dim3 grid((neigbhors.numel() + block.x - 1) / block.x);
  _RepeatKernel<int64_t><<<grid, block>>>(
      neighbors_ptr.data_ptr<int64_t>(), neighbors_key.data_ptr<int64_t>(),
      neighbors_ptr.numel(), neigbhors.numel());

  return _BatchCOOSlicing<int64_t>(axis, coo_row, coo_col, batch_ptr, neigbhors,
                                   neighbors_ptr, neighbors_key);
}

template <typename IdType>
__global__ void _COORowSlicingKernel(const IdType* const in_coo_row,
                                     IdType* const key_buffer,
                                     IdType* const value_buffer,
                                     IdType* const out_mask,
                                     const int num_items, const int dir_size) {
  IdType tid = threadIdx.x + blockIdx.x * blockDim.x;
  IdType stride = gridDim.x * blockDim.x;

  NodeQueryHashmap<IdType> hashmap(key_buffer, value_buffer, dir_size);

  while (tid < num_items) {
    IdType value = hashmap.Query(in_coo_row[tid]);
    out_mask[tid] = value != -1 ? 1 : 0;
    tid += stride;
  }
}

template <typename IdType>
std::tuple<torch::Tensor, torch::Tensor> _COORowSlicing(torch::Tensor coo_row,
                                                        torch::Tensor coo_col,
                                                        torch::Tensor row_ids) {
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

  constexpr int TILE_SIZE = 16;
  const dim3 block(256);
  const dim3 grid((num_items + TILE_SIZE - 1) / TILE_SIZE);

  CUDA_KERNEL_CALL((_COORowSlicingKernel<IdType>), grid, block,
                   coo_row.data_ptr<IdType>(), key_buffer.data_ptr<IdType>(),
                   value_buffer.data_ptr<IdType>(), out_mask.data_ptr<IdType>(),
                   num_items, dir_size);

  torch::Tensor select_index = torch::nonzero(out_mask).reshape(-1);
  return {coo_row.index({select_index}), coo_col.index({select_index})};
}

std::tuple<torch::Tensor, torch::Tensor> COORowSlicingGlobalIdCUDA(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor row_ids) {
  return _COORowSlicing<int64_t>(coo_row, coo_col, row_ids);
}

}  // namespace impl
}  // namespace gs