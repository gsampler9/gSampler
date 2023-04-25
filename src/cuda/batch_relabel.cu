#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "atomic.h"
#include "cuda_common.h"
#include "tensor_ops.h"
#include "utils.h"

namespace gs {
namespace impl {

template <typename IdType>
struct RelabelHashmap {
  __device__ inline RelabelHashmap(IdType* Kptr, IdType* Vptr, size_t numel)
      : kptr(Kptr), vptr(Vptr), capacity(numel){};

  __device__ inline void Update(IdType key, IdType value) {
    uint32_t delta = 1;
    uint32_t pos = hash(key);
    IdType prev = AtomicCAS(&kptr[pos], kEmptyKey, key);

    while (prev != key and prev != kEmptyKey) {
      pos = hash(pos + delta);
      delta += 1;
      prev = AtomicCAS(&kptr[pos], kEmptyKey, key);
    }

    AtomicMin(vptr + pos, value);
  }

  __device__ inline IdType SearchForPos(IdType key) {
    uint32_t delta = 1;
    uint32_t pos = hash(key);

    while (true) {
      if (kptr[pos] == key) {
        return pos;
      }
      if (kptr[pos] == kEmptyKey) {
        return -1;
      }
      pos = hash(pos + delta);
      delta += 1;
    }
  }

  __device__ inline IdType SearchForValue(IdType key) {
    uint32_t delta = 1;
    uint32_t pos = hash(key);

    while (true) {
      if (kptr[pos] == key) {
        return vptr[pos];
      };
      if (kptr[pos] == kEmptyKey) {
        return -1;
      }
      pos = hash(pos + delta);
      delta += 1;
    }
  }

  __device__ inline uint32_t hash(int32_t key) { return key & (capacity - 1); }

  __device__ inline uint32_t hash(uint32_t key) { return key & (capacity - 1); }

  __device__ inline uint32_t hash(int64_t key) { return key & (capacity - 1); }

  __device__ inline uint32_t hash(uint64_t key) { return key & (capacity - 1); }

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

///////////////////////////// BatchCSRRelabel ////////////////////////////////

template <typename IdType>
__global__ void _2TensorInsertHashmaps(
    IdType* __restrict__ data1, IdType* __restrict__ data1_key,
    IdType* __restrict__ data2, IdType* __restrict__ data2_key,
    IdType* __restrict__ hashmap_key_tensor,
    IdType* __restrict__ hashmap_value_tensor, IdType* __restrict__ hashmap_ptr,
    int64_t data1_len, int64_t num_items) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int64_t index = tid; index < num_items;
       index += gridDim.x * blockDim.x) {
    if (index < data1_len) {
      // insert data1
      int64_t batch_index = data1_key[index];
      int64_t hashmap_begin = hashmap_ptr[batch_index];
      int64_t dir_size = hashmap_ptr[batch_index + 1] - hashmap_begin;
      RelabelHashmap<IdType> table(hashmap_key_tensor + hashmap_begin,
                                   hashmap_value_tensor + hashmap_begin,
                                   dir_size);
      table.Update(data1[index], index);
    } else {
      // insert data2
      int64_t local_index = index - data1_len;
      int64_t batch_index = data2_key[local_index];
      int64_t hashmap_begin = hashmap_ptr[batch_index];
      int64_t dir_size = hashmap_ptr[batch_index + 1] - hashmap_begin;
      RelabelHashmap<IdType> table(hashmap_key_tensor + hashmap_begin,
                                   hashmap_value_tensor + hashmap_begin,
                                   dir_size);
      table.Update(data2[local_index], index);
    }
  }
}

template <typename IdType>
__global__ void _2TensorSearchHashmapsForPrefix(
    IdType* __restrict__ data1, IdType* __restrict__ data1_key,
    IdType* __restrict__ data1_ptr, IdType* __restrict__ data2,
    IdType* __restrict__ data2_key, IdType* __restrict__ data2_ptr,
    IdType* __restrict__ hashmap_key_tensor,
    IdType* __restrict__ hashmap_value_tensor, IdType* __restrict__ hashmap_ptr,
    IdType* __restrict__ item_prefix_tensor, int64_t data1_len,
    int64_t num_items) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int64_t index = tid; index < num_items;
       index += gridDim.x * blockDim.x) {
    if (index < data1_len) {
      // search data1
      int64_t batch_index = data1_key[index];
      int64_t hashmap_begin = hashmap_ptr[batch_index];
      int64_t dir_size = hashmap_ptr[batch_index + 1] - hashmap_begin;
      RelabelHashmap<IdType> table(hashmap_key_tensor + hashmap_begin,
                                   hashmap_value_tensor + hashmap_begin,
                                   dir_size);
      IdType result = table.SearchForValue(data1[index]);
      item_prefix_tensor[index + data2_ptr[batch_index]] =
          result == index ? 1 : 0;
    } else {
      // search data2
      int64_t local_index = index - data1_len;
      int64_t batch_index = data2_key[local_index];
      int64_t hashmap_begin = hashmap_ptr[batch_index];
      int64_t dir_size = hashmap_ptr[batch_index + 1] - hashmap_begin;
      RelabelHashmap<IdType> table(hashmap_key_tensor + hashmap_begin,
                                   hashmap_value_tensor + hashmap_begin,
                                   dir_size);
      IdType result = table.SearchForValue(data2[local_index]);
      item_prefix_tensor[local_index + data1_ptr[batch_index + 1]] =
          result == index ? 1 : 0;
    }
  }
}

template <typename IdType>
__global__ void _2TensorSearchHashmapsForUnique(
    IdType* __restrict__ data1, IdType* __restrict__ data1_key,
    IdType* __restrict__ data1_ptr, IdType* __restrict__ data2,
    IdType* __restrict__ data2_key, IdType* __restrict__ data2_ptr,
    IdType* __restrict__ hashmap_key_tensor,
    IdType* __restrict__ hashmap_index_tensor,
    IdType* __restrict__ hashmap_value_tensor, IdType* __restrict__ hashmap_ptr,
    IdType* __restrict__ item_prefix_tensor, IdType* __restrict__ unique_tensor,
    IdType* __restrict__ unique_tensor_ptr, int64_t data1_len,
    int64_t num_items) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int64_t index = tid; index < num_items;
       index += gridDim.x * blockDim.x) {
    if (index < data1_len) {
      // search data1
      int64_t batch_index = data1_key[index];
      int64_t hashmap_begin = hashmap_ptr[batch_index];
      int64_t dir_size = hashmap_ptr[batch_index + 1] - hashmap_begin;
      RelabelHashmap<IdType> table(hashmap_key_tensor + hashmap_begin,
                                   hashmap_index_tensor + hashmap_begin,
                                   dir_size);
      IdType pos = table.SearchForPos(data1[index]) + hashmap_begin;
      if (hashmap_index_tensor[pos] == index) {
        IdType local_pos = item_prefix_tensor[index + data2_ptr[batch_index]];
        hashmap_value_tensor[pos] = local_pos - unique_tensor_ptr[batch_index];
        unique_tensor[local_pos] = data1[index];
      }

    } else {
      // search data2
      int64_t local_index = index - data1_len;
      int64_t batch_index = data2_key[local_index];
      int64_t hashmap_begin = hashmap_ptr[batch_index];
      int64_t dir_size = hashmap_ptr[batch_index + 1] - hashmap_begin;
      RelabelHashmap<IdType> table(hashmap_key_tensor + hashmap_begin,
                                   hashmap_index_tensor + hashmap_begin,
                                   dir_size);
      IdType pos = table.SearchForPos(data2[local_index]) + hashmap_begin;
      if (hashmap_index_tensor[pos] == index) {
        IdType local_pos =
            item_prefix_tensor[local_index + data1_ptr[batch_index + 1]];
        hashmap_value_tensor[pos] = local_pos - unique_tensor_ptr[batch_index];
        unique_tensor[local_pos] = data2[local_index];
      }
    }
  }
}

template <typename IdType>
__global__ void _IndicesSearchHashmapsForRelabel(
    IdType* __restrict__ data2, IdType* __restrict__ data2_key,
    IdType* __restrict__ out_data2, IdType* __restrict__ hashmap_key_tensor,
    IdType* __restrict__ hashmap_value_tensor, IdType* __restrict__ hashmap_ptr,
    int64_t data2_len) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int64_t index = tid; index < data2_len;
       index += gridDim.x * blockDim.x) {
    // search data2
    int64_t batch_index = data2_key[index];
    int64_t hashmap_begin = hashmap_ptr[batch_index];
    int64_t dir_size = hashmap_ptr[batch_index + 1] - hashmap_begin;
    RelabelHashmap<IdType> table(hashmap_key_tensor + hashmap_begin,
                                 hashmap_value_tensor + hashmap_begin,
                                 dir_size);
    out_data2[index] = table.SearchForValue(data2[index]);
  }
}

template <typename IdType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
_BatchCSRRelabelByKey(torch::Tensor seeds, torch::Tensor seeds_ptr,
                      torch::Tensor seeds_key, torch::Tensor indices,
                      torch::Tensor indices_ptr, torch::Tensor indices_key) {
  int64_t num_items = seeds.numel() + indices.numel();
  int64_t num_batchs = seeds_ptr.numel() - 1;

  torch::Tensor hashmap_ptr = torch::empty_like(seeds_ptr);
  // Create Hashmaps
  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(
      it(0), it(num_batchs),
      [in1 = seeds_ptr.data_ptr<IdType>(), in2 = indices_ptr.data_ptr<IdType>(),
       out = hashmap_ptr.data_ptr<IdType>()] __device__(IdType i) mutable {
        out[i] = (1 << static_cast<uint32_t>(
                      log2(in1[i + 1] - in1[i] + in2[i + 1] - in2[i]) + 1));
      });

  cub_exclusiveSum<IdType>(hashmap_ptr.data_ptr<IdType>(), num_batchs + 1);
  thrust::device_ptr<IdType> wrapper_hashmap_ptr(
      static_cast<IdType*>(hashmap_ptr.data_ptr<IdType>()));
  IdType total_dir_size = wrapper_hashmap_ptr[num_batchs];

  IdType MAX = std::numeric_limits<IdType>::max();
  torch::Tensor key_tensor = torch::full(total_dir_size, -1, seeds.options());
  torch::Tensor index_tensor =
      torch::full(total_dir_size, MAX, seeds.options());

  // Insert hashmap
  constexpr int BLOCK_SIZE = 256;
  int num_blocks = (num_items + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 blocks(BLOCK_SIZE);
  dim3 grids(num_blocks);

  _2TensorInsertHashmaps<IdType><<<grids, blocks>>>(
      seeds.data_ptr<IdType>(), seeds_key.data_ptr<IdType>(),
      indices.data_ptr<IdType>(), indices_key.data_ptr<IdType>(),
      key_tensor.data_ptr<IdType>(), index_tensor.data_ptr<IdType>(),
      hashmap_ptr.data_ptr<IdType>(), seeds.numel(), num_items);

  // Scan to get the first elements
  torch::Tensor prefix_tensor = torch::empty(num_items + 1, seeds.options());

  _2TensorSearchHashmapsForPrefix<IdType><<<grids, blocks>>>(
      seeds.data_ptr<IdType>(), seeds_key.data_ptr<IdType>(),
      seeds_ptr.data_ptr<IdType>(), indices.data_ptr<IdType>(),
      indices_key.data_ptr<IdType>(), indices_ptr.data_ptr<IdType>(),
      key_tensor.data_ptr<IdType>(), index_tensor.data_ptr<IdType>(),
      hashmap_ptr.data_ptr<IdType>(), prefix_tensor.data_ptr<IdType>(),
      seeds.numel(), num_items);

  // Get unique_tensor
  thrust::device_ptr<IdType> wrapper_prefix_tensor(
      static_cast<IdType*>(prefix_tensor.data_ptr<IdType>()));
  cub_exclusiveSum<IdType>(thrust::raw_pointer_cast(wrapper_prefix_tensor),
                           num_items + 1);
  int64_t unique_size = wrapper_prefix_tensor[num_items];

  torch::Tensor unique_tensor_ptr =
      prefix_tensor.index({(seeds_ptr + indices_ptr).to(torch::kInt64)});
  torch::Tensor unique_tensor = torch::empty(unique_size, seeds.options());
  torch::Tensor value_tensor = torch::empty_like(index_tensor);

  _2TensorSearchHashmapsForUnique<IdType><<<grids, blocks>>>(
      seeds.data_ptr<IdType>(), seeds_key.data_ptr<IdType>(),
      seeds_ptr.data_ptr<IdType>(), indices.data_ptr<IdType>(),
      indices_key.data_ptr<IdType>(), indices_ptr.data_ptr<IdType>(),
      key_tensor.data_ptr<IdType>(), index_tensor.data_ptr<IdType>(),
      value_tensor.data_ptr<IdType>(), hashmap_ptr.data_ptr<IdType>(),
      prefix_tensor.data_ptr<IdType>(), unique_tensor.data_ptr<IdType>(),
      unique_tensor_ptr.data_ptr<IdType>(), seeds.numel(), num_items);

  // Relabel indices
  torch::Tensor out_indices = torch::empty_like(indices);

  dim3 blocks_relabel(BLOCK_SIZE);
  dim3 grids_relabel((indices.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE);

  _IndicesSearchHashmapsForRelabel<IdType><<<grids_relabel, blocks_relabel>>>(
      indices.data_ptr<IdType>(), indices_key.data_ptr<IdType>(),
      out_indices.data_ptr<IdType>(), key_tensor.data_ptr<IdType>(),
      value_tensor.data_ptr<IdType>(), hashmap_ptr.data_ptr<IdType>(),
      indices.numel());
  return {unique_tensor, unique_tensor_ptr, out_indices, indices_ptr};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
BatchCSRRelabelByKeyCUDA(torch::Tensor seeds, torch::Tensor seeds_ptr,
                         torch::Tensor seeds_key, torch::Tensor indices,
                         torch::Tensor indices_ptr, torch::Tensor indices_key) {
  return _BatchCSRRelabelByKey<int64_t>(seeds, seeds_ptr, seeds_key, indices,
                                        indices_ptr, indices_key);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
BatchCSRRelabelCUDA(torch::Tensor seeds, torch::Tensor seeds_ptr,
                    torch::Tensor indices, torch::Tensor indices_ptr) {
  torch::Tensor seeds_key = torch::empty_like(seeds);
  dim3 block(128);
  dim3 grid((seeds_key.numel() + block.x - 1) / block.x);
  _RepeatKernel<int64_t><<<grid, block>>>(seeds_ptr.data_ptr<int64_t>(),
                                          seeds_key.data_ptr<int64_t>(),
                                          seeds_ptr.numel(), seeds_key.numel());

  torch::Tensor indices_key = torch::empty_like(indices);
  dim3 block2(128);
  dim3 grid2((indices_key.numel() + block.x - 1) / block.x);
  _RepeatKernel<int64_t><<<grid2, block2>>>(
      indices_ptr.data_ptr<int64_t>(), indices_key.data_ptr<int64_t>(),
      indices_ptr.numel(), indices_key.numel());

  return _BatchCSRRelabelByKey<int64_t>(seeds, seeds_ptr, seeds_key, indices,
                                        indices_ptr, indices_key);
}

/////////////////////// BatchCOORelabel ////////////////////////////////
template <typename IdType>
__global__ void _2COOTensorSearchHashmapsForRelabel(
    IdType* __restrict__ coo_col, IdType* __restrict__ coo_row,
    IdType* __restrict__ out_col, IdType* __restrict__ out_row,
    IdType* __restrict__ coo_key, IdType* __restrict__ hashmap_key_tensor,
    IdType* __restrict__ hashmap_value_tensor, IdType* __restrict__ hashmap_ptr,
    int64_t num_items) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int64_t index = tid; index < num_items;
       index += gridDim.x * blockDim.x) {
    // search data2
    int64_t batch_index = coo_key[index];
    int64_t hashmap_begin = hashmap_ptr[batch_index];
    int64_t dir_size = hashmap_ptr[batch_index + 1] - hashmap_begin;
    RelabelHashmap<IdType> table(hashmap_key_tensor + hashmap_begin,
                                 hashmap_value_tensor + hashmap_begin,
                                 dir_size);
    out_col[index] = table.SearchForValue(coo_col[index]);
    out_row[index] = table.SearchForValue(coo_row[index]);
  }
}

template <typename IdType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
_BatchCOORelabelByKey(torch::Tensor seeds, torch::Tensor seeds_ptr,
                      torch::Tensor seeds_key, torch::Tensor coo_col,
                      torch::Tensor coo_row, torch::Tensor coo_ptr,
                      torch::Tensor coo_key) {
  int64_t num_items = seeds.numel() + coo_row.numel();
  int64_t num_batchs = seeds_ptr.numel() - 1;

  torch::Tensor hashmap_ptr = torch::empty_like(seeds_ptr);
  // Create Hashmaps
  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(
      it(0), it(num_batchs),
      [in1 = seeds_ptr.data_ptr<IdType>(), in2 = coo_ptr.data_ptr<IdType>(),
       out = hashmap_ptr.data_ptr<IdType>()] __device__(IdType i) mutable {
        out[i] = (1 << static_cast<uint32_t>(
                      log2(in1[i + 1] - in1[i] + in2[i + 1] - in2[i]) + 1));
      });

  cub_exclusiveSum<IdType>(hashmap_ptr.data_ptr<IdType>(), num_batchs + 1);
  thrust::device_ptr<IdType> wrapper_hashmap_ptr(
      static_cast<IdType*>(hashmap_ptr.data_ptr<IdType>()));
  IdType total_dir_size = wrapper_hashmap_ptr[num_batchs];

  IdType MAX = std::numeric_limits<IdType>::max();
  torch::Tensor key_tensor = torch::full(total_dir_size, -1, seeds.options());
  torch::Tensor index_tensor =
      torch::full(total_dir_size, MAX, seeds.options());

  // Insert hashmap
  constexpr int BLOCK_SIZE = 256;
  int num_blocks = (num_items + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 blocks(BLOCK_SIZE);
  dim3 grids(num_blocks);

  _2TensorInsertHashmaps<IdType><<<grids, blocks>>>(
      seeds.data_ptr<IdType>(), seeds_key.data_ptr<IdType>(),
      coo_row.data_ptr<IdType>(), coo_key.data_ptr<IdType>(),
      key_tensor.data_ptr<IdType>(), index_tensor.data_ptr<IdType>(),
      hashmap_ptr.data_ptr<IdType>(), seeds.numel(), num_items);

  // Scan to get the first elements
  torch::Tensor prefix_tensor = torch::empty(num_items + 1, seeds.options());

  _2TensorSearchHashmapsForPrefix<IdType><<<grids, blocks>>>(
      seeds.data_ptr<IdType>(), seeds_key.data_ptr<IdType>(),
      seeds_ptr.data_ptr<IdType>(), coo_row.data_ptr<IdType>(),
      coo_key.data_ptr<IdType>(), coo_ptr.data_ptr<IdType>(),
      key_tensor.data_ptr<IdType>(), index_tensor.data_ptr<IdType>(),
      hashmap_ptr.data_ptr<IdType>(), prefix_tensor.data_ptr<IdType>(),
      seeds.numel(), num_items);

  // Get unique_tensor
  thrust::device_ptr<IdType> wrapper_prefix_tensor(
      static_cast<IdType*>(prefix_tensor.data_ptr<IdType>()));
  cub_exclusiveSum<IdType>(thrust::raw_pointer_cast(wrapper_prefix_tensor),
                           num_items + 1);
  int64_t unique_size = wrapper_prefix_tensor[num_items];

  torch::Tensor unique_tensor_ptr =
      prefix_tensor.index({(seeds_ptr + coo_ptr).to(torch::kInt64)});
  torch::Tensor unique_tensor = torch::empty(unique_size, seeds.options());
  torch::Tensor value_tensor = torch::empty_like(index_tensor);

  _2TensorSearchHashmapsForUnique<IdType><<<grids, blocks>>>(
      seeds.data_ptr<IdType>(), seeds_key.data_ptr<IdType>(),
      seeds_ptr.data_ptr<IdType>(), coo_row.data_ptr<IdType>(),
      coo_key.data_ptr<IdType>(), coo_ptr.data_ptr<IdType>(),
      key_tensor.data_ptr<IdType>(), index_tensor.data_ptr<IdType>(),
      value_tensor.data_ptr<IdType>(), hashmap_ptr.data_ptr<IdType>(),
      prefix_tensor.data_ptr<IdType>(), unique_tensor.data_ptr<IdType>(),
      unique_tensor_ptr.data_ptr<IdType>(), seeds.numel(), num_items);

  // Relabel indices
  torch::Tensor out_col = torch::empty_like(coo_col);
  torch::Tensor out_row = torch::empty_like(coo_row);

  dim3 blocks_relabel(BLOCK_SIZE);
  dim3 grids_relabel((coo_col.numel() + BLOCK_SIZE - 1) / BLOCK_SIZE);

  _2COOTensorSearchHashmapsForRelabel<IdType>
      <<<grids_relabel, blocks_relabel>>>(
          coo_col.data_ptr<IdType>(), coo_row.data_ptr<IdType>(),
          out_col.data_ptr<IdType>(), out_row.data_ptr<IdType>(),
          coo_key.data_ptr<IdType>(), key_tensor.data_ptr<IdType>(),
          value_tensor.data_ptr<IdType>(), hashmap_ptr.data_ptr<IdType>(),
          coo_col.numel());
  return {unique_tensor, unique_tensor_ptr, out_row, out_col, coo_ptr};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
BatchCOORelabelByKeyCUDA(torch::Tensor seeds, torch::Tensor seeds_ptr,
                         torch::Tensor seeds_key, torch::Tensor coo_col,
                         torch::Tensor coo_row, torch::Tensor coo_ptr,
                         torch::Tensor coo_key) {
  return _BatchCOORelabelByKey<int64_t>(seeds, seeds_ptr, seeds_key, coo_col,
                                        coo_row, coo_ptr, coo_key);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
BatchCOORelabelCUDA(torch::Tensor seeds, torch::Tensor seeds_ptr,
                    torch::Tensor coo_col, torch::Tensor coo_row,
                    torch::Tensor coo_ptr) {
  torch::Tensor seeds_key = torch::empty_like(seeds);
  dim3 block(128);
  dim3 grid((seeds_key.numel() + block.x - 1) / block.x);
  _RepeatKernel<int64_t><<<grid, block>>>(seeds_ptr.data_ptr<int64_t>(),
                                          seeds_key.data_ptr<int64_t>(),
                                          seeds_ptr.numel(), seeds_key.numel());

  torch::Tensor coo_key = torch::empty_like(coo_col);
  dim3 block2(128);
  dim3 grid2((coo_key.numel() + block.x - 1) / block.x);
  _RepeatKernel<int64_t><<<grid2, block2>>>(coo_ptr.data_ptr<int64_t>(),
                                            coo_key.data_ptr<int64_t>(),
                                            coo_ptr.numel(), coo_key.numel());

  return _BatchCOORelabelByKey<int64_t>(seeds, seeds_ptr, seeds_key, coo_col,
                                        coo_row, coo_ptr, coo_key);
}
}  // namespace impl
}  // namespace gs