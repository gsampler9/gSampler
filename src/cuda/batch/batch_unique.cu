#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "../atomic.h"
#include "../cuda_common.h"
#include "../utils.h"

#include "batch_ops.h"

namespace gs {
namespace impl {
namespace batch {

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
__global__ void _InsertHashmaps(IdType* __restrict__ data_tensor,
                                IdType* __restrict__ data_key_tensor,
                                IdType* __restrict__ hashmap_key_tensor,
                                IdType* __restrict__ hashmap_value_tensor,
                                IdType* __restrict__ hashmap_ptr,
                                int64_t num_items) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int64_t index = tid; index < num_items;
       index += gridDim.x * blockDim.x) {
    int64_t batch_index = data_key_tensor[index];
    int64_t hashmap_begin = hashmap_ptr[batch_index];
    int64_t dir_size = hashmap_ptr[batch_index + 1] - hashmap_begin;
    RelabelHashmap<IdType> table(hashmap_key_tensor + hashmap_begin,
                                 hashmap_value_tensor + hashmap_begin,
                                 dir_size);
    table.Update(data_tensor[index], index);
  }
}

template <typename IdType>
__global__ void _SearchHashmapsForUnique(
    IdType* __restrict__ data_tensor, IdType* __restrict__ data_key_tensor,
    IdType* __restrict__ hashmap_key_tensor,
    IdType* __restrict__ hashmap_value_tensor, IdType* __restrict__ hashmap_ptr,
    IdType* __restrict__ item_prefix_tensor, int64_t num_items) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int64_t index = tid; index < num_items;
       index += gridDim.x * blockDim.x) {
    int64_t batch_index = data_key_tensor[index];
    int64_t hashmap_begin = hashmap_ptr[batch_index];
    int64_t dir_size = hashmap_ptr[batch_index + 1] - hashmap_begin;
    RelabelHashmap<IdType> table(hashmap_key_tensor + hashmap_begin,
                                 hashmap_value_tensor + hashmap_begin,
                                 dir_size);
    IdType result = table.SearchForValue(data_tensor[index]);
    item_prefix_tensor[index] = result == index ? 1 : 0;
  }
}

///////////////////////////// BatchUniqueByKey ////////////////////////////////
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
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> _BatchUniqueByKey(
    torch::Tensor data_tensor, torch::Tensor data_ptr, torch::Tensor data_key) {
  int64_t num_batchs = data_ptr.numel() - 1;
  int64_t num_items = data_tensor.numel();
  torch::Tensor hashmap_ptr = torch::empty_like(data_ptr);

  // Create Hashmaps
  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(
      it(0), it(num_batchs),
      [in = data_ptr.data_ptr<IdType>(),
       out = hashmap_ptr.data_ptr<IdType>()] __device__(IdType i) mutable {
        out[i] = (1 << static_cast<uint32_t>(log2(in[i + 1] - in[i]) + 1));
      });

  cub_exclusiveSum<IdType>(hashmap_ptr.data_ptr<IdType>(), num_batchs + 1);
  thrust::device_ptr<IdType> wrapper_hashmap_ptr(
      static_cast<IdType*>(hashmap_ptr.data_ptr<IdType>()));
  IdType total_dir_size = wrapper_hashmap_ptr[num_batchs];

  IdType MAX = std::numeric_limits<IdType>::max();
  torch::Tensor key_tensor =
      torch::full(total_dir_size, -1, data_tensor.options());
  torch::Tensor index_tensor =
      torch::full(total_dir_size, MAX, data_tensor.options());

  constexpr int BLOCK_SIZE = 256;
  int num_blocks = (num_items + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 blocks(BLOCK_SIZE);
  dim3 grids(num_blocks);

  _InsertHashmaps<IdType><<<grids, blocks>>>(
      data_tensor.data_ptr<IdType>(), data_key.data_ptr<IdType>(),
      key_tensor.data_ptr<IdType>(), index_tensor.data_ptr<IdType>(),
      hashmap_ptr.data_ptr<IdType>(), num_items);

  torch::Tensor mask_tensor = torch::empty(num_items, data_tensor.options());

  _SearchHashmapsForUnique<IdType><<<grids, blocks>>>(
      data_tensor.data_ptr<IdType>(), data_key.data_ptr<IdType>(),
      key_tensor.data_ptr<IdType>(), index_tensor.data_ptr<IdType>(),
      hashmap_ptr.data_ptr<IdType>(), mask_tensor.data_ptr<IdType>(),
      num_items);

  torch::Tensor data_unique_index = torch::nonzero(mask_tensor).reshape({-1});
  torch::Tensor sub_data = data_tensor.index({data_unique_index});
  torch::Tensor sub_data_key = data_key.index({data_unique_index});

  torch::Tensor sub_data_ptr = torch::zeros_like(data_ptr);

  dim3 block(128);
  dim3 grid((num_batchs + block.x - 1) / block.x);
  _SortedSearchKernelUpperBound<IdType>
      <<<grid, block>>>(sub_data_key.data_ptr<IdType>(), sub_data_key.numel(),
                        num_batchs, sub_data_ptr.data_ptr<IdType>() + 1);

  return {sub_data, sub_data_ptr, sub_data_key};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> BatchUniqueByKeyCUDA(
    torch::Tensor data_tensor, torch::Tensor data_ptr, torch::Tensor data_key) {
  return _BatchUniqueByKey<int64_t>(data_tensor, data_ptr, data_key);
}

///////////////////////////// BatchUnique ////////////////////////////////
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> BatchUniqueCUDA(
    torch::Tensor data_tensor, torch::Tensor data_ptr) {
  torch::Tensor data_key = torch::empty_like(data_tensor);
  dim3 block(128);
  dim3 grid((data_tensor.numel() + block.x - 1) / block.x);
  _RepeatKernel<int64_t><<<grid, block>>>(
      data_ptr.data_ptr<int64_t>(), data_key.data_ptr<int64_t>(),
      data_ptr.numel(), data_tensor.numel());
  return _BatchUniqueByKey<int64_t>(data_tensor, data_ptr, data_key);
}

///////////////////////////// BatchUnique2 ////////////////////////////////
template <typename IdType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> _BatchUniqueByKey2(
    torch::Tensor data, torch::Tensor data_ptr, torch::Tensor data_key) {
  torch::Tensor segment_sorted_data = torch::empty_like(data);
  int64_t num_segments = data_ptr.numel() - 1;
  int64_t num_items = data.numel();

  cub_segmentSort<IdType>(data.data_ptr<IdType>(),
                          segment_sorted_data.data_ptr<IdType>(),
                          data_ptr.data_ptr<IdType>(), num_items, num_segments);

  torch::Tensor unique_data = torch::empty_like(segment_sorted_data);
  torch::Tensor unique_data_key = torch::empty_like(data_key);

  int64_t* d_num_selected_out = reinterpret_cast<int64_t*>(
      c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(int64_t)));

  cub_uniqueByKey<IdType, IdType>(
      segment_sorted_data.data_ptr<IdType>(), unique_data.data_ptr<IdType>(),
      data_key.data_ptr<IdType>(), unique_data_key.data_ptr<IdType>(),
      num_items, d_num_selected_out);

  thrust::device_ptr<int64_t> wrapper_d_num_selected_out(d_num_selected_out);
  int64_t num_unique = wrapper_d_num_selected_out[0];

  torch::Tensor narrow_unique_tensor = unique_data.slice(0, 0, num_unique, 1);
  torch::Tensor narrow_unique_tensor_key =
      unique_data_key.slice(0, 0, num_unique, 1);

  torch::Tensor narrow_unique_tensor_ptr = torch::zeros_like(data_ptr);

  dim3 block(128);
  dim3 grid((num_segments + block.x - 1) / block.x);
  _SortedSearchKernelUpperBound<IdType>
      <<<grid, block>>>(narrow_unique_tensor_key.data_ptr<IdType>(),
                        narrow_unique_tensor_key.numel(), num_segments,
                        narrow_unique_tensor_ptr.data_ptr<IdType>() + 1);

  return {narrow_unique_tensor, narrow_unique_tensor_ptr,
          narrow_unique_tensor_key};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> BatchUniqueByKey2CUDA(
    torch::Tensor data, torch::Tensor data_ptr, torch::Tensor data_key) {
  return _BatchUniqueByKey2<int64_t>(data, data_ptr, data_key);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> BatchUnique2CUDA(
    torch::Tensor data, torch::Tensor data_ptr) {
  torch::Tensor data_key = torch::empty_like(data);
  dim3 block(128);
  dim3 grid((data.numel() + block.x - 1) / block.x);
  _RepeatKernel<int64_t><<<grid, block>>>(data_ptr.data_ptr<int64_t>(),
                                          data_key.data_ptr<int64_t>(),
                                          data_ptr.numel(), data.numel());
  return _BatchUniqueByKey2<int64_t>(data, data_ptr, data_key);
}
}  // namespace batch
}  // namespace impl
}  // namespace gs