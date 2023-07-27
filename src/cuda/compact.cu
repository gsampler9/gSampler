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

inline int UpPower(int key) {
  int ret = 1 << static_cast<uint32_t>(std::log2(key) + 1);
  return ret;
}

// Compact
template <typename IdType>
std::tuple<torch::Tensor, torch::Tensor> TensorCompactCUDA(torch::Tensor data) {
  torch::Tensor unique_tensor = std::get<0>(torch::_unique2(data));
  int num_items = unique_tensor.numel();
  int dir_size = UpPower(num_items);

  IdType MAX = std::numeric_limits<IdType>::max();
  torch::Tensor key_tensor = torch::full(dir_size, -1, data.options());
  torch::Tensor value_tensor = torch::full(dir_size, MAX, data.options());

  // insert
  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(it(0), it(num_items),
                   [key = key_tensor.data_ptr<IdType>(),
                    value = value_tensor.data_ptr<IdType>(),
                    in = unique_tensor.data_ptr<IdType>(), num_items,
                    dir_size] __device__(IdType i) mutable {
                     RelabelHashmap<IdType> table(key, value, dir_size);
                     table.Update(in[i], i);
                   });

  // remap
  torch::Tensor compact_tensor = torch::empty_like(data);
  thrust::for_each(
      it(0), it(data.numel()),
      [key = key_tensor.data_ptr<IdType>(),
       value = value_tensor.data_ptr<IdType>(), in = data.data_ptr<IdType>(),
       out = compact_tensor.data_ptr<IdType>(), num_items,
       dir_size] __device__(IdType i) mutable {
        RelabelHashmap<IdType> table(key, value, dir_size);
        out[i] = table.SearchForValue(in[i]);
      });

  return {unique_tensor, compact_tensor};
}

std::tuple<torch::Tensor, torch::Tensor> TensorCompact(torch::Tensor data) {
  return TensorCompactCUDA<int64_t>(data);
}

}  // namespace impl
}  // namespace gs