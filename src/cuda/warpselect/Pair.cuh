#ifndef GS_CUDA_WARPSELECT_PAIR_H_
#define GS_CUDA_WARPSELECT_PAIR_H_
#include <cuda.h>
#include "Utils.cuh"

namespace gs {
namespace impl {
namespace warpselect {

/// A simple pair type for CUDA device usage
template <typename K, typename V>
struct Pair {
  constexpr __device__ inline Pair() {}

  constexpr __device__ inline Pair(K key, V value) : k(key), v(value) {}

  __device__ inline bool operator==(const Pair<K, V> &rhs) const {
    return eq(k, rhs.k) && eq(v, rhs.v);
  }

  __device__ inline bool operator!=(const Pair<K, V> &rhs) const {
    return !operator==(rhs);
  }

  __device__ inline bool operator<(const Pair<K, V> &rhs) const {
    return lt(k, rhs.k) || (eq(k, rhs.k) && lt(v, rhs.v));
  }

  __device__ inline bool operator>(const Pair<K, V> &rhs) const {
    return gt(k, rhs.k) || (eq(k, rhs.k) && gt(v, rhs.v));
  }

  K k;
  V v;
};

}  // namespace warpselect
}  // namespace impl
}  // namespace gs
#endif