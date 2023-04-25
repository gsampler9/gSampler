#ifndef GS_CUDA_WARPSELECT_REDUCTIONS_H_
#define GS_CUDA_WARPSELECT_REDUCTIONS_H_

#include <cuda.h>
#include <limits>
#include "Pair.cuh"
#include "Utils.cuh"

namespace gs {
namespace impl {
namespace warpselect {

template <typename T>
struct _Limits {};

// Unfortunately we can't use constexpr because there is no
// constexpr constructor for half
// FIXME: faiss CPU uses +/-FLT_MAX instead of +/-infinity
constexpr float _kFloatMax = std::numeric_limits<float>::max();
constexpr float _kFloatMin = std::numeric_limits<float>::lowest();

template <>
struct _Limits<float> {
  static __device__ __host__ inline float getMin() { return _kFloatMin; }
  static __device__ __host__ inline float getMax() { return _kFloatMax; }
};

constexpr int32_t _kIntMax = std::numeric_limits<int32_t>::max();
constexpr int32_t _kIntMin = std::numeric_limits<int32_t>::lowest();

template <>
struct _Limits<int32_t> {
  static __device__ __host__ inline int32_t getMin() { return _kIntMin; }
  static __device__ __host__ inline int32_t getMax() { return _kIntMax; }
};

constexpr int64_t _kLongMax = std::numeric_limits<int64_t>::max();
constexpr int64_t _kLongMin = std::numeric_limits<int64_t>::lowest();

template <>
struct _Limits<int64_t> {
  static __device__ __host__ inline int64_t getMin() { return _kLongMin; }
  static __device__ __host__ inline int64_t getMax() { return _kLongMax; }
};

template <typename K, typename V>
struct _Limits<Pair<K, V>> {
  static __device__ __host__ inline Pair<K, V> getMin() {
    return Pair<K, V>(_Limits<K>::getMin(), _Limits<V>::getMin());
  }

  static __device__ __host__ inline Pair<K, V> getMax() {
    return Pair<K, V>(_Limits<K>::getMax(), _Limits<V>::getMax());
  }
};

template <typename T>
struct Max {
  __device__ inline T operator()(T a, T b) const { return gt(a, b) ? a : b; }

  inline __device__ T identity() const { return _Limits<T>::getMin(); }
};

template <typename T>
struct Min {
  __device__ inline T operator()(T a, T b) const { return lt(a, b) ? a : b; }

  inline __device__ T identity() const { return _Limits<T>::getMax(); }
};

template <typename T, typename Op, int ReduceWidth = kWarpSize>
__device__ inline T warpReduceAll(T val, Op op) {
#pragma unroll
  for (int mask = ReduceWidth / 2; mask > 0; mask >>= 1) {
    val = op(val, shfl_xor(val, mask));
  }

  return val;
}

}  // namespace warpselect
}  // namespace impl
}  // namespace gs

#endif