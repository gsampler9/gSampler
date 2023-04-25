#ifndef GS_CUDA_WARPSELECT_UTILS_H_
#define GS_CUDA_WARPSELECT_UTILS_H_
#include "Pair.cuh"

namespace gs {
namespace impl {
namespace warpselect {

constexpr int kWarpSize = 32;

template <typename T>
constexpr __host__ __device__ bool isPowerOf2(T v) {
  return (v && !(v & (v - 1)));
}

template <typename T>
__device__ static inline bool lt(T a, T b) {
  return a < b;
}

template <typename T>
__device__ static inline bool gt(T a, T b) {
  return a > b;
}

__device__ __forceinline__ int _getLaneId() {
  int laneId;
  asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
  return laneId;
}

template <typename T>
static inline __device__ bool eq(T a, T b) {
  return a == b;
}

template <typename U, typename V>
constexpr __host__ __device__ auto divDown(U a, V b) -> decltype(a + b) {
  return (a / b);
}

template <typename U, typename V>
constexpr __host__ __device__ auto roundDown(U a, V b) -> decltype(a + b) {
  return divDown(a, b) * b;
}

template <typename U, typename V>
constexpr __host__ __device__ auto divUp(U a, V b) -> decltype(a + b) {
  return (a + b - 1) / b;
}

template <typename T>
inline __device__ void swap(bool swap, T &x, T &y) {
  T tmp = x;
  x = swap ? y : x;
  y = swap ? tmp : y;
}

template <typename T>
inline __device__ void assign(bool assign, T &x, T y) {
  x = assign ? y : x;
}

template <typename T>
struct Comparator {
  __device__ static inline bool lt(T a, T b) { return a < b; }

  __device__ static inline bool gt(T a, T b) { return a > b; }
};

template <typename T>
inline __device__ T shfl(const T val, int srcLane, int width = kWarpSize) {
  return __shfl_sync(0xffffffff, val, srcLane, width);
}

// CUDA SDK does not provide specializations for T*
template <typename T>
inline __device__ T *shfl(T *const val, int srcLane, int width = kWarpSize) {
  static_assert(sizeof(T *) == sizeof(long long), "pointer size");
  long long v = (long long)val;

  return (T *)shfl(v, srcLane, width);
}

template <typename T>
inline __device__ T shfl_xor(const T val, int laneMask, int width = kWarpSize) {
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(0xffffffff, val, laneMask, width);
#else
  return __shfl_xor(val, laneMask, width);
#endif
}

// CUDA SDK does not provide specializations for T*
template <typename T>
inline __device__ T *shfl_xor(T *const val, int laneMask,
                              int width = kWarpSize) {
  static_assert(sizeof(T *) == sizeof(long long), "pointer size");
  long long v = (long long)val;
  return (T *)shfl_xor(v, laneMask, width);
}

template <typename T, typename U>
inline __device__ Pair<T, U> shfl_xor(const Pair<T, U> &pair, int laneMask,
                                      int width = kWarpSize) {
  return Pair<T, U>(shfl_xor(pair.k, laneMask, width),
                    shfl_xor(pair.v, laneMask, width));
}
}  // namespace warpselect
}  // namespace impl
}  // namespace gs

#endif