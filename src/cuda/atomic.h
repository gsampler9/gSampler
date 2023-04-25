#ifndef GS_CUDA_ATOMIC_H_
#define GS_CUDA_ATOMIC_H_

#include <cuda.h>

namespace gs {
namespace impl {
inline __device__ int64_t AtomicMax(int64_t *const address, const int64_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = long long int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicMax(reinterpret_cast<Type *>(address), static_cast<Type>(val));
}

inline __device__ float AtomicMax(float *const address, const float val) {
  float old;
  old = (val >= 0)
            ? __int_as_float(atomicMax((int *)address, __float_as_int(val)))
            : __uint_as_float(
                  atomicMin((unsigned int *)address, __float_as_uint(val)));

  return old;
}

inline __device__ int32_t AtomicMax(int32_t *const address, const int32_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicMax(reinterpret_cast<Type *>(address), static_cast<Type>(val));
}

inline __device__ int64_t AtomicCAS(int64_t *const address,
                                    const int64_t compare, const int64_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = unsigned long long int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicCAS(reinterpret_cast<Type *>(address),
                   static_cast<Type>(compare), static_cast<Type>(val));
}

inline __device__ uint64_t AtomicCAS(uint64_t *const address,
                                     const uint64_t compare,
                                     const uint64_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = unsigned long long int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicCAS(reinterpret_cast<Type *>(address),
                   static_cast<Type>(compare), static_cast<Type>(val));
}

inline __device__ int32_t AtomicCAS(int32_t *const address,
                                    const int32_t compare, const int32_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicCAS(reinterpret_cast<Type *>(address),
                   static_cast<Type>(compare), static_cast<Type>(val));
}

inline __device__ uint32_t AtomicCAS(uint32_t *const address,
                                     const uint32_t compare,
                                     const uint32_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicCAS(reinterpret_cast<Type *>(address),
                   static_cast<Type>(compare), static_cast<Type>(val));
}

inline __device__ int64_t AtomicMin(int64_t *const address, const int64_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = long long int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicMin(reinterpret_cast<Type *>(address), static_cast<Type>(val));
}

inline __device__ uint64_t AtomicMin(uint64_t *const address,
                                     const uint64_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = unsigned long long int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicMin(reinterpret_cast<Type *>(address), static_cast<Type>(val));
}

inline __device__ int32_t AtomicMin(int32_t *const address, const int32_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicMin(reinterpret_cast<Type *>(address), static_cast<Type>(val));
}

inline __device__ uint32_t AtomicMin(uint32_t *const address,
                                     const uint32_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = unsigned int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicMin(reinterpret_cast<Type *>(address), static_cast<Type>(val));
}

inline __device__ float AtomicMin(float *const address, const float val) {
  float old;
  old = (val >= 0)
            ? __int_as_float(atomicMin((int *)address, __float_as_int(val)))
            : __uint_as_float(
                  atomicMax((unsigned int *)address, __float_as_uint(val)));

  return old;
}

inline __device__ int32_t AtomicAdd(int32_t *const address, const int32_t val) {
  using Type = int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");
  return atomicAdd(reinterpret_cast<Type *>(address), static_cast<Type>(val));
}

inline __device__ int64_t AtomicAdd(int64_t *const address, const int64_t val) {
  using Type = unsigned long long int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");
  return atomicAdd(reinterpret_cast<Type *>(address), static_cast<Type>(val));
}

inline __device__ float AtomicAdd(float *const address, const float val) {
  using Type = float;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");
  return atomicAdd(reinterpret_cast<Type *>(address), static_cast<Type>(val));
}

inline __device__ int32_t AtomicSub(int32_t *const address, const int32_t val) {
  using Type = int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");
  return atomicSub(reinterpret_cast<Type *>(address), static_cast<Type>(val));
}

inline __device__ int64_t AtomicSub(int64_t *const address, const int64_t val) {
  return AtomicAdd(address, (-val));
}

inline __device__ float AtomicSub(float *const address, const float val) {
  return AtomicAdd(address, (-val));
}
}  // namespace impl
}  // namespace gs
#endif