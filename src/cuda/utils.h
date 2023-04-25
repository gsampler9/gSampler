#ifndef GS_CUDA_UTILS_H_
#define GS_CUDA_UTILS_H_

#include <c10/cuda/CUDACachingAllocator.h>
#include <cub/cub.cuh>
#include "../logging.h"
#include "cuda_common.h"

namespace gs {
namespace impl {

#define CUDA_MAX_NUM_BLOCKS_X 0x7FFFFFFF
#define CUDA_MAX_NUM_BLOCKS_Y 0xFFFF
#define CUDA_MAX_NUM_BLOCKS_Z 0xFFFF
#define CUDA_MAX_NUM_THREADS 256  // The max number of threads per block

template <typename IdType>
void cub_exclusiveSum(IdType* arrays, const IdType array_length) {
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, arrays,
                                arrays, array_length);
  d_temp_storage =
      c10::cuda::CUDACachingAllocator::raw_alloc(temp_storage_bytes);
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, arrays,
                                arrays, array_length);
  c10::cuda::CUDACachingAllocator::raw_delete(d_temp_storage);
}

template <typename IdType>
void cub_inclusiveSum(IdType* arrays, int32_t array_length) {
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, arrays,
                                arrays, array_length);

  d_temp_storage =
      c10::cuda::CUDACachingAllocator::raw_alloc(temp_storage_bytes);
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, arrays,
                                arrays, array_length);
  c10::cuda::CUDACachingAllocator::raw_delete(d_temp_storage);
}

template <typename KeyType, typename ValueType>
void cub_sortPairs(KeyType* d_keys_in, KeyType* d_keys_out,
                   ValueType* d_values_in, ValueType* d_values_out,
                   int32_t num_items, int num_bits) {
  if (num_bits == 0) {
    num_bits = sizeof(KeyType) * 8;
  }
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in,
                                  d_keys_out, d_values_in, d_values_out,
                                  num_items, 0, num_bits);
  d_temp_storage =
      c10::cuda::CUDACachingAllocator::raw_alloc(temp_storage_bytes);
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in,
                                  d_keys_out, d_values_in, d_values_out,
                                  num_items, 0, num_bits);
  c10::cuda::CUDACachingAllocator::raw_delete(d_temp_storage);
}

template <typename KeyType, typename ValueType>
void cub_sortPairsDescending(KeyType* d_keys_in, KeyType* d_keys_out,
                             ValueType* d_values_in, ValueType* d_values_out,
                             int32_t num_items, int num_bits) {
  if (num_bits == 0) {
    num_bits = sizeof(KeyType) * 8;
  }
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, 0, num_bits);
  // Allocate temporary storage
  d_temp_storage =
      c10::cuda::CUDACachingAllocator::raw_alloc(temp_storage_bytes);
  // Run sorting operation
  cub::DeviceRadixSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, 0, num_bits);
  c10::cuda::CUDACachingAllocator::raw_delete(d_temp_storage);
}

template <typename KeyType, typename ValueType>
void cub_segmentedSortPairsDescending(KeyType* d_keys_in, KeyType* d_keys_out,
                                      ValueType* d_values_in,
                                      ValueType* d_values_out,
                                      int64_t* d_offsets, int32_t num_items,
                                      int32_t num_segments) {
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, num_segments, d_offsets, d_offsets + 1);
  // Allocate temporary storage
  d_temp_storage =
      c10::cuda::CUDACachingAllocator::raw_alloc(temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in,
      d_values_out, num_items, num_segments, d_offsets, d_offsets + 1);
  c10::cuda::CUDACachingAllocator::raw_delete(d_temp_storage);
}

template <typename IdType, typename DType>
void cub_segmentedSum(DType* d_in, DType* d_out, IdType* d_offsets,
                      int64_t num_segments) {
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_in,
                                  d_out, num_segments, d_offsets,
                                  d_offsets + 1);
  // Allocate temporary storage
  d_temp_storage =
      c10::cuda::CUDACachingAllocator::raw_alloc(temp_storage_bytes);
  // Run sum-reduction
  cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_in,
                                  d_out, num_segments, d_offsets,
                                  d_offsets + 1);
  c10::cuda::CUDACachingAllocator::raw_delete(d_temp_storage);
}

template <typename IdType>
void cub_consecutiveUnique(IdType* d_in, IdType* d_out,
                           IdType* d_num_selected_out, int64_t num_items) {
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes, d_in, d_out,
                            d_num_selected_out, num_items);
  // Allocate temporary storage
  d_temp_storage =
      c10::cuda::CUDACachingAllocator::raw_alloc(temp_storage_bytes);
  // Run sum-reduction
  cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes, d_in, d_out,
                            d_num_selected_out, num_items);
  c10::cuda::CUDACachingAllocator::raw_delete(d_temp_storage);
}

template <typename KeyType, typename ValueType>
void cub_uniqueByKey(KeyType* d_keys_in, KeyType* d_keys_out,
                     ValueType* d_values_in, ValueType* d_values_out,
                     int64_t num_items, int64_t* d_num_selected_out) {
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::UniqueByKey(d_temp_storage, temp_storage_bytes, d_keys_in,
                                 d_values_in, d_keys_out, d_values_out,
                                 d_num_selected_out, num_items);
  // Allocate temporary storage
  d_temp_storage =
      c10::cuda::CUDACachingAllocator::raw_alloc(temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSelect::UniqueByKey(d_temp_storage, temp_storage_bytes, d_keys_in,
                                 d_values_in, d_keys_out, d_values_out,
                                 d_num_selected_out, num_items);
  c10::cuda::CUDACachingAllocator::raw_delete(d_temp_storage);
}

template <typename IdType>
void cub_segmentSort(IdType* d_keys_in, IdType* d_keys_out, IdType* d_offsets,
                     int64_t num_items, int64_t num_segments) {
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items,
      num_segments, d_offsets, d_offsets + 1);
  // Allocate temporary storage
  d_temp_storage =
      c10::cuda::CUDACachingAllocator::raw_alloc(temp_storage_bytes);
  // Run sorting operation
  cub::DeviceSegmentedRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items,
      num_segments, d_offsets, d_offsets + 1);
  c10::cuda::CUDACachingAllocator::raw_delete(d_temp_storage);
}

/*! \brief Calculate the number of threads needed given the dimension length.
 *
 * It finds the biggest number that is smaller than min(dim, max_nthrs)
 * and is also power of two.
 */
inline int FindNumThreads(int dim, int max_nthrs = CUDA_MAX_NUM_THREADS) {
  CHECK_GE(dim, 0);
  if (dim == 0) return 1;
  int ret = max_nthrs;
  while (ret > dim) {
    ret = ret >> 1;
  }
  return ret;
}

/*
 * !\brief Find number of blocks is smaller than nblks and max_nblks
 * on the given axis ('x', 'y' or 'z').
 */
template <char axis>
inline int FindNumBlocks(int nblks, int max_nblks = -1) {
  int default_max_nblks = -1;
  switch (axis) {
    case 'x':
      default_max_nblks = CUDA_MAX_NUM_BLOCKS_X;
      break;
    case 'y':
      default_max_nblks = CUDA_MAX_NUM_BLOCKS_Y;
      break;
    case 'z':
      default_max_nblks = CUDA_MAX_NUM_BLOCKS_Z;
      break;
    default:
      LOG(FATAL) << "Axis " << axis << " not recognized";
      break;
  }
  if (max_nblks == -1) max_nblks = default_max_nblks;
  CHECK_NE(nblks, 0);
  if (nblks < max_nblks) return nblks;
  return max_nblks;
}

template <typename T>
__device__ __forceinline__ T _ldg(T* addr) {
#if __CUDA_ARCH__ >= 350
  return __ldg(addr);
#else
  return *addr;
#endif
}

#define SWITCH_BITS(bits, DType, ...)                              \
  do {                                                             \
    if ((bits) == 32) {                                            \
      typedef float DType;                                         \
      { __VA_ARGS__ }                                              \
    } else if ((bits) == 64) {                                     \
      typedef double DType;                                        \
      { __VA_ARGS__ }                                              \
    } else {                                                       \
      LOG(FATAL) << "Data type not recognized with bits " << bits; \
    }                                                              \
  } while (0)
}  // namespace impl
}  // namespace gs

#endif