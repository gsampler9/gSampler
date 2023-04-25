#include "graph_ops.h"

#include <curand_kernel.h>
#include "atomic.h"
#include "cuda_common.h"
#include "macro.h"
#include "utils.h"

namespace gs {
namespace impl {
//////////////////////// CSCSumCUDA //////////////////////////
/**
 * @brief SpMV for CSCSum
 */
template <typename IdType, typename DType, bool UseEMap, bool UseNMap>
__global__ void _SegmentSumKernel(IdType* indptr, IdType* EMap, IdType* NMap,
                                  DType* data, int num_rows, int powk,
                                  int out_len, DType* out) {
  // SPMM with CSR.
  int ty = blockIdx.x * blockDim.y + threadIdx.y;
  const IdType stride_y = blockDim.y * gridDim.x;
  const int stride_x = blockDim.x * gridDim.y;
  while (ty < num_rows) {
    int tx = blockIdx.y * blockDim.x + threadIdx.x;
    while (tx < out_len) {
      DType local_accum = 0;
      for (IdType i = indptr[ty]; i < indptr[ty + 1]; ++i) {
        const IdType data_idx = UseEMap ? EMap[i] : i;
        const DType* dataoff = data + data_idx * out_len;
        DType tmp = powk == 1 ? dataoff[tx] : __powf(dataoff[tx], powk);
        local_accum += tmp;
      }
      int out_pos = UseNMap ? NMap[ty * out_len + tx] : ty * out_len + tx;
      out[out_pos] = local_accum;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/**
 * @brief SpMMCOO for graphSum
 */
template <typename IdType, typename DType, bool UseEMap>
__global__ void _SegmentSumCOOKernel(IdType* target, IdType* EMap, DType* data,
                                     int64_t E, int powk, int out_len,
                                     DType* out) {
  // SPMM with COO.
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  const IdType stride_y = blockDim.y * gridDim.y;
  const int64_t stride_x = blockDim.x * gridDim.x;
  while (ty < E) {
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const IdType data_idx = UseEMap ? EMap[ty] : ty;
    const DType* dataoff = data + data_idx * out_len;
    DType* outoff = out + target[ty] * out_len;
    while (tx < out_len) {
      DType val = powk == 1 ? dataoff[tx] : __powf(dataoff[tx], powk);
      AtomicAdd(outoff + tx, val);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

template <typename IdType, typename DType>
void CSCSum(torch::Tensor indptr, torch::optional<torch::Tensor> e_ids,
            torch::optional<torch::Tensor> n_ids, torch::Tensor data,
            torch::Tensor out_data, int64_t powk) {
  auto num_element = indptr.numel() - 1;
  auto use_n_map = n_ids.has_value(), use_e_map = e_ids.has_value();
  auto n_ids_map = use_n_map ? n_ids.value().data_ptr<IdType>() : nullptr;
  auto e_ids_map = use_e_map ? e_ids.value().data_ptr<IdType>() : nullptr;

  // Aligning DGL
  const int out_len = 1;

  const int ntx = 1;
  const int nty = 256;
  const int nby = (out_len + ntx - 1) / ntx;
  const int nbx = (num_element + nty - 1) / nty;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  SWITCH_IDX(use_e_map, use_n_map, {
    CUDA_KERNEL_CALL((_SegmentSumKernel<IdType, DType, UseEMap, UseNMap>),
                     nblks, nthrs, indptr.data_ptr<IdType>(), e_ids_map,
                     n_ids_map, data.data_ptr<DType>(), num_element, powk,
                     out_len, out_data.data_ptr<DType>());
  });
}

void CSCSumCUDA(torch::Tensor indptr, torch::optional<torch::Tensor> e_ids,
                torch::optional<torch::Tensor> n_ids, torch::Tensor data,
                torch::Tensor out_data, int64_t powk) {
  CSCSum<int64_t, float>(indptr, e_ids, n_ids, data, out_data, powk);
}

template <typename IdType, typename DType>
void COOSum(torch::Tensor target, torch::optional<torch::Tensor> e_ids,
            torch::Tensor data, torch::Tensor out_data, int64_t powk) {
  int64_t E = target.numel();
  auto use_e_map = e_ids.has_value();
  auto e_ids_map = use_e_map ? e_ids.value().data_ptr<IdType>() : nullptr;

  // Aligning DGL
  const int out_len = 1;

  const int ntx = FindNumThreads(out_len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (out_len + ntx - 1) / ntx;
  const int nby = FindNumBlocks<'y'>((E + nty - 1) / nty);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  SWITCH_IDX(use_e_map, false, {
    CUDA_KERNEL_CALL((_SegmentSumCOOKernel<IdType, DType, UseEMap>), nblks,
                     nthrs, target.data_ptr<IdType>(), e_ids_map,
                     data.data_ptr<DType>(), E, powk, out_len,
                     out_data.data_ptr<DType>());
  });
}

void COOSumCUDA(torch::Tensor target, torch::optional<torch::Tensor> e_ids,
                torch::Tensor data, torch::Tensor out_data, int64_t powk) {
  COOSum<int64_t, float>(target, e_ids, data, out_data, powk);
}

//////////////////////// CSCNormalizeCUDA //////////////////////////
/**
 * @brief SpMV for CSCNormalize
 */
template <typename IdType, typename DType, bool UseEMap>
__global__ void _SegmentNormalizeKernel(IdType* indptr, IdType* EMap,
                                        DType* data, int num_rows, int out_len,
                                        DType* out) {
  // SPMM with CSR.
  int ty = blockIdx.x * blockDim.y + threadIdx.y;
  const IdType stride_y = blockDim.y * gridDim.x;
  const int stride_x = blockDim.x * gridDim.y;
  while (ty < num_rows) {
    int tx = blockIdx.y * blockDim.x + threadIdx.x;
    while (tx < out_len) {
      DType local_accum = 0;
      for (IdType i = indptr[ty]; i < indptr[ty + 1]; ++i) {
        const IdType data_idx = UseEMap ? EMap[i] : i;
        const DType* dataoff = data + data_idx * out_len;
        local_accum += dataoff[tx];
      }
      for (IdType i = indptr[ty]; i < indptr[ty + 1]; ++i) {
        const IdType data_idx = UseEMap ? EMap[i] : i;
        const DType* indataoff = data + data_idx * out_len;
        DType* outdataoff = out + data_idx * out_len;
        outdataoff[tx] = indataoff[tx] / local_accum;
      }
      tx += stride_x;
    }
    ty += stride_y;
  }
}

template <typename IdType, typename DType>
void CSCNormalize(torch::Tensor indptr, torch::optional<torch::Tensor> e_ids,
                  torch::Tensor data, torch::Tensor out_data) {
  auto num_element = indptr.numel() - 1;
  auto use_e_map = e_ids.has_value();
  auto e_ids_map = use_e_map ? e_ids.value().data_ptr<IdType>() : nullptr;

  // Aligning DGL
  const int out_len = 1;

  const int ntx = 1;
  const int nty = 256;
  const int nby = (out_len + ntx - 1) / ntx;
  const int nbx = (num_element + nty - 1) / nty;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  SWITCH_IDX(use_e_map, false, {
    CUDA_KERNEL_CALL((_SegmentNormalizeKernel<IdType, DType, UseEMap>), nblks,
                     nthrs, indptr.data_ptr<IdType>(), e_ids_map,
                     data.data_ptr<DType>(), num_element, out_len,
                     out_data.data_ptr<DType>());
  });
}

void CSCNormalizeCUDA(torch::Tensor indptr,
                      torch::optional<torch::Tensor> e_ids, torch::Tensor data,
                      torch::Tensor out_data) {
  CSCNormalize<int64_t, float>(indptr, e_ids, data, out_data);
}

template <typename IdType>
__global__ void _CSCSplitKernel(IdType* indptr, IdType* out, IdType* out_sizes,
                                int64_t num_split, int64_t split_size,
                                int64_t last_size) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
  while (row < num_split) {
    int64_t inoff = row * split_size;
    int64_t outoff = row * (split_size + 1);
    int64_t size = (row == num_split - 1) ? last_size : split_size;
    IdType prefix = indptr[inoff];
    out[outoff] = 0;
    out_sizes[row] = indptr[inoff + size] - prefix;
    for (int idx = threadIdx.x; idx < size; idx += blockDim.x) {
      out[outoff + idx + 1] = indptr[inoff + idx + 1] - prefix;
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType>
std::vector<std::vector<torch::Tensor>> _CSCSplit(
    torch::Tensor indptr, torch::Tensor indices,
    torch::optional<torch::Tensor> eid, int64_t split_size) {
  std::vector<torch::Tensor> sub_indptrs;
  std::vector<torch::Tensor> sub_indices;
  std::vector<torch::Tensor> sub_eids;

  auto total_element = indptr.numel() - 1;
  auto num_split = total_element / split_size;
  auto redundant = total_element - num_split * split_size;
  int64_t total_indptrs_len;
  if (redundant != 0) {
    total_indptrs_len = num_split * (split_size + 1) + redundant + 1;
    num_split += 1;
  } else {
    total_indptrs_len = num_split * (split_size + 1);
    redundant = split_size;
  }
  auto total_indptrs = torch::empty(total_indptrs_len, indptr.options());
  auto indices_split_sizes = torch::empty(num_split, indices.options());

  dim3 block(64, 8);
  dim3 grid((num_split + block.y - 1) / block.y);
  CUDA_KERNEL_CALL((_CSCSplitKernel<IdType>), grid, block,
                   indptr.data_ptr<IdType>(), total_indptrs.data_ptr<IdType>(),
                   indices_split_sizes.data_ptr<IdType>(), num_split,
                   split_size, redundant);

  sub_indptrs = torch::split(total_indptrs, split_size + 1);

  indices_split_sizes = indices_split_sizes.to(torch::kCPU);
  auto data_ptr = indices_split_sizes.data_ptr<IdType>();
  std::vector<IdType> subindices_sizes(data_ptr, data_ptr + num_split);
  sub_indices = torch::split_with_sizes(indices, subindices_sizes);
  if (eid.has_value())
    sub_eids = torch::split_with_sizes(eid.value(), subindices_sizes);

  return {sub_indptrs, sub_indices, sub_eids};
}

std::vector<std::vector<torch::Tensor>> CSCSplitCUDA(
    torch::Tensor indptr, torch::Tensor indices,
    torch::optional<torch::Tensor> eid, int64_t split_size) {
  return _CSCSplit<int64_t>(indptr, indices, eid, split_size);
}
}  // namespace impl
}  // namespace gs