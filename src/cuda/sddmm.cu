#include "./sddmm.h"

#include "./cuda_common.h"
#include "./macro.h"
#include "./selector.h"
#include "./utils.h"

namespace gs {
namespace impl {

/**
 * @brief CUDA kernel of g-SDDMM on Coo format.
 * @note it uses edge parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different edges. Threadblocks
 *       on the x-axis are responsible for the computation on different
 * positions in feature dimension.
 */
template <typename Idx, typename DType, typename BinaryOp,
          bool UseBcast = false, bool UseIdx = false, int LhsTarget = 0,
          int RhsTarget = 2>
__global__ void SDDMMCooKernel(
    const DType* __restrict__ lhs, const DType* __restrict__ rhs,
    DType* __restrict__ out, const Idx* __restrict__ row,
    const Idx* __restrict__ col, const Idx* __restrict__ edge_map, int64_t E,
    int64_t reduce_size, const int64_t* __restrict__ lhs_off,
    const int64_t* __restrict__ rhs_off, int64_t lhs_len, int64_t rhs_len,
    int64_t out_len) {
  // SDDMM with COO.
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = UseIdx ? _ldg(edge_map + ty) : ty;
    const DType* lhsoff =
        BinaryOp::use_lhs
            ? (lhs + Selector<LhsTarget>::Call(src, eid, dst) * lhs_len)
            : nullptr;
    const DType* rhsoff =
        BinaryOp::use_rhs
            ? (rhs + Selector<RhsTarget>::Call(src, eid, dst) * rhs_len)
            : nullptr;
    DType* outoff = out + eid * out_len;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride_x = blockDim.x * gridDim.x;
    while (tx < out_len) {
      const Idx lhs_add = UseBcast ? lhs_off[tx] : tx;
      const Idx rhs_add = UseBcast ? rhs_off[tx] : tx;
      DType val = BinaryOp::Call(lhsoff + lhs_add * reduce_size,
                                 rhsoff + rhs_add * reduce_size, reduce_size);
      outoff[tx] = val;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/**
 * @brief CUDA kernel of SDDMM-dot on Coo format, accelerated with tree
 * reduction.
 * @note it uses edge parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different edges. Threadblocks
 *       on the x-axis are responsible for the computation on different
 * positions in feature dimension.
 */
template <typename Idx, typename DType, bool UseBcast = false,
          bool UseIdx = false, int LhsTarget = 0, int RhsTarget = 2>
__global__ void SDDMMCooTreeReduceKernel(
    const DType* __restrict__ lhs, const DType* __restrict__ rhs,
    DType* __restrict__ out, const Idx* __restrict__ row,
    const Idx* __restrict__ col, const Idx* __restrict__ edge_map, int64_t E,
    int64_t reduce_size, const int64_t* __restrict__ lhs_off,
    const int64_t* __restrict__ rhs_off, int64_t lhs_len, int64_t rhs_len,
    int64_t out_len) {
  Idx ty = blockIdx.x * blockDim.y + threadIdx.y;
  if (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = UseIdx ? _ldg(edge_map + ty) : ty;
    const DType* lhsoff =
        lhs + Selector<LhsTarget>::Call(src, eid, dst) * lhs_len;
    const DType* rhsoff =
        rhs + Selector<RhsTarget>::Call(src, eid, dst) * rhs_len;
    DType* outoff = out + eid * out_len;
    int tx = threadIdx.x;  // tx < 32
    for (int i = blockIdx.y; i < out_len;
         i += gridDim.y) {  // over output feature dimension
      const Idx lhs_add = UseBcast ? __ldg(lhs_off + i) : i;
      const Idx rhs_add = UseBcast ? __ldg(rhs_off + i) : i;
      DType val = reduce::Sum<Idx, DType>::zero();
      for (int j = tx; j < reduce_size; j += 64) {
        val += lhsoff[lhs_add * reduce_size + j] *
               rhsoff[rhs_add * reduce_size + j];
        if (j + 32 < reduce_size)
          val += lhsoff[lhs_add * reduce_size + j + 32] *
                 rhsoff[rhs_add * reduce_size + j + 32];
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
      if (tx == 0) outoff[i] = val;
    }
  }
}

/**
 * @brief CUDA kernel of g-SDDMM on CSC format.
 * @note it uses edge parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different edges. Threadblocks
 *       on the x-axis are responsible for the computation on different
 *       positions in feature dimension. To efficiently find the source node idx
 *       and destination node index of an given edge on CSC format, it uses
 *       binary search (time complexity O(log N)).
 */
template <typename Idx, typename DType, typename BinaryOp,
          bool UseBcast = false, bool UseIdx = false, bool UseNMap = false,
          int LhsTarget = 0, int RhsTarget = 2>
__global__ void SDDMMCSCKernel(
    const DType* __restrict__ lhs, const DType* __restrict__ rhs,
    DType* __restrict__ out, const Idx* __restrict__ indptr,
    const Idx* __restrict__ indices, const Idx* __restrict__ edge_map,
    const Idx* __restrict__ nid_map, int64_t N, int64_t E, int64_t reduce_size,
    const int64_t* __restrict__ lhs_off, const int64_t* __restrict__ rhs_off,
    int64_t lhs_len, int64_t rhs_len, int64_t out_len) {
  // SDDMM with CSC.
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = _ldg(indices + ty);
    const Idx dst = UseNMap ? _ldg(nid_map + cub::UpperBound(indptr, N, ty) - 1)
                            : cub::UpperBound(indptr, N, ty) - 1;
    const Idx eid = UseIdx ? _ldg(edge_map + ty) : ty;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    const DType* lhsoff =
        BinaryOp::use_lhs
            ? (lhs + Selector<LhsTarget>::Call(src, eid, dst) * lhs_len)
            : nullptr;
    const DType* rhsoff =
        BinaryOp::use_rhs
            ? (rhs + Selector<RhsTarget>::Call(src, eid, dst) * rhs_len)
            : nullptr;
    DType* outoff = out + eid * out_len;
    while (tx < out_len) {
      const Idx lhs_add = UseBcast ? lhs_off[tx] : tx;
      const Idx rhs_add = UseBcast ? rhs_off[tx] : tx;
      DType val = BinaryOp::Call(lhsoff + lhs_add * reduce_size,
                                 rhsoff + rhs_add * reduce_size, reduce_size);
      outoff[tx] = val;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/**
 * @brief CUDA implementation of g-SDDMM on Coo format.
 * @param bcast Broadcast information.
 * @param coo The Coo matrix.
 * @param lhs The left hand side operand feature.
 * @param rhs The right hand size operand feature.
 * @param out The result feature on edges.
 */
template <typename Idx, typename DType, typename Op, int LhsTarget = 0,
          int RhsTarget = 2>
void SDDMMCOOCUDA(const BcastOff& bcast, std::shared_ptr<COO> coo,
                  torch::Tensor lhs, torch::Tensor rhs, torch::Tensor out) {
  const int64_t nnz = coo->row.numel();
  const bool use_idx = coo->e_ids.has_value();

  const Idx* row = coo->row.data_ptr<Idx>();
  const Idx* col = coo->col.data_ptr<Idx>();
  const Idx* edge_map = use_idx ? coo->e_ids.value().data_ptr<Idx>() : nullptr;
  const DType* lhs_data = lhs.data_ptr<DType>();
  const DType* rhs_data = rhs.data_ptr<DType>();
  DType* out_data = out.data_ptr<DType>();

  int64_t *lhs_off = nullptr, *rhs_off = nullptr;
  int64_t len = bcast.out_len, lhs_len = bcast.lhs_len, rhs_len = bcast.rhs_len;
  int64_t reduce_dim = bcast.reduce_size;

  if (std::is_same<Op, binary::Dot<DType> >::value && reduce_dim >= 32) {
    const int ntx = 32;  // on feature dimension
    const int nty = 8;   // on out dimension
    const int nbx = (nnz + nty - 1) / nty;
    const int nby = FindNumBlocks<'y'>(len);
    const dim3 nblks(nbx, nby);
    const dim3 nthrs(ntx, nty);
    BCAST_IDX_CTX_SWITCH(bcast, use_idx, lhs_off, rhs_off, {
      CUDA_KERNEL_CALL((SDDMMCooTreeReduceKernel<Idx, DType, UseBcast, UseIdx,
                                                 LhsTarget, RhsTarget>),
                       nblks, nthrs, lhs_data, rhs_data, out_data, row, col,
                       edge_map, nnz, reduce_dim, lhs_off, rhs_off, lhs_len,
                       rhs_len, len);
    });
  } else {
    const int ntx = FindNumThreads(len);
    const int nty = CUDA_MAX_NUM_THREADS / ntx;
    const int nbx = (len + ntx - 1) / ntx;
    const int nby = FindNumBlocks<'y'>((nnz + nty - 1) / nty);
    const dim3 nblks(nbx, nby);
    const dim3 nthrs(ntx, nty);
    BCAST_IDX_CTX_SWITCH(bcast, use_idx, lhs_off, rhs_off, {
      CUDA_KERNEL_CALL((SDDMMCooKernel<Idx, DType, Op, UseBcast, UseIdx,
                                       LhsTarget, RhsTarget>),
                       nblks, nthrs, lhs_data, rhs_data, out_data, row, col,
                       edge_map, nnz, reduce_dim, lhs_off, rhs_off, lhs_len,
                       rhs_len, len);
    });
  }
}

/**
 * @brief CUDA implementation of g-SDDMM on CSC format.
 * @param bcast Broadcast information.
 * @param csc The CSC Graph.
 * @param lhs The left hand side operand feature.
 * @param rhs The right hand size operand feature.
 * @param out The result feature on edges.
 */
template <typename Idx, typename DType, typename Op, int LhsTarget = 0,
          int RhsTarget = 2>
void SDDMMCSCCUDA(const BcastOff& bcast, std::shared_ptr<CSC> csc,
                  torch::optional<torch::Tensor> n_ids, torch::Tensor lhs,
                  torch::Tensor rhs, torch::Tensor out) {
  const bool use_idx = csc->e_ids.has_value();
  const bool use_nid = n_ids.has_value();

  const Idx* indptr = csc->indptr.data_ptr<Idx>();
  const Idx* indices = csc->indices.data_ptr<Idx>();
  const Idx* edge_map = use_idx ? csc->e_ids.value().data_ptr<Idx>() : nullptr;
  const Idx* nid_map = use_nid ? n_ids.value().data_ptr<Idx>() : nullptr;
  const DType* lhs_data = lhs.data_ptr<DType>();
  const DType* rhs_data = rhs.data_ptr<DType>();
  DType* out_data = out.data_ptr<DType>();
  int64_t N = csc->indptr.numel(), E = csc->indices.numel();

  int64_t *lhs_off = nullptr, *rhs_off = nullptr;
  int64_t len = bcast.out_len, lhs_len = bcast.lhs_len, rhs_len = bcast.rhs_len;
  int64_t reduce_dim = bcast.reduce_size;

  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = FindNumBlocks<'y'>((E + nty - 1) / nty);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);

  BCAST_IDX_CTX_SWITCH(bcast, use_idx, lhs_off, rhs_off, {
    SWITCH_IDX(use_idx, use_nid, {
      CUDA_KERNEL_CALL((SDDMMCSCKernel<Idx, DType, Op, UseBcast, UseIdx,
                                       UseNMap, LhsTarget, RhsTarget>),
                       nblks, nthrs, lhs_data, rhs_data, out_data, indptr,
                       indices, edge_map, nid_map, N, E, reduce_dim, lhs_off,
                       rhs_off, lhs_len, rhs_len, len);
    });
  });
}

/**
 * @brief CUDA implementation of g-SDDMM on CSC format.
 */
void SDDMMCSC(const std::string& op, const BcastOff& bcast,
              std::shared_ptr<CSC> csc, torch::optional<torch::Tensor> n_ids,
              torch::Tensor lhs, torch::Tensor rhs, torch::Tensor out,
              int lhs_target, int rhs_target) {
  SWITCH_BITS(32, DType, {
    SWITCH_OP(op, Op, {
      SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, {
        SDDMMCSCCUDA<int64_t, DType, Op, LhsTarget, RhsTarget>(
            bcast, csc, n_ids, lhs, rhs, out);
      });
    });
  });
}

/**
 * @brief CUDA implementation of g-SDDMM on Coo format.
 */
void SDDMMCOO(const std::string& op, const BcastOff& bcast,
              std::shared_ptr<COO> coo, torch::Tensor lhs, torch::Tensor rhs,
              torch::Tensor out, int lhs_target, int rhs_target) {
  SWITCH_BITS(32, DType, {
    SWITCH_OP(op, Op, {
      SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, {
        SDDMMCOOCUDA<int64_t, DType, Op, LhsTarget, RhsTarget>(bcast, coo, lhs,
                                                               rhs, out);
      });
    });
  });
}

}  // namespace impl
}  // namespace gs