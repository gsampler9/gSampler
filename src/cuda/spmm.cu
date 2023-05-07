#include "./spmm.h"

#include "./cuda_common.h"
#include "./macro.h"
#include "./selector.h"
#include "./utils.h"

namespace gs {
namespace impl {
/**
 * @brief CUDA kernel of g-SpMM on Coo format.
 * @note it uses edge parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different edges. Threadblocks
 *       on the x-axis are responsible for the computation on different
 * positions in feature dimension. To avoid possible data hazards, it uses
 * atomic operators for reduction.
 */
template <typename Idx, typename DType, typename BinaryOp, typename ReduceOp,
          bool UseBcast = false, bool UseIdx = false>
__global__ void SpMMCOOKernel(
    const DType* __restrict__ ufeat, const DType* __restrict__ efeat,
    DType* __restrict__ out, Idx* __restrict__ arg_u, Idx* __restrict__ arg_e,
    const Idx* __restrict__ row, const Idx* __restrict__ col,
    const Idx* __restrict__ edge_map, int64_t E,
    const int64_t* __restrict__ ubcast_off,
    const int64_t* __restrict__ ebcast_off, int64_t ufeat_len,
    int64_t efeat_len, int64_t out_len, int64_t u_target) {
  // SPMM with COO.
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = (u_target == 0) ? _ldg(row + ty) : _ldg(col + ty);
    const Idx dst = (u_target == 0) ? _ldg(col + ty) : _ldg(row + ty);
    const Idx eid = UseIdx ? _ldg(edge_map + ty) : ty;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    const DType* uoff = BinaryOp::use_lhs ? (ufeat + src * ufeat_len) : nullptr;
    const DType* eoff = BinaryOp::use_rhs ? (efeat + eid * efeat_len) : nullptr;
    DType* outoff = out + dst * out_len;
    while (tx < out_len) {
      const int64_t lhs_add = UseBcast ? ubcast_off[tx] : tx;
      const int64_t rhs_add = UseBcast ? ebcast_off[tx] : tx;
      DType val = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
      Idx* arguoff = nullptr;  // arguoff is not used in SpMMCoo.
      Idx* argeoff = nullptr;  // argeoff is not used in SpMMCoo.
      ReduceOp::Call(outoff + tx, arguoff, argeoff, val, src, eid);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/**
 * @brief CUDA kernel to compute argu and arge in g-SpMM on Coo format.
 * @note it uses edge parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different edges. Threadblocks
 *       on the x-axis are responsible for the computation on different
 * positions in feature dimension.
 */
template <typename Idx, typename DType, typename BinaryOp, typename ReduceOp,
          bool UseBcast = false, bool UseIdx = false>
__global__ void ArgSpMMCOOKernel(
    const DType* __restrict__ ufeat, const DType* __restrict__ efeat,
    DType* __restrict__ out, Idx* __restrict__ arg_u, Idx* __restrict__ arg_e,
    const Idx* __restrict__ row, const Idx* __restrict__ col,
    const Idx* __restrict__ edge_map, int64_t E,
    const int64_t* __restrict__ ubcast_off,
    const int64_t* __restrict__ ebcast_off, int64_t ufeat_len,
    int64_t efeat_len, int64_t out_len, int64_t u_target) {
  // SPMM with COO arg max/min.
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = (u_target == 0) ? _ldg(row + ty) : _ldg(col + ty);
    const Idx dst = (u_target == 0) ? _ldg(col + ty) : _ldg(row + ty);
    const Idx eid = UseIdx ? _ldg(edge_map + ty) : ty;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    const DType* uoff = BinaryOp::use_lhs ? (ufeat + src * ufeat_len) : nullptr;
    const DType* eoff = BinaryOp::use_rhs ? (efeat + eid * efeat_len) : nullptr;
    const DType* outoff = out + dst * out_len;
    Idx* arguoff = BinaryOp::use_lhs ? (arg_u + dst * out_len) : nullptr;
    Idx* argeoff = BinaryOp::use_rhs ? (arg_e + dst * out_len) : nullptr;
    while (tx < out_len) {
      int64_t lhs_add = UseBcast ? ubcast_off[tx] : tx;
      int64_t rhs_add = UseBcast ? ebcast_off[tx] : tx;
      DType val = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
      ReduceOp::CallArg(tx, arguoff, argeoff, val, outoff[tx], src, eid);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/**
 * @brief CUDA kernel of g-SpMM on CSC format.
 * @note it uses node parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different destination nodes.
 *       Threadblocks on the x-axis are responsible for the computation on
 *       different positions in feature dimension.
 */
template <typename Idx, typename DType, typename BinaryOp, typename ReduceOp,
          bool UseBcast = false, bool UseIdx = false>
__global__ void SpMMCSCKernel(
    const DType* __restrict__ ufeat, const DType* __restrict__ efeat,
    DType* __restrict__ out, Idx* __restrict__ arg_u, Idx* __restrict__ arg_e,
    const Idx* __restrict__ indptr, const Idx* __restrict__ indices,
    const Idx* __restrict__ edge_map, int64_t num_cols,
    const int64_t* __restrict__ ubcast_off,
    const int64_t* __restrict__ ebcast_off, int64_t ufeat_len,
    int64_t efeat_len, int64_t out_len) {
  // SPMM with CSC.
  int ty = blockIdx.x * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.x;
  const int stride_x = blockDim.x * gridDim.y;
  while (ty < num_cols) {
    int tx = blockIdx.y * blockDim.x + threadIdx.x;
    while (tx < out_len) {
      DType local_accum = 0;
      Idx local_argu = 0, local_arge = 0;
      const int lhs_add = UseBcast ? ubcast_off[tx] : tx;
      const int rhs_add = UseBcast ? ebcast_off[tx] : tx;
      for (Idx i = indptr[ty]; i < indptr[ty + 1]; ++i) {
        const Idx eid = UseIdx ? _ldg(edge_map + i) : i;
        const Idx cid = _ldg(indices + i);
        const DType* uoff =
            BinaryOp::use_lhs ? (ufeat + cid * ufeat_len) : nullptr;
        const DType* eoff =
            BinaryOp::use_rhs ? (efeat + eid * efeat_len) : nullptr;
        DType out = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
        ReduceOp::Call(&local_accum, &local_argu, &local_arge, out, cid, eid);
      }
      int out_pos = ty * out_len + tx;
      out[out_pos] += local_accum;
      if (ReduceOp::require_arg && BinaryOp::use_lhs)
        arg_u[out_pos] = local_argu;
      if (ReduceOp::require_arg && BinaryOp::use_rhs)
        arg_e[out_pos] = local_arge;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/**
 * @brief CUDA implementation of g-SpMM on COO format.
 * @param bcast Broadcast information.
 * @param coo The Coo matrix.
 * @param ufeat The feature on source nodes.
 * @param efeat The feature on edges.
 * @param out The result feature on destination nodes.
 * @param argu Arg-Min/Max on source nodes, which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 * @param arge Arg-Min/Max on edges. which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 */
template <typename Idx, typename DType, typename BinaryOp, typename ReduceOp>
void SpMMCOOCUDA(const BcastOff& bcast, const std::shared_ptr<COO> coo,
                 torch::Tensor ufeat, torch::Tensor efeat, torch::Tensor out,
                 int64_t u_target, torch::Tensor argu, torch::Tensor arge) {
  const bool use_idx = coo->e_ids.has_value();

  const Idx *row = coo->row.data_ptr<Idx>(), *col = coo->col.data_ptr<Idx>(),
            *edge_map = use_idx ? coo->e_ids.value().data_ptr<Idx>() : nullptr;
  const DType *ufeat_data =
                  BinaryOp::use_lhs ? ufeat.data_ptr<DType>() : nullptr,
              *efeat_data =
                  BinaryOp::use_rhs ? efeat.data_ptr<DType>() : nullptr;
  Idx *argu_data = (ReduceOp::require_arg && BinaryOp::use_lhs)
                       ? argu.data_ptr<Idx>()
                       : nullptr,
      *arge_data = (ReduceOp::require_arg && BinaryOp::use_rhs)
                       ? arge.data_ptr<Idx>()
                       : nullptr;
  DType* out_data = out.data_ptr<DType>();
  const int64_t E = coo->row.numel();

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = bcast.out_len, lhs_len = bcast.lhs_len, rhs_len = bcast.rhs_len;

  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = FindNumBlocks<'y'>((E + nty - 1) / nty);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);

  BCAST_IDX_CTX_SWITCH(bcast, use_idx, ubcast_off, ebcast_off, {
    CUDA_KERNEL_CALL(
        (SpMMCOOKernel<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
        nblks, nthrs, ufeat_data, efeat_data, out_data, argu_data, arge_data,
        row, col, edge_map, E, ubcast_off, ebcast_off, lhs_len, rhs_len, len,
        u_target);
    if (ReduceOp::require_arg) {
      CUDA_KERNEL_CALL(
          (ArgSpMMCOOKernel<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
          nblks, nthrs, ufeat_data, efeat_data, out_data, argu_data, arge_data,
          row, col, edge_map, E, ubcast_off, ebcast_off, lhs_len, rhs_len, len,
          u_target);
    }
  });
}

/**
 * @brief CUDA implementation of g-SpMM on Csr format.
 * @param bcast Broadcast information.
 * @param csr The Csr matrix.
 * @param ufeat The feature on source nodes.
 * @param efeat The feature on edges.
 * @param out The result feature on destination nodes.
 * @param argu Arg-Min/Max on source nodes, which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 * @param arge Arg-Min/Max on edges. which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max
 * reducer.
 */
template <typename Idx, typename DType, typename BinaryOp, typename ReduceOp>
void SpMMCSCCUDA(const BcastOff& bcast, const std::shared_ptr<CSC> csc,
                 torch::Tensor ufeat, torch::Tensor efeat, torch::Tensor out,
                 torch::Tensor argu, torch::Tensor arge) {
  const bool use_idx = csc->e_ids.has_value();

  const Idx* indptr = csc->indptr.data_ptr<Idx>();
  const Idx* indices = csc->indices.data_ptr<Idx>();
  const Idx* edge_map = use_idx ? csc->e_ids.value().data_ptr<Idx>() : nullptr;
  const DType* ufeat_data =
      BinaryOp::use_lhs ? ufeat.data_ptr<DType>() : nullptr;
  const DType* efeat_data =
      BinaryOp::use_rhs ? efeat.data_ptr<DType>() : nullptr;
  Idx* argu_data = (ReduceOp::require_arg && BinaryOp::use_lhs)
                       ? argu.data_ptr<Idx>()
                       : nullptr;
  Idx* arge_data = (ReduceOp::require_arg && BinaryOp::use_rhs)
                       ? arge.data_ptr<Idx>()
                       : nullptr;
  DType* out_data = out.data_ptr<DType>();
  const int64_t num_cols = csc->indptr.numel() - 1;

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = bcast.out_len, lhs_len = bcast.lhs_len, rhs_len = bcast.rhs_len;
  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nby = (len + ntx - 1) / ntx;
  const int nbx = FindNumBlocks<'x'>((num_cols + nty - 1) / nty);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);

  BCAST_IDX_CTX_SWITCH(bcast, use_idx, ubcast_off, ebcast_off, {
    SWITCH_IDX(use_idx, {
      CUDA_KERNEL_CALL(
          (SpMMCSCKernel<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
          nblks, nthrs, ufeat_data, efeat_data, out_data, argu_data, arge_data,
          indptr, indices, edge_map, num_cols, ubcast_off, ebcast_off, lhs_len,
          rhs_len, len);
    });
  });
}

/**
 * @brief CUDA implementation of g-SpMM on CSC format.
 */
void SpMMCSC(const std::string& op, const std::string& reduce,
             const BcastOff& bcast, const std::shared_ptr<CSC> csc,
             torch::Tensor ufeat, torch::Tensor efeat, torch::Tensor out,
             std::vector<torch::Tensor> out_aux) {
  if (out.scalar_type() != torch::kFloat32)
    LOG(FATAL)
        << "Currently g-SpMM on COO format only support 32 bits float data.";
  if (reduce == "sum") {
    ID_TYPE_SWITCH(csc->indices.scalar_type(), IdType, {
      typedef float DType;
      SWITCH_OP(op, Op, {
        SpMMCSCCUDA<IdType, DType, Op, impl::reduce::Sum<IdType, DType> >(
            bcast, csc, ufeat, efeat, out, out_aux[0], out_aux[1]);
      });
    });
  } else if (reduce == "max") {
    ID_TYPE_SWITCH(csc->indices.scalar_type(), IdType, {
      typedef float DType;
      SWITCH_OP(op, Op, {
        SpMMCSCCUDA<IdType, DType, Op, impl::reduce::Max<IdType, DType> >(
            bcast, csc, ufeat, efeat, out, out_aux[0], out_aux[1]);
      });
    });
  } else if (reduce == "min") {
    ID_TYPE_SWITCH(csc->indices.scalar_type(), IdType, {
      typedef float DType;
      SWITCH_OP(op, Op, {
        SpMMCSCCUDA<IdType, DType, Op, impl::reduce::Min<IdType, DType> >(
            bcast, csc, ufeat, efeat, out, out_aux[0], out_aux[1]);
      });
    });
  } else {
    LOG(FATAL) << "Not implemented warning";
  }
}

/**
 * @brief CUDA implementation of g-SpMM on COO format.
 */
void SpMMCOO(const std::string& op, const std::string& reduce,
             const BcastOff& bcast, const std::shared_ptr<COO> coo,
             torch::Tensor ufeat, torch::Tensor efeat, torch::Tensor out,
             int64_t u_target, std::vector<torch::Tensor> out_aux) {
  if (out.scalar_type() != torch::kFloat32)
    LOG(FATAL)
        << "Currently g-SpMM on COO format only support 32 bits float data.";
  if (reduce == "sum") {
    ID_TYPE_SWITCH(coo->row.scalar_type(), IdType, {
      typedef float DType;
      SWITCH_OP(op, Op, {
        SpMMCOOCUDA<IdType, DType, Op, impl::reduce::Sum<IdType, DType, true> >(
            bcast, coo, ufeat, efeat, out, u_target, torch::Tensor(),
            torch::Tensor());
      });
    });
  } else if (reduce == "max") {
    ID_TYPE_SWITCH(coo->row.scalar_type(), IdType, {
      typedef float DType;
      SWITCH_OP(op, Op, {
        SpMMCOOCUDA<IdType, DType, Op, impl::reduce::Max<IdType, DType, true> >(
            bcast, coo, ufeat, efeat, out, u_target, out_aux[0], out_aux[1]);
      });
    });
  } else if (reduce == "min") {
    ID_TYPE_SWITCH(coo->row.scalar_type(), IdType, {
      typedef float DType;
      SWITCH_OP(op, Op, {
        SpMMCOOCUDA<IdType, DType, Op, impl::reduce::Min<IdType, DType, true> >(
            bcast, coo, ufeat, efeat, out, u_target, out_aux[0], out_aux[1]);
      });
    });
  } else {
    LOG(FATAL) << "Not implemented warning";
  }
}
}  // namespace impl
}  // namespace gs