#include <thrust/execution_policy.h>
#include "../atomic.h"
#include "../cuda_common.h"
#include "../utils.h"
#include "fused_coo_u_op_v.h"
#include "../macro.h"

namespace gs {
namespace impl {

namespace fusion {


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
__global__ void FUSED_COO_U_OP_V_Kernel(
    const DType* __restrict__ lhs1, const DType* __restrict__ rhs1,DType* __restrict__ out1, 
    const DType* __restrict__ lhs2, const DType* __restrict__ rhs2,DType* __restrict__ out2, 
    const Idx* __restrict__ row,
    const Idx* __restrict__ col, int64_t E,
    int64_t reduce_size, const int64_t* __restrict__ lhs_off,
    const int64_t* __restrict__ rhs_off, int64_t lhs_len, int64_t rhs_len,
    int64_t out_len) {
  // SDDMM with COO.
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = gs::impl::_ldg(row + ty);
    const Idx dst = gs::impl::_ldg(col + ty);
    const Idx eid =  ty;
    const DType* lhsoff1 = (lhs1 + src * lhs_len);
    const DType* rhsoff1 =(rhs1 + dst * rhs_len);
    const DType* lhsoff2 = (lhs2 + src * lhs_len);
    const DType* rhsoff2 =(rhs2 + dst * rhs_len);
    DType* outoff1 = out1 + eid * out_len;
    DType* outoff2 = out2 + eid * out_len;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride_x = blockDim.x * gridDim.x;
    while (tx < out_len) {
      const Idx lhs_add1 = UseBcast ? lhsoff1[tx] : tx;
      const Idx rhs_add1 = UseBcast ? rhsoff1[tx] : tx;
      DType val1 = BinaryOp::Call(lhsoff1 + lhs_add1 * reduce_size,
                                 rhsoff1 + rhs_add1 * reduce_size, reduce_size);
      outoff1[tx] = val1;
      const Idx lhs_add2 = UseBcast ? lhsoff2[tx] : tx;
      const Idx rhs_add2 = UseBcast ? rhsoff2[tx] : tx;
      DType val2 = BinaryOp::Call(lhsoff2 + lhs_add2 * reduce_size,
                                 rhsoff2 + rhs_add2 * reduce_size, reduce_size);
      outoff2[tx] = val2;
      tx += stride_x;
    }
    ty += stride_y;
  }
}




template <typename Idx, typename DType, typename Op, int LhsTarget = 0,int RhsTarget = 2>
void FUSED_COO_U_OP_V_CUDA(const BcastOff& bcast,
const std::shared_ptr<COO> coo,
torch::Tensor lhs1,
                  torch::Tensor rhs1,torch::Tensor out1,
                  torch::Tensor lhs2,
                  torch::Tensor rhs2,torch::Tensor out2) {
  const int64_t nnz = coo->row.numel();
  const Idx* row = coo->row.data_ptr<Idx>();
  const Idx* col = coo->col.data_ptr<Idx>();
  const DType* lhs_data1 = Op::use_lhs ? lhs1.data_ptr<DType>() : nullptr;
  const DType* rhs_data1 = Op::use_rhs ? rhs1.data_ptr<DType>() : nullptr;
  const DType* lhs_data2 = Op::use_lhs ? lhs2.data_ptr<DType>() : nullptr;
  const DType* rhs_data2 = Op::use_rhs ? rhs2.data_ptr<DType>() : nullptr;
  DType* out_data1 = out1.data_ptr<DType>();
    DType* out_data2 = out2.data_ptr<DType>();
  int64_t *lhs_off = nullptr;
  int64_t *rhs_off = nullptr;
  int64_t len = bcast.out_len;
  int64_t lhs_len = bcast.lhs_len;
  int64_t rhs_len = bcast.rhs_len;
  int64_t reduce_dim =  bcast.reduce_size;
    const int ntx = FindNumThreads(len);
    const int nty = CUDA_MAX_NUM_THREADS / ntx;
    const int nbx = (len + ntx - 1) / ntx;
    const int nby = FindNumBlocks<'y'>((nnz + nty - 1) / nty);
    const dim3 nblks(nbx, nby);
    const dim3 nthrs(ntx, nty);
    BCAST_IDX_CTX_SWITCH(bcast, false, lhs_off, rhs_off, {
      CUDA_KERNEL_CALL((FUSED_COO_U_OP_V_Kernel<Idx, DType, Op>),
                       nblks, nthrs, lhs_data1, rhs_data1, out_data1, lhs_data2, rhs_data2, out_data2,row, col,
                        nnz, reduce_dim, lhs_off, rhs_off, lhs_len,
                       rhs_len, len);
    });
}




/**
 * @brief CUDA implementation of g-SDDMM on COO format.
 */
void FUSED_COO_U_OP_V(const std::string& op, const BcastOff& bcast,
              const std::shared_ptr<COO> coo, torch::Tensor lhs1,
              torch::Tensor rhs1, torch::Tensor out1, torch::Tensor lhs2,
              torch::Tensor rhs2, torch::Tensor out2) {
  ID_TYPE_SWITCH(coo->row.scalar_type(), IdType, {         
    FLOAT_TYPE_SWITCH(out1.scalar_type(), DType, {
      SWITCH_OP(op, Op, {   
      FUSED_COO_U_OP_V_CUDA<IdType, DType, Op>(bcast, coo,lhs1,rhs1, out1,lhs2,rhs2, out2);
                    });
          });
    });
}

    }  // namespace fusion
}  // namespace impl
}  // namespace gs