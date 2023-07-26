#include <thrust/execution_policy.h>
#include "../atomic.h"
#include "../cuda_common.h"
#include "../utils.h"
#include "fused_coo_e_div_u_sum.h"
#include "../macro.h"

namespace gs {
namespace impl {

namespace fusion {


template <typename Idx, typename DType, typename BinaryOp, typename ReduceOp,
          bool UseBcast = false, bool UseIdx = false>
__global__ void COOEDivUSumKernel(
    const DType* __restrict__ ufeat, const DType* __restrict__ efeat,
    DType* __restrict__ out, Idx* __restrict__ arg_u, Idx* __restrict__ arg_e,
    const Idx* __restrict__ row, const Idx* __restrict__ col,
    const Idx* __restrict__ edge_map, int64_t E,
    const int64_t* __restrict__ ubcast_off,
    const int64_t* __restrict__ ebcast_off, int64_t ufeat_len,
    int64_t efeat_len, int64_t out_len) {

  int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
  while (tx < E) {
    const Idx src =  _ldg(row + tx);
    const Idx dst =  _ldg(col + tx);
    const Idx eid = UseIdx ? _ldg(edge_map + tx) : tx;
    const DType* uoff = BinaryOp::use_lhs ? (ufeat + src * ufeat_len) : nullptr;
    const DType* eoff = BinaryOp::use_rhs ? (efeat + eid * efeat_len) : nullptr;
    DType* outoff = out + dst * out_len;
    DType val = BinaryOp::Call(eoff,uoff);

    Idx* arguoff = nullptr;  // arguoff is not used in SpMMCoo.
    Idx* argeoff = nullptr;  // argeoff is not used in SpMMCoo.
    ReduceOp::Call(outoff, arguoff, argeoff, val, src, eid);
  
    
    tx += stride_x;
  }
}

template <typename Idx, typename DType, typename BinaryOp, typename ReduceOp>
void  COOEDivUSumCUDA(const BcastOff& bcast, const std::shared_ptr<COO> coo,
                 torch::Tensor ufeat, torch::Tensor efeat, torch::Tensor out,
                 torch::Tensor argu, torch::Tensor arge) {
  const bool use_idx = coo->e_ids.has_value();

  const Idx *row = coo->row.data_ptr<Idx>(), *col = coo->col.data_ptr<Idx>();
  const Idx* edge_map = use_idx ? coo->e_ids.value().data_ptr<Idx>() : nullptr;
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
  const int64_t E = coo->row.numel();

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = bcast.out_len, lhs_len = bcast.lhs_len, rhs_len = bcast.rhs_len;

  const int ntx = FindNumThreads(len);
  // const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  // const int nby = FindNumBlocks<'y'>((E + nty - 1) / nty);
  const dim3 nblks(nbx);
  const dim3 nthrs(ntx);

//   BCAST_IDX_CTX_SWITCH(bcast, use_idx, ubcast_off, ebcast_off, {
//     SWITCH_IDX(use_idx, {
      CUDA_KERNEL_CALL(
          (COOEDivUSumKernel<Idx, DType, BinaryOp, ReduceOp, false, false>),
          nblks, nthrs, ufeat_data, efeat_data, out_data, argu_data, arge_data,
          row, col, edge_map, E, ubcast_off, ebcast_off, lhs_len,
          rhs_len, len);
//     });
//   });
}


void EDivUSumCOO(const std::string& op, const std::string& reduce,
             const BcastOff& bcast, const std::shared_ptr<COO> coo,
             torch::Tensor ufeat, torch::Tensor efeat, torch::Tensor out,
             std::vector<torch::Tensor> out_aux) {
  if (out.scalar_type() != torch::kFloat32)
    LOG(FATAL)
        << "Currently g-SpMM on COO format only support 32 bits float data.";
  if (reduce == "sum") {
    ID_TYPE_SWITCH(coo->row.scalar_type(), IdType, {
      typedef float DType;
      SWITCH_OP(op, Op, {
        COOEDivUSumCUDA<IdType, DType, Op, impl::reduce::Sum<IdType, DType> >(
            bcast, coo, ufeat, efeat, out, out_aux[0], out_aux[1]);
      });
    });
  }else {
    LOG(FATAL) << "Not implemented warning";
  }
}


    }  // namespace fusion
}  // namespace impl
}  // namespace gs