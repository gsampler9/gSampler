#ifndef GS_CUDA_MACRO_H_
#define GS_CUDA_MACRO_H_

#include "./cuda_common.h"
#include "./functor.h"

#define SWITCH_OP(op, Op, ...)                                        \
  do {                                                                \
    if ((op) == "add") {                                              \
      typedef impl::binary::Add<DType> Op;                            \
      { __VA_ARGS__ }                                                 \
    } else if ((op) == "sub") {                                       \
      typedef impl::binary::Sub<DType> Op;                            \
      { __VA_ARGS__ }                                                 \
    } else if ((op) == "mul") {                                       \
      typedef impl::binary::Mul<DType> Op;                            \
      { __VA_ARGS__ }                                                 \
    } else if ((op) == "div") {                                       \
      typedef impl::binary::Div<DType> Op;                            \
      { __VA_ARGS__ }                                                 \
    } else if ((op) == "copy_lhs") {                                  \
      typedef impl::binary::CopyLhs<DType> Op;                        \
      { __VA_ARGS__ }                                                 \
    } else if ((op) == "copy_rhs") {                                  \
      typedef impl::binary::CopyRhs<DType> Op;                        \
      { __VA_ARGS__ }                                                 \
    } else if ((op) == "dot") {                                       \
      typedef impl::binary::Dot<DType> Op;                            \
      { __VA_ARGS__ }                                                 \
    } else {                                                          \
      LOG(FATAL) << "Unsupported SpMM/SDDMM binary operator: " << op; \
    }                                                                 \
  } while (0)

#define SWITCH_RHS(rhs_target, RhsTarget, ...)             \
  do {                                                     \
    if ((rhs_target) == 0) {                               \
      constexpr int RhsTarget = 0;                         \
      { __VA_ARGS__ }                                      \
    } else if ((rhs_target) == 1) {                        \
      constexpr int RhsTarget = 1;                         \
      { __VA_ARGS__ }                                      \
    } else if ((rhs_target) == 2) {                        \
      constexpr int RhsTarget = 2;                         \
      { __VA_ARGS__ }                                      \
    } else {                                               \
      LOG(INFO) << "Invalid rhs target: " << (rhs_target); \
    }                                                      \
  } while (0)

#define SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, ...) \
  do {                                                                   \
    if ((lhs_target) == 0) {                                             \
      constexpr int LhsTarget = 0;                                       \
      SWITCH_RHS(rhs_target, RhsTarget, __VA_ARGS__);                    \
    } else if ((lhs_target) == 1) {                                      \
      constexpr int LhsTarget = 1;                                       \
      SWITCH_RHS(rhs_target, RhsTarget, __VA_ARGS__);                    \
    } else if ((lhs_target) == 2) {                                      \
      constexpr int LhsTarget = 2;                                       \
      SWITCH_RHS(rhs_target, RhsTarget, __VA_ARGS__);                    \
    } else {                                                             \
      LOG(INFO) << "Invalid lhs target: " << (lhs_target);               \
    }                                                                    \
  } while (0)

/* Macro used for switching between broadcasting and non-broadcasting kernels.
 * It also copies the auxiliary information for calculating broadcasting offsets
 * to GPU.
 */
#define BCAST_IDX_CTX_SWITCH(BCAST, EDGE_MAP, LHS_OFF, RHS_OFF, ...)   \
  do {                                                                 \
    const BcastOff &info = (BCAST);                                    \
    if (!info.use_bcast) {                                             \
      constexpr bool UseBcast = false;                                 \
      if ((EDGE_MAP)) {                                                \
        constexpr bool UseIdx = true;                                  \
        { __VA_ARGS__ }                                                \
      } else {                                                         \
        constexpr bool UseIdx = false;                                 \
        { __VA_ARGS__ }                                                \
      }                                                                \
    } else {                                                           \
      constexpr bool UseBcast = true;                                  \
      CUDA_CALL(cudaMalloc((void **)&LHS_OFF,                          \
                           sizeof(int64_t) * info.lhs_offset.size())); \
      CUDA_CALL(cudaMemcpy((LHS_OFF), &info.lhs_offset[0],             \
                           sizeof(int64_t) * info.lhs_offset.size(),   \
                           cudaMemcpyHostToDevice));                   \
      CUDA_CALL(cudaMalloc((void **)&RHS_OFF,                          \
                           sizeof(int64_t) * info.rhs_offset.size())); \
      CUDA_CALL(cudaMemcpy((RHS_OFF), &info.rhs_offset[0],             \
                           sizeof(int64_t) * info.rhs_offset.size(),   \
                           cudaMemcpyHostToDevice));                   \
      if ((EDGE_MAP)) {                                                \
        constexpr bool UseIdx = true;                                  \
        { __VA_ARGS__ }                                                \
      } else {                                                         \
        constexpr bool UseIdx = false;                                 \
        { __VA_ARGS__ }                                                \
      }                                                                \
      CUDA_CALL(cudaFree(LHS_OFF));                                    \
      CUDA_CALL(cudaFree(RHS_OFF));                                    \
    }                                                                  \
  } while (0)

#define SWITCH_IDX(EDGE_MAP, ...)     \
  do {                                \
    if ((EDGE_MAP)) {                 \
      constexpr bool UseEMap = true;  \
    } else {                          \
      constexpr bool UseEMap = false; \
    }                                 \
  } while (0)

#endif
