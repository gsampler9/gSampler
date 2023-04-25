#include "./bcast.h"

namespace gs {
/*!
 * \brief Determine whether use broadcasting or not, given the operator
 *        type, lhs array and rhs array.
 */
bool UseBcast(const std::string& op, torch::Tensor lhs, torch::Tensor rhs) {
  if (op == "copy_lhs" || op == "copy_rhs")
    return false;  // broadcasting is not required for copy_u/copy_e
  if (lhs.dim() != rhs.dim()) return true;
  for (int i = 1; i < lhs.dim(); ++i) {
    if (lhs.size(i) != rhs.size(i)) return true;
  }
  return false;
}

/*!
 * \brief: Compute broadcast and auxiliary information given operator
 *         and operands for kernel computation.
 * \note: Expect lhs, rhs to have ndim >= 2 and the shape of lhs/rhs
 *        valid for the op computation.
 */
BcastOff CalcBcastOff(const std::string& op, torch::Tensor lhs,
                      torch::Tensor rhs) {
  BcastOff rst;
  rst.lhs_len = 1;
  rst.rhs_len = 1;
  for (int i = 1; i < lhs.dim(); ++i) rst.lhs_len *= lhs.size(i);
  for (int i = 1; i < rhs.dim(); ++i) rst.rhs_len *= rhs.size(i);
  rst.use_bcast = UseBcast(op, lhs, rhs);
  rst.reduce_size = 1;  // defaults to 1, except for the case op == 'dot'.
  if (rst.use_bcast) {
    const int max_ndim = std::max(lhs.dim(), rhs.dim()) - 1;
    int out_len = 1, j = 0;
    if (op == "dot") {
      rst.reduce_size = lhs.size(lhs.dim() - 1);  // set reduce_size for dot.
      ++j;  // do not consider reduce axis in computing lhs_offset and
            // rhs_offset.
    }
    int stride_l = 1, stride_r = 1;
    rst.lhs_offset.push_back(0);  // lhs_offset[0] is always 0
    rst.rhs_offset.push_back(0);  // rhs_offset[0] is always 0
    for (; j < max_ndim; ++j) {   // iterate the axis from back to front.
      // dl refers to the size of lhs array in the current axis, likewise for
      // dr.
      const int dl = (lhs.dim() - 1 - j < 1) ? 1 : lhs.size(lhs.dim() - 1 - j);
      const int dr = (rhs.dim() - 1 - j < 1) ? 1 : rhs.size(rhs.dim() - 1 - j);
      for (int i = 1; i < std::max(dl, dr); ++i) {
        for (int k = 0; k < out_len; ++k) {
          /* Explanation:
           * if current dimension is not broadcast dimension for lhs array
           *   lhs_offset[i * out_len + k] = lhs_offset[k] + i * stride_l
           * else
           *   lhs_offset[i * out_len + k] = lhs_offset[k]
           * likewise for rhs_offset.
           */
          rst.lhs_offset.push_back(rst.lhs_offset[k] + i * (i < dl) * stride_l);
          rst.rhs_offset.push_back(rst.rhs_offset[k] + i * (i < dr) * stride_r);
        }
      }
      out_len *= std::max(dl, dr);
      stride_l *= dl;
      stride_r *= dr;
    }
    rst.out_len = out_len;
  } else {
    rst.out_len = (op == "copy_rhs") ? rst.rhs_len : rst.lhs_len;
    if (op == "dot") {
      rst.reduce_size = lhs.size(lhs.dim() - 1);  // set reduce_size for dot.
      rst.out_len /=
          rst.reduce_size;  // out_len is divied by reduce_size in dot.
    }
  }
  return rst;
}
}  // namespace gs
