#ifndef GS_CUDA_SELECTOR_H_
#define GS_CUDA_SELECTOR_H_

#include <cuda_runtime.h>
#include "../logging.h"

namespace gs {
namespace impl {
#define DGLDEVICE __device__
#define DGLINLINE __forceinline__

/*!
 * \brief Select among src/edge/dst feature/idx.
 * \note the integer argument target specifies which target
 *       to choose, 0: src, 1: edge, 2: dst.
 */
template <int target>
struct Selector {
  template <typename T>
  static DGLDEVICE DGLINLINE T Call(T src, T edge, T dst) {
    LOG(INFO) << "Target " << target << " not recognized.";
    return src;
  }
};

template <>
template <typename T>
DGLDEVICE DGLINLINE T Selector<0>::Call(T src, T edge, T dst) {
  return src;
}

template <>
template <typename T>
DGLDEVICE DGLINLINE T Selector<1>::Call(T src, T edge, T dst) {
  return edge;
}

template <>
template <typename T>
DGLDEVICE DGLINLINE T Selector<2>::Call(T src, T edge, T dst) {
  return dst;
}
}  // namespace impl
}  // namespace gs

#endif  // GS_SELECTOR_H_
