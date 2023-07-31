#include "../logging.h"
#include "omp.h"
#include "stdlib.h"
#include "tensor_ops.h"

#define ID_TYPE_SWITCH(val, IdType, ...)               \
  do {                                                 \
    if ((val).scalar_type() == torch::kInt32) {        \
      typedef int32_t IdType;                          \
      { __VA_ARGS__ }                                  \
    } else if ((val).scalar_type() == torch::kInt64) { \
      typedef int64_t IdType;                          \
      { __VA_ARGS__ }                                  \
    } else {                                           \
      LOG(FATAL) << "ID can only be int32 or int64";   \
    }                                                  \
  } while (0)

namespace gs {
namespace impl {

template <typename IdType>
int cmpfunc(const void *a, const void *b) {
  return (*(IdType *)a - *(IdType *)b);
}

torch::Tensor SortIndicesCPU(torch::Tensor indptr, torch::Tensor indices) {
  // auto ret_indices =
  //      torch::empty_like(indices,
  //      torch::TensorOptions().pinned_memory(true));

  int64_t num_segments = indptr.numel() - 1;

#pragma omp parallel for
  for (int64_t i = 0; i < num_segments; i++) {
    ID_TYPE_SWITCH(indptr, ETyep, {
      ID_TYPE_SWITCH(indices, NType, {
        ETyep array_len =
            indptr.data_ptr<ETyep>()[i + 1] - indptr.data_ptr<ETyep>()[i];
        NType *array_beg =
            indices.data_ptr<NType>() + indptr.data_ptr<ETyep>()[i];
        qsort(array_beg, array_len, sizeof(NType), cmpfunc<NType>);
      });
    });
  }

  return indices;
}
}  // namespace impl
}  // namespace gs