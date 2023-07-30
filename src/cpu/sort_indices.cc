#include "omp.h"
#include "stdlib.h"
#include "tensor_ops.h"

namespace gs {
namespace impl {

int cmpfunc(const void *a, const void *b) {
  return (*(int64_t *)a - *(int64_t *)b);
}

torch::Tensor SortIndicesCPU(torch::Tensor indptr, torch::Tensor indices) {
  // auto ret_indices =
  //      torch::empty_like(indices,
  //      torch::TensorOptions().pinned_memory(true));

  int64_t num_segments = indptr.numel() - 1;

#pragma omp parallel for
  for (int64_t i = 0; i < num_segments; i++) {
    int64_t array_len =
        indptr.data_ptr<int64_t>()[i + 1] - indptr.data_ptr<int64_t>()[i];
    int64_t *array_beg =
        indices.data_ptr<int64_t>() + indptr.data_ptr<int64_t>()[i];

    qsort(array_beg, array_len, sizeof(int64_t), cmpfunc);
  }

  return indices;
}
}  // namespace impl
}  // namespace gs