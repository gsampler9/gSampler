#ifndef GS_CUDA_FUSION_COLUMN_ROW_SLICING_H_
#define GS_CUDA_FUSION_COLUMN_ROW_SLICING_H_

#include <torch/torch.h>

namespace gs {
namespace impl {
namespace fusion {
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CSCColRowSlicingCUDA(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor column_ids,
    torch::Tensor row_ids);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
BatchCSCColRowSlicingCUDA(torch::Tensor indptr, torch::Tensor indices,
                          torch::Tensor column_ids, torch::Tensor col_ptr,
                          torch::Tensor row_ids, torch::Tensor row_ptr,
                          int64_t encoding_size);

}  // namespace fusion
}  // namespace impl
}  // namespace gs

#endif