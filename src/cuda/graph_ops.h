#ifndef GS_CUDA_GRAPH_OPS_H_
#define GS_CUDA_GRAPH_OPS_H_

#include <torch/torch.h>
#include "./logging.h"

namespace gs {
namespace impl {

// slicing
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
OnIndptrSlicingCUDA(torch::Tensor indptr, torch::Tensor indices,
                    torch::Tensor seeds, bool with_coo);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
OnIndicesSlicingCUDA(torch::Tensor indptr, torch::Tensor indices,
                     torch::Tensor row_ids, bool with_coo);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> COORowSlicingCUDA(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor row_ids);

// sampling
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
CSCColSamplingCUDA(torch::Tensor indptr, torch::Tensor indices, int64_t fanout,
                   bool replace, bool with_out);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
CSCColSamplingProbsCUDA(torch::Tensor indptr, torch::Tensor indices,
                        torch::Tensor probs, int64_t fanout, bool replace,
                        bool with_out);

std::pair<torch::Tensor, torch::Tensor> CSC2COOCUDA(torch::Tensor indptr,
                                                    torch::Tensor indices);

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
COO2CSCCUDA(torch::Tensor row, torch::Tensor col, int64_t num_cols,
            bool col_sorted);

// unique & relabel
torch::Tensor TensorUniqueCUDA(torch::Tensor node_ids);

std::tuple<torch::Tensor, std::vector<torch::Tensor>> TensorRelabelCUDA(
    const std::vector<torch::Tensor>& mapping_tensor,
    const std::vector<torch::Tensor>& data_requiring_relabel);

}  // namespace impl
}  // namespace gs

#endif