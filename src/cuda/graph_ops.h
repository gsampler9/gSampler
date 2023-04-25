#ifndef GS_CUDA_GRAPH_OPS_H_
#define GS_CUDA_GRAPH_OPS_H_

#include <torch/torch.h>
#include "./logging.h"

namespace gs {
namespace impl {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
CSCColSlicingCUDA(torch::Tensor indptr, torch::Tensor indices,
                  torch::Tensor column_ids, bool with_coo);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor>
BatchCSCColSlicingCUDA(torch::Tensor indptr, torch::Tensor indices,
                       torch::Tensor column_ids, torch::Tensor nid_ptr,
                       int64_t encoding_size, bool with_coo, bool encoding);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
DCSCColSlicingCUDA(torch::Tensor indptr, torch::Tensor indices,
                   torch::Tensor nid_map, torch::Tensor column_ids,
                   bool with_coo);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
CSCRowSlicingCUDA(torch::Tensor indptr, torch::Tensor indices,
                  torch::Tensor row_ids, bool with_coo);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> COORowSlicingCUDA(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor row_ids);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
BatchCOORowSlicingCUDA(torch::Tensor coo_row, torch::Tensor coo_col,
                       torch::Tensor row_ids, torch::Tensor indices_ptr,
                       torch::Tensor nodeids_ptr);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
CSCColSamplingCUDA(torch::Tensor indptr, torch::Tensor indices, int64_t fanout,
                   bool replace, bool with_out);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
CSCColSamplingProbsCUDA(torch::Tensor indptr, torch::Tensor indices,
                        torch::Tensor probs, int64_t fanout, bool replace,
                        bool with_out);

void CSCSumCUDA(torch::Tensor indptr, torch::optional<torch::Tensor> e_ids,
                torch::optional<torch::Tensor> n_ids, torch::Tensor data,
                torch::Tensor out_data, int64_t powk);

void COOSumCUDA(torch::Tensor target, torch::optional<torch::Tensor> e_ids,
                torch::Tensor data, torch::Tensor out_data, int64_t powk);

void CSCNormalizeCUDA(torch::Tensor indptr,
                      torch::optional<torch::Tensor> e_ids, torch::Tensor data,
                      torch::Tensor out_data);

std::pair<torch::Tensor, torch::Tensor> CSC2COOCUDA(torch::Tensor indptr,
                                                    torch::Tensor indices);

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
COO2CSCCUDA(torch::Tensor row, torch::Tensor col, int64_t num_cols,
            bool col_sorted);

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>,
           torch::Tensor>
COO2DCSCCUDA(torch::Tensor row, torch::Tensor col, int64_t max_num_cols,
             bool col_sorted);

std::pair<torch::Tensor, torch::Tensor> DCSC2COOCUDA(torch::Tensor indptr,
                                                     torch::Tensor indices,
                                                     torch::Tensor ids);

std::vector<std::vector<torch::Tensor>> CSCSplitCUDA(
    torch::Tensor indptr, torch::Tensor indices,
    torch::optional<torch::Tensor> eid, int64_t split_size);

}  // namespace impl
}  // namespace gs

#endif