#ifndef GS_GRAPH_OPS_H_
#define GS_GRAPH_OPS_H_

#include <torch/torch.h>

#include "graph_storage.h"

namespace gs {

std::pair<std::shared_ptr<CSC>, torch::Tensor> FusedCSCColRowSlicing(
    std::shared_ptr<CSC> csc, torch::Tensor column_ids, torch::Tensor row_ids);

std::pair<std::shared_ptr<_TMP>, torch::Tensor> CSCColSlicing(
    std::shared_ptr<CSC> csc, torch::Tensor node_ids, bool with_coo);

std::pair<std::shared_ptr<_TMP>, torch::Tensor> DCSCColSlicing(
    std::shared_ptr<CSC> csc, torch::Tensor nid_map, torch::Tensor node_ids,
    bool with_coo);

std::pair<std::shared_ptr<_TMP>, torch::Tensor> CSCRowSlicing(
    std::shared_ptr<CSC> csc, torch::Tensor node_ids, bool with_coo);

std::pair<std::shared_ptr<COO>, torch::Tensor> COOColSlicing(
    std::shared_ptr<COO> coo, torch::Tensor node_ids, int64_t axis);

std::pair<std::shared_ptr<_TMP>, torch::Tensor> CSCColSampling(
    std::shared_ptr<CSC> csc, int64_t fanout, bool replace, bool with_coo);

std::pair<std::shared_ptr<_TMP>, torch::Tensor> CSCColSamplingProbs(
    std::shared_ptr<CSC> csc, torch::Tensor edge_probs, int64_t fanout,
    bool replace, bool with_coo);

std::pair<std::shared_ptr<CSC>, torch::Tensor> FusedCSCColSlicingAndSampling(
    std::shared_ptr<CSC> csc, torch::Tensor node_ids, int64_t fanout,
    bool replace);

torch::Tensor TensorUnique(torch::Tensor node_ids);

std::tuple<torch::Tensor, std::vector<torch::Tensor>> BatchTensorRelabel(
    const std::vector<torch::Tensor> &mapping_tensors,
    const std::vector<torch::Tensor> &to_be_relabeled_tensors);

void CSCGraphSum(std::shared_ptr<CSC> csc, torch::optional<torch::Tensor> n_ids,
                 torch::Tensor data, torch::Tensor out_data, int64_t powk);

void COOGraphSum(std::shared_ptr<COO> coo, torch::Tensor data,
                 torch::Tensor out_data, int64_t powk, int target_side);

void CSCGraphDiv(std::shared_ptr<CSC> csc, torch::optional<torch::Tensor> n_ids,
                 torch::Tensor data, torch::Tensor divisor,
                 torch::Tensor out_data);

void COOGraphDiv(std::shared_ptr<COO> coo, torch::Tensor data,
                 torch::Tensor divisor, torch::Tensor out_data,
                 int target_side);

std::shared_ptr<COO> GraphCSC2COO(std::shared_ptr<CSC> csc, bool CSC2COO);

std::shared_ptr<CSC> GraphCOO2CSC(std::shared_ptr<COO> coo, int64_t num_items,
                                  bool COO2CSC);

std::pair<std::shared_ptr<CSC>, torch::Tensor> GraphCOO2DCSC(
    std::shared_ptr<COO> coo, int64_t num_items, bool COO2DCSC);

std::shared_ptr<COO> GraphDCSC2COO(std::shared_ptr<CSC> csc, torch::Tensor ids,
                                   bool DCSC2COO);

torch::Tensor FusedRandomWalk(std::shared_ptr<CSC> csc, torch::Tensor seeds,
                              int64_t walk_length);

}  // namespace gs

#endif