#include "graph.h"

#include <sstream>
#include "bcast.h"
#include "cuda/batch/batch_ops.h"
#include "cuda/fusion/column_row_slicing.h"
#include "cuda/fusion/edge_map_reduce.h"
#include "cuda/tensor_ops.h"
#include "graph_ops.h"

namespace gs {
// batch api
std::tuple<c10::intrusive_ptr<Graph>, torch::Tensor> Graph::BatchColSlicing(
    torch::Tensor seeds, torch::Tensor batch_ptr, bool encoding) {
  int64_t axis = 1;
  int64_t on_format = _CSC;
  int64_t output_format = _CSC + _COO;
  CreateSparseFormat(on_format);
  torch::Tensor select_index;
  std::shared_ptr<COO> coo_ptr = nullptr;
  std::shared_ptr<CSC> csc_ptr = nullptr;
  std::shared_ptr<CSR> csr_ptr = nullptr;
  std::shared_ptr<_TMP> tmp_ptr = nullptr;
  bool with_coo = output_format & _COO;
  int64_t new_num_cols, new_num_rows;
  torch::optional<torch::Tensor> e_ids = torch::nullopt;
  int64_t batch_num = batch_ptr.numel() - 1;
  col_bptr_ = batch_ptr;

  if (axis == 1) {
    new_num_cols = seeds.numel();
    new_num_rows = GetNumRows() * batch_num;
  } else {
    LOG(FATAL) << "batch slicing only suppurt column wise";
  }
  auto ret = c10::intrusive_ptr<Graph>(
      std::unique_ptr<Graph>(new Graph(new_num_rows, new_num_cols)));

  if (on_format == _CSC) {
    CHECK(output_format != _CSR)
        << "Error in Slicing, Not implementation [on_format = CSC, "
           "output_forat = CSR] !";
    auto csc = GetCSC();
    e_ids = csc->e_ids;

    std::tie(tmp_ptr, select_index, edge_bptr_) = BatchOnIndptrSlicing(
        csc, seeds, batch_ptr, with_coo, encoding, GetNumRows());

    if (output_format & _CSC)
      csc_ptr = std::make_shared<CSC>(
          CSC{tmp_ptr->indptr, tmp_ptr->coo_in_indices, torch::nullopt});
    if (output_format & _COO) {
      coo_ptr = std::make_shared<COO>(COO{tmp_ptr->coo_in_indices,
                                          tmp_ptr->coo_in_indptr,
                                          torch::nullopt, false, true});
    }

  } else {
    LOG(FATAL) << "Not Implementatin Error";
  }

  ret->SetNumEdges(select_index.numel());
  ret->SetCOO(coo_ptr);
  ret->SetCSC(csc_ptr);
  ret->SetCSR(csr_ptr);
  ret->SetColBptr(col_bptr_);
  ret->SetEdgeBptr(csc_->indptr.index({col_bptr_}));

  torch::Tensor split_index;
  if (e_ids.has_value()) {
    split_index = e_ids.value().index_select(0, select_index);
  } else {
    split_index = select_index;
  }

  return {ret, split_index};
}

std::tuple<c10::intrusive_ptr<Graph>, torch::Tensor> Graph::BatchRowSampling(
    int64_t fanout, bool replace) {
  auto ret = Sampling(1, fanout, replace, _CSC, _CSC + _COO);
  auto graph_ptr = std::get<0>(ret);
  auto split_index = std::get<1>(ret);
  graph_ptr->SetColBptr(col_bptr_);
  graph_ptr->SetEdgeBptr(graph_ptr->csc_->indptr.index({col_bptr_}));
  return {graph_ptr, split_index};
}

std::tuple<c10::intrusive_ptr<Graph>, torch::Tensor>
Graph::BatchRowSamplingProbs(int64_t fanout, bool replace,
                             torch::Tensor edge_probs) {
  auto ret = SamplingProbs(1, edge_probs, fanout, replace, _CSC, _CSC + _COO);
  auto graph_ptr = std::get<0>(ret);
  auto split_index = std::get<1>(ret);
  graph_ptr->SetColBptr(col_bptr_);
  graph_ptr->SetEdgeBptr(graph_ptr->csc_->indptr.index({col_bptr_}));
  return {graph_ptr, split_index};
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>>
Graph::BatchGraphRelabel(torch::Tensor col_seeds, torch::Tensor row_ids) {
  if (edge_bptr_.numel() == 0)
    LOG(FATAL) << "Relabel BatchGraph on COO must has edge batch pointer";

  CreateSparseFormat(_COO);
  auto coo = GetCOO();

  if (!coo->col_sorted)
    LOG(FATAL)
        << "Relabel BatchGraph on COO must require COO to be column-sorted";

  torch::Tensor coo_col = coo->col;
  torch::Tensor coo_row =
      row_ids.numel() > 0 ? row_ids.index({coo->row}) : coo->row;

  torch::Tensor unique_tensor, unique_tensor_bptr;
  torch::Tensor out_coo_row, out_coo_col, out_coo_bptr;
  std::tie(unique_tensor, unique_tensor_bptr, out_coo_row, out_coo_col,
           out_coo_bptr) =
      impl::batch::BatchCOORelabelCUDA(col_seeds, col_bptr_, coo_col, coo_row,
                                       edge_bptr_);

  auto frontier_vector =
      impl::batch::SplitByOffset(unique_tensor, unique_tensor_bptr);
  auto coo_row_vector = impl::batch::SplitByOffset(out_coo_row, out_coo_bptr);
  auto coo_col_vector = impl::batch::SplitByOffset(out_coo_col, out_coo_bptr);
  std::vector<torch::Tensor> eid_vector;
  if (coo_->e_ids.has_value())
    eid_vector = impl::batch::SplitByOffset(coo->e_ids.value(), out_coo_bptr);
  else {
    eid_vector.resize(out_coo_bptr.numel() - 1);
    std::fill(eid_vector.begin(), eid_vector.end(), torch::Tensor());
  }

  return {frontier_vector, coo_row_vector, coo_col_vector, eid_vector};
}

std::tuple<torch::Tensor, torch::Tensor> Graph::BatchGetValidNodes(
    torch::Tensor col_seeds, torch::Tensor row_ids) {
  if (edge_bptr_.numel() == 0)
    LOG(FATAL) << "Relabel BatchGraph on COO must has edge batch pointer";

  CreateSparseFormat(_COO);
  auto coo = GetCOO();

  if (!coo->col_sorted)
    LOG(FATAL)
        << "Relabel BatchGraph on COO must require COO to be column-sorted";

  torch::Tensor coo_col = coo->col;
  torch::Tensor coo_row =
      row_ids.numel() > 0 ? row_ids.index({coo->row}) : coo->row;

  torch::Tensor unique_tensor, unique_tensor_bptr;
  torch::Tensor out_coo_row, out_coo_col, out_coo_bptr;
  std::tie(unique_tensor, unique_tensor_bptr, out_coo_row, out_coo_col,
           out_coo_bptr) =
      impl::batch::BatchCOORelabelCUDA(col_seeds, col_bptr_, coo_col, coo_row,
                                       edge_bptr_);

  return {unique_tensor, unique_tensor_bptr};
};

}  // namespace gs