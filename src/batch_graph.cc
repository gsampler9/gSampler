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
    torch::Tensor seeds, torch::Tensor col_bptr, bool encoding) {
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

  torch::optional<torch::Tensor> e_ids = torch::nullopt;
  int64_t num_batch = col_bptr.numel() - 1;

  int64_t row_encoding_size = GetNumRows();

  torch::Tensor orig_row_ids;
  torch::Tensor row_bptr;
  torch::Tensor unique_encoding_rows;

  if (on_format == _CSC) {
    CHECK(output_format != _CSR)
        << "Error in Slicing, Not implementation [on_format = CSC, "
           "output_forat = CSR] !";
    auto csc = GetCSC();
    e_ids = csc->e_ids;

    std::tie(tmp_ptr, select_index, edge_bptr_) = BatchOnIndptrSlicing(
        csc, seeds, col_bptr, with_coo, encoding, row_encoding_size);

    if (encoding) {
      torch::Tensor unique_encoding_rows, new_indices;
      std::tie(unique_encoding_rows, new_indices) =
          impl::TensorCompact(tmp_ptr->coo_in_indices);
      tmp_ptr->coo_in_indices = new_indices;

      std::tie(row_bptr, orig_row_ids) = impl::batch::GetBatchOffsets(
          unique_encoding_rows, num_batch, row_encoding_size);
    }

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

  int64_t new_num_cols = seeds.numel();
  int64_t new_num_rows;
  if (encoding) {
    new_num_rows = orig_row_ids.numel();
  } else {
    new_num_rows = num_rows_ * num_batch;
  }

  auto ret = c10::intrusive_ptr<Graph>(
      std::unique_ptr<Graph>(new Graph(new_num_rows, new_num_cols)));

  ret->SetNumEdges(select_index.numel());
  ret->SetCOO(coo_ptr);
  ret->SetCSC(csc_ptr);
  ret->SetCSR(csr_ptr);
  ret->SetColBptr(col_bptr);
  ret->SetOrigColIds(seeds);
  ret->SetEdgeBptr(csc_ptr->indptr.index({col_bptr}));

  if (encoding) {
    ret->SetRowBptr(row_bptr);
    ret->SetOrigRowIds(orig_row_ids);
    ret->row_encoding_size_ = row_encoding_size;
  }
  // ret->unique_encoding_rows_ = unique_encoding_rows;

  torch::Tensor split_index;
  if (e_ids.has_value()) {
    split_index = (e_ids.value().is_pinned())
                      ? impl::IndexSelectCPUFromGPU(e_ids.value(), select_index)
                      : e_ids.value().index_select(0, select_index);
  } else {
    split_index = select_index;
  }

  return {ret, split_index};
}


std::tuple<c10::intrusive_ptr<Graph>, torch::Tensor> Graph::BatchColRowSlcing(
    torch::Tensor col, torch::Tensor col_bptr, torch::Tensor row,
    torch::Tensor row_bptr) {
  torch::Tensor select_index, out_data, indptr, indices;
  std::shared_ptr<CSC> csc_ptr;

  std::tie(indptr, indices, select_index) =
      impl::fusion::BatchCSCColRowSlicingCUDA(csc_->indptr, csc_->indices,
                                              col, col_bptr, row,
                                              row_bptr, num_rows_);
  csc_ptr = std::make_shared<CSC>(CSC{indptr, indices, torch::nullopt});


  int64_t new_num_cols = col.numel();
  int64_t new_num_rows = row.numel();

  auto ret = c10::intrusive_ptr<Graph>(
      std::unique_ptr<Graph>(new Graph(new_num_rows, new_num_cols)));

  ret->SetCSC(csc_ptr);
  ret->SetNumEdges(select_index.numel());
  ret->SetColBptr(col_bptr);
  ret->SetRowBptr(row_bptr);
  ret->SetOrigColIds(col);
  ret->SetOrigRowIds(row);
  ret->SetEdgeBptr(csc_ptr->indptr.index({col_bptr}));
  torch::Tensor split_index;
  if (csc_->e_ids.has_value()) {
    split_index = (csc_->e_ids.value().is_pinned())
                      ? impl::IndexSelectCPUFromGPU(csc_->e_ids.value(), select_index)
                      : csc_->e_ids.value().index_select(0, select_index);
  } else {
    split_index = select_index;
  }
  return {ret, split_index};
}

std::tuple<c10::intrusive_ptr<Graph>, torch::Tensor> Graph::BatchRowSlicing(
    torch::Tensor row_ids, torch::Tensor row_bptr) {
  auto ret = Slicing(row_ids, 0, _CSC, _CSC + _COO);
  auto graph_ptr = std::get<0>(ret);
  auto split_index = std::get<1>(ret);
  graph_ptr->SetColBptr(col_bptr_);
  graph_ptr->SetRowBptr(row_bptr);
  graph_ptr->SetOrigColIds(orig_col_ids_);
  graph_ptr->SetOrigRowIds(row_ids);
  // graph_ptr->unique_encoding_rows_ = encoding_row_ids;
  graph_ptr->SetEdgeBptr(graph_ptr->csc_->indptr.index({col_bptr_}));
  graph_ptr->row_encoding_size_ = row_encoding_size_;
  return {graph_ptr, split_index};
}

std::tuple<c10::intrusive_ptr<Graph>, torch::Tensor> Graph::BatchRowSampling(
    int64_t fanout, bool replace) {
  auto ret = Sampling(1, fanout, replace, _CSC, _CSC + _COO);
  auto graph_ptr = std::get<0>(ret);
  auto split_index = std::get<1>(ret);
  graph_ptr->SetColBptr(col_bptr_);
  graph_ptr->SetRowBptr(row_bptr_);
  graph_ptr->SetOrigColIds(orig_col_ids_);
  graph_ptr->SetOrigRowIds(orig_row_ids_);
  graph_ptr->SetEdgeBptr(graph_ptr->csc_->indptr.index({col_bptr_}));
  // graph_ptr->unique_encoding_rows_ = unique_encoding_rows_;
  graph_ptr->row_encoding_size_ = row_encoding_size_;
  return {graph_ptr, split_index};
}

std::tuple<c10::intrusive_ptr<Graph>, torch::Tensor>
Graph::BatchRowSamplingProbs(int64_t fanout, bool replace,
                             torch::Tensor edge_probs) {
  auto ret = SamplingProbs(1, edge_probs, fanout, replace, _CSC, _CSC + _COO);
  auto graph_ptr = std::get<0>(ret);
  auto split_index = std::get<1>(ret);
  graph_ptr->SetColBptr(col_bptr_);
  graph_ptr->SetRowBptr(row_bptr_);
  graph_ptr->SetOrigColIds(orig_col_ids_);
  graph_ptr->SetOrigRowIds(orig_row_ids_);
  graph_ptr->SetEdgeBptr(graph_ptr->csc_->indptr.index({col_bptr_}));
  // graph_ptr->unique_encoding_rows_ = unique_encoding_rows_;
  graph_ptr->row_encoding_size_ = row_encoding_size_;
  return {graph_ptr, split_index};
}

std::tuple<c10::intrusive_ptr<Graph>, torch::Tensor>
Graph::BatchFusedSlicingSampling(torch::Tensor seeds, torch::Tensor col_bptr,
                                 int64_t fanout, bool replace) {
  int64_t axis = 1;
  auto ret =
      FusedSlicingSampling(axis, seeds, fanout, replace, _CSC, _CSC + _COO);
  auto graph_ptr = std::get<0>(ret);
  auto select_index = std::get<1>(ret);
  graph_ptr->SetColBptr(col_bptr);
  graph_ptr->SetOrigColIds(seeds);
  graph_ptr->SetEdgeBptr(graph_ptr->csc_->indptr.index({col_bptr}));
  torch::Tensor split_index;
  torch::optional<torch::Tensor> e_ids = csc_->e_ids;
  if (e_ids.has_value()) {
    split_index = (e_ids.value().is_pinned())
                      ? impl::IndexSelectCPUFromGPU(e_ids.value(), select_index)
                      : e_ids.value().index_select(0, select_index);
  } else {
    split_index = select_index;
  }

  return {graph_ptr, split_index};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::optional<torch::Tensor>, torch::Tensor>
Graph::BatchGraphRelabel(torch::Tensor col_seeds, torch::Tensor row_ids) {
  if (edge_bptr_.numel() == 0)
    LOG(FATAL) << "Relabel BatchGraph on COO must has edge batch pointer";

  CreateSparseFormat(_COO);
  auto coo = GetCOO();

  if (!coo->col_sorted)
    LOG(FATAL)
        << "Relabel BatchGraph on COO must require COO to be column-sorted";

  torch::Tensor coo_col = col_seeds.index({coo->col});
  // torch::Tensor coo_row =
  //    row_ids.numel() > 0 ? row_ids.index({coo->row}) : coo->row;

  torch::Tensor coo_row;
  torch::Tensor tmp;
  if (row_ids.numel() > 0) {
    std::tie(tmp, coo_row) = impl::batch::GetBatchOffsets(
        row_ids.index({coo->row}), edge_bptr_.numel() - 1, row_encoding_size_);
  } else {
    coo_row = coo->row;
  }

  torch::Tensor unique_tensor, unique_tensor_bptr;
  torch::Tensor out_coo_row, out_coo_col, out_coo_bptr;
  std::tie(unique_tensor, unique_tensor_bptr, out_coo_row, out_coo_col,
           out_coo_bptr) =
      impl::batch::BatchCOORelabelCUDA(col_seeds, col_bptr_, coo_col, coo_row,
                                       edge_bptr_);

  return {unique_tensor, unique_tensor_bptr, out_coo_row,
          out_coo_col,   coo_->e_ids,        out_coo_bptr};
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

  torch::Tensor coo_col = col_seeds.index({coo->col});
  // torch::Tensor coo_row =
  //    row_ids.numel() > 0 ? row_ids.index({coo->row}) : coo->row;

  torch::Tensor coo_row;
  torch::Tensor tmp;
  if (row_ids.numel() > 0) {
    std::tie(tmp, coo_row) = impl::batch::GetBatchOffsets(
        row_ids.index({coo->row}), edge_bptr_.numel() - 1, row_encoding_size_);
  } else {
    coo_row = coo->row;
  };

  torch::Tensor unique_tensor, unique_tensor_bptr;
  torch::Tensor out_coo_row, out_coo_col, out_coo_bptr;
  std::tie(unique_tensor, unique_tensor_bptr, out_coo_row, out_coo_col,
           out_coo_bptr) =
      impl::batch::BatchCOORelabelCUDA(col_seeds, col_bptr_, coo_col, coo_row,
                                       edge_bptr_);

  return {unique_tensor, unique_tensor_bptr};
};

}  // namespace gs