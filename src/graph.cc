#include "./graph.h"

#include <sstream>
#include "bcast.h"
#include "cuda/fusion/column_row_slicing.h"
#include "cuda/fusion/edge_map_reduce.h"
#include "cuda/graph_ops.h"
#include "cuda/sddmm.h"
#include "cuda/tensor_ops.h"
#include "graph_ops.h"

namespace gs {

void Graph::LoadCSC(torch::Tensor indptr, torch::Tensor indices) {
  csc_ = std::make_shared<CSC>();
  csc_->indptr = indptr;
  csc_->indices = indices;
  num_cols_ = indptr.numel() - 1;
  num_rows_ = indptr.numel() - 1;
  num_edges_ = indices.numel();
  LOG(INFO) << "Loaded CSC with " << indptr.numel() - 1 << " nodes and "
            << indices.numel() << " edges";
}

void Graph::LoadCOO(torch::Tensor row, torch::Tensor col) {
  coo_ = std::make_shared<COO>();
  coo_->row = row;
  coo_->col = col;
  auto num_nodes = std::max(std::get<0>(torch::_unique(row)).numel(),
                            std::get<0>(torch::_unique(col)).numel());
  num_cols_ = num_rows_ = num_nodes;
  num_edges_ = row.numel();
  LOG(INFO) << "Loaded COO with " << num_nodes << " nodes and " << num_edges_
            << " edges";
}

void Graph::LoadCSR(torch::Tensor indptr, torch::Tensor indices) {
  csr_ = std::make_shared<CSR>();
  csr_->indptr = indptr;
  csr_->indices = indices;
  num_rows_ = indptr.numel() - 1;
  num_cols_ = indptr.numel() - 1;
  num_edges_ = indices.numel();
  LOG(INFO) << "Loaded CSR with " << indptr.numel() - 1 << " nodes and "
            << indices.numel() << " edges";
}

void Graph::LoadCSCWithColIds(torch::Tensor column_ids, torch::Tensor indptr,
                              torch::Tensor indices) {
  if (column_ids.numel() != indptr.numel() - 1) {
    LOG(FATAL) << "length of column_ids is not aligned with indptr";
  }
  csc_ = std::make_shared<CSC>();
  col_ids_ = column_ids;
  csc_->indptr = indptr;
  csc_->indices = indices;
  num_cols_ = column_ids.numel();
  num_rows_ = indices.max().item<int64_t>() + 1;
  num_edges_ = indices.numel();
}

std::shared_ptr<CSC> Graph::GetCSC() { return csc_; }

std::shared_ptr<CSR> Graph::GetCSR() { return csr_; }

std::shared_ptr<COO> Graph::GetCOO() { return coo_; }

torch::optional<torch::Tensor> Graph::GetData(std::string order) {
  auto data =
      data_.has_value()
          ? data_.value()
          : torch::ones(num_edges_,
                        torch::dtype(torch::kFloat32).device(torch::kCUDA));
  bool need_idx = false;
  torch::Tensor idx;
  if (order == "col") {
    CreateSparseFormat(_CSC);
    if (csc_->e_ids.has_value()) {
      need_idx = true;
      idx = csc_->e_ids.value();
    }
  } else if (order == "row") {
    CreateSparseFormat(_CSR);
    if (csr_->e_ids.has_value()) {
      need_idx = true;
      idx = csr_->e_ids.value();
    }
  }
  return need_idx ? data[idx] : data;
}

int64_t Graph::GetNumCols() { return num_cols_; }

int64_t Graph::GetNumRows() { return num_rows_; }

int64_t Graph::GetNumEdges() { return num_edges_; }

void Graph::SetCSC(std::shared_ptr<CSC> csc) { csc_ = csc; }

void Graph::SetCSR(std::shared_ptr<CSR> csr) { csr_ = csr; }

void Graph::SetCOO(std::shared_ptr<COO> coo) { coo_ = coo; }

void Graph::SetCOOByTensor(torch::Tensor row, torch::Tensor col,
                           torch::optional<torch::Tensor> e_ids,
                           bool row_sorted, bool col_sorted) {
  coo_ = std::make_shared<COO>(COO{row, col, e_ids, row_sorted, col_sorted});
}

void Graph::SetData(torch::Tensor data) { data_ = data; }

void Graph::SetValidCols(torch::Tensor val_cols) { val_col_ids_ = val_cols; }

void Graph::SetValidRows(torch::Tensor val_rows) { val_row_ids_ = val_rows; }

c10::intrusive_ptr<Graph> Graph::FusedBidirSlicing(torch::Tensor column_seeds,
                                                   torch::Tensor row_seeds) {
  torch::Tensor select_index, out_data;
  std::shared_ptr<CSC> csc_ptr;
  torch::Tensor col_ids = (col_ids_.has_value())
                              ? col_ids_.value().index({column_seeds})
                              : column_seeds;
  torch::Tensor row_ids =
      (row_ids_.has_value()) ? row_ids_.value().index({row_seeds}) : row_seeds;
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(new Graph(
      true, col_ids, row_ids, column_seeds.numel(), row_seeds.numel())));
  std::tie(csc_ptr, select_index) =
      FusedCSCColRowSlicing(csc_, col_ids, row_ids);
  ret->SetCSC(csc_ptr);
  if (data_.has_value()) {
    if (csc_->e_ids.has_value()) {
      out_data =
          data_.value().index({csc_->e_ids.value().index({select_index})});
    } else {
      out_data = data_.value().index({select_index});
    }
    ret->SetData(out_data);
  }
  return ret;
}

c10::intrusive_ptr<Graph> Graph::BatchFusedBidirSlicing(
    torch::Tensor column_seeds, torch::Tensor col_ptr, torch::Tensor row_seeds,
    torch::Tensor row_ptr) {
  torch::Tensor select_index, out_data, indptr, indices;
  std::shared_ptr<CSC> csc_ptr;
  torch::Tensor col_ids = (col_ids_.has_value())
                              ? col_ids_.value().index({column_seeds})
                              : column_seeds;
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(true, col_ids, row_ids_, column_seeds.numel(), num_rows_)));
  std::tie(indptr, indices, select_index) =
      impl::fusion::BatchCSCColRowSlicingCUDA(csc_->indptr, csc_->indices,
                                              col_ids, col_ptr, row_seeds,
                                              row_ptr, num_rows_);
  csc_ptr = std::make_shared<CSC>(CSC{indptr, indices, torch::nullopt});
  ret->SetCSC(csc_ptr);
  if (data_.has_value()) {
    if (csc_->e_ids.has_value()) {
      out_data =
          data_.value().index({csc_->e_ids.value().index({select_index})});
    } else {
      out_data = data_.value().index({select_index});
    }
    ret->SetData(out_data);
  }
  return ret;
}

void Graph::SetNumEdges(int64_t num_edges) { num_edges_ = num_edges; }

// axis = 0 for col; axis = 1 for row.
c10::intrusive_ptr<Graph> Graph::Slicing(torch::Tensor n_ids, int64_t axis,
                                         int64_t on_format,
                                         int64_t output_format, bool relabel) {
  CreateSparseFormat(on_format);
  std::shared_ptr<COO> coo_ptr;
  std::shared_ptr<CSC> csc_ptr;
  std::shared_ptr<CSR> csr_ptr;
  std::shared_ptr<_TMP> tmp_ptr;
  torch::Tensor select_index;
  torch::optional<torch::Tensor> e_ids, new_col_ids, new_row_ids, new_val_cols,
      new_val_rows;
  int64_t new_num_cols, new_num_rows, new_num_edges;
  bool with_coo = output_format & _COO;

  if (axis == 0) {
    new_col_ids =
        col_ids_.has_value() ? col_ids_.value().index({n_ids}) : n_ids;
    new_row_ids = row_ids_;
    new_num_cols = n_ids.numel();
    new_num_rows = num_rows_;
  } else {
    new_col_ids = col_ids_;
    new_row_ids =
        row_ids_.has_value() ? row_ids_.value().index({n_ids}) : n_ids;
    new_num_cols = num_cols_;
    new_num_rows = n_ids.numel();
  }

  if (on_format == _COO) {
    std::tie(coo_ptr, select_index) = COOColSlicing(coo_, n_ids, axis);
    new_num_edges = coo_ptr->row.numel();
    e_ids = coo_->e_ids;

  } else if (on_format == _CSC || on_format == _DCSC) {
    if (output_format == _CSR) {
      LOG(FATAL) << "Error in Slicing, Not implementation [on_format = CSC, "
                    "output_forat = CSR]";
    }
    if (axis == 0) {
      // for col
      if (val_col_ids_.has_value())
        std::tie(tmp_ptr, select_index) =
            DCSCColSlicing(csc_, val_col_ids_.value(), n_ids, with_coo);
      else
        std::tie(tmp_ptr, select_index) = CSCColSlicing(csc_, n_ids, with_coo);

      if (relabel) {
        torch::Tensor frontier;
        std::vector<torch::Tensor> relabeled_result;
        std::tie(frontier, relabeled_result) = BatchTensorRelabel(
            {tmp_ptr->coo_in_indices}, {tmp_ptr->coo_in_indices});
        tmp_ptr->coo_in_indices = relabeled_result[0];
        new_row_ids = frontier;
        new_num_rows = frontier.numel();
      }
    } else {
      // for row
      std::tie(tmp_ptr, select_index) = CSCRowSlicing(csc_, n_ids, with_coo);
      new_val_cols = val_col_ids_;

      if (relabel) LOG(FATAL) << "Not implemented warning";
    }

    if (output_format & _CSC)
      csc_ptr = std::make_shared<CSC>(
          CSC{tmp_ptr->indptr, tmp_ptr->coo_in_indices, torch::nullopt});
    if (output_format & _COO)
      coo_ptr = std::make_shared<COO>(COO{tmp_ptr->coo_in_indices,
                                          tmp_ptr->coo_in_indptr,
                                          torch::nullopt, false, true});
    new_num_edges = tmp_ptr->coo_in_indices.numel();
    e_ids = csc_->e_ids;

  } else if (on_format == _CSR || on_format == _DCSR) {
    if (output_format == _CSC) {
      LOG(FATAL) << "Error in Slicing, Not implementation [on_format = CSR, "
                    "output_forat = CSC]";
    }
    if (axis == 0) {
      // for col
      std::tie(tmp_ptr, select_index) = CSCRowSlicing(csr_, n_ids, with_coo);
      new_val_rows = val_row_ids_;

      if (relabel) LOG(FATAL) << "Not implemented warning";
    } else {
      if (val_row_ids_.has_value())
        std::tie(tmp_ptr, select_index) =
            DCSCColSlicing(csr_, val_row_ids_.value(), n_ids, with_coo);
      else
        std::tie(tmp_ptr, select_index) = CSCColSlicing(csr_, n_ids, with_coo);

      if (relabel) {
        torch::Tensor frontier;
        std::vector<torch::Tensor> relabeled_result;
        std::tie(frontier, relabeled_result) = BatchTensorRelabel(
            {tmp_ptr->coo_in_indices}, {tmp_ptr->coo_in_indices});
        tmp_ptr->coo_in_indices = relabeled_result[0];
        new_col_ids = frontier;
        new_num_cols = frontier.numel();
      }
    }

    if (output_format & _CSR)
      csr_ptr = std::make_shared<CSR>(
          CSR{tmp_ptr->indptr, tmp_ptr->coo_in_indices, torch::nullopt});
    if (output_format & _COO)
      coo_ptr = std::make_shared<COO>(COO{tmp_ptr->coo_in_indptr,
                                          tmp_ptr->coo_in_indices,
                                          torch::nullopt, true, false});
    new_num_edges = tmp_ptr->coo_in_indices.numel();
    e_ids = csr_->e_ids;
  } else {
    LOG(FATAL) << "Not implemented warning";
  }
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(true, new_col_ids, new_row_ids, new_num_cols, new_num_rows)));
  ret->SetNumEdges(new_num_edges);
  ret->SetCOO(coo_ptr);
  ret->SetCSC(csc_ptr);
  ret->SetCSR(csr_ptr);
  if (new_val_cols.has_value()) ret->SetValidCols(new_val_cols.value());
  if (new_val_rows.has_value()) ret->SetValidRows(new_val_rows.value());
  if (data_.has_value()) {
    torch::Tensor out_data, data_index;
    if (e_ids.has_value())
      data_index =
          (e_ids.value().is_pinned())
              ? impl::IndexSelectCPUFromGPU(e_ids.value(), select_index)
              : e_ids.value().index({select_index});
    else
      data_index = select_index;
    out_data = (data_.value().is_pinned())
                   ? impl::IndexSelectCPUFromGPU(data_.value(), data_index)
                   : data_.value().index({data_index});
    ret->SetData(out_data);
  }
  return ret;
}

std::tuple<c10::intrusive_ptr<Graph>, torch::Tensor> Graph::BatchSlicing(
    torch::Tensor n_ids, torch::Tensor nid_ptr, int64_t axis, int64_t on_format,
    int64_t output_format, bool relabel, bool encoding) {
  CreateSparseFormat(on_format);
  std::shared_ptr<COO> coo_ptr;
  std::shared_ptr<CSC> csc_ptr;
  std::shared_ptr<CSR> csr_ptr;
  std::shared_ptr<_TMP> tmp_ptr;
  torch::Tensor select_index, coo_offsets;
  torch::optional<torch::Tensor> e_ids, new_col_ids, new_row_ids, new_val_cols,
      new_val_rows;
  int64_t new_num_cols, new_num_rows, new_num_edges;
  bool with_coo = output_format & _COO;
  int64_t batch_num = nid_ptr.numel() - 1;

  if (axis == 0) {
    new_col_ids = col_ids_.has_value()
                      ? col_ids_.value().index({n_ids})
                      : impl::BatchEncodeCUDA(n_ids, nid_ptr, num_rows_);
    new_row_ids = row_ids_;
    new_num_cols = n_ids.numel();
    new_num_rows = num_rows_ * batch_num;
  } else {
    LOG(FATAL) << "batch slicing only suppurt column slicing now";
  }

  if (on_format == _CSC) {
    torch::Tensor sub_indptr, coo_col, coo_row;
    std::tie(sub_indptr, coo_col, coo_row, select_index, coo_offsets) =
        impl::BatchCSCColSlicingCUDA(csc_->indptr, csc_->indices, n_ids,
                                     nid_ptr, num_rows_, with_coo, encoding);
    tmp_ptr = std::make_shared<_TMP>(_TMP{sub_indptr, coo_col, coo_row});

    if (relabel) {
      torch::Tensor frontier;
      std::vector<torch::Tensor> relabeled_result;
      std::tie(frontier, relabeled_result) = BatchTensorRelabel(
          {tmp_ptr->coo_in_indices}, {tmp_ptr->coo_in_indices});
      tmp_ptr->coo_in_indices = relabeled_result[0];
      new_row_ids = frontier;
      new_num_rows = frontier.numel();
    }

    if (output_format & _CSC)
      csc_ptr = std::make_shared<CSC>(
          CSC{tmp_ptr->indptr, tmp_ptr->coo_in_indices, torch::nullopt});
    if (output_format & _COO)
      coo_ptr = std::make_shared<COO>(COO{tmp_ptr->coo_in_indices,
                                          tmp_ptr->coo_in_indptr,
                                          torch::nullopt, false, true});
    new_num_edges = tmp_ptr->coo_in_indices.numel();
    e_ids = csc_->e_ids;
  } else {
    LOG(FATAL) << "Not Implementatin Error";
  }

  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(true, new_col_ids, new_row_ids, new_num_cols, new_num_rows)));
  ret->SetNumEdges(new_num_edges);
  ret->SetCOO(coo_ptr);
  ret->SetCSC(csc_ptr);
  ret->SetCSR(csr_ptr);
  if (new_val_cols.has_value()) ret->SetValidCols(new_val_cols.value());
  if (new_val_rows.has_value()) ret->SetValidRows(new_val_rows.value());
  if (data_.has_value()) {
    torch::Tensor out_data, data_index;
    if (e_ids.has_value())
      data_index =
          (e_ids.value().is_pinned())
              ? impl::IndexSelectCPUFromGPU(e_ids.value(), select_index)
              : e_ids.value().index({select_index});
    else
      data_index = select_index;
    out_data = (data_.value().is_pinned())
                   ? impl::IndexSelectCPUFromGPU(data_.value(), data_index)
                   : data_.value().index({data_index});
    ret->SetData(out_data);
  }
  return {ret, coo_offsets};
}

c10::intrusive_ptr<Graph> Graph::Sampling(int64_t axis, int64_t fanout,
                                          bool replace, int64_t on_format,
                                          int64_t output_format) {
  CreateSparseFormat(on_format);
  torch::Tensor select_index, out_data;
  std::shared_ptr<_TMP> tmp_ptr;
  bool with_coo = output_format & _COO;

  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(true, col_ids_, row_ids_, num_cols_, num_rows_)));

  if (axis == 0 && on_format == _CSC) {
    if (output_format == _CSR)
      LOG(FATAL) << "Error in Sampling, Not implementation [on_format = CSC, "
                    "output_forat = CSR]";

    std::tie(tmp_ptr, select_index) =
        CSCColSampling(csc_, fanout, replace, with_coo);

    if (output_format & _CSC)
      ret->SetCSC(std::make_shared<CSC>(
          CSC{tmp_ptr->indptr, tmp_ptr->coo_in_indices, torch::nullopt}));
    if (output_format & _COO)
      ret->SetCOO(std::make_shared<COO>(COO{tmp_ptr->coo_in_indices,
                                            tmp_ptr->coo_in_indptr,
                                            torch::nullopt, false, true}));
    ret->SetNumEdges(tmp_ptr->coo_in_indices.numel());
    if (data_.has_value()) {
      if (csc_->e_ids.has_value()) {
        out_data =
            data_.value().index({csc_->e_ids.value().index({select_index})});
      } else {
        out_data = data_.value().index({select_index});
      }
      ret->SetData(out_data);
    }

  } else if (axis == 1 && on_format == _CSR) {
    if (output_format == _CSC)
      LOG(FATAL) << "Error in Sampling, Not implementation [on_format = CSR, "
                    "output_forat = CSC]";

    std::tie(tmp_ptr, select_index) =
        CSCColSampling(csr_, fanout, replace, with_coo);
    if (output_format & _CSR)
      ret->SetCSR(std::make_shared<CSR>(
          CSR{tmp_ptr->indptr, tmp_ptr->coo_in_indices, torch::nullopt}));
    if (output_format & _COO)
      ret->SetCOO(std::make_shared<COO>(COO{tmp_ptr->coo_in_indptr,
                                            tmp_ptr->coo_in_indices,
                                            torch::nullopt, true, false}));
    ret->SetNumEdges(tmp_ptr->coo_in_indices.numel());
    if (data_.has_value()) {
      if (csr_->e_ids.has_value()) {
        out_data =
            data_.value().index({csr_->e_ids.value().index({select_index})});
      } else {
        out_data = data_.value().index({select_index});
      }
      ret->SetData(out_data);
    }

  } else {
    LOG(FATAL) << "No implementation!";
  }
  return ret;
}

c10::intrusive_ptr<Graph> Graph::SamplingProbs(int64_t axis,
                                               torch::Tensor edge_probs,
                                               int64_t fanout, bool replace,
                                               int64_t on_format,
                                               int64_t output_format) {
  CreateSparseFormat(on_format);
  torch::Tensor select_index, out_data;
  std::shared_ptr<_TMP> tmp_ptr;
  bool with_coo = output_format & _COO;

  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(true, col_ids_, row_ids_, num_cols_, num_rows_)));

  if (axis == 0 && on_format == _CSC) {
    if (output_format == _CSR)
      LOG(FATAL) << "Error in Sampling, Not implementation [on_format = CSC, "
                    "output_forat = CSR]";

    std::tie(tmp_ptr, select_index) =
        CSCColSamplingProbs(csc_, edge_probs, fanout, replace, with_coo);

    if (output_format & _CSC)
      ret->SetCSC(std::make_shared<CSC>(
          CSC{tmp_ptr->indptr, tmp_ptr->coo_in_indices, torch::nullopt}));
    if (output_format & _COO)
      ret->SetCOO(std::make_shared<COO>(COO{tmp_ptr->coo_in_indices,
                                            tmp_ptr->coo_in_indptr,
                                            torch::nullopt, false, true}));
    ret->SetNumEdges(tmp_ptr->coo_in_indices.numel());
    if (data_.has_value()) {
      if (csc_->e_ids.has_value()) {
        out_data =
            data_.value().index({csc_->e_ids.value().index({select_index})});
      } else {
        out_data = data_.value().index({select_index});
      }
      ret->SetData(out_data);
    }

  } else if (axis == 1 && on_format == _CSR) {
    if (output_format == _CSC)
      LOG(FATAL) << "Error in Sampling, Not implementation [on_format = CSR, "
                    "output_forat = CSC]";

    std::tie(tmp_ptr, select_index) =
        CSCColSamplingProbs(csr_, edge_probs, fanout, replace, with_coo);
    if (output_format & _CSR)
      ret->SetCSR(std::make_shared<CSR>(
          CSR{tmp_ptr->indptr, tmp_ptr->coo_in_indices, torch::nullopt}));
    if (output_format & _COO)
      ret->SetCOO(std::make_shared<COO>(COO{tmp_ptr->coo_in_indptr,
                                            tmp_ptr->coo_in_indices,
                                            torch::nullopt, true, false}));
    ret->SetNumEdges(tmp_ptr->coo_in_indices.numel());
    if (data_.has_value()) {
      if (csr_->e_ids.has_value()) {
        out_data =
            data_.value().index({csr_->e_ids.value().index({select_index})});
      } else {
        out_data = data_.value().index({select_index});
      }
      ret->SetData(out_data);
    }

  } else {
    LOG(FATAL) << "No implementation!";
  }
  return ret;
}

c10::intrusive_ptr<Graph> Graph::ColumnwiseFusedSlicingAndSampling(
    torch::Tensor column_index, int64_t fanout, bool replace) {
  torch::Tensor select_index, out_data;
  std::shared_ptr<CSC> csc_ptr;
  torch::Tensor col_ids = (col_ids_.has_value())
                              ? col_ids_.value().index({column_index})
                              : column_index;
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(true, col_ids, row_ids_, col_ids.numel(), num_rows_)));
  std::tie(csc_ptr, select_index) =
      FusedCSCColSlicingAndSampling(csc_, column_index, fanout, replace);
  ret->SetCSC(csc_ptr);
  ret->SetNumEdges(csc_ptr->indices.numel());
  if (data_.has_value()) {
    torch::Tensor out_data, data_index;
    if (csc_->e_ids.has_value())
      data_index =
          (csc_->e_ids.value().is_pinned())
              ? impl::IndexSelectCPUFromGPU(csc_->e_ids.value(), select_index)
              : csc_->e_ids.value().index({select_index});
    else
      data_index = select_index;
    out_data = (data_.value().is_pinned())
                   ? impl::IndexSelectCPUFromGPU(data_.value(), data_index)
                   : data_.value().index({data_index});
    ret->SetData(out_data);
  }
  return ret;
}

void Graph::CSC2CSR() {
  SetCOO(GraphCSC2COO(csc_, true));
  SetCSR(GraphCOO2CSC(coo_, num_rows_, false));
}

void Graph::CSC2DCSR() {
  SetCOO(GraphCSC2COO(csc_, true));
  std::shared_ptr<CSR> csr_ptr;
  std::tie(csr_ptr, val_row_ids_) = GraphCOO2DCSC(coo_, num_rows_, false);
  SetCSR(csr_ptr);
}

void Graph::CSR2CSC() {
  SetCOO(GraphCSC2COO(csr_, false));
  SetCSC(GraphCOO2CSC(coo_, num_cols_, true));
}

void Graph::CSR2DCSC() {
  SetCOO(GraphCSC2COO(csr_, false));
  std::shared_ptr<CSC> csc_ptr;
  std::tie(csc_ptr, val_col_ids_) = GraphCOO2DCSC(coo_, num_cols_, true);
  SetCSC(csc_ptr);
}

void Graph::CreateSparseFormat(int64_t format) {
  if (format == _COO) {
    if (coo_ != nullptr) return;
    if (csc_ != nullptr) {
      if (val_col_ids_.has_value())
        SetCOO(GraphDCSC2COO(csc_, val_col_ids_.value(), true));
      else
        SetCOO(GraphCSC2COO(csc_, true));
    } else {
      if (val_row_ids_.has_value())
        SetCOO(GraphDCSC2COO(csr_, val_row_ids_.value(), false));
      else
        SetCOO(GraphCSC2COO(csr_, false));
    }
  } else if (format == _CSC) {
    if (csc_ != nullptr) {
      if (val_col_ids_.has_value())
        LOG(FATAL) << "Require CSC, get DCSC instead";
      return;
    }
    if (coo_ != nullptr)
      SetCSC(GraphCOO2CSC(coo_, num_cols_, true));
    else
      CSR2CSC();
  } else if (format == _DCSC) {
    if (csc_ != nullptr) {
      if (!val_col_ids_.has_value())
        LOG(FATAL) << "Require DCSC, get CSC instead";
      return;
    }
    if (coo_ != nullptr) {
      std::shared_ptr<CSC> csc_ptr;
      std::tie(csc_ptr, val_col_ids_) = GraphCOO2DCSC(coo_, num_cols_, true);
      SetCSC(csc_ptr);
    } else
      CSR2DCSC();
  } else if (format == _CSR) {
    if (csr_ != nullptr) {
      if (val_row_ids_.has_value())
        LOG(FATAL) << "Require CSR, get DCSR instead";
      return;
    }
    if (coo_ != nullptr)
      SetCSR(GraphCOO2CSC(coo_, num_rows_, false));
    else
      CSC2CSR();
  } else if (format == _DCSR) {
    if (csr_ != nullptr) {
      if (!val_row_ids_.has_value())
        LOG(FATAL) << "Require DCSR, get CSR instead";
      return;
    }
    if (coo_ != nullptr) {
      std::shared_ptr<CSR> csr_ptr;
      std::tie(csr_ptr, val_row_ids_) = GraphCOO2DCSC(coo_, num_rows_, false);
      SetCSR(csr_ptr);
    } else
      CSC2DCSR();
  }
}

torch::Tensor Graph::RandomWalk(torch::Tensor seeds, int64_t walk_length) {
  return FusedRandomWalk(this->csc_, seeds, walk_length);
}

torch::Tensor Graph::Sum(int64_t axis, int64_t powk, int64_t on_format) {
  CreateSparseFormat(on_format);
  auto in_data =
      data_.has_value()
          ? data_.value()
          : torch::ones(num_edges_,
                        torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto out_size = (axis == 0) ? num_cols_ : num_rows_;
  torch::Tensor out_data = torch::zeros(out_size, in_data.options());

  if (on_format == _COO) {
    COOGraphSum(coo_, in_data, out_data, powk, 1 - axis);
  } else if (axis == 0 && (on_format == _CSC || on_format == _DCSC)) {
    CSCGraphSum(csc_, val_col_ids_, in_data, out_data, powk);
  } else if (axis == 1 && (on_format == _CSR || on_format == _DCSR)) {
    CSCGraphSum(csr_, val_row_ids_, in_data, out_data, powk);
  } else {
    LOG(FATAL) << "axis should be 0 or 1? on_format and axis do not match?";
  }
  return out_data;
}

c10::intrusive_ptr<Graph> Graph::Normalize(int64_t axis, int64_t on_format) {
  CreateSparseFormat(on_format);
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(is_subgraph_, col_ids_, row_ids_, num_cols_, num_rows_)));
  auto in_data =
      data_.has_value()
          ? data_.value()
          : torch::ones(num_edges_,
                        torch::dtype(torch::kFloat32).device(torch::kCUDA));
  auto out_size = (axis == 0) ? num_cols_ : num_rows_;
  torch::Tensor out_data = torch::empty_like(in_data);

  if (on_format == _COO) {
    LOG(FATAL) << "Not Implementation Error";
  } else if (axis == 0 && (on_format == _CSC || on_format == _DCSC)) {
    impl::CSCNormalizeCUDA(csc_->indptr, csc_->e_ids, in_data, out_data);
  } else if (axis == 1 && (on_format == _CSR || on_format == _DCSR)) {
    impl::CSCNormalizeCUDA(csr_->indptr, csr_->e_ids, in_data, out_data);
  } else {
    LOG(FATAL) << "Not Implementation Error";
  }

  ret->SetCSC(csc_);
  ret->SetCSR(csr_);
  ret->SetCOO(coo_);
  ret->SetNumEdges(num_edges_);
  ret->SetData(out_data);
  if (val_col_ids_.has_value()) ret->SetValidCols(val_col_ids_.value());
  if (val_row_ids_.has_value()) ret->SetValidRows(val_row_ids_.value());
  return ret;
}

c10::intrusive_ptr<Graph> Graph::Divide(torch::Tensor divisor, int64_t axis,
                                        int64_t on_format) {
  CreateSparseFormat(on_format);
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(is_subgraph_, col_ids_, row_ids_, num_cols_, num_rows_)));
  auto in_data =
      data_.has_value()
          ? data_.value()
          : torch::ones(num_edges_,
                        torch::dtype(torch::kFloat32).device(torch::kCUDA));
  torch::Tensor out_data = torch::zeros(num_edges_, in_data.options());
  if (on_format == _COO) {
    COOGraphDiv(coo_, in_data, divisor, out_data, 1 - axis);
  } else if (axis == 0 && (on_format == _CSC || on_format == _DCSC)) {
    CSCGraphDiv(csc_, val_col_ids_, in_data, divisor, out_data);
  } else if (axis == 1 && (on_format == _CSR || on_format == _DCSR)) {
    CSCGraphDiv(csr_, val_row_ids_, in_data, divisor, out_data);
  }
  ret->SetCSC(csc_);
  ret->SetCSR(csr_);
  ret->SetCOO(coo_);
  ret->SetNumEdges(num_edges_);
  ret->SetData(out_data);
  if (val_col_ids_.has_value()) ret->SetValidCols(val_col_ids_.value());
  if (val_row_ids_.has_value()) ret->SetValidRows(val_row_ids_.value());
  return ret;
}

std::tuple<c10::intrusive_ptr<Graph>, torch::Tensor> Graph::EDivUSum(
    torch::Tensor divisor) {
  CreateSparseFormat(_COO);
  auto ret = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
      new Graph(is_subgraph_, col_ids_, row_ids_, num_cols_, num_rows_)));
  auto in_data =
      data_.has_value()
          ? data_.value()
          : torch::ones(num_edges_,
                        torch::dtype(torch::kFloat32).device(torch::kCUDA));
  torch::Tensor out_data = torch::zeros(num_edges_, in_data.options());
  torch::Tensor out_sum = torch::zeros(num_cols_, in_data.options());
  impl::fusion::COOEDivUSum(coo_->row, coo_->col, in_data, divisor, out_data,
                            out_sum);

  ret->SetCSC(csc_);
  ret->SetCSR(csr_);
  ret->SetCOO(coo_);
  ret->SetNumEdges(num_edges_);
  ret->SetData(out_data);
  if (val_col_ids_.has_value()) ret->SetValidCols(val_col_ids_.value());
  if (val_row_ids_.has_value()) ret->SetValidRows(val_row_ids_.value());
  return {ret, out_sum};
}

torch::Tensor Graph::AllValidNode() {
  // for sampling, col_ids_ is necessary!
  if (!col_ids_.has_value()) {
    LOG(ERROR) << "For sampling, col_ids_ is necessary!";
  }

  torch::Tensor col_ids = col_ids_.value();
  torch::Tensor row_ids;
  if (csc_ != nullptr) {
    row_ids = (row_ids_.has_value()) ? row_ids_.value() : csc_->indices;
  } else if (coo_ != nullptr) {
    row_ids = (row_ids_.has_value()) ? row_ids_.value() : coo_->row;
  } else if (csr_ != nullptr) {
    // turn csr2coo
    SetCOO(GraphCSC2COO(csr_, false));
    row_ids = (row_ids_.has_value()) ? row_ids_.value() : coo_->row;
  } else {
    LOG(ERROR) << "Error in AllValidNode!";
  }

  return TensorUnique(torch::cat({col_ids, row_ids}));
}

std::tuple<torch::Tensor, int64_t, int64_t, torch::Tensor, torch::Tensor,
           torch::optional<torch::Tensor>, std::string>
Graph::Relabel() {
  // for sampling, col_ids_ is necessary!
  if (!col_ids_.has_value()) {
    LOG(ERROR) << "For sampling, col_ids_ is necessary!";
  }
  torch::Tensor col_ids = col_ids_.value();

  if (csc_ != nullptr) {
    if (val_col_ids_.has_value())
      col_ids = col_ids.index({val_col_ids_.value()});
    torch::Tensor row_indices = row_ids_.has_value()
                                    ? row_ids_.value().index({csc_->indices})
                                    : csc_->indices;

    torch::Tensor frontier;
    std::vector<torch::Tensor> relabeled_result;

    std::tie(frontier, relabeled_result) =
        BatchTensorRelabel({col_ids, row_indices}, {row_indices});

    torch::Tensor relabeled_indptr = csc_->indptr.clone();
    torch::Tensor relabeled_indices = relabeled_result[0];

    return {frontier,
            frontier.numel(),
            col_ids.numel(),
            relabeled_indptr,
            relabeled_indices,
            csc_->e_ids,
            "csc"};

  } else if (csr_ != nullptr or coo_ != nullptr) {
    if (coo_ == nullptr) {
      SetCOO(GraphCSC2COO(csr_, false));
    }
    torch::Tensor coo_col = col_ids.index({coo_->col});
    torch::Tensor coo_row = (row_ids_.has_value())
                                ? row_ids_.value().index({coo_->row})
                                : coo_->row;

    torch::Tensor frontier;
    std::vector<torch::Tensor> relabeled_result;
    std::tie(frontier, relabeled_result) =
        BatchTensorRelabel({col_ids, coo_row}, {coo_col, coo_row});

    return {frontier,
            frontier.numel(),
            col_ids.numel(),
            relabeled_result[1],
            relabeled_result[0],
            coo_->e_ids,
            "coo"};

  } else {
    LOG(ERROR) << "Error in Relabel!";
    return {};
  }
}

torch::Tensor Graph::GetRows() {
  if (row_ids_.has_value()) {
    return row_ids_.value();
  } else {
    if (csr_ != nullptr) {
      return torch::arange(num_rows_, csr_->indices.options());
    } else {
      return torch::arange(num_rows_, csc_->indices.options());
    }
  }
}

torch::Tensor Graph::GetCols() {
  if (col_ids_.has_value()) {
    return col_ids_.value();
  } else {
    if (csc_ != nullptr) {
      return torch::arange(num_cols_, csc_->indices.options());
    } else {
      return torch::arange(num_cols_, csr_->indices.options());
    }
  }
}

torch::Tensor Graph::GetValidRows() {
  torch::Tensor valid_row_local_ids =
      val_row_ids_.has_value() ? val_row_ids_.value()
                               : std::get<0>(torch::_unique(GetCOORows(false)));
  return row_ids_.has_value() ? row_ids_.value().index({valid_row_local_ids})
                              : valid_row_local_ids;
}

torch::Tensor Graph::GetValidCols() {
  torch::Tensor valid_col_local_ids =
      val_col_ids_.has_value() ? val_col_ids_.value()
                               : std::get<0>(torch::_unique(GetCOOCols(false)));
  return col_ids_.has_value() ? col_ids_.value().index({valid_col_local_ids})
                              : valid_col_local_ids;
}

torch::Tensor Graph::GetCOORows(bool is_original) {
  torch::Tensor coo_rows;
  if (coo_ != nullptr) {
    coo_rows = coo_->row;
  } else if (csc_ != nullptr && !csc_->e_ids.has_value()) {
    coo_rows = csc_->indices;
  } else if (csr_ != nullptr && !csr_->e_ids.has_value()) {
    SetCOO(GraphCSC2COO(csr_, false));
    coo_rows = coo_->row;
  } else {
    LOG(ERROR) << "Error in GetCOORows!";
  }

  return (is_original && row_ids_.has_value())
             ? row_ids_.value().index({coo_rows})
             : coo_rows;
};

torch::Tensor Graph::GetCOOCols(bool is_original) {
  torch::Tensor coo_cols;
  if (coo_ != nullptr) {
    coo_cols = coo_->col;
  } else if (csc_ != nullptr && !csc_->e_ids.has_value()) {
    SetCOO(GraphCSC2COO(csc_, true));
    coo_cols = coo_->col;
  } else if (csr_ != nullptr && !csr_->e_ids.has_value()) {
    coo_cols = csr_->indices;
  } else {
    LOG(ERROR) << "Error in GetCOOCols!";
  }
  return (is_original && col_ids_.has_value())
             ? col_ids_.value().index({coo_cols})
             : coo_cols;
};

std::vector<torch::Tensor> Graph::MetaData() {
  torch::Tensor col_ids, row_ids, data;
  col_ids = (col_ids_.has_value()) ? col_ids_.value() : torch::Tensor();
  row_ids = (row_ids_.has_value()) ? row_ids_.value() : torch::Tensor();
  if (csc_ != nullptr) {
    data =
        (data_.has_value())
            ? data_.value()
            : torch::ones(csc_->indices.numel(),
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));
    return {col_ids, row_ids, data, csc_->indptr, csc_->indices};
  } else if (csr_ != nullptr) {
    data =
        (data_.has_value())
            ? data_.value()
            : torch::ones(csr_->indices.numel(),
                          torch::dtype(torch::kFloat32).device(torch::kCUDA));
    return {col_ids, row_ids, data, csr_->indptr, csr_->indices};
  } else {
    LOG(INFO) << "Warning in MetaData: no CSC nor CSR.";
    return {coo_->row, coo_->col};
  }
}

std::vector<torch::Tensor> Graph::COOMetaData() {
  torch::Tensor col_ids, row_ids, data;
  col_ids = (col_ids_.has_value()) ? col_ids_.value() : torch::Tensor();
  row_ids = (row_ids_.has_value()) ? row_ids_.value() : torch::Tensor();
  data = data_.has_value() ? data_.value() : torch::Tensor();
  if (coo_ != nullptr) {
    return {col_ids, row_ids, data, coo_->row, coo_->col};
  } else {
    LOG(INFO) << "Warning in MetaData: no COO.";
    return {};
  }
}

std::vector<torch::Tensor> Graph::CSCMetaData() {
  torch::Tensor col_ids, row_ids, data;
  col_ids = (col_ids_.has_value()) ? col_ids_.value() : torch::Tensor();
  row_ids = (row_ids_.has_value()) ? row_ids_.value() : torch::Tensor();
  data = data_.has_value() ? data_.value() : torch::Tensor();
  if (csc_ != nullptr) {
    return {col_ids, row_ids, data, csc_->indptr, csc_->indices};
  } else {
    LOG(INFO) << "Warning in MetaData: no CSC.";
    return {};
  }
}

/*! \brief Generalized Sampled Dense-Dense Matrix Multiplication. */
void Graph::SDDMM(const std::string& op, torch::Tensor lhs, torch::Tensor rhs,
                  torch::Tensor out, int64_t lhs_target, int64_t rhs_target,
                  int64_t on_format) {
  CreateSparseFormat(on_format);
  const auto& bcast = CalcBcastOff(op, lhs, rhs);
  if (on_format == _COO) {
    impl::SDDMMCOO(op, bcast, coo_, lhs, rhs, out, lhs_target, rhs_target);
  } else if (on_format == _CSR || on_format == _DCSR) {
    lhs_target = lhs_target == 1 ? lhs_target : (2 - lhs_target);
    rhs_target = rhs_target == 1 ? rhs_target : (2 - rhs_target);
    impl::SDDMMCSC(op, bcast, csr_, val_row_ids_, lhs, rhs, out, lhs_target,
                   rhs_target);
  } else if (on_format == _CSC || on_format == _DCSC) {
    impl::SDDMMCSC(op, bcast, csc_, val_col_ids_, lhs, rhs, out, lhs_target,
                   rhs_target);
  } else {
    LOG(FATAL) << "SDDMM only supports CSR and COO formats";
  }
}

std::vector<c10::intrusive_ptr<Graph>> Graph::Split(int64_t split_size) {
  std::vector<c10::intrusive_ptr<Graph>> ret;
  std::vector<torch::Tensor> splitted_colid;
  std::vector<torch::Tensor> indptr, indices, eid, nid;
  std::shared_ptr<CSC> csc_ptr;
  c10::intrusive_ptr<Graph> ret_graph;

  if (col_ids_.has_value()) nid = torch::split(col_ids_.value(), split_size);
  auto ret_vec =
      impl::CSCSplitCUDA(csc_->indptr, csc_->indices, csc_->e_ids, split_size);
  indptr = ret_vec[0], indices = ret_vec[1], eid = ret_vec[2];
  int64_t num_res = indptr.size();
  for (int i = 0; i < num_res; ++i) {
    if (csc_->e_ids.has_value())
      csc_ptr = std::make_shared<CSC>(CSC{indptr[i], indices[i], eid[i]});
    else
      csc_ptr =
          std::make_shared<CSC>(CSC{indptr[i], indices[i], torch::nullopt});
    if (col_ids_.has_value())
      ret_graph = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(
          new Graph(true, nid[i], row_ids_, nid[i].numel(), num_rows_)));
    else
      ret_graph = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(new Graph(
          true, torch::nullopt, row_ids_, indptr[i].numel() - 1, num_rows_)));
    ret_graph->SetCSC(csc_ptr);
    ret_graph->SetNumEdges(indices[i].numel());
    ret.push_back(ret_graph);
  }

  return ret;
}

std::vector<torch::Tensor> Graph::GetCSCTensor() {
  CreateSparseFormat(_CSC);
  auto eid = csc_->e_ids.has_value() ? csc_->e_ids.value() : torch::Tensor();
  return {csc_->indptr, csc_->indices, eid};
}

std::vector<torch::Tensor> Graph::GetCOOTensor() {
  CreateSparseFormat(_COO);
  CHECK_EQ(coo_->col_sorted, true);
  if (!col_ids_.has_value()) {
    LOG(ERROR) << "For sampling, col_ids_ is necessary!";
  }
  auto eid = coo_->e_ids.has_value() ? coo_->e_ids.value() : torch::Tensor();
  return {coo_->row, coo_->col, eid};
}

void Graph::Decode(int64_t encoding_size) {
  if (col_ids_.has_value() || row_ids_.has_value())
    LOG(FATAL) << "Error: Decoding should have both row and column id";
  col_ids_ = impl::BatchDecodeCUDA(col_ids_.value(), encoding_size);
  row_ids_ = impl::BatchDecodeCUDA(row_ids_.value(), encoding_size);
}

}  // namespace gs
