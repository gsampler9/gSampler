#include "graph_ops.h"
#include "bcast.h"
#include "cuda/batch/batch_ops.h"
#include "cuda/fusion/column_row_slicing.h"
#include "cuda/fusion/node2vec.h"
#include "cuda/fusion/random_walk.h"
#include "cuda/fusion/slice_sampling.h"
#include "cuda/graph_ops.h"
#include "cuda/sddmm.h"
#include "cuda/tensor_ops.h"

namespace gs {

std::shared_ptr<COO> GraphCSC2COO(std::shared_ptr<CSC> csc, bool CSC2COO) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor row, col;
    bool row_sorted = false, col_sorted = false;
    if (CSC2COO) {
      std::tie(row, col) = impl::CSC2COOCUDA(csc->indptr, csc->indices);
      col_sorted = true;
    } else {
      std::tie(col, row) = impl::CSC2COOCUDA(csc->indptr, csc->indices);
      row_sorted = true;
    }
    return std::make_shared<COO>(
        COO{row, col, csc->e_ids, row_sorted, col_sorted});
  } else {
    LOG(FATAL) << "Not implemented warning";
    return std::make_shared<COO>(COO{});
  }
}

std::shared_ptr<CSC> GraphCOO2CSC(std::shared_ptr<COO> coo, int64_t num_items,
                                  bool COO2CSC) {
  if (coo->row.device().type() == torch::kCUDA) {
    torch::Tensor indptr, indices;
    torch::optional<torch::Tensor> sorted_e_ids = torch::nullopt,
                                   sort_index = torch::nullopt;
    if (COO2CSC) {
      std::tie(indptr, indices, sort_index) =
          impl::COO2CSCCUDA(coo->row, coo->col, num_items, coo->col_sorted);
    } else {
      std::tie(indptr, indices, sort_index) =
          impl::COO2CSCCUDA(coo->col, coo->row, num_items, coo->row_sorted);
    }

    if (coo->e_ids.has_value()) {
      if (sort_index.has_value()) {
        sorted_e_ids = coo->e_ids.value().index({sort_index});
      } else {
        sorted_e_ids = coo->e_ids.value();
      }
    } else {
      sorted_e_ids = sort_index;
    }
    return std::make_shared<CSC>(CSC{indptr, indices, sorted_e_ids});
  } else {
    LOG(FATAL) << "Not implemented warning";
    return std::make_shared<CSC>(CSC{});
  }
}

// Slicing Operators
std::pair<std::shared_ptr<_TMP>, torch::Tensor> OnIndptrSlicing(
    std::shared_ptr<CSC> csc, torch::Tensor node_ids, bool with_coo) {
  auto csc_type = csc->indptr.device().type();
  if (csc_type == torch::kCUDA || csc->indptr.is_pinned()) {
    torch::Tensor sub_indptr, coo_col, coo_row, select_index;
    std::tie(sub_indptr, coo_col, coo_row, select_index) =
        impl::OnIndptrSlicingCUDA(csc->indptr, csc->indices, node_ids,
                                  with_coo);
    return {std::make_shared<_TMP>(_TMP{sub_indptr, coo_col, coo_row}),
            select_index};
  } else {
    LOG(FATAL) << "Not implemented warning";
    return {std::make_shared<_TMP>(_TMP{}), torch::Tensor()};
  }
}

std::pair<std::shared_ptr<_TMP>, torch::Tensor> OnIndicesSlicing(
    std::shared_ptr<CSC> csc, torch::Tensor node_ids, bool with_coo) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, coo_col, coo_row, select_index;
    std::tie(sub_indptr, coo_col, coo_row, select_index) =
        impl::OnIndicesSlicingCUDA(csc->indptr, csc->indices, node_ids,
                                   with_coo);
    return {std::make_shared<_TMP>(_TMP{sub_indptr, coo_col, coo_row}),
            select_index};
  } else {
    LOG(FATAL) << "Not implemented warning";
    return {std::make_shared<_TMP>(_TMP{}), torch::Tensor()};
  }
}

// axis == 0 for row, axis == 1 for column
std::pair<std::shared_ptr<COO>, torch::Tensor> COOSlicing(
    std::shared_ptr<COO> coo, torch::Tensor node_ids, int64_t axis) {
  if (coo->col.device().type() == torch::kCUDA) {
    torch::Tensor sub_coo_row, sub_coo_col, select_index;
    if (axis == 0)
      std::tie(sub_coo_row, sub_coo_col, select_index) =
          impl::COORowSlicingCUDA(coo->row, coo->col, node_ids);
    else
      std::tie(sub_coo_col, sub_coo_row, select_index) =
          impl::COORowSlicingCUDA(coo->col, coo->row, node_ids);
    return {std::make_shared<COO>(COO{sub_coo_row, sub_coo_col, torch::nullopt,
                                      coo->row_sorted, coo->col_sorted}),
            select_index};
  } else {
    LOG(FATAL) << "Not implemented warning";
    return {std::make_shared<COO>(COO{}), torch::Tensor()};
  }
}

// sampling operators
std::pair<std::shared_ptr<_TMP>, torch::Tensor> CSCColSampling(
    std::shared_ptr<CSC> csc, int64_t fanout, bool replace, bool with_coo) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_coo_col, sub_indices, select_index;
    std::tie(sub_indptr, sub_coo_col, sub_indices, select_index) =
        impl::CSCColSamplingCUDA(csc->indptr, csc->indices, fanout, replace,
                                 with_coo);
    return {std::make_shared<_TMP>(_TMP{sub_indptr, sub_coo_col, sub_indices}),
            select_index};
  } else {
    LOG(FATAL) << "Not implemented warning";
    return {std::make_shared<_TMP>(_TMP{}), torch::Tensor()};
  }
}

std::pair<std::shared_ptr<_TMP>, torch::Tensor> CSCColSamplingProbs(
    std::shared_ptr<CSC> csc, torch::Tensor edge_probs, int64_t fanout,
    bool replace, bool with_coo) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_coo_col, sub_indices, select_index;
    torch::Tensor input_edge_probs;

    if (csc->e_ids.has_value()) {
      input_edge_probs = edge_probs.index_select(0, csc->e_ids.value());
    } else {
      input_edge_probs = edge_probs;
    }

    std::tie(sub_indptr, sub_coo_col, sub_indices, select_index) =
        impl::CSCColSamplingProbsCUDA(csc->indptr, csc->indices,
                                      input_edge_probs, fanout, replace,
                                      with_coo);
    return {std::make_shared<_TMP>(_TMP{sub_indptr, sub_coo_col, sub_indices}),
            select_index};
  } else {
    LOG(FATAL) << "Not implemented warning";
    return {std::make_shared<_TMP>(_TMP{}), torch::Tensor()};
  }
}

std::pair<std::shared_ptr<_TMP>, torch::Tensor> CSCSlicingSampling(
    std::shared_ptr<CSC> csc, torch::Tensor node_ids, int64_t fanout,
    bool replace, bool with_coo) {
  if (csc->indptr.device().type() == torch::kCUDA || csc->indptr.is_pinned()) {
    torch::Tensor sub_indptr, sub_coo_col, sub_indices, select_index;
    std::tie(sub_indptr, sub_coo_col, sub_indices, select_index) =
        impl::fusion::FusedCSCColSlicingSamplingCUDA(
            csc->indptr, csc->indices, fanout, node_ids, replace, with_coo);
    return {std::make_shared<_TMP>(_TMP{sub_indptr, sub_coo_col, sub_indices}),
            select_index};
  } else {
    LOG(FATAL) << "Not implemented warning";
    return {std::make_shared<_TMP>(_TMP{}), torch::Tensor()};
  }
}

torch::Tensor FusedRandomWalk(std::shared_ptr<CSC> csc, torch::Tensor seeds,
                              int64_t walk_length) {
  torch::Tensor paths = impl::fusion::FusedRandomWalkCUDA(
      seeds, walk_length, csc->indices.data_ptr<int64_t>(),
      csc->indptr.data_ptr<int64_t>());
  return paths;
}

torch::Tensor FusedNode2Vec(std::shared_ptr<CSC> csc, torch::Tensor seeds,
                            int64_t walk_length, double p, double q) {
  torch::Tensor paths = impl::fusion::FusedNode2VecCUDA(
      seeds, walk_length, csc->indices.data_ptr<int64_t>(),
      csc->indptr.data_ptr<int64_t>(), p, q);
  return paths;
}

// Slicing Operators
std::tuple<std::shared_ptr<_TMP>, torch::Tensor, torch::Tensor>
BatchOnIndptrSlicing(std::shared_ptr<CSC> csc, torch::Tensor node_ids,
                     torch::Tensor batch_ptr, bool with_coo, bool encoding,
                     int64_t encoding_size) {
  auto csc_type = csc->indptr.device().type();
  if (csc_type == torch::kCUDA || csc->indptr.is_pinned()) {
    torch::Tensor sub_indptr, coo_col, coo_row, select_index, indices_ptr;
    std::tie(sub_indptr, coo_col, coo_row, select_index, indices_ptr) =
        impl::batch::BatchOnIndptrSlicingCUDA(csc->indptr, csc->indices,
                                              node_ids, batch_ptr, with_coo,
                                              encoding, encoding_size);
    return {std::make_shared<_TMP>(_TMP{sub_indptr, coo_col, coo_row}),
            select_index, indices_ptr};
  } else {
    LOG(FATAL) << "Not implemented warning";
    return {std::make_shared<_TMP>(_TMP{}), torch::Tensor(), torch::Tensor()};
  }
}
}  // namespace gs
