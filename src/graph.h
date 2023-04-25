#ifndef GS_GRAPH_H_
#define GS_GRAPH_H_

#include <torch/custom_class.h>
#include <torch/script.h>

#include "./graph_storage.h"

namespace gs {

class Graph : public torch::CustomClassHolder {
 public:
  Graph(bool is_subgraph) { is_subgraph_ = is_subgraph; }
  Graph(bool is_subgraph, torch::optional<torch::Tensor> col_ids,
        torch::optional<torch::Tensor> row_ids, int64_t num_cols,
        int64_t num_rows)
      : is_subgraph_{is_subgraph},
        num_cols_{num_cols},
        num_rows_{num_rows},
        col_ids_{col_ids},
        row_ids_{row_ids} {}
  void LoadCSC(torch::Tensor indptr, torch::Tensor indices);
  void LoadCOO(torch::Tensor row, torch::Tensor col);
  void LoadCSR(torch::Tensor indptr, torch::Tensor indices);
  void LoadCSCWithColIds(torch::Tensor column_ids, torch::Tensor indptr,
                         torch::Tensor indices);
  void SetMetaData(torch::optional<torch::Tensor> col_ids,
                   torch::optional<torch::Tensor> row_ids, int64_t num_cols,
                   int64_t num_rows) {
    num_cols_ = num_cols;
    num_rows_ = num_rows;
    col_ids_ = col_ids;
    row_ids_ = row_ids;
  }
  void SetCSC(std::shared_ptr<CSC> csc);
  void SetCSR(std::shared_ptr<CSR> csr);
  void SetCOO(std::shared_ptr<COO> coo);
  void SetCOOByTensor(torch::Tensor row, torch::Tensor col,
                      torch::optional<torch::Tensor> e_ids, bool row_sorted,
                      bool col_sorted);
  void SetData(torch::Tensor data);
  void SetValidCols(torch::Tensor val_cols);
  void SetValidRows(torch::Tensor val_rows);
  void SetNumEdges(int64_t num_edges);
  void CSC2CSR();
  void CSC2DCSR();
  void CSR2CSC();
  void CSR2DCSC();
  void CreateSparseFormat(int64_t format);
  std::shared_ptr<CSC> GetCSC();
  std::shared_ptr<CSR> GetCSR();
  std::shared_ptr<COO> GetCOO();
  std::vector<torch::Tensor> GetCSCTensor();
  std::vector<torch::Tensor> GetCOOTensor();
  torch::optional<torch::Tensor> GetData(std::string order = "default");
  int64_t GetNumRows();
  int64_t GetNumCols();
  int64_t GetNumEdges();
  c10::intrusive_ptr<Graph> FusedBidirSlicing(torch::Tensor column_seeds,
                                              torch::Tensor row_seeds);
  c10::intrusive_ptr<Graph> BatchFusedBidirSlicing(torch::Tensor column_seeds,
                                                   torch::Tensor col_ptr,
                                                   torch::Tensor row_seeds,
                                                   torch::Tensor row_ptr);
  c10::intrusive_ptr<Graph> Slicing(torch::Tensor n_ids, int64_t axis,
                                    int64_t on_format, int64_t output_format,
                                    bool relabel = false);
  std::tuple<c10::intrusive_ptr<Graph>, torch::Tensor> BatchSlicing(
      torch::Tensor n_ids, torch::Tensor nid_ptr, int64_t axis,
      int64_t on_format, int64_t output_format, bool relabel, bool encoding);

  c10::intrusive_ptr<Graph> Sampling(int64_t axis, int64_t fanout, bool replace,
                                     int64_t on_format, int64_t output_format);
  c10::intrusive_ptr<Graph> SamplingProbs(int64_t axis,
                                          torch::Tensor edge_probs,
                                          int64_t fanout, bool replace,
                                          int64_t on_format,
                                          int64_t output_format);
  c10::intrusive_ptr<Graph> ColumnwiseFusedSlicingAndSampling(
      torch::Tensor column_index, int64_t fanout, bool replace);
  torch::Tensor Sum(int64_t axis, int64_t powk, int64_t on_format);
  c10::intrusive_ptr<Graph> Normalize(int64_t axis, int64_t on_format);
  c10::intrusive_ptr<Graph> Divide(torch::Tensor divisor, int64_t axis,
                                   int64_t on_format);
  std::tuple<c10::intrusive_ptr<Graph>, torch::Tensor> EDivUSum(
      torch::Tensor divisor);
  // A "valid" node means that the node is required by the user or that it is
  // not an isolated node.
  torch::Tensor AllValidNode();
  torch::Tensor GetRows();       // return row_ids
  torch::Tensor GetCols();       // return col_ids
  torch::Tensor GetValidRows();  // return valid row_ids.
  torch::Tensor GetValidCols();  // return valid col_ids.

  //  If is_original, it return  in global_id else in local_id.
  torch::Tensor GetCOORows(bool is_original);  // return coo_row, which is
                                               // coo[0]
  torch::Tensor GetCOOCols(bool is_original);  // return coo_row, which is
                                               // coo[1]
  std::tuple<torch::Tensor, int64_t, int64_t, torch::Tensor, torch::Tensor,
             torch::optional<torch::Tensor>, std::string>
  Relabel();
  std::vector<torch::Tensor> MetaData();
  std::vector<torch::Tensor> COOMetaData();
  std::vector<torch::Tensor> CSCMetaData();
  torch::Tensor RandomWalk(torch::Tensor seeds, int64_t walk_length);
  void SDDMM(const std::string& op, torch::Tensor lhs, torch::Tensor rhs,
             torch::Tensor out, int64_t lhs_target, int64_t rhs_target,
             int64_t on_format);
  std::vector<c10::intrusive_ptr<Graph>> Split(int64_t split_size);
  void Decode(int64_t encoding_size);

  // todo: return global_e_id
 private:
  bool is_subgraph_ = false;
  int64_t num_cols_ = 0;  // total number of cols in a matrix
  int64_t num_rows_ = 0;  // total number of rows in a matrix
  int64_t num_edges_ = 0;
  std::shared_ptr<CSC> csc_;
  std::shared_ptr<CSR> csr_;
  std::shared_ptr<COO> coo_;
  torch::optional<torch::Tensor> data_;
  torch::optional<torch::Tensor> col_ids_;      // column id in matrix
  torch::optional<torch::Tensor> row_ids_;      // row id in matrix
  torch::optional<torch::Tensor> val_col_ids_;  // valid column id in matrix
  torch::optional<torch::Tensor> val_row_ids_;  // valid row id in matrix
};

}  // namespace gs

#endif
