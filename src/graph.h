#ifndef GS_GRAPH_H_
#define GS_GRAPH_H_

#include <torch/custom_class.h>
#include <torch/script.h>

#include "./graph_storage.h"

namespace gs {

class Graph : public torch::CustomClassHolder {
 public:
  // init graph
  // python code will make sure that all inputs are legitimate
  Graph(int64_t num_rows, int64_t num_cols) {
    num_rows_ = num_rows;
    num_cols_ = num_cols;
  }
  void LoadCSC(torch::Tensor indptr, torch::Tensor indices) {
    csc_ = std::make_shared<CSC>();
    csc_->indptr = indptr;
    csc_->indices = indices;
    num_edges_ = indices.numel();
    LOG(INFO) << "Loaded CSC with " << num_rows_ << " rows, " << num_cols_
              << " cols, " << num_edges_ << " edges!";
  }
  void LoadCOO(torch::Tensor row, torch::Tensor col, bool row_sorted,
               bool col_sorted) {
    coo_ = std::make_shared<COO>();
    coo_->row = row;
    coo_->col = col;
    coo_->row_sorted = row_sorted;
    coo_->col_sorted = col_sorted;
    num_edges_ = row.numel();
    LOG(INFO) << "Loaded COO with " << num_rows_ << " rows, " << num_cols_
              << " cols, " << num_edges_ << " edges!";
  }
  void LoadCSR(torch::Tensor indptr, torch::Tensor indices) {
    csr_ = std::make_shared<CSR>();
    csr_->indptr = indptr;
    csr_->indices = indices;
    num_edges_ = indices.numel();
    LOG(INFO) << "Loaded CSR with " << num_rows_ << " rows, " << num_cols_
              << " cols, " << num_edges_ << " edges!";
  }

  // set private member
  void SetCSC(std::shared_ptr<CSC> csc) { csc_ = csc; }
  void SetCSR(std::shared_ptr<CSR> csr) { csr_ = csr; }
  void SetCOO(std::shared_ptr<COO> coo) { coo_ = coo; }
  void SetNumEdges(int64_t num_edges) { num_edges_ = num_edges; }
  void SetNumCols(int64_t num_cols) { num_cols_ = num_cols; }
  void SetNumRows(int64_t num_rows) { num_rows_ = num_rows; }

  // get private member
  std::shared_ptr<CSC> GetCSC() { return csc_; }
  std::shared_ptr<CSR> GetCSR() { return csr_; }
  std::shared_ptr<COO> GetCOO() { return coo_; }
  int64_t GetNumRows() { return num_rows_; }
  int64_t GetNumCols() { return num_cols_; }
  int64_t GetNumEdges() { return num_edges_; }

  // format coversion
  void CreateSparseFormat(int64_t format);

  // return format
  torch::Tensor GetCSCIndptr();
  torch::Tensor GetCSCIndices();
  torch::Tensor GetCSCEids();
  torch::Tensor GetCOORows();
  torch::Tensor GetCOOCols();
  torch::Tensor GetCOOEids();
  torch::Tensor GetCSRIndptr();
  torch::Tensor GetCSRIndices();
  torch::Tensor GetCSREids();

  // graph operation
  std::tuple<c10::intrusive_ptr<Graph>, torch::Tensor> Slicing(
      torch::Tensor seeds, int64_t axis, int64_t on_format,
      int64_t output_format);

  std::tuple<c10::intrusive_ptr<Graph>, torch::Tensor> Sampling(
      int64_t axis, int64_t fanout, bool replace, int64_t on_format,
      int64_t output_format);
  std::tuple<c10::intrusive_ptr<Graph>, torch::Tensor> SamplingProbs(
      int64_t axis, torch::Tensor edge_probs, int64_t fanout, bool replace,
      int64_t on_format, int64_t output_format);
  void SDDMM(const std::string& op, torch::Tensor lhs, torch::Tensor rhs,
             torch::Tensor out, int64_t lhs_target, int64_t rhs_target,
             int64_t on_format);
  void SpMM(const std::string& op, const std::string& reduce,
            torch::Tensor ufeat, torch::Tensor efeat, torch::Tensor out,
            torch::Tensor argu, torch::Tensor arge, int64_t u_target,
            int64_t on_format);

  std::tuple<torch::Tensor, int64_t, int64_t, torch::Tensor, torch::Tensor,
             torch::optional<torch::Tensor>, std::string>
  GraphRelabel(torch::Tensor col_seeds, torch::Tensor row_ids);
  torch::Tensor GetValidNodes(torch::Tensor col_seeds, torch::Tensor row_ids);

  torch::Tensor RandomWalk(torch::Tensor seeds, int64_t walk_length);
  torch::Tensor Node2Vec(torch::Tensor seeds, int64_t walk_length, double p,
                         double q);

 private:
  int64_t num_cols_ = 0;   // total number of cols in matrix
  int64_t num_rows_ = 0;   // total number of rows in matrix
  int64_t num_edges_ = 0;  // total number of edges in matrix
  std::shared_ptr<CSC> csc_;
  std::shared_ptr<CSR> csr_;
  std::shared_ptr<COO> coo_;
};

}  // namespace gs

#endif
