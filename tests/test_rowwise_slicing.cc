#include <gtest/gtest.h>
#include "graph.h"
#include <torch/torch.h>

using namespace gs;

TEST(RowwiseSlicing, test1)
{
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    torch::Tensor indptr = torch::cat({torch::zeros(10, options), torch::arange(0, 3, options) * 2});
    torch::Tensor indices = torch::arange(5, 7, options).repeat({2});
    torch::Tensor expected_indices = torch::arange(10, 12, options);
    Graph A(indptr.numel() - 1, indptr.numel() - 1);
    A.LoadCSC(indptr, indices);

    torch::Tensor row_ids = torch::arange(5, 6, options);
    c10::intrusive_ptr<Graph> subA;
    torch::Tensor select_index;
    std::tie(subA, select_index) = A.Slicing(row_ids, 0, _CSR, _CSR);
    auto csr_ptr = subA->GetCSR();

    EXPECT_EQ(csr_ptr->indptr.numel(), 2);
    EXPECT_EQ(csr_ptr->indices.numel(), 2);
    EXPECT_TRUE(csr_ptr->indices.equal(expected_indices));
}