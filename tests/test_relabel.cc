#include <gtest/gtest.h>
#include "graph.h"
#include <torch/torch.h>

using namespace gs;

TEST(SlicingRelabel, test1)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    torch::Tensor col_ids = torch::arange(0, 20, options);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    A.LoadCSCWithColIds(col_ids, indptr, indices);

    torch::Tensor ids = torch::arange(1, 4, options);
    auto subA = A.Slicing(ids, 0, _CSC, _CSC, true);
    EXPECT_EQ(A.GetNumCols(), 20);
    EXPECT_EQ(A.GetNumRows(), 100);
    EXPECT_EQ(subA->GetNumCols(), 3);
    EXPECT_EQ(subA->GetNumRows(), 15);

    auto csc_ptr = subA->GetCSC();
    EXPECT_TRUE(csc_ptr->indptr.equal(torch::arange(0, 4, options) * 5));
    EXPECT_TRUE(csc_ptr->indices.equal(torch::arange(0, 15, options)));
}

TEST(SlicingRelabel, test2)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    torch::Tensor indptr = torch::cat({torch::zeros(10, options), torch::arange(0, 3, options) * 2});
    torch::Tensor indices = torch::arange(5, 7, options).repeat({2});
    torch::Tensor expected_indices = torch::arange(0, 2, options);
    A.LoadCSC(indptr, indices);
    A.CSC2DCSR();

    torch::Tensor row_ids = torch::arange(5, 6, options);
    auto subA = A.Slicing(row_ids, 1, _DCSR, _CSR, true);
    EXPECT_EQ(A.GetNumCols(), 12);
    EXPECT_EQ(A.GetNumRows(), 12);
    EXPECT_EQ(subA->GetNumCols(), 2);
    EXPECT_EQ(subA->GetNumRows(), 1);

    auto csr_ptr = subA->GetCSR();
    EXPECT_EQ(csr_ptr->indptr.numel(), 2);
    EXPECT_EQ(csr_ptr->indices.numel(), 2);
    EXPECT_TRUE(csr_ptr->indices.equal(expected_indices));
}