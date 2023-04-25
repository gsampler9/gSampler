#include <gtest/gtest.h>
#include "graph.h"
#include <torch/torch.h>

using namespace gs;

TEST(RowwiseSlicing, test1)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    torch::Tensor indptr = torch::arange(0, 21, options).to(torch::kCUDA) * 5;
    torch::Tensor indices = torch::arange(0, 100, options).to(torch::kCUDA);
    A.LoadCSC(indptr, indices);

    torch::Tensor row_ids = torch::arange(0, 100, 5, options).to(torch::kCUDA);
    auto subA = A.Slicing(row_ids, 1, _CSC, _CSC);
    auto csc_ptr = subA->GetCSC();
    EXPECT_TRUE(csc_ptr->indptr.equal(torch::arange(21).to(torch::kCUDA) * 1));
    EXPECT_TRUE(csc_ptr->indices.equal(torch::arange(20).to(torch::kCUDA)));
}

TEST(RowwiseSlicing, test2)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    torch::Tensor indptr = torch::cat({torch::zeros(10, options), torch::arange(0, 3, options) * 2});
    torch::Tensor indices = torch::arange(5, 7, options).repeat({2});
    torch::Tensor expected_indices = torch::arange(10, 12, options);
    A.LoadCSC(indptr, indices);
    A.CSC2DCSR();

    torch::Tensor row_ids = torch::arange(5, 6, options);
    auto subA = A.Slicing(row_ids, 1, _DCSR, _CSR);
    auto csr_ptr = subA->GetCSR();

    EXPECT_EQ(csr_ptr->indptr.numel(), 2);
    EXPECT_EQ(csr_ptr->indices.numel(), 2);
    EXPECT_TRUE(csr_ptr->indices.equal(expected_indices));
}