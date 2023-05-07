#include <gtest/gtest.h>
#include "graph.h"
#include <torch/torch.h>

using namespace gs;

TEST(ColwiseSlicing, test1)
{
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    Graph A(indptr.numel() - 1, indptr.numel() - 1);
    A.LoadCSC(indptr, indices);
    torch::Tensor col_ids = torch::arange(0, 3, options);
    c10::intrusive_ptr<Graph> subA;
    torch::Tensor select_index;
    std::tie(subA, select_index) = A.Slicing(col_ids, 1, _CSC, _CSC);
    auto csc_ptr = subA->GetCSC();
    EXPECT_TRUE(csc_ptr->indptr.equal(torch::arange(0, 4, options) * 5));
    EXPECT_TRUE(csc_ptr->indices.equal(torch::arange(0, 15, options)));
}
