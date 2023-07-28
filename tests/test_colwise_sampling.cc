#include <gtest/gtest.h>
#include "graph.h"
#include <torch/torch.h>

using namespace gs;

TEST(ColwiseSampling, test1)
{

    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    Graph A(indptr.numel() - 1, indptr.numel() - 1);
    A.LoadCSC(indptr, indices);

    int64_t fanout = 6;
    c10::intrusive_ptr<Graph> subA;
    torch::Tensor select_index;
    std::tie(subA, select_index) = A.Sampling(1, fanout, false, _CSC, _CSC);
    auto csc_ptr = subA->GetCSC();
    EXPECT_TRUE(csc_ptr->indptr.equal(indptr));
    EXPECT_TRUE(csc_ptr->indices.equal(indices));
}

TEST(ColwiseSampling, test2)
{
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    Graph A(indptr.numel() - 1, indptr.numel() - 1);
    A.LoadCSC(indptr, indices);
    int64_t fanout = 6;
    c10::intrusive_ptr<Graph> subA;
    torch::Tensor select_index;
    std::tie(subA, select_index) = A.Sampling(1, fanout, true, _CSC, _CSC);
    auto csc_ptr = subA->GetCSC();
    EXPECT_TRUE(csc_ptr->indptr.equal(torch::arange(0, 21, options) * 6));
    EXPECT_FALSE(csc_ptr->indices.equal(indices));
    EXPECT_TRUE(csc_ptr->indices.numel() == (indptr.numel() - 1) * fanout);
    ASSERT_TRUE(std::get<0>(torch::_unique(csc_ptr->indices)).numel() <= 100);
}

TEST(ColwiseSampling, test3)
{

    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    Graph A(indptr.numel() - 1, indptr.numel() - 1);
    A.LoadCSC(indptr, indices);
    int64_t fanout = 4;
    c10::intrusive_ptr<Graph> subA;
    torch::Tensor select_index;
    std::tie(subA, select_index) = A.Sampling(1, fanout, true, _CSC, _CSC);
    auto csc_ptr = subA->GetCSC();
    EXPECT_TRUE(csc_ptr->indptr.equal(torch::arange(0, 21, options) * 4));
    EXPECT_FALSE(csc_ptr->indices.equal(indices));
    EXPECT_TRUE(csc_ptr->indices.numel() == (indptr.numel() - 1) * fanout);
    ASSERT_TRUE(std::get<0>(torch::_unique(csc_ptr->indices)).numel() <= 100);
}
